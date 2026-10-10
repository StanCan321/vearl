#!/usr/bin/env python3
"""ONNX import -> verifiable int8 VEARL program, for SEQUENTIAL networks: convolutions (symmetric stride and padding, groups = 1),
BatchNormalization (folded), Relu / Gelu / Tanh / Sigmoid (a single activation per network), MaxPool / AveragePool (window = stride, no
padding), Flatten, Gemm, MatMul + Add (dense layers with bias). Nothing else is accepted: a graph with branches (residuals),
attention, LayerNormalization, etc. is REFUSED with an explanation (it is not silently approximated).

    python3 import_onnx.py model.onnx calib.npy [--labels y.npy] [--out model.json] [--n 1024] [--shape 1,8,8]
    calib.npy: (N, C, H, W) or (N, D) in [0, 1]; the report compares float (onnx) / int8 if --labels is supplied.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantize as Q
import quantize_cnn as QC

ACT_OPS = {"Relu": "relu", "Gelu": "gelu", "Tanh": "tanh", "Sigmoid": "sigmoid", "Silu": "silu"}


def _attr(node, name, default=None):
    import onnx
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default


def load_layers(path):
    """(kind, layers, act, input_shape); kind = "cnn" | "mlp"."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(path)
    init = {t.name: numpy_helper.to_array(t).astype(np.float64) for t in m.graph.initializer}
    dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    layers, act, cur = [], None, m.graph.input[0].name
    nodes = list(m.graph.node)
    i = 0
    while i < len(nodes):
        nd = nodes[i]
        t = nd.op_type
        ins = [x for x in nd.input if x]
        if not ins or (ins[0] != cur and t not in ("Constant",)):
            raise ValueError(f"node {t} '{nd.name}': input {ins[0] if ins else '-'} is not the output of the previous node (non-sequential graph: branch / residual not supported)")
        if t == "Conv":
            if int(_attr(nd, "group", 1)) != 1 or any(int(d) != 1 for d in _attr(nd, "dilations", [1, 1])):
                raise ValueError("Conv: groups / dilations not supported")
            pads, strides = list(_attr(nd, "pads", [0, 0, 0, 0])), list(_attr(nd, "strides", [1, 1]))
            if len(set(pads)) != 1 or len(set(strides)) != 1:
                raise ValueError(f"Conv: padding {pads} / stride {strides} not symmetric")
            W = init[ins[1]]
            b = init[ins[2]] if len(ins) > 2 else np.zeros(W.shape[0])
            layers.append({"t": "conv", "W": W, "b": b, "act": False, "pad": int(pads[0]), "stride": int(strides[0])})
        elif t == "BatchNormalization":
            if not layers or layers[-1]["t"] not in ("conv", "linear"):
                raise ValueError("BatchNormalization must follow a convolution or a dense layer")
            eps = float(_attr(nd, "epsilon", 1e-5))
            scale, bias, mean, var = (init[x] for x in ins[1:5])
            sc = scale / np.sqrt(var + eps)
            L = layers[-1]
            L["W"] = L["W"] * sc.reshape(-1, *([1] * (L["W"].ndim - 1)))
            L["b"] = (L["b"] - mean) * sc + bias
        elif t in ACT_OPS:
            if not layers or layers[-1]["t"] not in ("conv", "linear") or layers[-1]["act"]:
                raise ValueError(f"{t} must follow a convolution or a dense layer")
            if act not in (None, ACT_OPS[t]):
                raise ValueError(f"a single activation per network ({act} and {ACT_OPS[t]} found)")
            act = ACT_OPS[t]
            layers[-1]["act"] = True
        elif t in ("MaxPool", "AveragePool"):
            ks, st = list(_attr(nd, "kernel_shape")), list(_attr(nd, "strides", None) or _attr(nd, "kernel_shape"))
            if len(set(ks)) != 1 or ks != st or any(int(p) != 0 for p in _attr(nd, "pads", [0, 0, 0, 0])):
                raise ValueError(f"{t}: square window, stride = window, no padding")
            layers.append({"t": "pool", "mode": "max" if t == "MaxPool" else "avg", "k": int(ks[0])})
        elif t in ("Flatten", "Reshape"):
            layers.append({"t": "flatten"})
        elif t == "Gemm":
            if float(_attr(nd, "alpha", 1.0)) != 1.0 or float(_attr(nd, "beta", 1.0)) != 1.0 or int(_attr(nd, "transA", 0)):
                raise ValueError("Gemm: alpha, beta, transA not supported")
            B = init[ins[1]]
            W = B if int(_attr(nd, "transB", 0)) else B.T
            b = init[ins[2]].reshape(-1) if len(ins) > 2 else np.zeros(W.shape[0])
            layers.append({"t": "linear", "W": W, "b": b, "act": False})
        elif t == "MatMul":
            if ins[1] not in init:
                raise ValueError("MatMul between two computed tensors: not supported")
            layers.append({"t": "linear", "W": init[ins[1]].T, "b": np.zeros(init[ins[1]].shape[1]), "act": False})
        elif t == "Add" and layers and layers[-1]["t"] == "linear" and len(ins) == 2 and ins[1] in init:
            layers[-1]["b"] = layers[-1]["b"] + init[ins[1]].reshape(-1)
        elif t in ("Identity", "Dropout"):
            pass
        else:
            raise ValueError(f"ONNX operator {t} not supported (node '{nd.name}')")
        cur = nd.output[0]
        i += 1
    if not layers or layers[-1]["t"] != "linear":
        raise ValueError("the network must end with a dense layer (scores)")
    kind = "cnn" if any(L["t"] in ("conv", "pool") for L in layers) else "mlp"
    return kind, layers, act or "relu", dims[1:]


def convert(path, calib, n, labels=None, shape=None):
    kind, layers, act, dims = load_layers(path)
    if kind == "cnn":
        model, rep = QC.quantize_cnn(layers, calib, n, labels, shape=tuple(shape or dims), act=act)
    else:
        layers = [L for L in layers if L["t"] == "linear"]
        model, rep = Q.quantize_mlp([L["W"] for L in layers], calib.reshape(len(calib), -1), n, labels, bs=[L["b"] for L in layers], act=act)
    return model, rep


def main():
    a = sys.argv[1:]
    if len(a) < 2:
        print(__doc__)
        return
    calib = np.load(a[1]).astype(np.float64)
    labels = np.load(a[a.index("--labels") + 1]) if "--labels" in a else None
    n = int(a[a.index("--n") + 1]) if "--n" in a else 1024
    shape = tuple(int(v) for v in a[a.index("--shape") + 1].split(",")) if "--shape" in a else None
    out = a[a.index("--out") + 1] if "--out" in a else "model.json"
    model, rep = convert(a[0], calib, n, labels, shape)
    json.dump(model, open(out, "w"))
    json.dump(rep, open(out.replace(".json", ".report.json"), "w"), indent=1)
    print("model written:", out, os.path.getsize(out) // 1024, "Kio")


def selftest():
    """Exports a PyTorch CNN (stride 2, BatchNorm, GELU) and an MLP (bias, tanh) to ONNX, imports them, compares with onnxruntime-free float, quantises."""
    import torch
    from sklearn.datasets import load_digits
    d = load_digits()
    x, y = (d.data / 16.0).reshape(-1, 1, 8, 8), d.target
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(x))
    xtr, ytr, xte, yte = x[idx[:1300]], y[idx[:1300]], x[idx[1300:]], y[idx[1300:]]
    cases = {
        "cnn": (torch.nn.Sequential(torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.BatchNorm2d(8), torch.nn.GELU(), torch.nn.MaxPool2d(2),
                                    torch.nn.Conv2d(8, 16, 3, stride=2, padding=1), torch.nn.GELU(), torch.nn.Flatten(), torch.nn.Linear(64, 10)), 256),
        "mlp": (torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 48), torch.nn.Tanh(), torch.nn.Linear(48, 10)), 128),
    }
    for name, (net, n) in cases.items():
        torch.manual_seed(1)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        X, Y = torch.tensor(xtr, dtype=torch.float32), torch.tensor(ytr)
        for ep in range(30):
            net.train()
            perm = torch.randperm(len(X))
            for s in range(0, len(X), 64):
                b = perm[s:s + 64]
                opt.zero_grad()
                torch.nn.functional.cross_entropy(net(X[b]), Y[b]).backward()
                opt.step()
        net.eval()
        path = f"/tmp/selftest-{name}.onnx"
        torch.onnx.export(net, torch.zeros(1, 1, 8, 8), path, dynamo=False, input_names=["x"])
        kind, layers, act, dims = load_layers(path)
        assert kind == name, kind
        with torch.no_grad():
            ref = net(torch.tensor(xte, dtype=torch.float32)).numpy()
        if kind == "cnn":
            mine = QC.float_forward(layers, xte, Q.ACTS[act])
        else:
            a_ = xte.reshape(len(xte), -1).T
            for i, L in enumerate([L for L in layers if L["t"] == "linear"]):
                a_ = L["W"] @ a_ + L["b"][:, None]
                if i < 1:
                    a_ = Q.ACTS[act](a_)
            mine = a_.T
        err = np.abs(ref - mine).max()
        assert err < 1e-3, (name, err)
        model, rep = convert(path, xte, n, yte)
        assert rep["float_acc"] > 0.9 and rep["int8_acc"] > rep["float_acc"] - 0.05, rep
        json.dump(model, open(f"/tmp/selftest-onnx-{name}.json", "w"))
        print(f"selftest ONNX {name} OK: exact import (gap {err:.1e}), activation {act},{100 * rep['int8_acc']:.1f} % int8 against {100 * rep['float_acc']:.1f} % float")
    # an unsupported graph is refused with an explanation
    import onnx
    from onnx import helper, TensorProto
    g = helper.make_graph([helper.make_node("LayerNormalization", ["x", "s", "b"], ["y"])], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])])
    onnx.save(helper.make_model(g), "/tmp/bad.onnx")
    try:
        load_layers("/tmp/bad.onnx")
        raise SystemExit("should have been refused")
    except ValueError as e:
        print("selftest ONNX: unsupported graph refused:", str(e)[:80])


if __name__ == "__main__":
    selftest() if sys.argv[1:2] == ["selftest"] else main()
