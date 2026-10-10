#!/usr/bin/env python3
"""'Bring your own model': converts a floating-point MLP (weights of a model that YOU trained) into a verifiable int8 VEARL program,
with calibration of the shifts and a float / int8 accuracy report.

Input: a .npz file of matrices W0, W1, ... (shape (output, input), y = W x, NO bias), a .npy of calibration samples
(N, input) with values in [0, 1], and (optionally) a .npy of labels to measure the accuracy.
Each hidden layer is followed by a ReLU; the last layer gives the scores (classes).

    python3 quantize.py mlp weights.npz calib.npy --labels labels.npy --out model.json [--n 1024]
    python3 quantize.py selftest                       # checks the whole chain on an MLP trained on the fly (sklearn digits)

Export from PyTorch (without bias: Linear(bias=False)):
    np.savez("weights.npz", **{f"W{i}": l.weight.detach().numpy() for i, l in enumerate(linear_layers)})
The output is used with `vearl-cli prog-submit NODE WORKER KEY ESCROW SAMPLE_INDEX model:model.json` on a chain of the same width n.
Limits (docs/11): MLP only here; input and output <= n, wider hidden layers split into blocks (<= 240 operations,
64 weight blocks); per-tensor quantisation. BIASES supported (Bias ops); BatchNorm1d folded into the linear layer (PyTorch import).

    python3 quantize.py torch state.pt calib.npy --labels labels.npy --out model.json [--act relu|gelu|silu|tanh|sigmoid]   # state.pt = state_dict
    python3 quantize.py selftest-torch                  # trains a PyTorch MLP with bias + BatchNorm, converts it, compares
"""
import json, os, sys
import numpy as np
import vearl_ref as R


import math

ACTS = {
    "relu": lambda x: np.maximum(x, 0),
    "gelu": lambda x: 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2))),
    "silu": lambda x: x / (1 + np.exp(-x)),
    "tanh": np.tanh,
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
}


def quantize_mlp(Ws, calib, n, labels=None, batch=None, verbose=True, bs=None, act="relu"):
    """Ws: list of float matrices (out, in). calib: (N, in) in [0, 1]. Returns (model dict, report dict).
    bs: (optional) list of BIAS vectors (one per layer, length = output): each layer becomes Gemm (relu = false) then
    Bias{relu} (docs/07 section 12.2); the biases of all the layers are stored in ONE extra weight block (one row per output
    block). A bias is rounded to the activation scale of its layer and saturated to +-64 (the report counts the saturated biases).
    act: activation of the hidden layers, "relu" (fused into the Gemm) or "gelu" / "silu" / "tanh" / "sigmoid": Lut op, TABLE of 128
    entries (one row of the same weight block as the biases): table[k] = round(f((k - 64) * sa) / sy), sy = max|f| / 64 (output scale).

    A layer wider than the chain (width > n) is SPLIT INTO n x n BLOCKS: each output block i is the sum over k of the
    products W_ik . X_k (one Gemm per product, requantised), joined by Adds. The semantics is that of the PROGRAM (each partial
    product is requantised with the same shift, +ceil(log2(#products)) bits of margin so that the sum fits in [-64, 64]):
    it does not exactly equal a monolithic product, the calibration measures what this costs. The input must fit in a tensor
    (input <= n); so must the output (classes). At most 240 operations and 64 weight blocks."""
    R.N = n
    sizes = [Ws[0].shape[1]] + [w.shape[0] for w in Ws]
    # WIDE INPUT (more than n features): nin feature blocks laid SIDE BY SIDE in the columns of T_0 (block b: columns
    # [b*B, (b+1)*B)), B = n // nin samples per job; the Shift operation extracts each block (docs/12).
    nin = -(-sizes[0] // n)
    B = n // nin
    if B < 1:
        raise ValueError(f"input of {sizes[0]} features: too wide for n = {n}")
    batch = min(batch or min(n, len(calib)), B) if nin > 1 else (batch or min(n, len(calib)))
    if sizes[-1] > n:
        raise ValueError(f"the output ({sizes[-1]}) must fit in a tensor n={n}")
    if calib.min() < 0 or calib.max() > 1:
        raise ValueError("the calibration inputs must be in [0, 1] (the integer input is round(64 * x))")
    if bs is not None and len(bs) != len(Ws):
        raise ValueError("one bias vector per layer")
    if act not in ACTS:
        raise ValueError(f"unknown activation {act} ({', '.join(ACTS)})")
    f_act = ACTS[act]
    need_table = bs is not None or act != "relu"
    L = len(Ws)
    nb = lambda d: -(-d // n)
    sw = [np.abs(w).max() / 63.0 for w in Ws]                    # weight scale of each layer
    Wq = [np.clip(np.round(w / s), -63, 63).astype(np.int64) for w, s in zip(Ws, sw)]
    blocks = {}   # (layer, i, k) -> index in the weight list
    W = []
    for l, w in enumerate(Wq):
        for i in range(nb(w.shape[0])):
            for k in range(nb(w.shape[1])):
                sub = w[i * n:(i + 1) * n, k * n:(k + 1) * n]
                if not sub.any():
                    continue
                o = np.zeros((n, n), np.int64)
                o[:sub.shape[0], :sub.shape[1]] = sub
                blocks[(l, i, k)] = len(W)
                W.append(o)
    n_table_rows = (sum(nb(w.shape[0]) for w in Ws) if bs is not None else 0) + (sum(nb(w.shape[0]) for w in Ws[:-1]) if act != "relu" else 0)
    if len(W) + (1 if need_table else 0) > 64:
        raise ValueError(f"{len(W) + (1 if need_table else 0)} weight blocks (max 64 per program)")
    if n_table_rows > n:
        raise ValueError("too many output blocks for the bias / activation table")
    clipped = [0]

    def build(shifts):
        """(operations, Gemm indices per layer, bias table or None) for these shifts. Activation scale: x0 = 1/64,
        then sa_l = sw_l * sx_l * 2^shift_l; the integer bias of layer l is round(b_l / sa_l)."""
        ops, gemm_ops, act_t = [], [[] for _ in range(L)], [0]
        for b in range(1, nin):
            ops.append({"Map": {"x": 0, "map": {"Shift": {"dr": 0, "dc": b * B}}}})
            act_t.append(len(ops))
        bias_w = np.zeros((n, n), np.int64) if need_table else None
        row, sx = 0, 1.0 / 64
        clipped[0] = 0
        for l in range(L):
            hidden = l < L - 1
            sa = sw[l] * sx * 2.0 ** shifts[l]
            new = []
            for i in range(nb(Wq[l].shape[0])):
                parts = []
                ks = [k for k in range(nb(Wq[l].shape[1])) if (l, i, k) in blocks]
                fuse_relu = hidden and len(ks) == 1 and bs is None and act == "relu"
                for k in ks:
                    ops.append({"Gemm": {"w": blocks[(l, i, k)], "x": act_t[k], "shift": shifts[l], "relu": fuse_relu}})
                    gemm_ops[l].append(len(ops) - 1)
                    parts.append(len(ops))
                acc = parts[0]
                for j, pt in enumerate(parts[1:], 1):
                    ops.append({"Add": {"a": acc, "b": pt, "relu": hidden and j == len(parts) - 1 and bs is None and act == "relu"}})
                    acc = len(ops)
                if bs is not None:
                    b = np.zeros(n)
                    seg = bs[l][i * n:(i + 1) * n]
                    b[:len(seg)] = seg
                    bi = np.round(b / sa)
                    clipped[0] += int((np.abs(bi) > 64).sum())
                    bias_w[row, :] = np.clip(bi, -64, 64).astype(np.int64)
                    ops.append({"Bias": {"x": acc, "w": len(W), "row": row, "relu": hidden and act == "relu"}})
                    row += 1
                    acc = len(ops)
                if hidden and act != "relu":
                    # activation table: input = integer v (sa units), output = f(v * sa) / sy, sy = output scale (full use of [-64, 64])
                    xs = (np.arange(128) - 64) * sa
                    fy = f_act(xs)
                    sy_ = max(np.abs(fy).max(), 1e-12) / 64.0
                    if not hasattr(build, "sy"):
                        build.sy = {}
                    build.sy[l] = sy_
                    bias_w[row, :128] = np.clip(np.round(fy / sy_), -64, 64).astype(np.int64)
                    ops.append({"Lut": {"x": acc, "w": len(W), "row": row}})
                    row += 1
                    acc = len(ops)
                new.append(acc)
            act_t = new
            sx = build.sy[l] if (hidden and act != "relu") else sa
        return ops, gemm_ops, bias_w

    n_ops = len(build([0] * L)[0])
    if n_ops > 240:
        raise ValueError(f"{n_ops} operations (max 240 per program)")

    def inp(x):
        t = np.zeros((n, n), np.int64)
        for b in range(nin):
            seg = x[:, b * n:(b + 1) * n]
            t[:seg.shape[1], b * B:b * B + len(x)] = np.round(seg.T * 64).astype(np.int64)
        return t

    def weights_of(bias_w):
        return W + ([bias_w] if bias_w is not None else [])

    shifts = [0] * L
    cal = [inp(calib[i * batch:(i + 1) * batch]) for i in range(max(1, min(2, len(calib) // batch)))]
    sx_run = 1.0 / 64
    for l in range(L):
        ops0, gemm_ops, bw0 = build(shifts)
        if l > 0 and act != "relu":
            sx_run = build.sy[l - 1]        # output scale of the table of the previous layer
        zs = []
        for x0 in cal:
            _, Z = R.run_program(x0, weights_of(bw0), ops0)
            for k in gemm_ops[l]:
                zs.append(np.abs(Z[k][:, :batch]).ravel())
        z99 = np.percentile(np.concatenate(zs), 99.9)
        if bs is not None:
            # the bias is added in z units: b / (sw * sx); the range to cover is that of z + bias
            z99 = z99 + np.abs(bs[l]).max() / (sw[l] * sx_run)
        per_out = len(gemm_ops[l]) / max(nb(Wq[l].shape[0]), 1)   # partial products per output block
        shifts[l] = max(int(np.ceil(np.log2(max(z99, 1) / 64.0))), 0) + int(np.ceil(np.log2(max(per_out, 1))))
        sx_run = sw[l] * sx_run * 2.0 ** shifts[l]
    ops, gemm_ops, bias_w = build(shifts)
    n_clipped = clipped[0]
    W_all = weights_of(bias_w)
    n_lut = sum(1 for o in ops if "Lut" in o)
    if verbose:
        extra = (f" and {sum(1 for o in ops if 'Bias' in o)} Bias" if bs is not None else "") + (f" and {n_lut} Lut ({act})" if n_lut else "")
        print(f"program: {len(ops)} operations of which {sum(len(g) for g in gemm_ops)} Gemm{extra}, {len(W_all)} weight blocks; calibrated shifts: {shifts}"
              + (f" ; {n_clipped} saturated biases" if n_clipped else ""))
    report = {"shifts": shifts, "layers": sizes, "ops": len(ops), "weights": len(W_all), "clipped_biases": n_clipped}
    samples = []
    if labels is not None:
        def fwd(x):
            a = x.T
            for i, w in enumerate(Ws):
                a = w @ a + (bs[i][:, None] if bs is not None else 0)
                if i < L - 1:
                    a = f_act(a)
            return a.argmax(0)

        pred = []
        for i in range(0, len(calib), batch):
            x = calib[i:i + batch]
            T, _ = R.run_program(inp(x), W_all, ops)
            pred += list(T[-1][:sizes[-1], :len(x)].argmax(0))
        pred = np.array(pred)
        fp = fwd(calib)
        report.update(float_acc=float((fp == labels).mean()), int8_acc=float((pred == labels).mean()), agreement=float((pred == fp).mean()), images=int(len(labels)))
        if verbose:
            print(f"FLOAT {100 * report['float_acc']:.2f} % | INT8 {100 * report['int8_acc']:.2f} % | agreement{100 * report['agreement']:.2f} % | loss{100 * (report['float_acc'] - report['int8_acc']):+.2f} points ({len(labels)} images)")
        samples = [{"x0": R.enc(inp(calib[:batch])), "labels": [int(v) for v in labels[:batch]]}]
    model = {"name": "custom-mlp-int8", "n": n, "ops": ops, "weights": [R.enc(w) for w in W_all], "classes": sizes[-1], "batch": batch, "input_scale": 1 / 64, "samples": samples}
    return model, report


def train_tiny(sizes, xtr, ytr, epochs=60, seed=0):
    """numpy MLP (ReLU, no bias): used for the self-verification tests."""
    rng = np.random.default_rng(seed)
    Ws = [rng.normal(0, np.sqrt(2 / a), (b, a)) for a, b in zip(sizes[:-1], sizes[1:])]
    m = [np.zeros_like(w) for w in Ws]
    v = [np.zeros_like(w) for w in Ws]
    t = 0
    for ep in range(epochs):
        for s in range(0, len(xtr), 64):
            xb, yb = xtr[s:s + 64], ytr[s:s + 64]
            acts = [xb.T]
            for i, w in enumerate(Ws):
                z = w @ acts[-1]
                acts.append(np.maximum(z, 0) if i < len(Ws) - 1 else z)
            pr = np.exp(acts[-1] - acts[-1].max(0))
            pr /= pr.sum(0)
            dl = pr.copy()
            dl[yb, np.arange(len(yb))] -= 1
            dl /= len(yb)
            t += 1
            for i in reversed(range(len(Ws))):
                g = dl @ acts[i].T
                if i:
                    dl = (Ws[i].T @ dl) * (acts[i] > 0)
                m[i] = .9 * m[i] + .1 * g
                v[i] = .999 * v[i] + .001 * g * g
                Ws[i] -= 2e-3 * (m[i] / (1 - .9 ** t)) / (np.sqrt(v[i] / (1 - .999 ** t)) + 1e-8)
    return Ws


def train_tiny_bias(sizes, xtr, ytr, epochs=60, seed=0):
    """numpy MLP WITH bias (ReLU): returns (weights, biases)."""
    rng = np.random.default_rng(seed)
    Ws = [rng.normal(0, np.sqrt(2 / a), (b, a)) for a, b in zip(sizes[:-1], sizes[1:])]
    Bs = [np.zeros(b) for b in sizes[1:]]
    P_ = Ws + Bs
    m = [np.zeros_like(p) for p in P_]
    v = [np.zeros_like(p) for p in P_]
    t = 0
    L = len(Ws)
    for ep in range(epochs):
        for s in range(0, len(xtr), 64):
            xb, yb = xtr[s:s + 64], ytr[s:s + 64]
            acts = [xb.T]
            for i in range(L):
                z = Ws[i] @ acts[-1] + Bs[i][:, None]
                acts.append(np.maximum(z, 0) if i < L - 1 else z)
            pr = np.exp(acts[-1] - acts[-1].max(0))
            pr /= pr.sum(0)
            dl = pr.copy()
            dl[yb, np.arange(len(yb))] -= 1
            dl /= len(yb)
            t += 1
            grads = [None] * (2 * L)
            for i in reversed(range(L)):
                grads[i] = dl @ acts[i].T
                grads[L + i] = dl.sum(1)
                if i:
                    dl = (Ws[i].T @ dl) * (acts[i] > 0)
            for j in range(2 * L):
                m[j] = .9 * m[j] + .1 * grads[j]
                v[j] = .999 * v[j] + .001 * grads[j] * grads[j]
                P_[j] -= 2e-3 * (m[j] / (1 - .9 ** t)) / (np.sqrt(v[j] / (1 - .999 ** t)) + 1e-8)
    return Ws, Bs


def selftest():
    from sklearn.datasets import load_digits
    d = load_digits()
    x, y = d.data / 16.0, d.target
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(x))
    xtr, ytr, xte, yte = x[idx[:1300]], y[idx[:1300]], x[idx[1300:]], y[idx[1300:]]
    # 1) layers narrower than the chain: one Gemm per layer
    Ws = train_tiny([64, 48, 10], xtr, ytr)
    model, rep = quantize_mlp(Ws, xte, 128, yte, batch=128)
    assert rep["float_acc"] > 0.9 and rep["int8_acc"] > rep["float_acc"] - 0.04, rep
    assert len(model["ops"]) == 2
    path = "/tmp/selftest-model.json"
    json.dump(model, open(path, "w"))
    # 2) hidden layer WIDER than the chain (200 > n=128): split into 128x128 blocks + Add
    Wb = train_tiny([64, 200, 10], xtr, ytr, epochs=40)
    modelb, repb = quantize_mlp(Wb, xte, 128, yte, batch=128)
    assert repb["ops"] > 2 and repb["int8_acc"] > repb["float_acc"] - 0.06, repb
    pathb = "/tmp/selftest-blocks.json"
    json.dump(modelb, open(pathb, "w"))
    # 3) MLP WITH BIASES (Gemm + Bias ops): y = W x + b, b learned (non-zero); one more weight block carries the bias vectors
    Wbi, Bbi = train_tiny_bias([64, 48, 10], xtr, ytr, epochs=40)
    assert max(np.abs(b).max() for b in Bbi) > 0.05, "the learned biases are non-trivial"
    modelbi, repbi = quantize_mlp(Wbi, xte, 128, yte, batch=128, bs=Bbi)
    assert any("Bias" in o for o in modelbi["ops"]) and repbi["int8_acc"] > repbi["float_acc"] - 0.05 and repbi["float_acc"] > 0.9, repbi
    pathbi = "/tmp/selftest-bias.json"
    json.dump(modelbi, open(pathbi, "w"))
    print(f"selftest OK; models written: {path} ({rep['ops']} ops), {pathb} ({repb['ops']} ops: hidden layer 200 > n=128 split into blocks) and {pathbi} ({repbi['ops']} ops of which Bias: {100 * repbi['int8_acc']:.1f} % int8 against {100 * repbi['float_acc']:.1f} % float, {repbi['clipped_biases']} saturated biases)")
    return path, rep


def mlp_from_state_dict(sd, eps=1e-5):
    """(weights, biases) of a PyTorch MLP (nn.Sequential of Linear / BatchNorm1d, ReLU between the linear layers, Dropout ignored) read from a
    state_dict. BatchNorm1d is FOLDED into the linear layer that precedes it: W' = g W, b' = g (b - mean) + beta, g = gamma / sqrt(var + eps).
    Refuses what is not an MLP (convolutions, ...) and says so. The keys are grouped by module, in the order of the state_dict."""
    groups = {}
    for k, v in sd.items():
        mod, _, name = k.rpartition(".")
        groups.setdefault(mod, {})[name] = np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v, dtype=np.float64)
    Ws, bs = [], []
    for mod, g in groups.items():
        if "running_mean" in g:
            if not Ws:
                raise ValueError(f"BatchNorm '{mod}' before any linear layer: not supported")
            gam, bet = g.get("weight", np.ones_like(g["running_mean"])), g.get("bias", np.zeros_like(g["running_mean"]))
            sc = gam / np.sqrt(g["running_var"] + eps)
            Ws[-1] = Ws[-1] * sc[:, None]
            bs[-1] = (bs[-1] - g["running_mean"]) * sc + bet
        elif "weight" in g and g["weight"].ndim == 2:
            Ws.append(g["weight"])
            bs.append(g.get("bias", np.zeros(g["weight"].shape[0])))
        elif "weight" in g:
            raise ValueError(f"module '{mod}': weights of shape {g['weight'].shape} (convolution?) not supported by this MLP import")
    if not Ws:
        raise ValueError("no linear layer found")
    return Ws, bs


def selftest_torch(act="relu"):
    """Trains a PyTorch MLP WITH bias and BatchNorm1d, exports it as a state_dict, converts it (BN folded) and compares."""
    import torch
    from sklearn.datasets import load_digits
    d = load_digits()
    x, y = d.data / 16.0, d.target
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(x))
    xtr, ytr, xte, yte = x[idx[:1300]], y[idx[:1300]], x[idx[1300:]], y[idx[1300:]]
    torch.manual_seed(1)
    net = torch.nn.Sequential(torch.nn.Linear(64, 48), torch.nn.BatchNorm1d(48), {"relu": torch.nn.ReLU(), "gelu": torch.nn.GELU(), "tanh": torch.nn.Tanh()}[act], torch.nn.Dropout(0.1), torch.nn.Linear(48, 10))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    X, Y = torch.tensor(xtr, dtype=torch.float32), torch.tensor(ytr)
    for ep in range(60):
        perm = torch.randperm(len(X))
        for s in range(0, len(X), 64):
            b = perm[s:s + 64]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(X[b]), Y[b]).backward()
            opt.step()
    net.eval()
    torch.save(net.state_dict(), f"/tmp/selftest-torch-{act}.pt")
    sd = torch.load(f"/tmp/selftest-torch-{act}.pt", weights_only=True)       # never an arbitrary pickle
    Ws, bs = mlp_from_state_dict(sd)
    assert len(Ws) == 2
    # the folding of the BatchNorm is exact (in float): same outputs as the PyTorch module in evaluation mode
    with torch.no_grad():
        ref = net(torch.tensor(xte, dtype=torch.float32)).numpy()
    a = ACTS[act](Ws[0] @ xte.T + bs[0][:, None])
    mine = (Ws[1] @ a + bs[1][:, None]).T
    assert np.abs(ref - mine).max() < 1e-3, np.abs(ref - mine).max()
    model, rep = quantize_mlp(Ws, xte, 128, yte, batch=128, bs=bs, act=act)
    assert rep["float_acc"] > 0.9 and rep["int8_acc"] > rep["float_acc"] - 0.05, rep
    json.dump(model, open("/tmp/selftest-torch.json" if act == "relu" else f"/tmp/selftest-torch-{act}.json", "w"))
    print(f"selftest torch ({act}) OK: BatchNorm folded exactly (max gap {np.abs(ref - mine).max():.1e}); {100 * rep['int8_acc']:.1f} % int8 against {100 * rep['float_acc']:.1f} % float; /tmp/selftest-torch.json")


def main():
    a = sys.argv[1:]
    if a and a[0] == "selftest":
        selftest()
        return
    if a and a[0] == "selftest-torch":
        for act in (a[1:] or ["relu", "gelu", "tanh"]):
            selftest_torch(act)
        return
    if a and a[0] == "torch":
        # python3 quantize.py torch state.pt calib.npy [--labels labels.npy] [--out model.json] [--n 1024]   (state_dict, never a pickle model)
        import torch
        sd = torch.load(a[1], weights_only=True, map_location="cpu")
        Ws, bs = mlp_from_state_dict(sd)
        calib = np.load(a[2]).astype(np.float64)
        labels = np.load(a[a.index("--labels") + 1]) if "--labels" in a else None
        n = int(a[a.index("--n") + 1]) if "--n" in a else 1024
        out = a[a.index("--out") + 1] if "--out" in a else "model.json"
        model, rep = quantize_mlp(Ws, calib, n, labels, bs=bs, act=a[a.index("--act") + 1] if "--act" in a else "relu")
        json.dump(model, open(out, "w"))
        json.dump(rep, open(out.replace(".json", ".report.json"), "w"), indent=1)
        print("model written:", out, os.path.getsize(out) // 1024, "Kio")
        return
    if len(a) < 3 or a[0] != "mlp":
        print(__doc__)
        sys.exit(2)
    z = np.load(a[1])
    Ws = [z[f"W{i}"].astype(np.float64) for i in range(len([k for k in z.files if k.startswith("W")]))]
    calib = np.load(a[2]).astype(np.float64)
    labels = np.load(a[a.index("--labels") + 1]) if "--labels" in a else None
    n = int(a[a.index("--n") + 1]) if "--n" in a else 1024
    out = a[a.index("--out") + 1] if "--out" in a else "model.json"
    model, rep = quantize_mlp(Ws, calib, n, labels)
    json.dump(model, open(out, "w"))
    json.dump(rep, open(out.replace(".json", ".report.json"), "w"), indent=1)
    print("model written:", out, os.path.getsize(out) // 1024, "Kio")


if __name__ == "__main__":
    main()
