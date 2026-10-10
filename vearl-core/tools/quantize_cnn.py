#!/usr/bin/env python3
"""'Bring your own CNN': converts a floating-point convolutional network (convolutions of any stride and padding: token `conv/s2/p0`; by default stride 1 and padding kh//2; bias, folded BatchNorm, ReLU / GELU / tanh...,
max or mean pooling, flattening, dense layers) into a verifiable int8 VEARL PROGRAM, with calibration of the shifts and an accuracy report.

Layout (docs/07 section 12): an n x n tensor puts the channel in rows and (y, x, sample) in columns: a job processes B = n // (h * w)
images at a time (28 x 28 on n = 1024: B = 1; 8 x 8 on n = 128: B = 2; 16 x 16 on n = 1024: B = 4). Convolution = Im2col then Gemm
(the weights are (c_out, c_in * kh * kw)) then Bias; pooling = Pool op (Avg or Max); Flatten moves on to the dense classifier.

Limits: c_in * kh * kw <= n; c_out <= n; h * w * B <= n at the input of each convolution;
after flattening, dense layers of width <= n; h and w divisible by the pooling window size; per-tensor quantisation.

    python3 quantize_cnn.py selftest            # trains a PyTorch CNN (sklearn digits 8x8) with BatchNorm + max-pooling, converts it, compares
    python3 quantize_cnn.py torch state.pt images.npy --shape 1,8,8 --arch "conv,bn,relu,maxpool2,conv,relu,maxpool2,flatten,linear" \\
                            [--labels y.npy] [--out model.json] [--n 128] [--act relu]
"""
import json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vearl_ref as R
from quantize import ACTS


def _conv_float(x, W, b, pad, stride=1):
    """x (B, ci, h, w), W (co, ci, kh, kw) -> (B, co, oh, ow), padding `pad`, stride `stride`."""
    B, ci, h, w = x.shape
    co, _, kh, kw = W.shape
    oh, ow = (h + 2 * pad - kh) // stride + 1, (w + 2 * pad - kw) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((B, co, oh, ow))
    for dy in range(kh):
        for dx in range(kw):
            out += np.einsum("bchw,oc->bohw", xp[:, :, dy:dy + (oh - 1) * stride + 1:stride, dx:dx + (ow - 1) * stride + 1:stride], W[:, :, dy, dx])
    return out + b[None, :, None, None]


def _pool_float(x, mode, k):
    B, c, h, w = x.shape
    a = x.reshape(B, c, h // k, k, w // k, k)
    return a.max(axis=(3, 5)) if mode == "max" else a.mean(axis=(3, 5))


def float_forward(layers, x, f_act):
    for L in layers:
        t = L["t"]
        if t == "conv":
            x = _conv_float(x, L["W"], L["b"], L.get("pad", L["W"].shape[2] // 2), L.get("stride", 1))
        elif t == "linear":
            x = x.reshape(len(x), -1) @ L["W"].T + L["b"]
        elif t == "pool":
            x = _pool_float(x, L["mode"], L["k"])
        elif t == "flatten":
            x = x.reshape(len(x), -1)
        if L.get("act"):
            x = f_act(x)
    return x


def quantize_cnn(layers, calib, n, labels=None, shape=None, act="relu", verbose=True, max_samples=8):
    """layers: list of dicts {"t": "conv", "W": (co, ci, kh, kw), "b": (co,), "act": bool} | {"t": "pool", "mode": "max"|"avg", "k": k} |
    {"t": "flatten"} | {"t": "linear", "W": (out, in), "b": (out,), "act": bool}. calib: (N, c, h, w) in [0, 1]."""
    c0, h0, w0 = shape or calib.shape[1:]
    B = n // (h0 * w0)
    if B < 1:
        raise ValueError(f"an image {h0}x{w0} does not fit in n = {n} columns")
    if calib.min() < 0 or calib.max() > 1:
        raise ValueError("the calibration images must be in [0, 1]")
    R.N = n
    f_act = ACTS[act]
    gemm_layers = [L for L in layers if L["t"] in ("conv", "linear")]
    sw = {id(L): np.abs(L["W"]).max() / 63.0 for L in gemm_layers}
    W_blocks, w_of = [], {}
    for L in gemm_layers:
        Wm = L["W"].reshape(L["W"].shape[0], -1)
        if Wm.shape[0] > n or Wm.shape[1] > n:
            raise ValueError(f"layer {L['t']} of shape {Wm.shape}: exceeds n = {n}")
        blk = np.zeros((n, n), np.int64)
        blk[:Wm.shape[0], :Wm.shape[1]] = np.clip(np.round(Wm / sw[id(L)]), -63, 63)
        w_of[id(L)] = len(W_blocks)
        W_blocks.append(blk)
    need_table = any(np.abs(L["b"]).max() > 0 for L in gemm_layers) or act != "relu"
    clip = [0]

    def build(shifts):
        """(operations, index of the Gemm operation per dense/conv layer, table or None, valid (rows, cols) of each Gemm)."""
        ops, gemm_at, valid = [], {}, {}
        table = np.zeros((n, n), np.int64) if need_table else None
        row, sx, c, h, w, cur, gi = 0, 1.0 / 64, c0, h0, w0, 0, 0
        clip[0] = 0
        nt = len(W_blocks)     # index of the table block
        for L in layers:
            t = L["t"]
            if t == "conv":
                co, ci, kh, kw = L["W"].shape
                if ci != c or ci * kh * kw > n or h * w * B > n:
                    raise ValueError(f"convolution inconsistent with the current tensor ({c} channels {h}x{w}, batch {B})")
                pad, stride = L.get("pad", kh // 2), L.get("stride", 1)
                if stride == 1 and pad == kh // 2 and kh == kw:
                    ops.append({"Map": {"x": cur, "map": {"Im2col": {"c_in": c, "h": h, "w": w, "batch": B, "kh": kh, "kw": kw, "pad": pad}}}})
                    oh, ow = h, w
                else:
                    ops.append({"Map": {"x": cur, "map": {"Conv": {"c_in": c, "h": h, "w": w, "batch": B, "kh": kh, "kw": kw, "pad": pad, "stride": stride}}}})
                    oh, ow = (h + 2 * pad - kh) // stride + 1, (w + 2 * pad - kw) // stride + 1
                cur = len(ops)
                rows, cols = co, oh * ow * B
                h, w = oh, ow
                c = co
            elif t == "linear":
                if L["W"].shape[1] != c:
                    raise ValueError(f"linear layer: {L['W'].shape[1]} inputs expected, the current tensor has {c}")
                rows, cols = L["W"].shape[0], B
            elif t == "pool":
                k = L["k"]
                if h % k or w % k:
                    raise ValueError("the pooling must divide h and w")
                ops.append({"Pool": {"x": cur, "pool": {"Max" if L["mode"] == "max" else "Avg": {"c": c, "h": h, "w": w, "batch": B, "k": k}}}})
                cur = len(ops)
                h, w = h // k, w // k
                continue
            elif t == "flatten":
                if c * h * w > n:
                    raise ValueError(f"flattening: {c * h * w} > n = {n}")
                ops.append({"Map": {"x": cur, "map": {"Flatten": {"c": c, "h": h, "w": w, "batch": B}}}})
                cur = len(ops)
                c, h, w = c * h * w, 1, 1
                continue
            # --- Gemm (+ Bias) (+ activation)
            has_bias = np.abs(L["b"]).max() > 0
            sa = sw[id(L)] * sx * 2.0 ** shifts[gi]
            fuse_relu = L.get("act") and act == "relu" and not has_bias
            ops.append({"Gemm": {"w": w_of[id(L)], "x": cur, "shift": shifts[gi], "relu": bool(fuse_relu)}})
            gemm_at[gi] = len(ops) - 1
            valid[gi] = (rows, cols)
            cur = len(ops)
            if has_bias:
                b = np.zeros(n)
                b[:len(L["b"])] = L["b"]
                bi = np.round(b / sa)
                clip[0] += int((np.abs(bi) > 64).sum())
                table[row, :] = np.clip(bi, -64, 64)
                ops.append({"Bias": {"x": cur, "w": nt, "row": row, "relu": bool(L.get("act") and act == "relu")}})
                row += 1
                cur = len(ops)
            sx_next = sa
            if L.get("act") and act != "relu":
                xs = (np.arange(128) - 64) * sa
                fy = f_act(xs)
                sy = max(np.abs(fy).max(), 1e-12) / 64.0
                table[row, :128] = np.clip(np.round(fy / sy), -64, 64)
                ops.append({"Lut": {"x": cur, "w": nt, "row": row}})
                row += 1
                cur = len(ops)
                sx_next = sy
            sx = sx_next
            if t == "linear":
                c = rows
            gi += 1
        return ops, gemm_at, table, valid

    G = len(gemm_layers)
    if len(build([0] * G)[0]) > 240:
        raise ValueError("more than 240 operations")
    if len(W_blocks) + (1 if need_table else 0) > 64:
        raise ValueError("more than 64 weight blocks")

    def inp(imgs):
        t = np.zeros((n, n), np.int64)
        arr = np.round(imgs * 64).astype(np.int64)                     # (b, c, h, w)
        for bb in range(len(imgs)):                                     # column = (y * w + x) * B + b
            cols = (np.arange(h0 * w0) * B + bb)
            t[:c0, cols] = arr[bb].reshape(c0, -1)
        return t

    shifts = [0] * G
    cal_batches = [calib[i * B:(i + 1) * B] for i in range(max(1, min(3, len(calib) // B)))]
    cal_inputs = [inp(x) for x in cal_batches]
    for gi, L in enumerate(gemm_layers):
        ops0, gemm_at, table0, valid = build(shifts)
        W_all = W_blocks + ([table0] if need_table else [])
        zs = []
        for x0 in cal_inputs:
            _, Z = R.run_program(x0, W_all, ops0)
            r, cnum = valid[gi]
            zs.append(np.abs(Z[gemm_at[gi]][:r, :cnum]).ravel())
        z99 = np.percentile(np.concatenate(zs), 99.9)
        if np.abs(L["b"]).max() > 0:
            sx_in = 1.0 / 64 if gi == 0 else _scale_after(gemm_layers, shifts, gi - 1, sw, act, f_act)
            z99 += np.abs(L["b"]).max() / (sw[id(L)] * sx_in)
        shifts[gi] = max(int(math.ceil(math.log2(max(z99, 1) / 64.0))), 0)
    ops, gemm_at, table, valid = build(shifts)
    W_all = W_blocks + ([table] if need_table else [])
    report = {"shifts": shifts, "ops": len(ops), "weights": len(W_all), "batch": B, "clipped_biases": clip[0]}
    if verbose:
        print(f"program: {len(ops)} operations of which {len(gemm_layers)} Gemm, {len(W_all)} weight blocks; batch of {B} image(s) per job; calibrated shifts: {shifts}"
              + (f" ; {clip[0]} saturated biases" if clip[0] else ""))
    classes = layers[-1]["W"].shape[0]
    samples = []
    if labels is not None:
        pred = []
        for i in range(0, len(calib) - len(calib) % B, B):
            T, _ = R.run_program(inp(calib[i:i + B]), W_all, ops)
            pred += list(T[-1][:classes, :B].argmax(0))
        m = len(pred)
        pred = np.array(pred)
        fp = float_forward(layers, calib[:m], f_act).argmax(1)
        report.update(float_acc=float((fp == labels[:m]).mean()), int8_acc=float((pred == labels[:m]).mean()), agreement=float((pred == fp).mean()), images=m)
        if verbose:
            print(f"FLOAT {100 * report['float_acc']:.2f} % | INT8 {100 * report['int8_acc']:.2f} % | agreement {100 * report['agreement']:.2f} % | loss {100 * (report['float_acc'] - report['int8_acc']):+.2f} points ({m} images)")
    for s in range(min(max_samples, len(calib) // B)):
        idx = slice(s * B, (s + 1) * B)
        samples.append({"x0": R.enc(inp(calib[idx])), "labels": [int(v) for v in labels[idx]] if labels is not None else []})
    model = {"name": "custom-cnn-int8", "n": n, "ops": ops, "weights": [R.enc(w) for w in W_all], "classes": int(classes), "batch": B, "input_scale": 1 / 64, "samples": samples}
    return model, report


def _scale_after(gemm_layers, shifts, gi, sw, act, f_act):
    """Scale of the activations at the output of the dense/conv layer of index `gi` (after the possible activation table)."""
    sx = 1.0 / 64
    for k, L in enumerate(gemm_layers[:gi + 1]):
        sa = sw[id(L)] * sx * 2.0 ** shifts[k]
        if L.get("act") and act != "relu":
            fy = f_act((np.arange(128) - 64) * sa)
            sx = max(np.abs(fy).max(), 1e-12) / 64.0
        else:
            sx = sa
    return sx


def cnn_from_state_dict(sd, arch, eps=1e-5):
    """arch: tokens separated by commas among conv | bn | linear | relu (or act) | maxpoolK | avgpoolK | flatten; the tensors are taken in
    the order of the state_dict (one key group per module with weights). BatchNorm is folded into the convolution / linear layer that precedes it."""
    groups = []
    for k, v in sd.items():
        mod, _, name = k.rpartition(".")
        if not groups or groups[-1][0] != mod:
            groups.append((mod, {}))
        groups[-1][1][name] = np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v, dtype=np.float64)
    gi, layers = 0, []
    for tok in [t.strip() for t in arch.split(",") if t.strip()]:
        if tok.split("/")[0] in ("conv", "linear"):
            opts = tok.split("/")
            tok = opts[0]
            g = groups[gi][1]
            gi += 1
            if (tok == "conv") != (g["weight"].ndim == 4):
                raise ValueError(f"token {tok} but module {groups[gi - 1][0]} has weights of shape {g['weight'].shape}")
            L = {"t": tok, "W": g["weight"], "b": g.get("bias", np.zeros(g["weight"].shape[0])), "act": False}
            for o in opts[1:]:                                   # conv/s2/p0: stride 2, padding 0
                if o[0] == "s":
                    L["stride"] = int(o[1:])
                elif o[0] == "p":
                    L["pad"] = int(o[1:])
                else:
                    raise ValueError(f"unknown option {o}")
            layers.append(L)
        elif tok == "bn":
            g = groups[gi][1]
            gi += 1
            sc = g.get("weight", 1.0) / np.sqrt(g["running_var"] + eps)
            L = layers[-1]
            L["W"] = L["W"] * sc.reshape(-1, *([1] * (L["W"].ndim - 1)))
            L["b"] = (L["b"] - g["running_mean"]) * sc + g.get("bias", 0.0)
        elif tok in ("relu", "act"):
            layers[-1]["act"] = True
        elif tok.startswith("maxpool") or tok.startswith("avgpool"):
            layers.append({"t": "pool", "mode": tok[:3], "k": int(tok[7:])})
        elif tok == "flatten":
            layers.append({"t": "flatten"})
        else:
            raise ValueError(f"unknown token {tok}")
    return layers


def selftest():
    import torch
    from sklearn.datasets import load_digits
    d = load_digits()
    x, y = (d.data / 16.0).reshape(-1, 1, 8, 8), d.target
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(x))
    xtr, ytr, xte, yte = x[idx[:1300]], y[idx[:1300]], x[idx[1300:]], y[idx[1300:]]
    torch.manual_seed(2)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.BatchNorm2d(8), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(8, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Flatten(), torch.nn.Linear(64, 10))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    X, Y = torch.tensor(xtr, dtype=torch.float32), torch.tensor(ytr)
    for ep in range(40):
        net.train()
        perm = torch.randperm(len(X))
        for s in range(0, len(X), 64):
            b = perm[s:s + 64]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(X[b]), Y[b]).backward()
            opt.step()
    net.eval()
    torch.save(net.state_dict(), "/tmp/selftest-cnn.pt")
    sd = torch.load("/tmp/selftest-cnn.pt", weights_only=True)
    layers = cnn_from_state_dict(sd, "conv,bn,relu,maxpool2,conv,relu,maxpool2,flatten,linear")
    with torch.no_grad():
        ref = net(torch.tensor(xte, dtype=torch.float32)).numpy()
    mine = float_forward(layers, xte, ACTS["relu"])
    assert np.abs(ref - mine).max() < 1e-3, np.abs(ref - mine).max()
    model, rep = quantize_cnn(layers, xte, 128, yte, max_samples=64)
    assert rep["float_acc"] > 0.9 and rep["int8_acc"] > rep["float_acc"] - 0.05, rep
    json.dump(model, open("/tmp/selftest-cnn.json", "w"))
    print(f"selftest CNN OK: exact BatchNorm folding (gap {np.abs(ref - mine).max():.1e}); {100 * rep['int8_acc']:.1f} % int8 against {100 * rep['float_acc']:.1f} % float; /tmp/selftest-cnn.json")


def selftest_stride():
    """CNN with convolution WITHOUT padding (8x8 -> 6x6) then convolution of STRIDE 2: general Conv op (n = 256, 4 images per job)."""
    import torch
    from sklearn.datasets import load_digits
    d = load_digits()
    x, y = (d.data / 16.0).reshape(-1, 1, 8, 8), d.target
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(x))
    xtr, ytr, xte, yte = x[idx[:1300]], y[idx[:1300]], x[idx[1300:]], y[idx[1300:]]
    torch.manual_seed(4)
    net = torch.nn.Sequential(torch.nn.Conv2d(1, 8, 3, padding=0), torch.nn.ReLU(), torch.nn.Conv2d(8, 16, 3, stride=2, padding=1), torch.nn.BatchNorm2d(16), torch.nn.ReLU(),
                              torch.nn.Flatten(), torch.nn.Linear(144, 10))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    X, Y = torch.tensor(xtr, dtype=torch.float32), torch.tensor(ytr)
    for ep in range(40):
        net.train()
        perm = torch.randperm(len(X))
        for s0 in range(0, len(X), 64):
            b = perm[s0:s0 + 64]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(X[b]), Y[b]).backward()
            opt.step()
    net.eval()
    torch.save(net.state_dict(), "/tmp/selftest-cnn-stride.pt")
    sd = torch.load("/tmp/selftest-cnn-stride.pt", weights_only=True)
    layers = cnn_from_state_dict(sd, "conv/p0,relu,conv/s2/p1,bn,relu,flatten,linear")
    with torch.no_grad():
        ref = net(torch.tensor(xte, dtype=torch.float32)).numpy()
    mine = float_forward(layers, xte, ACTS["relu"])
    assert np.abs(ref - mine).max() < 1e-3, np.abs(ref - mine).max()
    model, rep = quantize_cnn(layers, xte, 256, yte, max_samples=32)
    assert any("Conv" in o.get("Map", {}).get("map", {}) for o in model["ops"] if "Map" in o and isinstance(o["Map"]["map"], dict))
    assert rep["float_acc"] > 0.9 and rep["int8_acc"] > rep["float_acc"] - 0.05, rep
    json.dump(model, open("/tmp/selftest-cnn-stride.json", "w"))
    print(f"selftest CNN stride 2 / no padding OK: {100 * rep['int8_acc']:.1f} % int8 against {100 * rep['float_acc']:.1f} % float; /tmp/selftest-cnn-stride.json")


def main():
    a = sys.argv[1:]
    if a and a[0] == "selftest":
        selftest()
        return selftest_stride()
    if a and a[0] == "torch":
        import torch
        sd = torch.load(a[1], weights_only=True, map_location="cpu")
        arch = a[a.index("--arch") + 1]
        shape = tuple(int(v) for v in a[a.index("--shape") + 1].split(","))
        layers = cnn_from_state_dict(sd, arch)
        calib = np.load(a[2]).astype(np.float64)
        labels = np.load(a[a.index("--labels") + 1]) if "--labels" in a else None
        n = int(a[a.index("--n") + 1]) if "--n" in a else 1024
        act = a[a.index("--act") + 1] if "--act" in a else "relu"
        out = a[a.index("--out") + 1] if "--out" in a else "model.json"
        model, rep = quantize_cnn(layers, calib, n, labels, shape=shape, act=act)
        json.dump(model, open(out, "w"))
        json.dump(rep, open(out.replace(".json", ".report.json"), "w"), indent=1)
        print("model written:", out, os.path.getsize(out) // 1024, "Kio")
        return
    print(__doc__)


if __name__ == "__main__":
    main()
