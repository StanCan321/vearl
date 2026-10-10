#!/usr/bin/env python3
"""Trains a small CNN (pure numpy) on sklearn digits (8x8), quantises it to int8 (post-training) and exports it in
the VEARL model format. Measures the float -> int8 accuracy loss (independent reference engine).

    python3 train_digits.py OUT.json [--vearl-model path] [--export-samples K]   (K exported image pairs, default: the whole test set)
Architecture (without bias): conv3x3(1->8)+ReLU+avgpool2 -> conv3x3(8->16)+ReLU+avgpool2 -> flatten(64) -> dense(10).
"""
import json, subprocess, sys
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import vearl_ref as R

N = 128
rng = np.random.default_rng(7)


# ---------------- floating-point CNN ----------------

def im2col(a, pad=1):  # a (B,H,W,C) -> (B,H,W,C*9), order (c,dy,dx)
    B, H, W, C = a.shape
    ap = np.zeros((B, H + 2 * pad, W + 2 * pad, C), a.dtype)
    ap[:, pad:pad + H, pad:pad + W] = a
    cols = np.stack([ap[:, dy:dy + H, dx:dx + W, :] for dy in range(3) for dx in range(3)], axis=-1)  # (B,H,W,C,9)
    return cols.reshape(B, H, W, C * 9)


def col2im(d, C, pad=1):  # d (B,H,W,C*9) -> (B,H,W,C)
    B, H, W, _ = d.shape
    d = d.reshape(B, H, W, C, 9)
    ap = np.zeros((B, H + 2 * pad, W + 2 * pad, C))
    for i, (dy, dx) in enumerate([(a, b) for a in range(3) for b in range(3)]):
        ap[:, dy:dy + H, dx:dx + W, :] += d[..., i]
    return ap[:, pad:pad + H, pad:pad + W]


def pool(a):
    B, H, W, C = a.shape
    return a.reshape(B, H // 2, 2, W // 2, 2, C).mean(axis=(2, 4))


def unpool(d):
    return np.repeat(np.repeat(d, 2, axis=1), 2, axis=2) / 4.0


def forward(P, x):
    c1 = im2col(x); z1 = c1 @ P[0].T; a1 = np.maximum(z1, 0); p1 = pool(a1)
    c2 = im2col(p1); z2 = c2 @ P[1].T; a2 = np.maximum(z2, 0); p2 = pool(a2)
    f = p2.transpose(0, 3, 1, 2).reshape(len(x), -1)  # order (c,y,x)
    return (c1, z1, p1, c2, z2, p2, f), f @ P[2].T


def train(xtr, ytr, epochs=60, bs=32, lr=2e-3):
    P = [rng.normal(0, np.sqrt(2 / 9), (8, 9)), rng.normal(0, np.sqrt(2 / 72), (16, 72)), rng.normal(0, np.sqrt(1 / 64), (10, 64))]
    m = [np.zeros_like(p) for p in P]
    v = [np.zeros_like(p) for p in P]
    t = 0
    for ep in range(epochs):
        idx = rng.permutation(len(xtr))
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            x, y = xtr[b], ytr[b]
            (c1, z1, p1, c2, z2, p2, f), lg = forward(P, x)
            pr = np.exp(lg - lg.max(1, keepdims=True))
            pr /= pr.sum(1, keepdims=True)
            dl = pr.copy()
            dl[np.arange(len(b)), y] -= 1
            dl /= len(b)
            g2 = dl.T @ f
            df = (dl @ P[2]).reshape(len(b), 16, 2, 2).transpose(0, 2, 3, 1)  # (B,2,2,16)
            dz2 = unpool(df) * (z2 > 0)
            g1 = np.einsum('bhwo,bhwk->ok', dz2, c2)
            dp1 = col2im(dz2 @ P[1], 8)
            dz1 = unpool(dp1) * (z1 > 0)
            g0 = np.einsum('bhwo,bhwk->ok', dz1, c1)
            t += 1
            for i, g in enumerate([g0, g1, g2]):
                m[i] = .9 * m[i] + .1 * g
                v[i] = .999 * v[i] + .001 * g * g
                P[i] -= lr * (m[i] / (1 - .9 ** t)) / (np.sqrt(v[i] / (1 - .999 ** t)) + 1e-8)
    return P


# ---------------- quantisation and VEARL program ----------------

def pad_mat(a):
    o = np.zeros((N, N), np.int8)
    o[:a.shape[0], :a.shape[1]] = a
    return o


def make_ops(shifts):
    im = lambda x, c, h, w: {"Map": {"x": x, "map": {"Im2col": {"c_in": c, "h": h, "w": w, "batch": 2, "kh": 3, "kw": 3, "pad": 1}}}}
    pl = lambda x, c, h, w: {"Pool": {"x": x, "pool": {"Avg": {"c": c, "h": h, "w": w, "batch": 2, "k": 2}}}}
    return [im(0, 1, 8, 8), {"Gemm": {"w": 0, "x": 1, "shift": shifts[0], "relu": True}}, pl(2, 8, 8, 8),
            im(3, 8, 4, 4), {"Gemm": {"w": 1, "x": 4, "shift": shifts[1], "relu": True}}, pl(5, 16, 4, 4),
            {"Map": {"x": 6, "map": {"Flatten": {"c": 16, "h": 2, "w": 2, "batch": 2}}}},
            {"Gemm": {"w": 2, "x": 7, "shift": shifts[2], "relu": False}}]


def input_tensor(imgs):  # imgs (2,8,8) pixels 0..16 -> integer 4*pixel (scale 1/64: pixel/16 = int/64)
    x = np.zeros((N, N), np.int8)
    for b in range(len(imgs)):
        x[0, b:128:2] = np.round(imgs[b].reshape(-1) * 4).astype(np.int8)  # column = position*batch + b
    return x


def main():
    out = sys.argv[1]
    exe = sys.argv[sys.argv.index("--vearl-model") + 1] if "--vearl-model" in sys.argv else None
    d = load_digits()
    X = d.images.astype(np.float64)
    y = d.target
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=396, random_state=3, stratify=y)
    f = lambda a: (a / 16.0)[..., None]
    P = train(f(xtr), ytr)
    facc_tr = (forward(P, f(xtr))[1].argmax(1) == ytr).mean()
    facc = (forward(P, f(xte))[1].argmax(1) == yte).mean()
    print(f"float (float64): train accuracy {facc_tr*100:.2f} %, test {facc*100:.2f} % ({len(xte)} images)")

    Wq = []
    for p in P:
        sw = np.abs(p).max() / 63.0
        Wq.append(np.clip(np.round(p / sw), -63, 63).astype(np.int8))
    W = [pad_mat(w) for w in Wq]
    # calibration of the shifts on 150 pairs of training images: smallest shift such that the 99.9th percentile of |Z|>>shift <= 64
    shifts = [0, 0, 0]
    cal = [(xtr[2 * i], xtr[2 * i + 1]) for i in range(150)]
    g_ops = [1, 4, 7]
    shape = [(8, 128), (16, 32), (10, 2)]
    for L in range(3):
        zs = []
        for pair in cal:
            _, Z = R.run_program(input_tensor(pair), W, make_ops(shifts))
            zs.append(np.abs(Z[g_ops[L]][:shape[L][0], :shape[L][1]]).ravel())
        p999 = np.percentile(np.concatenate(zs), 99.9)
        shifts[L] = max(int(np.ceil(np.log2(max(p999, 1) / 64.0))), 0)
    print("calibrated requantisation shifts:", shifts)

    ops = make_ops(shifts)
    pred = []
    for i in range(0, len(xte), 2):
        T, _ = R.run_program(input_tensor((xte[i], xte[i + 1])), W, ops)
        lg = T[-1][:10, :2].astype(int)
        pred += [int(np.argmax(lg[:, b])) for b in range(2)]  # first maximum, like the Rust
    pred = np.array(pred)
    iacc = (pred == yte).mean()
    fpred = forward(P, f(xte))[1].argmax(1)
    print(f"int8 (integer VEARL program, reference engine): test accuracy {iacc*100:.2f} % ; agreement with the float {100*(pred==fpred).mean():.2f} % ; loss {100*(facc-iacc):+.2f} points")

    keep = int(sys.argv[sys.argv.index("--export-samples") + 1]) * 2 if "--export-samples" in sys.argv else len(xte)
    samples = [{"x0": R.enc(input_tensor((xte[i], xte[i + 1]))), "labels": [int(yte[i]), int(yte[i + 1])]} for i in range(0, keep, 2)]
    model = {"name": "digits-cnn-int8", "n": N, "ops": ops, "weights": [R.enc(w) for w in W], "classes": 10, "batch": 2,
             "input_scale": 1 / 64, "samples": samples}
    json.dump(model, open(out, "w"))
    report = {"float_test_acc": float(facc), "int8_test_acc": float(iacc), "agreement": float((pred == fpred).mean()), "shifts": shifts, "test_images": int(len(xte))}
    json.dump(report, open(out.replace(".json", ".report.json"), "w"), indent=1)
    if exe:
        r = subprocess.run([exe, "eval", out], capture_output=True, text=True)
        print(r.stdout.strip(), r.stderr.strip())
        m = int((pred[:keep] == yte[:keep]).sum())
        assert f"{m}/{keep}" in r.stdout, "the Rust node and the reference engine diverge in accuracy!"
        print("=> same results as the Rust node")


if __name__ == "__main__":
    main()
