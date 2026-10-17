#!/usr/bin/env python3
"""INDEPENDENT reference engine (numpy + Python integers) for the VEARL tensor programs.

It shares no code with the Rust node: the exp2 table is recomputed here in decimal arithmetic, the integer
root uses math.isqrt, the operations are vectorised with numpy using a different formulation. Its only purpose is
to verify that the specified integer semantics (docs/02, docs/05) is exact and reproducible outside the node's code.

        python3 vearl_ref.py fuzz [NB_PROGRAMS] [path of vearl-model]
"""
import base64, json, math, random, subprocess, sys, tempfile
from decimal import Decimal, getcontext
import numpy as np

N = 128
getcontext().prec = 90


def exp2_table():
    # floor(65536 * 2^(-d/8)), d = 0..128, in 90-digit decimal. At exact powers of 2 (d multiple of 8) the
    # decimal computation gives x - epsilon: a 1e-40 guard is added (the other values are irrational, at >1e-40 from an integer).
    ln2 = Decimal(2).ln()
    guard = Decimal(10) ** -40
    return [int((Decimal(65536) * (-(Decimal(d) * ln2) / 8).exp() + guard).to_integral_value(rounding="ROUND_FLOOR")) for d in range(129)]


EXP2 = exp2_table()


def enc(a):
    return base64.b64encode(np.asarray(a, dtype=np.int8).tobytes()).decode()


def dec(s, n=N):
    return np.frombuffer(base64.b64decode(s), dtype=np.int8).reshape(n, n).copy()


def _mm(a, b):
    """EXACT integer product via floating-point BLAS: |inputs| <= 64 and n <= 4096 => |sum| <= 2^24 << 2^53 (no rounding error)."""
    return np.rint(np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)).astype(np.int64)


def _act(z, shift, relu):
    v = z >> shift
    return np.clip(v, 0 if relu else -64, 64).astype(np.int8)


def _softmax(t, ln, scale=0):
    out = np.zeros(t.shape, dtype=np.int64)
    if ln == 0:
        return out.astype(np.int8)
    x = t[:ln].astype(np.int64)
    m = x.max(axis=0)
    e = np.array(EXP2, dtype=np.int64)[np.clip((m[None, :] - x) << scale, 0, 128)]
    s = e.sum(axis=0)
    out[:ln] = (e * 64 + s[None, :] // 2) // s[None, :]
    return out.astype(np.int8)


def _softmax_lo(t, ln, scale=0):
    # fine part: round(4096 p) - 64 round(64 p), same e and S as _softmax (in [-32, 32])
    out = np.zeros(t.shape, dtype=np.int64)
    if ln == 0:
        return out.astype(np.int8)
    x = t[:ln].astype(np.int64)
    m = x.max(axis=0)
    e = np.array(EXP2, dtype=np.int64)[np.clip((m[None, :] - x) << scale, 0, 128)]
    s = e.sum(axis=0)
    hi = (e * 64 + s[None, :] // 2) // s[None, :]
    fine = (e * 4096 + s[None, :] // 2) // s[None, :]
    out[:ln] = np.clip(fine - 64 * hi, -64, 64)
    return out.astype(np.int8)


def _causal_softmax(t, ln, l, fine, scale=0):
    # CAUSAL softmax: column j = query q = j % l; only its first min(ln, q + 1) cells count (the others: 0);
    # fine: fine part (like _softmax_lo), otherwise output of _softmax; scale: exponent step 2^scale (CausalSoftmaxS)
    eff = np.minimum(ln, np.arange(t.shape[1]) % max(l, 1) + 1)
    x = t.astype(np.int64)
    valid = np.arange(t.shape[0])[:, None] < eff[None, :]
    m = np.where(valid, x, -10 ** 9).max(axis=0)
    e = np.where(valid, np.array(EXP2, dtype=np.int64)[np.clip((m[None, :] - x) << scale, 0, 128)], 0)
    s = e.sum(axis=0)[None, :]
    hi = (e * 64 + s // 2) // s
    if not fine:
        return hi.astype(np.int8)
    return np.clip((e * 4096 + s // 2) // s - 64 * hi, -64, 64).astype(np.int8)


def _layernorm(t, ln):
    out = np.zeros(t.shape, dtype=np.int64)
    if ln == 0:
        return out.astype(np.int8)
    x = t[:ln].astype(np.int64)
    mu = (x.sum(axis=0) + ln // 2) // ln
    var = ((x - mu[None, :]) ** 2).sum(axis=0) // ln
    std = np.array([max(1, (math.isqrt(4 * (int(v) + 1)) + 1) // 2) for v in var], dtype=np.int64)
    out[:ln] = np.clip(((x - mu[None, :]) * 16 + (std // 2)[None, :]) // std[None, :], -64, 64)
    return out.astype(np.int8)


def _rmsnorm(t, ln, group, shift=0):
    # RMS normalisation of each column, in groups of `group` rows over the first `ln` rows (LayerNorm without the mean):
    # r = isqrt(256 sum(v^2) // group + 256) (= 16 sqrt(ms + 1), 4 fractional bits), out = clip((256 v + r // 2) // r, -64, 64); rows >= ln: 0
    out = np.zeros(t.shape, dtype=np.int64)
    g = min(group, ln)
    if ln == 0 or g == 0:
        return out.astype(np.int8)
    for s in range(0, ln - g + 1, g):
        x = t[s:s + g].astype(np.int64)
        ms = (256 * (x * x).sum(axis=0)) // g
        rms = np.array([math.isqrt(int(v) + 256) << shift for v in ms], dtype=np.int64)
        out[s:s + g] = np.clip((x * 256 + (rms // 2)[None, :]) // rms[None, :], -64, 64)
    return out.astype(np.int8)


def _im2col(t, c_in, h, w, batch, kh, kw, pad):
    # the output grid stays h x w; sources outside the image are 0 (margin M >= kh + pad)
    M = 8
    out = np.zeros((N, N), dtype=np.int8)
    a = t[:c_in, : h * w * batch].reshape(c_in, h, w, batch)
    ap = np.zeros((c_in, h + 2 * M, w + 2 * M, batch), dtype=np.int8)
    ap[:, M : M + h, M : M + w, :] = a
    for c in range(c_in):
        for dy in range(kh):
            for dx in range(kw):
                r = c * kh * kw + dy * kw + dx
                y0, x0 = M + dy - pad, M + dx - pad
                out[r, : h * w * batch] = ap[c, y0 : y0 + h, x0 : x0 + w, :].reshape(-1)
    return out


def _pool(t, c, h, w, batch, k, mode="Avg"):
    oh, ow = h // k, w // k
    out = np.zeros((N, N), dtype=np.int8)
    a = t[:c, : h * w * batch].astype(np.int64).reshape(c, h, w, batch)[:, : oh * k, : ow * k, :]
    if mode == "Max":
        out[:c, : oh * ow * batch] = a.reshape(c, oh, k, ow, k, batch).max(axis=(2, 4)).reshape(c, -1).astype(np.int8)
        return out
    s = a.reshape(c, oh, k, ow, k, batch).sum(axis=(2, 4))
    out[:c, : oh * ow * batch] = ((s + (k * k) // 2) // (k * k)).reshape(c, -1).astype(np.int8)
    return out


def _conv(t, c_in, h, w, batch, kh, kw, pad, stride):
    oh, ow = (h + 2 * pad - kh) // stride + 1, (w + 2 * pad - kw) // stride + 1
    out = np.zeros((N, N), dtype=np.int8)
    a = t[:c_in, : h * w * batch].reshape(c_in, h, w, batch)
    ap = np.zeros((c_in, h + 2 * pad, w + 2 * pad, batch), dtype=np.int8)
    ap[:, pad:pad + h, pad:pad + w, :] = a
    for c in range(c_in):
        for dy in range(kh):
            for dx in range(kw):
                r = c * kh * kw + dy * kw + dx
                out[r, : oh * ow * batch] = ap[c, dy: dy + (oh - 1) * stride + 1: stride, dx: dx + (ow - 1) * stride + 1: stride, :].reshape(-1)
    return out


def _flatten(t, c, h, w, batch):
    out = np.zeros((N, N), dtype=np.int8)
    a = t[:c, : h * w * batch].reshape(c, h * w, batch)  # (c, pos, b)
    out[: c * h * w, :batch] = a.reshape(c * h * w, batch)
    return out


def eval_op(op, T, W):
    ((kind, p),) = op.items()
    if kind == "Gemm":
        z = _mm(W[p["w"]], T[p["x"]])
        return z, _act(z, p["shift"], p["relu"])
    if kind == "GemmAT":
        z = _mm(T[p["a"]].T, T[p["x"]])
        return z, _act(z, p["shift"], p["relu"])
    if kind == "Add":
        v = T[p["a"]].astype(np.int32) + T[p["b"]].astype(np.int32)
        return None, np.clip(v, 0 if p["relu"] else -64, 64).astype(np.int8)
    if kind == "Lut":
        tab = W[p["w"]][p["row"]].astype(np.int8)
        return None, tab[np.clip(T[p["x"]].astype(np.int32), -64, 63) + 64].astype(np.int8)
    if kind == "Mul":
        # per-row multiplication (gamma of a LayerNorm): out[i][j] = sat((x[i][j] * g[i]) >> shift), arithmetic shift
        g = W[p["w"]][p["row"]].astype(np.int64)[:, None]
        return None, np.clip((T[p["x"]].astype(np.int64) * g) >> p["shift"], -64, 64).astype(np.int8)
    if kind == "Bias":
        v = T[p["x"]].astype(np.int32) + W[p["w"]][p["row"]].astype(np.int32)[:, None]
        return None, np.clip(v, 0 if p["relu"] else -64, 64).astype(np.int8)
    if kind == "Softmax":
        return None, _softmax(T[p["x"]], p["len"])
    if kind == "SoftmaxLo":
        return None, _softmax_lo(T[p["x"]], p["len"])
    if kind == "LayerNorm":
        return None, _layernorm(T[p["x"]], p["len"])
    if kind == "RmsNorm":
        return None, _rmsnorm(T[p["x"]], p["len"], p["group"], p["shift"])
    if kind == "MulT":
        # element-wise product of two tensors: sat((a * b) >> shift), arithmetic shift (SwiGLU gate)
        return None, np.clip((T[p["a"]].astype(np.int64) * T[p["b"]].astype(np.int64)) >> p["shift"], -64, 64).astype(np.int8)
    if kind == "MulW":
        # element-wise product of a tensor and a weight block: sat((x * W) >> shift) (rotary position embedding tables)
        return None, np.clip((T[p["x"]].astype(np.int64) * W[p["w"]].astype(np.int64)) >> p["shift"], -64, 64).astype(np.int8)
    if kind == "CausalSoftmax":
        return None, _causal_softmax(T[p["x"]], p["len"], p["l"], p["fine"])
    if kind == "CausalSoftmaxS":
        return None, _causal_softmax(T[p["x"]], p["len"], p["l"], p["fine"], p["scale"])
    if kind == "Map":
        m = p["map"]
        x = T[p["x"]]
        if m == "Transpose":
            return None, x.T.copy()
        ((mk, mp),) = m.items()
        if mk == "Shift":
            out = np.zeros((N, N), dtype=np.int8)
            dr, dc = mp["dr"], mp["dc"]
            out[: N - dr, : N - dc] = x[dr:, dc:]
            return None, out
        if mk == "HeadSpread":
            # MULTI-head attention in ONE product: head h (rows h*dh .. h*dh+dh-1) goes into the column block h*l .. h*l+l-1
            dh, H, l = mp["dh"], mp["heads"], mp["l"]
            out = np.zeros((N, N), dtype=np.int8)
            for h in range(H):
                if h * l + l <= N and h * dh + dh <= N:
                    out[h * dh:(h + 1) * dh, h * l:(h + 1) * l] = x[h * dh:(h + 1) * dh, :l]
            return None, out
        if mk == "HeadGather":
            # inverse: row i (head h = i // dh) takes back the columns of block h
            dh, H, l = mp["dh"], mp["heads"], mp["l"]
            out = np.zeros((N, N), dtype=np.int8)
            for h in range(H):
                if h * l + l <= N and h * dh + dh <= N:
                    out[h * dh:(h + 1) * dh, :l] = x[h * dh:(h + 1) * dh, h * l:(h + 1) * l]
            return None, out
        if mk == "Conv":
            return None, _conv(x, mp["c_in"], mp["h"], mp["w"], mp["batch"], mp["kh"], mp["kw"], mp["pad"], mp["stride"])
        if mk == "Im2col":
            return None, _im2col(x, mp["c_in"], mp["h"], mp["w"], mp["batch"], mp["kh"], mp["kw"], mp["pad"])
        if mk == "Flatten":
            return None, _flatten(x, mp["c"], mp["h"], mp["w"], mp["batch"])
    if kind == "Pool":
        ((pm, mp),) = p["pool"].items()
        return None, _pool(T[p["x"]], mp["c"], mp["h"], mp["w"], mp["batch"], mp["k"], pm)
    raise ValueError(op)


def run_program(x0, W, ops):
    T = [np.asarray(x0, dtype=np.int8)]
    Z = []
    for op in ops:
        z, o = eval_op(op, T, W)
        Z.append(z)
        T.append(o)
    return T, Z


# ---------------- conformance fuzz against `vearl-model dump` ----------------

def rand_prog(rng):
    """Valid random program: sequence of 3..10 operations, each reading existing tensors."""
    nprng = np.random.default_rng(rng.randrange(1 << 30))

    def rt(sparse=0.0):
        a = nprng.integers(-64, 65, (N, N)).astype(np.int8)
        if sparse:
            a[nprng.random((N, N)) < sparse] = 0
        return a

    nw = rng.randint(1, 3)
    W = [rt() for _ in range(nw)]
    x0 = rt(rng.choice([0, 0.3]))
    ops = []
    for k in range(rng.randint(3, 10)):
        t = lambda: rng.randint(0, len(ops))  # tensors 0..len(ops) exist before the next operation
        kind = rng.choice(["Gemm", "GemmAT", "Add", "Transpose", "Im2col", "Conv", "Shift", "Pool", "Flatten", "Softmax", "SoftmaxLo", "LayerNorm", "Bias", "Lut",
                           "Mul", "HeadSpread", "HeadGather", "CausalSoftmax", "RmsNorm", "MulT", "MulW", "CausalSoftmaxS"])
        if kind == "RmsNorm":
            if ops:
                ln = rng.choice([v for v in (1, 2, 7, 12, 32, 64, 100, 128) if v <= N])
                gr = rng.choice([g for g in range(1, ln + 1) if ln % g == 0])
                ops.append({"RmsNorm": {"x": max(1, t()), "len": ln, "group": gr, "shift": rng.randint(0, 4)}})
        elif kind == "MulT":
            ops.append({"MulT": {"a": t(), "b": t(), "shift": rng.randint(0, 12)}})
        elif kind == "MulW":
            ops.append({"MulW": {"x": t(), "w": rng.randrange(nw), "shift": rng.randint(0, 12)}})
        elif kind == "CausalSoftmax":
            if ops:
                ops.append({"CausalSoftmax": {"x": max(1, t()), "len": rng.choice([v for v in (1, 2, 7, 12, 64, 128) if v <= N]),
                                              "l": rng.choice([v for v in (1, 3, 16, 64, 100, N) if v <= N]), "fine": rng.random() < .5}})
        elif kind == "CausalSoftmaxS":
            if ops:
                ops.append({"CausalSoftmaxS": {"x": max(1, t()), "len": rng.choice([v for v in (1, 2, 7, 12, 64, 128) if v <= N]),
                                               "l": rng.choice([v for v in (1, 3, 16, 64, 100, N) if v <= N]), "fine": rng.random() < .5,
                                               "scale": rng.randint(0, 4)}})
        elif kind == "Mul":
            ops.append({"Mul": {"x": t(), "w": rng.randrange(nw), "row": rng.choice([0, 1, 5, N - 1]), "shift": rng.randint(0, 12)}})
        elif kind in ("HeadSpread", "HeadGather"):
            dh, H = rng.choice([(1, 1), (8, 4), (32, 3), (3, 7), (16, 8)])
            l = rng.randint(1, N // H)
            if H * dh <= N and H * l <= N:
                ops.append({"Map": {"x": t(), "map": {kind: {"dh": dh, "heads": H, "l": l}}}})
        elif kind == "Lut":
            ops.append({"Lut": {"x": t(), "w": rng.randrange(nw), "row": rng.choice([0, 3, N - 1])}})
        elif kind == "Bias":
            ops.append({"Bias": {"x": t(), "w": rng.randrange(nw), "row": rng.choice([0, 1, 5, N - 1]), "relu": rng.random() < .5}})
        elif kind == "Gemm":
            ops.append({"Gemm": {"w": rng.randrange(nw), "x": t(), "shift": rng.randint(0, 16), "relu": rng.random() < .5}})
        elif kind == "GemmAT":
            ops.append({"GemmAT": {"a": t(), "x": t(), "shift": rng.randint(0, 16), "relu": rng.random() < .5}})
        elif kind == "Add":
            ops.append({"Add": {"a": t(), "b": t(), "relu": rng.random() < .5}})
        elif kind == "Transpose":
            ops.append({"Map": {"x": t(), "map": "Transpose"}})
        elif kind == "Im2col":
            c_in, h, w, b = rng.randint(1, 4), rng.randint(2, 6), rng.randint(2, 6), rng.randint(1, 3)
            kh, kw = rng.randint(1, 3), rng.randint(1, 3)
            pad = rng.randint(0, kh + kw - 1)
            if c_in * kh * kw <= N and h * w * b <= N:
                ops.append({"Map": {"x": t(), "map": {"Im2col": {"c_in": c_in, "h": h, "w": w, "batch": b, "kh": kh, "kw": kw, "pad": pad}}}})
        elif kind == "Shift":
            ops.append({"Map": {"x": t(), "map": {"Shift": {"dr": rng.choice([0, 0, 1, 5, N - 1]), "dc": rng.choice([0, 3, 64, N - 1])}}}})
        elif kind == "Conv":
            c_in, h, w, b = rng.randint(1, 4), rng.randint(2, 7), rng.randint(2, 7), rng.randint(1, 3)
            kh, kw = rng.randint(1, 3), rng.randint(1, 3)
            pad, st = rng.randint(0, kh + kw - 1), rng.randint(1, 3)
            if h + 2 * pad >= kh and w + 2 * pad >= kw and c_in * kh * kw <= N and h * w * b <= N:
                oh, ow = (h + 2 * pad - kh) // st + 1, (w + 2 * pad - kw) // st + 1
                if oh * ow * b <= N:
                    ops.append({"Map": {"x": t(), "map": {"Conv": {"c_in": c_in, "h": h, "w": w, "batch": b, "kh": kh, "kw": kw, "pad": pad, "stride": st}}}})
        elif kind == "Pool":
            kk = rng.randint(1, 3)
            c, h, w, b = rng.randint(1, 16), kk * rng.randint(1, 8 // kk), kk * rng.randint(1, 8 // kk), rng.randint(1, 2)
            if h * w * b <= N:
                ops.append({"Pool": {"x": t(), "pool": {rng.choice(["Avg", "Max"]): {"c": c, "h": h, "w": w, "batch": b, "k": kk}}}})
        elif kind == "Flatten":
            c, h, w, b = rng.randint(1, 8), rng.randint(1, 4), rng.randint(1, 4), rng.randint(1, 4)
            if c * h * w <= N and h * w * b <= N:
                ops.append({"Map": {"x": t(), "map": {"Flatten": {"c": c, "h": h, "w": w, "batch": b}}}})
        elif kind in ("Softmax", "SoftmaxLo"):
            if ops:
                ops.append({kind: {"x": max(1, t()), "len": rng.choice([v for v in (1, 2, 7, 32, 100, 128, 197) if v <= N])}})
        elif ops:
            ops.append({"LayerNorm": {"x": max(1, t()), "len": rng.choice([1, 2, 7, 32, 100, 128])}})
    return x0, W, ops


def fuzz(count, vearl_model, seed=1):
    rng = random.Random(seed)
    tested = mism = 0
    kinds = {}
    for it in range(count):
        x0, W, ops = rand_prog(rng)
        if not ops:
            continue
        ref, _ = run_program(x0, W, ops)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"n": N, "ops": ops, "weights": [enc(w) for w in W], "x0": enc(x0)}, f)
            path = f.name
        r = subprocess.run([vearl_model, "dump", path], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"program {it}: refused by vearl-model: {r.stderr.strip()}; ops={ops}")
            mism += 1
            continue
        got = [dec(s) for s in json.loads(r.stdout)["tensors"]]
        tested += 1
        for op in ops:
            kinds[next(iter(op))] = kinds.get(next(iter(op)), 0) + 1
        if len(got) != len(ref) or any(not np.array_equal(a, b) for a, b in zip(got, ref)):
            mism += 1
            bad = next(i for i, (a, b) in enumerate(zip(got, ref)) if not np.array_equal(a, b))
            print(f"DIVERGENCE program {it}, tensor {bad}, op {ops[bad - 1] if bad else 'input'}")
    print(f"conformance: {tested} programs compared tensor by tensor, {mism} divergences; operations covered: {kinds}")
    return mism == 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "fuzz":
        cnt = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        exe = sys.argv[3] if len(sys.argv) > 3 else "../target/release/vearl-model"
        sys.exit(0 if fuzz(cnt, exe) else 1)
    print(__doc__)
