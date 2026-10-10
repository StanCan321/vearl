#!/usr/bin/env python3
"""int8 quantisation of the small embedding transformer (tools/train_tiny_transformer.py) into a verifiable VEARL PROGRAM (docs/12 section 6).

Topology (chain n = 128, layout features x tokens: one column = one token; ONE text of 32 tokens per job):
  T1 Gemm(Win)  T2 LayerNorm  T3 Q  T4 K  T5 V  T6 GemmAT(K, Q) = scores  T7 Softmax  T8 Transpose(V)  T9 GemmAT(V^T, A) = V.A  T10 Gemm(Wo)
  T11 Add (residual)  T12 LayerNorm  T13 Gemm(W1, ReLU)  T14 Gemm(W2)  T15 Add (residual)  T16 LayerNorm  T17 Transpose  T18 Gemm(row of 1) = sum of the tokens
  T19 Transpose  T20 Gemm(Wout) = embedding of 128 dimensions (column 0).
The SCALES are computed analytically from the real distributions (99.9th percentile): residuals at a common scale s_r (the Adds require
tensors at the same scale), output of Wo / W2 brought back to s_r by choosing the weight scale ("target"), scores put at the scale of the integer
softmax (exp2(-d/8): one step = ln2/8 of logit) by adjusting the weight scale of Q. No floating point in the program: only integers.

    python3 quantize_transformer.py WEIGHTS.npz OUT.json      # after train_tiny_transformer.py; fidelity and quality report
"""
import json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vearl_ref as R
from embed_text import tokens

N, L, D, FF, DOUT, LAYERS = 128, 32, 64, 128, 128, 1


def configure(P):
    """Dimensions read from the weights (meta = [d, ff, dout, layers, L]; absent: old 1-layer format 64/128/128)."""
    global N, L, D, FF, DOUT, LAYERS
    if "meta" in P:
        D, FF, DOUT, LAYERS, L = (int(v) for v in P["meta"])
    else:
        D, FF, DOUT, LAYERS, L = 64, 128, 128, 1, 32
    N = 128 * -(-max(D, FF, DOUT, L) // 128)
    R.N = N


def W(P, layer, name):
    k = f"blocks_{layer}_{name}_weight"
    return P[k] if k in P else P[f"{name}_weight"]
C_SOFTMAX = math.log(2) / 8            # one input step of the integer softmax = C_SOFTMAX of logit


def ln_np(x, eps=1e-3):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)


def softmax_np(s, axis):
    e = np.exp(s - s.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def float_forward(P, tok):
    """Real quantities of the floating-point model: `g` (final tensors) and `layers` (one entry per layer, same formulas as torch)."""
    g, layers = {}, []
    x0 = P["E_weight"][tok] + P["P"]
    r0 = x0 @ P["Win_weight"].T
    r = r0
    rs = [r0]
    for l in range(LAYERS):
        h = ln_np(r)
        q, k, v = h @ W(P, l, "Wq").T, h @ W(P, l, "Wk").T, h @ W(P, l, "Wv").T
        sc = np.einsum("bid,bjd->bij", k, q)                              # (B, key i, query j) = K^T Q
        a = softmax_np(sc / math.sqrt(D), axis=1)
        o = np.einsum("bdi,bij->bjd", v.transpose(0, 2, 1), a)           # (B, query j, D): V . A
        ao = o @ W(P, l, "Wo").T
        r = r + ao
        rs.append(r)
        h2 = ln_np(r)
        hid = np.maximum(h2 @ W(P, l, "W1").T, 0)
        fo = hid @ W(P, l, "W2").T
        r = r + fo
        rs.append(r)
        layers.append(dict(q=q, k=k, v=v, sc=sc, a=a, o=o, ao=ao, hid=hid, fo=fo))
    h3 = ln_np(r)
    pooled = h3.sum(1)
    out = pooled @ P["Wout_weight"].T
    g.update(x0=x0, r0=r0, rs=rs, pooled=pooled, out=out)
    return g, layers


def p999(x):
    return float(np.percentile(np.abs(x), 99.9))


class Builder:
    """Accumulates operations and weight blocks, tracking the scale of each tensor."""

    def __init__(self):
        self.ops, self.W, self.scale = [], [], {0: None}

    def weight(self, wf, sw):
        blk = np.zeros((N, N), np.int64)
        m = np.clip(np.round(wf / sw), -63, 63).astype(np.int64)
        blk[:m.shape[0], :m.shape[1]] = m
        self.W.append(blk)
        return len(self.W) - 1

    def gemm_free(self, wf, x, s_in, out_p999, relu=False):
        sw = np.abs(wf).max() / 63.0
        nat = sw * s_in
        sh = max(0, math.ceil(math.log2(out_p999 / (63 * nat)))) if out_p999 > 63 * nat else 0
        self.ops.append({"Gemm": {"w": self.weight(wf, sw), "x": x, "shift": sh, "relu": relu}})
        return len(self.ops), nat * 2.0 ** sh

    def gemm_target(self, wf, x, s_in, s_t):
        """Output at the TARGET scale s_t: integer shift, adjusted weight scale (>= natural: no clipping of the weights)."""
        sw_nat = np.abs(wf).max() / 63.0
        sh = math.floor(math.log2(s_t / (s_in * sw_nat)))
        if sh < 0:
            sh, s_t_eff = 0, s_in * sw_nat
        else:
            s_t_eff = s_t
        sw = s_t_eff / (s_in * 2.0 ** sh)
        self.ops.append({"Gemm": {"w": self.weight(wf, sw), "x": x, "shift": sh, "relu": False}})
        return len(self.ops), s_t_eff


def build(P, calib_tok):
    g, gl = float_forward(P, calib_tok)
    b = Builder()
    s0 = p999(g["x0"]) / 63
    s_r = max(p999(r) for r in g["rs"]) / 63
    sL = 1.0 / 16                                                          # output of the integer LayerNorm: 16 steps per standard deviation
    ln = lambda x: (b.ops.append({"LayerNorm": {"x": x, "len": D}}), len(b.ops))[1]
    r, _ = b.gemm_target(P["Win_weight"], 0, s0, s_r)                     # input projection, output at the scale of the residual stream
    info = dict(s0=s0, s_r=s_r)
    for l in range(LAYERS):
        f = gl[l]
        t2 = ln(r)
        # free K and V, Q adjusted so that the product of scales matches the integer softmax
        t4, s_k = b.gemm_free(W(P, l, "Wk"), t2, sL, p999(f["k"]))
        t5, s_v = b.gemm_free(W(P, l, "Wv"), t2, sL, p999(f["v"]))
        sq_min = max(p999(f["q"]) / 63, sL * np.abs(W(P, l, "Wq")).max() / 63)
        sh_sc = max(0, math.floor(math.log2(math.sqrt(D) * C_SOFTMAX / (s_k * sq_min))))
        s_q_t = math.sqrt(D) * C_SOFTMAX * 2.0 ** (-sh_sc) / s_k
        t3, s_q = b.gemm_target(W(P, l, "Wq"), t2, sL, s_q_t)
        b.ops.append({"GemmAT": {"a": t4, "x": t3, "shift": sh_sc, "relu": False}})       # scores K^T Q >> sh_sc (softmax steps)
        t6 = len(b.ops)
        b.ops.append({"Softmax": {"x": t6, "len": L}})
        t7 = len(b.ops)
        b.ops.append({"Map": {"x": t5, "map": "Transpose"}})
        t8 = len(b.ops)
        zo = p999(f["o"]) * 64 / s_v                                       # V . A: A in [0, 64] (scale 1/64)
        sho = max(0, math.ceil(math.log2(max(zo, 1) / 63)))
        b.ops.append({"GemmAT": {"a": t8, "x": t7, "shift": sho, "relu": False}})
        t9 = len(b.ops)
        s_o = s_v / 64 * 2.0 ** sho
        t10, _ = b.gemm_target(W(P, l, "Wo"), t9, s_o, s_r)
        b.ops.append({"Add": {"a": r, "b": t10, "relu": False}})
        r1 = len(b.ops)
        t12 = ln(r1)
        t13, s_h = b.gemm_free(W(P, l, "W1"), t12, sL, p999(f["hid"]), relu=True)
        t14, _ = b.gemm_target(W(P, l, "W2"), t13, s_h, s_r)
        b.ops.append({"Add": {"a": r1, "b": t14, "relu": False}})
        r = len(b.ops)
        info.update({f"sh_sc{l}": sh_sc, f"sho{l}": sho})
    t16 = ln(r)
    b.ops.append({"Map": {"x": t16, "map": "Transpose"}})
    t17 = len(b.ops)
    ones = np.zeros((1, L))
    ones[0, :] = 1.0                                                       # sum of the tokens: row of 1
    pooled_int = p999(g["pooled"]) * 16                                    # |sum of the integer LN outputs|
    shp = max(0, math.ceil(math.log2(max(pooled_int, 1) / 63)))
    b.W.append(np.pad(ones.astype(np.int64), ((0, N - 1), (0, N - L))))
    b.ops.append({"Gemm": {"w": len(b.W) - 1, "x": t17, "shift": shp, "relu": False}})
    t18 = len(b.ops)
    s_p = 2.0 ** shp / 16
    b.ops.append({"Map": {"x": t18, "map": "Transpose"}})
    t19 = len(b.ops)
    t20, s_out = b.gemm_free(P["Wout_weight"], t19, s_p, p999(g["out"]))
    info.update(shp=shp, s_out=s_out)
    return b, s0, info


def enc_input(P, tok, s0):
    x0 = P["E_weight"][tok] + P["P"]                                       # (L, D)
    t = np.zeros((N, N), np.int64)
    t[:D, :L] = np.clip(np.round(x0.T / s0), -63, 63).astype(np.int64)
    return t


def main(wpath, out):
    P = dict(np.load(wpath))
    configure(P)
    vocab = json.load(open(wpath.replace(".npz", ".vocab.json")))["vocab"]
    index = {w: i + 2 for i, w in enumerate(vocab)}
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news")
    te_t, te_y = ds["test"]["text"], np.array(ds["test"]["label"])
    tr_t = ds["train"]["text"][:2000]

    def toks(texts):
        o = np.zeros((len(texts), L), dtype=np.int64)
        for r, t in enumerate(texts):
            ids = [index.get(w, 1) for w in tokens(t)][:L]
            o[r, :len(ids)] = ids
        return o

    b, s0, info = build(P, toks(tr_t))
    print("program:", f"n = {N}, {LAYERS} layer(s), d = {D}, ff = {FF}, output {DOUT};", len(b.ops), "operations of which", sum(1 for o in b.ops if next(iter(o)) in ("Gemm", "GemmAT")), "GEMM,", len(b.W), "weight blocks;", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in info.items()})
    M = 2000
    tok_te = toks(te_t[:M])
    gf = float_forward(P, tok_te)[0]["out"]
    outs = []
    for r in range(M):
        T, _ = R.run_program(enc_input(P, tok_te[r], s0), b.W, b.ops)
        outs.append(T[-1][:DOUT, 0].astype(np.float64))
    Q = np.array(outs)
    cos = float(np.mean(np.sum(Q * gf, 1) / (np.linalg.norm(Q, axis=1) * np.linalg.norm(gf, axis=1) + 1e-9)))

    def nn_purity(E, y):
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        S = E @ E.T
        np.fill_diagonal(S, -2)
        return float((y[S.argmax(1)] == y).mean())

    rep = dict(cos_int8_vs_float=cos, nn_float=nn_purity(gf, te_y[:M]), nn_int8=nn_purity(Q, te_y[:M]), texts=M, **{k: (float(v) if isinstance(v, float) else int(v)) for k, v in info.items()})
    print(f"fidelity: int8/float cosine {cos:.3f}; nearest neighbour of the same category ({M} texts): float {100 * rep['nn_float']:.2f} %, int8 {100 * rep['nn_int8']:.2f} %")
    samples = [{"x0": R.enc(enc_input(P, tok_te[r], s0)), "labels": [int(te_y[r])]} for r in range(256)]
    model = {"name": "tiny-transformer-int8", "n": N, "ops": b.ops, "weights": [R.enc(w) for w in b.W], "classes": DOUT, "batch": 1, "input_scale": s0, "samples": samples}
    json.dump(model, open(out, "w"))
    json.dump(rep, open(out.replace(".json", ".report.json"), "w"), indent=1)
    np.savez_compressed(out.replace(".json", ".eval.npz"), int8=Q.astype(np.int8), flt=gf.astype(np.float32), labels=te_y[:M])
    print("model written:", out, os.path.getsize(out) // 1024, "KiB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
