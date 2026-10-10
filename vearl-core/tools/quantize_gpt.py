#!/usr/bin/env python3
"""Small LANGUAGE MODEL (GPT type, normalisation BEFORE each block) converted into integer VEARL programs. By default
roneneldan/TinyStories-33M (MIT licence; GPT-Neo: 4 layers, 16 heads of 48, d = 768, FFN 3072 GELU, vocabulary 50,257; it writes
short stories in English). The dimensions are read from the configuration (gpt_neo or gpt2).

EXACT transformations of the floating-point model before quantisation (same function computed, verified to ~1e-9):
  1. zero-mean residual stream: the matrices that write into it are centred (LayerNorm becomes an RMS normalisation);
  2. gamma / beta of the LayerNorms folded into the matrices that read them (no more Mul / Bias after the LayerNorms);
  3. orthogonal rotation Q of the residual stream (Q 1 = 1, so the zero mean is kept): spreads out the outlier dimensions;
  4. shift of the keys PER QUERY (b_k -= u, u fitted by least squares): q.u is constant over the keys of a query, the softmax
     does not change, but the maximum score of each query falls under the saturation of the integer scores (64 steps of ln2/8);
  5. output head centred on the vocabulary (logits shifted by a constant per position: softmax unchanged).
Without training (probe tools/llm_probe.py), the simulated integer model keeps 85 % of the float's next token (perplexity 4.90 against 4.46).

Layout: features x tokens, <= L = 64 tokens, one text per job; CAUSAL attention: the 16 heads in one product
(HeadSpread: 16 x 64 = 1024 columns), CausalSoftmax high + fine (probabilities in 1/4096), V.A, HeadGather. The input embedding
(word table + positions, already rotated and quantised: 8-bit integers) is read by the client. TWO jobs per generation step:
  BODY: the layers (one text; output: residual stream D x L, verified);
  HEAD: final LayerNorm + 50 blocks of 1024 vocabulary words (Gemm; the biases are in the product, via witness
          rows), up to S = 20 texts at a time: the last state of text s is placed by the client in column CB + s (CB = n - S); block b is brought back (Shift) to columns
          b*S .. b*S + S - 1 and the 50 blocks are added (disjoint columns): output = logits (1024 rows x 50*S columns).

    python3 quantize_gpt.py OUT.json [--model roneneldan/TinyStories-33M] [--calib 64] [--eval 64] [--diag]
"""
import json, math, os, sys, time, types
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vearl_ref as R
import quantize_bert as QB

arg = QB.arg
NAME = arg("--model", "roneneldan/TinyStories-33M")
N, L = 1024, 64
R.N = N
C_SOFTMAX = math.log(2) / 8
PCT = arg("--pct", 99.9, float)
S_TEXTS = 20                      # texts per HEAD job
CB = N - S_TEXTS                  # column of the first text in the head input
MARK = 64                         # value of the witness rows (head biases)
KMAX_TARGET = arg("--kmax-target", 4.0, float)
SEED_ROT = 0


def gelu_new(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


class Cfg:
    def __init__(self, c):
        self.kind = c.model_type
        self.D = c.hidden_size
        self.H = c.num_attention_heads if hasattr(c, "num_attention_heads") else c.num_heads
        self.NL = c.num_hidden_layers if hasattr(c, "num_hidden_layers") else c.num_layers
        self.FF = getattr(c, "intermediate_size", None) or getattr(c, "n_inner", None) or 4 * self.D
        self.DH = self.D // self.H
        self.V = c.vocab_size
        self.eps = c.layer_norm_epsilon
        self.NB = (self.V + N - 1) // N
        assert self.D <= N and self.H * L <= N and self.NB * S_TEXTS <= N, "dimensions out of the chain's bounds"


def hf_load():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tk = AutoTokenizer.from_pretrained(NAME)
    m = AutoModelForCausalLM.from_pretrained(NAME).eval()
    return tk, m


def raw_params(m, cfg):
    """Weights of the GPT in uniform form (y = W x + b, rows = outputs), 64-bit floats, BEFORE transformations."""
    sd = {k: v.detach().double().numpy() for k, v in m.state_dict().items()}
    D, p = cfg.D, "transformer."
    P = {"wte": sd[p + "wte.weight"], "wpe": sd[p + "wpe.weight"], "head": sd.get("lm_head.weight", sd[p + "wte.weight"]),
         "lnf.g": sd[p + "ln_f.weight"], "lnf.b": sd[p + "ln_f.bias"]}
    for l in range(cfg.NL):
        q = lambda n: sd[f"{p}h.{l}.{n}"]
        P[f"l{l}.ln1.g"], P[f"l{l}.ln1.b"] = q("ln_1.weight"), q("ln_1.bias")
        P[f"l{l}.ln2.g"], P[f"l{l}.ln2.b"] = q("ln_2.weight"), q("ln_2.bias")
        if cfg.kind == "gpt2":
            Wa, ba = q("attn.c_attn.weight").T, q("attn.c_attn.bias")
            sc = 1 / math.sqrt(cfg.DH)
            P[f"l{l}.Wq"], P[f"l{l}.bq"] = Wa[:D] * sc, ba[:D] * sc
            P[f"l{l}.Wk"], P[f"l{l}.bk"] = Wa[D:2 * D], ba[D:2 * D]
            P[f"l{l}.Wv"], P[f"l{l}.bv"] = Wa[2 * D:], ba[2 * D:]
            P[f"l{l}.Wo"], P[f"l{l}.bo"] = q("attn.c_proj.weight").T, q("attn.c_proj.bias")
            P[f"l{l}.W1"], P[f"l{l}.b1"] = q("mlp.c_fc.weight").T, q("mlp.c_fc.bias")
            P[f"l{l}.W2"], P[f"l{l}.b2"] = q("mlp.c_proj.weight").T, q("mlp.c_proj.bias")
        else:                                                       # GPT-Neo: q, k, v without bias, scores NOT divided by sqrt(dh)
            z = np.zeros(D)
            P[f"l{l}.Wq"], P[f"l{l}.bq"] = q("attn.attention.q_proj.weight"), z.copy()
            P[f"l{l}.Wk"], P[f"l{l}.bk"] = q("attn.attention.k_proj.weight"), z.copy()
            P[f"l{l}.Wv"], P[f"l{l}.bv"] = q("attn.attention.v_proj.weight"), z.copy()
            P[f"l{l}.Wo"], P[f"l{l}.bo"] = q("attn.attention.out_proj.weight"), q("attn.attention.out_proj.bias")
            P[f"l{l}.W1"], P[f"l{l}.b1"] = q("mlp.c_fc.weight"), q("mlp.c_fc.bias")
            P[f"l{l}.W2"], P[f"l{l}.b2"] = q("mlp.c_proj.weight"), q("mlp.c_proj.bias")
    return P


def ln_plain(x, eps):
    """LayerNorm WITHOUT gamma or beta (folded elsewhere), per column (layout features x tokens)."""
    return (x - x.mean(0, keepdims=True)) / np.sqrt(x.var(0, keepdims=True) + eps)


def ln_affine(x, g, b, eps):
    return ln_plain(x, eps) * g[:, None] + b[:, None]


def transform(P, cfg):
    """Transformations 1, 2, 3 and 5 (exact). Returns a new dictionary."""
    P = dict(P)
    D = cfg.D
    # 1. zero-mean stream
    P["wte"] = P["wte"] - P["wte"].mean(1, keepdims=True)
    P["wpe"] = P["wpe"] - P["wpe"].mean(1, keepdims=True)
    for l in range(cfg.NL):
        for w, b in (("Wo", "bo"), ("W2", "b2")):
            P[f"l{l}.{w}"] = P[f"l{l}.{w}"] - P[f"l{l}.{w}"].mean(0, keepdims=True)
            P[f"l{l}.{b}"] = P[f"l{l}.{b}"] - P[f"l{l}.{b}"].mean()
    # 2. gamma / beta replies
    for l in range(cfg.NL):
        g, b = P.pop(f"l{l}.ln1.g"), P.pop(f"l{l}.ln1.b")
        for w, bb in (("Wq", "bq"), ("Wk", "bk"), ("Wv", "bv")):
            P[f"l{l}.{bb}"] = P[f"l{l}.{bb}"] + P[f"l{l}.{w}"] @ b
            P[f"l{l}.{w}"] = P[f"l{l}.{w}"] * g[None, :]
        g, b = P.pop(f"l{l}.ln2.g"), P.pop(f"l{l}.ln2.b")
        P[f"l{l}.b1"] = P[f"l{l}.b1"] + P[f"l{l}.W1"] @ b
        P[f"l{l}.W1"] = P[f"l{l}.W1"] * g[None, :]
    g, b = P.pop("lnf.g"), P.pop("lnf.b")
    P["head_b"] = P["head"] @ b
    P["head"] = P["head"] * g[None, :]
    # 3. rotation Q (Q 1 = 1): Householder reflection that brings R e onto e
    rng = np.random.RandomState(SEED_ROT)
    Rm, _ = np.linalg.qr(rng.randn(D, D))
    e = np.ones(D) / math.sqrt(D)
    u = Rm @ e - e
    Q = (np.eye(D) - 2 * np.outer(u, u) / (u @ u)) @ Rm
    assert np.allclose(Q @ e, e, atol=1e-10)
    P["wte"], P["wpe"] = P["wte"] @ Q.T, P["wpe"] @ Q.T
    for l in range(cfg.NL):
        for w in ("Wq", "Wk", "Wv", "W1"):
            P[f"l{l}.{w}"] = P[f"l{l}.{w}"] @ Q.T
        for w, b in (("Wo", "bo"), ("W2", "b2")):
            P[f"l{l}.{w}"], P[f"l{l}.{b}"] = Q @ P[f"l{l}.{w}"], Q @ P[f"l{l}.{b}"]
    P["head"] = P["head"] @ Q.T
    # 5. head centred on the vocabulary
    P["head"] = P["head"] - P["head"].mean(0, keepdims=True)
    P["head_b"] = P["head_b"] - P["head_b"].mean()
    return P


def attn_float(q, k, v, H, DH):
    """Floating-point causal attention, layout (features, tokens); scores (keys, (head, query)) concatenated per head."""
    Lr = q.shape[1]
    ctx, scs = np.zeros_like(v), []
    fut = np.tril(np.ones((Lr, Lr), bool), -1)                        # [key, query]: key > query (future)
    for h in range(H):
        s = slice(h * DH, (h + 1) * DH)
        sc = k[s].T @ q[s]
        a = np.exp(np.where(fut, -np.inf, sc) - np.where(fut, -np.inf, sc).max(0, keepdims=True))
        a /= a.sum(0, keepdims=True)
        ctx[s] = v[s] @ a
        scs.append(sc)
    return ctx, np.concatenate(scs, 1)


def float_forward(P, cfg, ids, transformed=True):
    """Floating-point model (transformed: LayerNorm without affine); logits (V, Lr) and per-layer intermediates."""
    D, H, DH, FF = cfg.D, cfg.H, cfg.DH, cfg.FF
    Lr = len(ids)
    x = (P["wte"][ids] + P["wpe"][:Lr]).T
    inter = []
    for l in range(cfg.NL):
        g = lambda n: P[f"l{l}.{n}"]
        n1 = ln_plain(x, cfg.eps) if transformed else ln_affine(x, g("ln1.g"), g("ln1.b"), cfg.eps)
        q, k, v = (g(f"W{c}") @ n1 + g(f"b{c}")[:, None] for c in "qkv")
        ctx, sc = attn_float(q, k, v, H, DH)
        ao = g("Wo") @ ctx + g("bo")[:, None]
        y1 = x + ao
        n2 = ln_plain(y1, cfg.eps) if transformed else ln_affine(y1, g("ln2.g"), g("ln2.b"), cfg.eps)
        f1 = g("W1") @ n2 + g("b1")[:, None]
        gg = gelu_new(f1)
        W2 = g("W2")
        f2parts = [W2[:, lo:lo + N] @ gg[lo:lo + N] for lo in range(0, FF, N)]
        f2 = sum(f2parts) + g("b2")[:, None]
        y2 = y1 + f2
        inter.append(dict(x=x, n1=n1, q=q, k=k, v=v, sc=sc, ctx=ctx, ao=ao, y1=y1, n2=n2, f1=f1, g=gg, f2parts=f2parts, f2=f2, y2=y2))
        x = y2
    nf = ln_plain(x, cfg.eps) if transformed else ln_affine(x, P["lnf.g"], P["lnf.b"], cfg.eps)
    logits = P["head"] @ nf + (P["head_b"][:, None] if transformed else 0)
    return logits, inter


def kmax_offsets(P, cfg, calib_ids):
    """Transformation 4: b_k -= u per head, q.u ~ max_k score(q, k) - KMAX_TARGET (least squares on the calibration)."""
    D, H, DH = cfg.D, cfg.H, cfg.DH
    acc = [[[], []] for _ in range(cfg.NL * H)]
    for ids in calib_ids:
        _, inter = float_forward(P, cfg, ids)
        Lr = len(ids)
        fut = np.tril(np.ones((Lr, Lr), bool), -1)                    # [key, query]: key > query (future)
        for l, it in enumerate(inter):
            for h in range(H):
                sc = it["sc"][:, h * Lr:(h + 1) * Lr]
                m = np.where(fut, -np.inf, sc).max(0)                 # max score of each query
                acc[l * H + h][0].append(it["q"][h * DH:(h + 1) * DH].T)
                acc[l * H + h][1].append(m - KMAX_TARGET)
    P = dict(P)
    for l in range(cfg.NL):
        bk = P[f"l{l}.bk"].copy()
        for h in range(H):
            Qm, t = np.concatenate(acc[l * H + h][0]), np.concatenate(acc[l * H + h][1])
            u = np.linalg.lstsq(Qm, t, rcond=None)[0]
            bk[h * DH:(h + 1) * DH] -= u
        P[f"l{l}.bk"] = bk
    return P


def pct(arrs):
    return float(np.percentile(np.abs(np.concatenate([np.asarray(a).ravel() for a in arrs])), PCT))


def rescale(b, t, s_from, s_to):
    """Changes the scale of the residual stream (s_to >= s_from): Mul by g / 64 (integer g <= 64, shift 6). Returns (tensor, exact scale)."""
    g = int(round(64 * s_from / max(s_to, s_from)))
    g = min(64, max(1, g))
    if g == 64:
        return t, s_from
    return b.op({"Mul": {"x": t, "w": 0, "row": b.row(("const", g)), "shift": 6}}), s_from * 64 / g


def layer(b, P, cfg, calib, l, x, s_x, pl):
    D, H, DH, FF = cfg.D, cfg.H, cfg.DH, cfg.FF
    col = lambda key: [c[l][key] for c in calib]
    K = lambda n: f"l{l}.{n}"
    s_n = 1 / 16
    # attention
    tn1 = b.tag(b.op({"LayerNorm": {"x": x, "len": D}}), "n1", l, s_n)
    s_k = max(pct(col("k")), 1e-9) / 63
    s_v = max(pct(col("v")), 1e-9) / 63
    tk, s_k, pl["k"] = b.gemm_scaled(P, K("Wk"), (0, D), (0, D), tn1, s_n, s_k)
    tk = b.tag(b.bias(tk, K("bk"), 0, D, s_k), "k", l, s_k)
    tv, s_v, pl["v"] = b.gemm_scaled(P, K("Wv"), (0, D), (0, D), tn1, s_n, s_v)
    tv = b.tag(b.bias(tv, K("bv"), 0, D, s_v), "v", l, s_v)
    sq_min = max(pct(col("q")) / 63, 1e-9)
    sh_sc = max(0, math.floor(math.log2(C_SOFTMAX / (s_k * sq_min))))
    s_q_t = C_SOFTMAX * 2.0 ** (-sh_sc) / s_k
    tq, s_q, pl["q"] = b.gemm_scaled(P, K("Wq"), (0, D), (0, D), tn1, s_n, s_q_t)
    tq = b.tag(b.bias(tq, K("bq"), 0, D, s_q), "q", l, s_q)
    pl.update(sh_sc=sh_sc)
    tqs = b.op({"Map": {"x": tq, "map": {"HeadSpread": {"dh": DH, "heads": H, "l": L}}}})
    tsc = b.tag(b.op({"GemmAT": {"a": tk, "x": tqs, "shift": sh_sc, "relu": False}}), "sc", l, C_SOFTMAX, "sc")
    thi = b.tag(b.op({"CausalSoftmax": {"x": tsc, "len": "LEN", "l": L, "fine": False}}), "a", l, 1 / 64, "sc")
    tlo = b.op({"CausalSoftmax": {"x": tsc, "len": "LEN", "l": L, "fine": True}})
    tvt = b.op({"Map": {"x": tv, "map": "Transpose"}})
    zo = pct(col("ctx")) * 64 / s_v
    sho = max(0, math.ceil(math.log2(max(zo, 1) / 63)))
    t1 = b.op({"GemmAT": {"a": tvt, "x": thi, "shift": sho, "relu": False}})
    t2 = b.op({"GemmAT": {"a": tvt, "x": tlo, "shift": sho + 6, "relu": False}})
    tcx = b.op({"Add": {"a": t1, "b": t2, "relu": False}})
    tcx = b.op({"Bias": {"x": tcx, "w": 0, "row": b.row(("const", 1)), "relu": False}})     # recentres the two roundings by default
    tg = b.op({"Map": {"x": tcx, "map": {"HeadGather": {"dh": DH, "heads": H, "l": L}}}})
    s_ctx = s_v / 64 * 2.0 ** sho
    b.tag(tg, "ctx", l, s_ctx)
    s_a = max(pct(col("y1")), pct(col("ao"))) / 63
    xr, s_a = rescale(b, x, s_x, s_a)
    to, _, pl["o"] = b.gemm_scaled(P, K("Wo"), (0, D), (0, D), tg, s_ctx, s_a)
    to = b.tag(b.bias(to, K("bo"), 0, D, s_a), "ao", l, s_a)
    ty1 = b.tag(b.op({"Add": {"a": xr, "b": to, "relu": False}}), "y1", l, s_a)
    # FFN
    tn2 = b.tag(b.op({"LayerNorm": {"x": ty1, "len": D}}), "n2", l, s_n)
    s_b = max(pct(col("y2")), pct(col("f2")), max(pct([c[l]["f2parts"][k] for c in calib]) for k in range(len(calib[0][l]["f2parts"])))) / 63
    yr, s_b = rescale(b, ty1, s_a, s_b)
    parts = []
    for lo in range(0, FF, N):
        hi = min(FF, lo + N)
        s1 = pct([c[l]["f1"][lo:hi] for c in calib]) / 63
        t1, s1, _ = b.gemm_scaled(P, K("W1"), (lo, hi), (0, D), tn2, s_n, s1)
        t1 = b.tag(b.bias(t1, K("b1"), lo, hi, s1), "f1", l, s1, (lo, hi))
        sg = pct([c[l]["g"][lo:hi] for c in calib]) / 63
        lut = np.clip(np.round(gelu_new(np.arange(-64, 64) * s1) / sg), -64, 64)
        tl = b.tag(b.op({"Lut": {"x": t1, "w": 0, "row": b.row(("lut", lut, s1, sg))}}), "g", l, sg, (lo, hi))
        t2x, _, _ = b.gemm_scaled(P, K("W2"), (0, D), (lo, hi), tl, sg, s_b)
        parts.append(t2x)
    t2 = parts[0]
    for tp in parts[1:]:
        t2 = b.op({"Add": {"a": t2, "b": tp, "relu": False}})
    t2 = b.tag(b.bias(t2, K("b2"), 0, D, s_b), "f2", l, s_b)
    ty2 = b.tag(b.op({"Add": {"a": yr, "b": t2, "relu": False}}), "y2", l, s_b)
    pl.update(s_a=s_a, s_b=s_b, sho=sho)
    return ty2, s_b


def build(P, cfg, calib, calib_logits):
    """calib: floating-point intermediates (one per text). Returns (BODY job, HEAD job, plan)."""
    s0 = pct([c[0]["x"] for c in calib]) / 63
    plan = {"s0": s0, "layers": [], "L": L, "D": cfg.D, "V": cfg.V, "S": S_TEXTS, "CB": CB, "NB": cfg.NB, "MARK": MARK}
    body = QB.Builder(cfg.D)
    # the per-column operations do not read the raw job input: identity copy first
    x = body.op({"Map": {"x": 0, "map": {"Shift": {"dr": 0, "dc": 0}}}})
    s_x = s0
    for l in range(cfg.NL):
        pl = {}
        x, s_x = layer(body, P, cfg, calib, l, x, s_x, pl)
        plan["layers"].append(pl)
    plan["s_out"] = s_x
    # HEAD: the vocabulary biases go INSIDE the product (columns D..n-1 of the weight block, multiplied by MARK witness rows
    # that the client only puts in the columns of the texts): the empty columns stay exactly zero, the 50 blocks can be
    # laid side by side (Shift) and added without polluting each other.
    head = QB.Builder(cfg.D)
    t = head.op({"Map": {"x": 0, "map": {"Shift": {"dr": 0, "dc": 0}}}})
    tn = head.tag(head.op({"LayerNorm": {"x": t, "len": cfg.D}}), "nf", cfg.NL, 1 / 16)
    head.wrec.append(("sel",))                                        # identity on the witness rows D..n-1
    tm = head.op({"Gemm": {"w": len(head.wrec) - 1, "x": t, "shift": 0, "relu": False}})
    tc = head.op({"Add": {"a": tn, "b": tm, "relu": False}})
    s_lg = float(max(calib_logits)) / 63                              # calib_logits: maximal |logit| of each text
    blocks = []
    for bk in range(cfg.NB):
        rs = (bk * N, min(cfg.V, (bk + 1) * N))
        sw_nat = np.abs(P["head"][rs[0]:rs[1]]).max() / 63.0
        sh = max(0, math.floor(math.log2(s_lg * 16 / sw_nat)))
        sw = s_lg * 16 / 2.0 ** sh
        head.wrec.append(("headblk", "head", "head_b", rs, sw))
        tb = head.op({"Gemm": {"w": len(head.wrec) - 1, "x": tc, "shift": sh, "relu": False}})
        blocks.append(head.op({"Map": {"x": tb, "map": {"Shift": {"dr": 0, "dc": CB - bk * S_TEXTS}}}}))
    acc = blocks[0]
    for tb in blocks[1:]:
        acc = head.op({"Add": {"a": acc, "b": tb, "relu": False}})
    plan["s_lg"] = s_lg
    return body, head, plan


def embed_tables(P, plan):
    """INTEGER input tables of the client: rotated words and positions, at step s0 (the embedding of a token = E[id] + Pos[i], saturated)."""
    E = np.clip(np.round(P["wte"] / plan["s0"]), -64, 64).astype(np.int8)
    Pp = np.clip(np.round(P["wpe"][:L] / plan["s0"]), -64, 64).astype(np.int8)
    return E, Pp


def enc_input(E, Pp, ids):
    t = np.zeros((N, N), np.int64)
    Lr = len(ids)
    t[:E.shape[1], :Lr] = np.clip(E[ids].astype(np.int64) + Pp[:Lr].astype(np.int64), -64, 64).T
    return t


def head_input(states):
    """Input of the HEAD job: final state (D,) of text s in column CB + s, witness rows D..n-1 = MARK in those columns only."""
    t = np.zeros((N, N), np.int64)
    for s, v in enumerate(states):
        t[:len(v), CB + s] = v
        t[len(v):, CB + s] = MARK
    return t


def bias_split(t, k):
    """Integers t (m,) each spread over k values of [-63, 63] of the same sign, of exact sum t: (m, k)."""
    a, sg = np.abs(t), np.sign(t)
    q, r = a // 63, a % 63
    j = np.arange(k)[None, :]
    return np.where(j < q[:, None], 63, np.where(j == q[:, None], r[:, None], 0)) * sg[:, None]


def materialize(b, params):
    """Weight blocks of a job (tables, matrices, selection of the witness rows, head blocks with their biases)."""
    out = []
    for rec in b.wrec:
        if rec[0] == "sel":
            blk = np.zeros((N, N), np.int64)
            for r in range(b.D, N):
                blk[r, r] = 1
        elif rec[0] == "headblk":
            _, key, bkey, rs, sw = rec
            blk = np.zeros((N, N), np.int64)
            rows = rs[1] - rs[0]
            blk[:rows, :b.D] = np.clip(np.round(params[key][rs[0]:rs[1]] / sw), -63, 63)
            k = N - b.D
            t = np.clip(np.round(params[bkey][rs[0]:rs[1]] / (sw * MARK / 16)), -63 * k, 63 * k).astype(np.int64)
            blk[:rows, b.D:] = bias_split(t, k)
        else:
            blk = QB.materialize(types.SimpleNamespace(wrec=[rec], trec=b.trec), params)[0]
        out.append(blk)
    return out


def run_body(body, W, E, Pp, ids):
    T, _ = R.run_program(enc_input(E, Pp, ids), W, QB.program_for(body, len(ids)))
    return T


def run_head(head, W, states):
    T, _ = R.run_program(head_input(states), W, head.ops)
    return T


def logits_of(out, plan, s):
    """Integer logits (V,) of text s in the output of the HEAD job."""
    V, S = plan["V"], plan["S"]
    return np.concatenate([out[:, b * S + s] for b in range(plan["NB"])])[:V].astype(np.int64)


def run_int(body, head, plan, params, E, Pp, ids, Ws=None):
    """INDEPENDENT reference engine: integer logits of the next token (last token of ids)."""
    Wb, Wh = Ws or (materialize(body, params), materialize(head, params))
    T = run_body(body, Wb, E, Pp, ids)
    st = T[-1][:plan["D"], len(ids) - 1].astype(np.int64)
    Th = run_head(head, Wh, [st])
    return logits_of(Th[-1], plan, 0)


def export(out, body, head, plan, params, E, Pp, extra=None):
    model = {"name": NAME.split("/")[-1] + "-int8", "hf": NAME, "n": N, "L": L, "D": plan["D"], "V": plan["V"], "tokenizer": NAME,
             "plan": plan, "embed": R.enc(E), "embed_shape": list(E.shape),
             "pos": R.enc(Pp.astype(np.int64)), "pos_shape": list(Pp.shape),
             "body": {"ops_template": body.ops, "weights": [R.enc(w) for w in materialize(body, params)]},
             "head": {"ops": head.ops, "weights": [R.enc(w) for w in materialize(head, params)]}, **(extra or {})}
    json.dump(model, open(out, "w"))
    return os.path.getsize(out)


def load_texts(n_cal, n_ev, split="validation"):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split=split)
    return list(ds["text"][:n_cal]), list(ds["text"][n_cal:n_cal + n_ev])


def ids_of(tk, text):
    return tk(text)["input_ids"][:L]


def prepare(tk, m, calib_texts):
    """HF model -> transformed parameters (1-5) + BODY job, HEAD job, plan (calibration on calib_texts)."""
    cfg = Cfg(m.config)
    P0 = raw_params(m, cfg)
    P = transform(P0, cfg)
    cal_ids = [ids_of(tk, t) for t in calib_texts]
    P = kmax_offsets(P, cfg, cal_ids[:32])
    inter, lmax = [], []
    for ids in cal_ids:
        lg, it = float_forward(P, cfg, ids)
        inter.append(it)
        lmax.append(float(np.abs(lg).max()))
    body, head, plan = build(P, cfg, inter, lmax)
    return cfg, P0, P, body, head, plan


def main(out):
    t0 = time.time()
    tk, m = hf_load()
    cfg = Cfg(m.config)
    print(f"{NAME} ({cfg.kind}): {cfg.NL} layers, {cfg.H} heads of {cfg.DH}, d = {cfg.D}, FFN {cfg.FF}, vocabulary {cfg.V}, "
          f"{sum(p.numel() for p in m.parameters()) / 1e6:.1f} M parameters")
    cal, ev = load_texts(arg("--calib", 64, int), arg("--eval", 64, int))
    cfg, P0, P, body, head, plan = prepare(tk, m, cal)
    import torch
    ids = ids_of(tk, ev[0])
    with torch.no_grad():
        ref = m(torch.tensor([ids])).logits[0].double().numpy().T
    raw = float_forward(P0, cfg, ids, transformed=False)[0]
    mine = float_forward(P, cfg, ids)[0]
    cen = lambda a: a - a.mean(0, keepdims=True)
    print(f"floating-point reimplementation vs transformers: max gap {np.abs(raw - ref).max():.2e} ; after exact transformations "
          f"(centred logits): {np.abs(cen(mine) - cen(ref)).max():.2e}")
    for name, b in (("BODY", body), ("HEAD", head)):
        n_gemm = sum(1 for o in b.ops if next(iter(o)) in ("Gemm", "GemmAT"))
        print(f"job {name}: {len(b.ops)} operations of which {n_gemm} GEMM, {len(b.wrec)} weight blocks, {len(b.trec)} table rows")
    E, Pp = embed_tables(P, plan)
    Ws = (materialize(body, P), materialize(head, P))
    agree, nll, tot = 0, 0.0, 0
    for t in ev[:arg("--eval-int", 8, int)]:
        ids = ids_of(tk, t)
        fl = float_forward(P, cfg, ids)[0]
        for k in range(8, len(ids) - 1, 8):
            lg = run_int(body, head, plan, P, E, Pp, ids[:k + 1], Ws).astype(np.float64) * plan["s_lg"]
            agree += int(lg.argmax() == fl[:, k].argmax())
            lp = lg - lg.max() - math.log(np.exp(lg - lg.max()).sum())
            nll -= lp[ids[k + 1]]
            tot += 1
    print(f"reference integer engine (without training): next token identical to the float {100 * agree / tot:.1f} % ; "
          f"perplexity {math.exp(nll / tot):.2f} ({tot} positions) in {time.time() - t0:.0f} s")
    size = export(out, body, head, plan, P, E, Pp)
    print(f"written {out} ({size >> 20} MiB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/tmp/gpt-int8.json")
