#!/usr/bin/env python3
"""Qwen3 EMBEDDING model (decoder with RMSNorm, QK-norm, RoPE, GQA, SwiGLU; embedding = state of the last token, normalised) converted
into CHAINED integer VEARL programs. By default Qwen/Qwen3-Embedding-0.6B (Apache-2.0: 28 layers, d = 1024, 16 query heads and 8 key/value
heads of 128, FFN 3072, 0.6 B parameters). The dimensions are read from the model configuration: the same code converts the other sizes of
the family as long as d <= n, kv_heads x head_dim <= n.

Protocol operations used (docs/07 section 12): Gemm, GemmAT, Add, Bias, Mul, Lut, Transpose, HeadSpread, HeadGather, CausalSoftmax, and the
three operations added for this family: RmsNorm (normalisation without mean, also per head), MulT (SwiGLU gate), MulW (RoPE tables).

Layout (like tools/quantize_bert.py): features x tokens, one text per job chain, <= L = n / kv_heads tokens (128 for n = 1024).
Exact transformations of the weights (checked to 1e-9 by tools/test_quantize_qwen3.py), done once off-chain:
  * the gains of the two RMSNorms of each layer are folded into the matrices that read them (the RmsNorm operation has no gain);
  * the residual stream is rotated by a random-sign Hadamard matrix (spreads the outlier channels: every tensor becomes near-Gaussian);
  * the q / k norm gains are rescaled per channel pair (smoothing of the q.k product), the up / down projections per channel (smoothing of
    the SwiGLU product);
  * the SwiGLU product is rotated by a block Hadamard matrix (3 blocks of 1024) before the down projection.
The attention-sink token. Qwen models give a huge residual state to the FIRST token of the sequence, which would blow the scale of the
whole tensor. The text is therefore prefixed by ONE fixed token (<|endoftext|>), whose keys and values are the same in every text (causal
attention: the first token only sees itself). Its column of the residual stream is meaningless (zero at the input, garbage afterwards, never read),
and in each layer its key / value columns are overwritten by constants carried in two weight blocks (MulW mask + Gemm of a marker tensor
built at the start of the job by a constant Lut + Add): the sink is exact, and absent from the quantised residual stream. The embedding is the one of the prefixed text (cosine ~0.97 with the unprefixed one,
docs/12 section 4 nonies).
RoPE: q' = q.cos + rot(q).sin = MulW(q, cos table) + MulW(Gemm(signed permutation, q), sin table) (tables in 1/64, committed with the weights).
The q / k norm gains (Mul) and the SwiGLU gate (Lut of silu, MulT of silu(gate) and up) use the usual operations.

The input embedding (token table, rotation, quantisation) and the final step (RMS of the last token, inverse rotation, final gain, L2
normalisation) are computed by the client, off-chain. The layers are split into jobs (bounds: 240 operations, 64 weight blocks per job);
the output of a job (residual stream D x L, verified) is the input of the next one.

    python3 quantize_qwen3.py OUT.json [--model Qwen/Qwen3-Embedding-0.6B] [--calib 48] [--eval 12] [--texts FILE] [--pct 99.99] [--diag]
"""
import json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vearl_ref as R

arg = lambda name, default, cast=str: cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
NAME = "Qwen/Qwen3-Embedding-0.6B"
N = 1024
R.N = N
C_SOFTMAX = math.log(2) / 8
PCT = 99.99                 # percentile of |activation| that fixes a scale
PREFIX_ID = 151643          # <|endoftext|>: the attention-sink token put before every text
ACT_PCT = float(os.environ.get("ACT_PCT", 99.9))        # percentile of |activation| for the context and feed-forward tensors
RES_PCT = float(os.environ.get("RES_PCT", 99.9))        # ... for the residual stream
QK_MARGIN = float(os.environ.get("QK_MARGIN", 1.5))      # clip of the normalised queries / keys = QK_MARGIN x their 99.99 percentile
QK_RAW_PCT = float(os.environ.get("QK_RAW_PCT", 99.99))   # percentile of |q|, |k| before their RMS normalisation (only the direction matters there)
SCORE_PCT = 99.9            # percentile of |score - sink score| (nats) that the score tensor must span
OPS_MAX, BLOCKS_MAX = 240, 64
MAX_RMS_SHIFT = 4           # largest output shift of RmsNorm
MAX_T = 4                   # largest exponent step shift of CausalSoftmaxS


def set_n(n):
    """Tensor size (1024 on the public chain; the tests use 128)."""
    global N
    N = n
    R.N = n


# ---------------------------------------------------------------------------------------------------------------------------------
# configuration and parameters
# ---------------------------------------------------------------------------------------------------------------------------------
class Cfg:
    def __init__(self, D, FF, NH, NKV, HD, NL, eps=1e-6, theta=1e6):
        self.D, self.FF, self.NH, self.NKV, self.HD, self.NL, self.eps, self.theta = D, FF, NH, NKV, HD, NL, eps, theta
        assert NH % NKV == 0 and HD % 2 == 0, "unsupported attention shape"
        self.r = NH // NKV                         # query heads per key/value head

    @property
    def L(self):
        """Tokens per text (prefix included): the NKV heads side by side in a score tensor of n columns."""
        return N // self.NKV

    def check(self):
        assert self.D <= N and self.NKV * self.HD <= N, "dimensions out of the chain's bounds"
        assert self.L >= 3, "n too small"

    @staticmethod
    def from_hf(c):
        theta = getattr(c, "rope_theta", None)
        if theta is None:
            theta = c.rope_parameters["rope_theta"]
        hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        return Cfg(c.hidden_size, c.intermediate_size, c.num_attention_heads, c.num_key_value_heads, hd, c.num_hidden_layers, c.rms_norm_eps, float(theta))


def params_from_sd(sd, cfg, dtype=None):
    """Weights in the layout used here (original basis)."""
    f = (lambda a: np.asarray(a, dtype=dtype)) if dtype else (lambda a: np.asarray(a))
    P = {"embed": f(sd["embed_tokens.weight"]), "final": f(sd["norm.weight"]), "rot": None, "layers": []}
    for l in range(cfg.NL):
        k = lambda n: f(sd[f"layers.{l}.{n}"])
        P["layers"].append(dict(ln1=k("input_layernorm.weight"), ln2=k("post_attention_layernorm.weight"), q=k("self_attn.q_proj.weight"),
                                k=k("self_attn.k_proj.weight"), v=k("self_attn.v_proj.weight"), o=k("self_attn.o_proj.weight"),
                                qn=k("self_attn.q_norm.weight"), kn=k("self_attn.k_norm.weight"), g=k("mlp.gate_proj.weight"),
                                u=k("mlp.up_proj.weight"), d=k("mlp.down_proj.weight"), hb=None))
    return P


def load(model=None, dtype=np.float32):
    import torch
    from transformers import AutoTokenizer, AutoModel
    name = model or NAME
    tk = AutoTokenizer.from_pretrained(name)
    m = AutoModel.from_pretrained(name, torch_dtype=torch.float32).eval()
    cfg = Cfg.from_hf(m.config)
    sd = {k: v.detach().numpy() for k, v in m.state_dict().items()}
    return tk, params_from_sd(sd, cfg, dtype), cfg


def hadamard(n, seed):
    """Orthogonal n x n matrix: random-sign Hadamard when n is a power of two (entries +-1/sqrt(n)), random rotation otherwise."""
    rng = np.random.default_rng(seed)
    if n & (n - 1) == 0:
        H = np.ones((1, 1))
        while H.shape[0] < n:
            H = np.block([[H, H], [H, -H]])
        return H * rng.choice([-1.0, 1.0], n)[None, :] / math.sqrt(n)
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))[None, :]


def fold_rotate(P, cfg, seed=0):
    """Exact: gains of the RMSNorms folded into the matrices that read them, residual stream rotated by `rot` (x' = x @ rot)."""
    rot = hadamard(cfg.D, seed)
    dt = P["embed"].dtype
    Q = P["final"]
    out = {"embed": (P["embed"].astype(np.float64) @ rot).astype(dt), "final": Q, "rot": rot, "layers": []}
    for w in P["layers"]:
        n = dict(w)
        for key in ("q", "k", "v"):
            n[key] = ((w[key].astype(np.float64) * w["ln1"][None, :]) @ rot).astype(dt)
        for key in ("g", "u"):
            n[key] = ((w[key].astype(np.float64) * w["ln2"][None, :]) @ rot).astype(dt)
        n["o"] = (rot.T @ w["o"].astype(np.float64)).astype(dt)
        n["d"] = (rot.T @ w["d"].astype(np.float64)).astype(dt)
        n["ln1"], n["ln2"] = np.ones_like(w["ln1"]), np.ones_like(w["ln2"])
        out["layers"].append(n)
    return out


def smooth(P, cfg, mx, alpha_qk=0.5, alpha_swi=0.5, clip_qk=(1 / 16, 16.0), clip_swi=(1 / 8, 8.0)):
    """Exact per-channel rescalings: q.k through the gains of the q / k norms (one factor per rope pair, geometric mean over the heads)
    and silu(gate).up -> down_proj. `mx[(name, l)]`: maximum |value| per channel over the calibration tokens. The factors are bounded: a "dead"
    channel (silu(gate) ~ 0 for every token) would otherwise get an enormous up projection and spoil the scale of the whole tensor."""
    H2 = cfg.HD // 2
    out = dict(P)
    out["layers"] = []
    for l, w in enumerate(P["layers"]):
        n = dict(w)
        qm, km = mx[("q", l)], mx[("k", l)]                                  # (NH, HD), (NKV, HD)
        qg = qm.reshape(cfg.NKV, cfg.r, cfg.HD).max(1)
        qp = np.maximum(qg[:, :H2], qg[:, H2:])
        kp = np.maximum(km[:, :H2], km[:, H2:])
        s = (np.maximum(kp, 1e-6) / np.maximum(qp, 1e-6)) ** alpha_qk
        s = np.concatenate([s, s], 1)                                         # (NKV, HD)
        sl = np.clip(np.exp(np.log(s).mean(0)), *clip_qk)
        n["qn"] = (w["qn"] * sl).astype(w["qn"].dtype)
        n["kn"] = (w["kn"] / sl).astype(w["kn"].dtype)
        sw = np.maximum(mx[("swi", l)], 1e-6)
        wd = np.maximum(np.abs(w["d"]).max(0), 1e-6)
        c = np.clip((sw ** alpha_swi) / (wd ** (1 - alpha_swi)), *clip_swi) if alpha_swi is not None else np.ones_like(sw)
        n["u"] = (w["u"] / c[:, None]).astype(w["u"].dtype)
        n["d"] = (w["d"] * c[None, :]).astype(w["d"].dtype)
        out["layers"].append(n)
    return out


def reorder_ffn(P, cfg, mx):
    """Exact: the feed-forward channels are (1) normalised: row c of the up projection divided by r_c = max |u_c|, column c of the down
    projection multiplied by it (every channel of u then spans the same range), and (2) sorted by decreasing amplitude of the SwiGLU product
    into the tiles of N channels: each tile gets its own scales, so that a tile of large channels no longer rounds the small ones to zero.
    `mx[("swi", l)]`, `mx[("u", l)]`: maximum |value| per channel over the calibration tokens."""
    out = dict(P)
    out["layers"] = []
    for l, w in enumerate(P["layers"]):
        n = dict(w)
        r = np.maximum(mx[("u", l)], 1e-6)
        u0 = float(np.median(r))
        order = np.argsort(-mx[("swi", l)], kind="stable")
        n["u"] = (w["u"].astype(np.float64) / r[:, None] * u0)[order].astype(w["u"].dtype)
        n["d"] = (w["d"].astype(np.float64) * (r / u0)[None, :])[:, order].astype(w["d"].dtype)
        n["g"] = w["g"][order]
        out["layers"].append(n)
    return out


def block_hadamard(P, cfg, seed=1):
    """Exact: the product silu(gate).up is rotated by an N x N Hadamard block per full tile of the feed-forward width before the down
    projection (the down matrix absorbs the inverse). The partial last tile, if any, is left as is."""
    hb = hadamard(N, seed)
    out = dict(P)
    out["layers"] = []
    for w in P["layers"]:
        n = dict(w)
        d = w["d"].astype(np.float64).copy()
        for lo in range(0, cfg.FF - N + 1, N):
            d[:, lo:lo + N] = d[:, lo:lo + N] @ hb.T
        n["d"] = d.astype(w["d"].dtype)
        n["hb"] = hb
        out["layers"].append(n)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# floating-point model (also used for the statistics)
# ---------------------------------------------------------------------------------------------------------------------------------
def rms_rows(x, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)


def rope_tables(cfg, n):
    inv = 1.0 / (cfg.theta ** (np.arange(0, cfg.HD, 2) / cfg.HD))
    f = np.outer(np.arange(n), inv)
    emb = np.concatenate([f, f], -1)
    return np.cos(emb), np.sin(emb)


def rot_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], -1)


def forward(P, cfg, ids, rec=None, cap=None, attn_fn=None, qk_fn=None, qn_fn=None):
    """Embedding (D,) of the token sequence `ids` (prefix included). rec(name, layer, array of shape (tokens - 1, ...)): statistics of the
    tensors, without the first (sink) token. cap: dict that receives per-layer tensors, all tokens included."""
    ids = list(ids)
    n = len(ids)
    dt = P["embed"].dtype
    x = P["embed"][ids].astype(dt)
    cos, sin = rope_tables(cfg, n)
    cos, sin = cos[:, None, :].astype(dt), sin[:, None, :].astype(dt)
    mask = np.triu(np.full((n, n), -np.inf), 1)
    R_ = (lambda name, l, a: rec(name, l, a[1:])) if rec else (lambda *a: None)
    for l, w in enumerate(P["layers"]):
        R_("x", l, x)
        h = rms_rows(x, cfg.eps) * w["ln1"]
        R_("n1h", l, h)
        if cap is not None:
            cap.setdefault("xin", []).append(x); cap.setdefault("n1", []).append(h)
        q = (h @ w["q"].T).reshape(n, cfg.NH, cfg.HD)
        k = (h @ w["k"].T).reshape(n, cfg.NKV, cfg.HD)
        v = (h @ w["v"].T).reshape(n, cfg.NKV, cfg.HD)
        R_("qraw", l, q); R_("kraw", l, k); R_("v", l, v)
        if cap is not None:
            cap.setdefault("qr0", []).append(q[:, 0::cfg.r].reshape(n, -1)); cap.setdefault("kr", []).append(k.reshape(n, -1))
        q = rms_rows(q, cfg.eps)
        k = rms_rows(k, cfg.eps)
        if qn_fn is not None:
            q, k = qn_fn(l, q, k)                                      # experiments: quantised normalised queries and keys
        R_("qhat", l, q); R_("khat", l, k)
        if cap is not None:
            cap.setdefault("qn0", []).append(q[:, 0::cfg.r].reshape(n, -1)); cap.setdefault("kn", []).append(k.reshape(n, -1))
        q = q * w["qn"]
        k = k * w["kn"]
        if cap is not None:
            cap.setdefault("qg0", []).append(q[:, 0::cfg.r].reshape(n, -1)); cap.setdefault("kg", []).append(k.reshape(n, -1))
        q = q * cos + rot_half(q) * sin
        k = k * cos + rot_half(k) * sin
        if qk_fn is not None:
            q, k = qk_fn(l, q, k)                                      # experiments: quantised queries and keys
        R_("q", l, q); R_("k", l, k); R_("kk", l, k - k[:1])
        if cap is not None:
            cap.setdefault("k", []).append(k.reshape(n, -1)); cap.setdefault("v", []).append(v.reshape(n, -1))
            cap.setdefault("kk", []).append((k - k[:1]).reshape(n, -1))
        sc = np.einsum("qhd,khd->hqk", q, np.repeat(k, cfg.r, 1)) / math.sqrt(cfg.HD)
        if rec and n > 1:
            rec("screl", l, np.concatenate([(sc[:, i, 1:i + 1] - sc[:, i, :1]).ravel() for i in range(1, n)]))
        if attn_fn is not None:
            sc = attn_fn(l, sc.astype(np.float64))                    # experiments: scores -> scores seen by the softmax
        sc = sc.astype(np.float64) + mask                              # float64: float32 softmax tails are denormals (100x slower)
        pr = np.exp(sc - sc.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        ctx = np.einsum("hqk,khd->qhd", pr.astype(dt), np.repeat(v, cfg.r, 1)).reshape(n, cfg.NH * cfg.HD)
        R_("ctx", l, ctx)
        if cap is not None:
            cap.setdefault("q0", []).append(q[:, 0::cfg.r].reshape(n, -1)); cap.setdefault("q1", []).append(q[:, cfg.r - 1::cfg.r].reshape(n, -1))
            c3 = ctx.reshape(n, cfg.NH, cfg.HD)
            cap.setdefault("ctx0", []).append(c3[:, 0::cfg.r].reshape(n, -1)); cap.setdefault("ctx1", []).append(c3[:, cfg.r - 1::cfg.r].reshape(n, -1))
        o = ctx @ w["o"].T
        y1 = x + o
        R_("o", l, o); R_("y1", l, y1)
        h2 = rms_rows(y1, cfg.eps) * w["ln2"]
        R_("n2h", l, h2)
        if cap is not None:
            cap.setdefault("n2", []).append(h2)
        g = h2 @ w["g"].T
        u = h2 @ w["u"].T
        a = g / (1.0 + np.exp(np.clip(-g, -80.0, 80.0)))
        s = a * u
        R_("u", l, u); R_("swi", l, s)
        for t_, lo_ in enumerate(range(0, cfg.FF, N)):
            sl = slice(lo_, lo_ + N)
            R_(f"g{t_}", l, g[:, sl]); R_(f"a{t_}", l, a[:, sl]); R_(f"u{t_}", l, u[:, sl]); R_(f"swi{t_}", l, s[:, sl])
        if w["hb"] is not None:
            s = s.copy()
            for lo in range(0, cfg.FF - N + 1, N):
                s[:, lo:lo + N] = s[:, lo:lo + N] @ w["hb"].T
        for t_, lo_ in enumerate(range(0, cfg.FF, N)):
            R_(f"swiH{t_}", l, s[:, lo_:lo_ + N])
        m = s @ w["d"].T
        if cap is not None:
            for t_, lo_ in enumerate(range(0, cfg.FF, N)):
                cap.setdefault(f"d{t_}", []).append(s[:, lo_:lo_ + N] @ w["d"][:, lo_:lo_ + N].T)
        x = y1 + m
        R_("mlp", l, m); R_("y2", l, x)
        if cap is not None:
            for nm, val in (("x", x), ("o", o), ("y1", y1), ("g", g), ("u", u), ("a", a), ("swi", a * u), ("swiH", s), ("mlp", m)):
                cap.setdefault(nm, []).append(val)
    return finish(P, cfg, x[-1])


def finish(P, cfg, last):
    """Client side, after the verified jobs: RMS of the last token, back to the original basis, final gain, L2 normalisation."""
    last = rms_rows(np.asarray(last, np.float64), cfg.eps)
    if P["rot"] is not None:
        last = last @ P["rot"].T
    last = last * P["final"]
    return last / np.linalg.norm(last)


class Top:
    """Exact upper percentiles of |values| in bounded memory: keeps the K largest values and the count."""
    K = 20000

    def __init__(self):
        self.v, self.n = np.zeros(0, np.float32), 0

    def add(self, a):
        a = np.abs(np.asarray(a, np.float32)).ravel()
        self.n += a.size
        if a.size > self.K:
            a = np.partition(a, a.size - self.K)[-self.K:]
        self.v = np.concatenate([self.v, a])
        if self.v.size > 4 * self.K:
            self.v = np.partition(self.v, self.v.size - self.K)[-self.K:]

    def pct(self, p):
        v = np.sort(self.v)[::-1]
        return float(v[min(int(self.n * (1 - p / 100.0)), v.size - 1)]) if v.size else 0.0


class Recorder:
    def __init__(self, maxnames=()):
        self.top, self.mx, self.maxnames = {}, {}, maxnames

    def __call__(self, name, l, a):
        self.top.setdefault((name, l), Top()).add(a)
        if name in self.maxnames:
            m = np.abs(a).max(0)
            self.mx[(name, l)] = m if (name, l) not in self.mx else np.maximum(self.mx[(name, l)], m)

    def pct(self, name, l, p=None):
        return self.top[(name, l)].pct(PCT if p is None else p)


def sink_kv(P, cfg):
    """Keys / values (post norm and rope: position 0 is the identity) of the prefix token in each layer: the same for every text."""
    cap = {}
    forward(P, cfg, [PREFIX_ID], cap=cap)
    return [c[0] for c in cap["k"]], [c[0] for c in cap["v"]]


def prepare(P0, cfg, calib_ids, seed=0, smoothing=None, log=print):
    """Original-basis parameters -> parameters of the programs (folded, rotated, smoothed, block-Hadamard), plus the statistics."""
    P1 = fold_rotate(P0, cfg, seed)
    rec = Recorder(maxnames=("q", "k", "swi"))
    for ids in calib_ids:
        forward(P1, cfg, ids, rec)
    P2 = smooth(P1, cfg, rec.mx, **(smoothing or {}))
    rec_u = Recorder(maxnames=("swi", "u"))
    for ids in calib_ids:
        forward(P2, cfg, ids, rec_u)
    P3 = block_hadamard(reorder_ffn(P2, cfg, rec_u.mx), cfg, seed + 1)
    rec2 = Recorder()
    for ids in calib_ids:
        forward(P3, cfg, ids, rec2)
    return P3, rec2


# ---------------------------------------------------------------------------------------------------------------------------------
# program builder
# ---------------------------------------------------------------------------------------------------------------------------------
def i8(a):
    return np.clip(np.rint(a), -64, 64).astype(np.int8)


class Builder:
    """A job: operations, weight blocks (block 0 = table of rows: gains, lookup tables, constants)."""

    def __init__(self):
        self.ops, self.W, self.tab, self.cache, self.tags = [], [np.zeros((N, N), np.int8)], [], {}, {}

    def op(self, o):
        self.ops.append(o)
        return len(self.ops)

    def block(self, key, make):
        if key not in self.cache:
            self.W.append(i8(make()))
            self.cache[key] = len(self.W) - 1
        return self.cache[key]

    def row(self, vec):
        v = np.zeros(N, np.int64)
        vec = np.asarray(vec)
        v[:len(vec)] = vec
        self.tab.append(i8(v))
        return len(self.tab) - 1

    def tag(self, t, name, l, scale, cols=None):
        self.tags[t] = (name, l, scale, cols)
        return t

    def tensor_table(self):
        t = np.zeros((N, N), np.int8)
        for r, v in enumerate(self.tab):
            t[r] = v
        return t

    def weights(self):
        W = list(self.W)
        W[0] = self.tensor_table()
        return W

    # --- scaled linear algebra ---
    def gemm(self, wf, x, s_in, s_t):
        """Gemm of the real matrix `wf` (rows x cols <= n x n) on tensor x of scale s_in, output scale >= s_t (returns the actual one)."""
        wf = np.asarray(wf, np.float64)
        sw_nat = max(float(np.abs(wf).max()), 1e-30) / 63.0
        sh = math.floor(math.log2(s_t / (s_in * sw_nat)))
        if sh < 0:
            sh, s_t = 0, s_in * sw_nat
        sw = s_t / (s_in * 2.0 ** sh)
        blk = np.zeros((N, N))
        blk[:wf.shape[0], :wf.shape[1]] = np.clip(np.rint(wf / sw), -63, 63)
        self.W.append(blk.astype(np.int8))
        self.last_w = (blk[:wf.shape[0], :wf.shape[1]].copy(), sw)        # (integer weights, real step): to predict the effect of a constant offset
        t = self.op({"Gemm": {"w": len(self.W) - 1, "x": x, "shift": sh, "relu": False}})
        return t, s_t

    def gemm_int(self, key, make, x, shift):
        """Gemm by an integer block (permutation, signs, markers, constants)."""
        return self.op({"Gemm": {"w": self.block(key, make), "x": x, "shift": shift, "relu": False}})

    def const_row(self, value, nrows):
        v = np.zeros(N, np.int64)
        v[:nrows] = value
        return self.row(v)

    def affine(self, t, gain, nrows, s_in, s_out):
        """Per-row gain on a tensor of scale s_in -> scale s_out (Mul: (x * g) >> sh with an 8-bit g)."""
        f = float(np.abs(gain).max()) * s_in / s_out
        sh = max(0, math.floor(math.log2(64 / max(f, 1e-30))))
        fac = 2.0 ** sh * s_in / s_out
        return self.op({"Mul": {"x": t, "w": 0, "row": self.row(np.clip(np.rint(gain * fac), -64, 64)), "shift": sh}})


def rope_blocks(cfg, b):
    """(permutation P, cos table, sin table) block indices of the job: shared by the queries and the keys of all the layers."""
    nk, hd, L = cfg.NKV, cfg.HD, cfg.L
    H2 = hd // 2

    def perm():
        p = np.zeros((N, N))
        for g in range(nk):
            for d in range(hd):
                p[g * hd + d, g * hd + (d + H2) % hd] = -1.0 if d < H2 else 1.0
        return p

    cos, sin = rope_tables(cfg, L)

    def table(a):
        t = np.zeros((N, N))
        for g in range(nk):
            t[g * hd:(g + 1) * hd, :L] = (a[:, :].T * 64.0)
        return t

    return (b.block("P", perm), b.block("C", lambda: table(cos)), b.block("S", lambda: table(sin)))


def sink_blocks(b):
    """Masks of the sink column: (1 at cell (0, 0) only, 1 everywhere except column 0)."""
    m00 = b.block("M00", lambda: _unit(0, 0))
    not0 = b.block("Mnot0", lambda: np.hstack([np.zeros((N, 1)), np.ones((N, N - 1))]))
    return m00, not0


def _unit(i, j):
    t = np.zeros((N, N))
    t[i, j] = 1
    return t


# ---------------------------------------------------------------------------------------------------------------------------------
# plan of the scales, then the layers
# ---------------------------------------------------------------------------------------------------------------------------------
def plan_layer(l, w, st, cfg, k0, v0):
    """Scales of one layer from the statistics (independent of the other layers)."""
    pc = lambda name: max(st.pct(name, l), 1e-9)
    pa = lambda name: max(st.pct(name, l, ACT_PCT), 1e-9)
    pr = lambda name: max(st.pct(name, l, RES_PCT), 1e-9)
    p = {}
    p["s_qraw"], p["s_kraw"] = max(st.pct("qraw", l, QK_RAW_PCT), 1e-9) / 63, max(st.pct("kraw", l, QK_RAW_PCT), 1e-9) / 63
    p["s_v"] = max(pc("v"), float(np.abs(v0).max())) / 63
    p["s_k"] = max(pc("k"), pc("kk")) / 63                              # keys, and keys minus the sink's key, at the same scale
    p["c_k0"] = np.rint(k0 / p["s_k"]).astype(np.int64)                 # the sink's key in that scale (subtracted from all the keys)
    # scores are taken relative to the sink (softmax unchanged); their unit is 2^t ln2/8 nats, t the smallest that spans their range
    rng = max(st.pct("screl", l, SCORE_PCT), 1e-9)
    p["t"] = next((t for t in range(MAX_T + 1) if 64 * C_SOFTMAX * 2 ** t >= rng), MAX_T)
    unit = C_SOFTMAX * 2.0 ** p["t"]
    sq_min = pc("q") / 63
    root = math.sqrt(cfg.HD)
    p["sh_sc"] = max(0, math.floor(math.log2(root * unit / (p["s_k"] * sq_min))))
    p["s_q"] = root * unit * 2.0 ** (-p["sh_sc"]) / p["s_k"]
    zo = pa("ctx") * 64 / p["s_v"]
    p["sho"] = max(0, math.ceil(math.log2(max(zo, 1) / 63)))
    p["s_ctx"] = p["s_v"] / 64 * 2.0 ** p["sho"]
    p["tiles"] = []
    for t in range(-(-cfg.FF // N)):                                    # per tile of N feed-forward channels (sorted by amplitude)
        p["tiles"].append(dict(s_g=pa(f"g{t}") / 63, s_a=pa(f"a{t}") / 63, s_u=pa(f"u{t}") / 63, s_sw=pa(f"swi{t}") / 63, s_swH_min=pa(f"swiH{t}") / 63))
    # smallest residual scale the weights of this layer can write at: Wo reads ctx, Wd reads the rotated SwiGLU product
    lb = 0.0
    for j in range(cfg.r):
        lb = max(lb, p["s_ctx"] * float(np.abs(w["o"]).max()) / 63)
    for t, tl in enumerate(p["tiles"]):
        lb = max(lb, tl["s_swH_min"] * float(np.abs(w["d"][:, t * N:(t + 1) * N]).max()) / 63)
    rshift = lambda name, margin: next((sft for sft in range(MAX_RMS_SHIFT + 1) if 4 * 2 ** sft >= margin * pc(name)), MAX_RMS_SHIFT)
    p["rs_n1"], p["rs_n2"] = rshift("n1h", 1.0), rshift("n2h", 1.0)      # output shift of the RMS norms: clip >= the 99.99 percentile of |x / rms|
    p["rs_q"], p["rs_k"] = rshift("qhat", QK_MARGIN), rshift("khat", QK_MARGIN)      # (queries and keys have heavy tails: more margin)
    p["s_x"] = pc("x") / 63                                            # (layer 0: the embedding, ~10x smaller than the layer's outputs)
    p["res"] = max(pr("x"), pr("o"), pr("y1"), pr("mlp"), pr("y2")) / 63
    p["res_lb"] = lb
    return p


def hadamard_scale(s_in, sh):
    return s_in * 2.0 ** sh / math.sqrt(N)


def layer(b, w, cfg, p, l, x, s_res, k0, v0, ctx_, x_norm=None):
    """One layer, residual stream x (tensor index) at scale s_res. Returns the new residual tensor. x_norm: finer copy of x read by the first
    normalisation (the embedding of the first layer is much smaller than what the layer writes)."""
    D, NK, HD, FF, r, L = cfg.D, cfg.NKV, cfg.HD, cfg.FF, cfg.r, cfg.L
    nrows_kv = NK * HD
    P_, C_, S_ = ctx_["rope"]
    E, not0 = ctx_["E"], ctx_["not0"]
    s_n1, s_n2 = 2.0 ** p["rs_n1"] / 16, 2.0 ** p["rs_n2"] / 16
    n1 = b.tag(b.op({"RmsNorm": {"x": x if x_norm is None else x_norm, "len": D, "group": D, "shift": p["rs_n1"]}}), "n1", l, s_n1)
    b.tag(x, "xin", l, s_res)

    def rope(t, offset):
        """Rotary embedding; `offset` (per row) is added at the end: +1 recentres the two floors, -c_k0 subtracts the sink's key."""
        pt = b.op({"Gemm": {"w": P_, "x": t, "shift": 0, "relu": False}})
        a = b.op({"MulW": {"x": t, "w": C_, "shift": 6}})
        c = b.op({"MulW": {"x": pt, "w": S_, "shift": 6}})
        out = b.op({"Add": {"a": a, "b": c, "relu": False}})
        offset = np.asarray(offset, np.int64)
        part = np.clip(offset, -64, 64)
        out = b.op({"Bias": {"x": out, "w": 0, "row": b.row(part), "relu": False}})
        if (offset != part).any():                                     # |offset| > 64: second Bias
            out = b.op({"Bias": {"x": out, "w": 0, "row": b.row(offset - part), "relu": False}})
        return out

    def qk(wmat, gain, s_raw, s_final, offset, rs, dbg=None):
        t, s_t = b.gemm(wmat, n1, s_n1, s_raw)
        if dbg:
            b.tag(t, dbg[0], l, s_t)
        t = b.op({"RmsNorm": {"x": t, "len": nrows_kv, "group": HD, "shift": rs}})
        if dbg:
            b.tag(t, dbg[1], l, 2.0 ** rs / 16)
        t = b.affine(t, np.tile(gain, NK), nrows_kv, 2.0 ** rs / 16, s_final)
        if dbg:
            b.tag(t, dbg[2], l, s_final)
        return rope(t, offset)

    def sink_fix(t, const, name):
        m = b.op({"MulW": {"x": t, "w": not0, "shift": 0}})
        blk = np.zeros((N, N))
        blk[:nrows_kv, 0] = const
        c = b.op({"Gemm": {"w": b.block(("sink", l, name), lambda: blk), "x": E, "shift": 6, "relu": False}})
        return b.op({"Add": {"a": m, "b": c, "relu": False}})

    wq = w["q"].reshape(NK, r, HD, D)                                    # head (g * r + j)
    # keys and values
    ones = np.ones(nrows_kv, np.int64)
    tk = qk(w["k"], w["kn"], p["s_kraw"], p["s_k"], ones - p["c_k0"], p["rs_k"], ("kr", "kn", "kg"))     # keys minus the sink's key
    tk = b.op({"MulW": {"x": tk, "w": not0, "shift": 0}})              # the sink's column is then exactly 0: its score is 0
    tk = b.tag(tk, "kk", l, p["s_k"])
    tv, _ = b.gemm(w["v"], n1, s_n1, p["s_v"])
    tv = sink_fix(tv, np.clip(np.rint(v0 / p["s_v"]), -64, 64), "v")
    tv = b.tag(tv, "v", l, p["s_v"])
    tvt = b.op({"Map": {"x": tv, "map": "Transpose"}})
    one_ctx = b.const_row(1, nrows_kv)
    os_ = []
    for j in range(r):
        wqj = wq[:, j].reshape(nrows_kv, D)
        tq = qk(wqj, w["qn"], p["s_qraw"], p["s_q"], ones, p["rs_q"], ("qr0", "qn0", "qg0") if j == 0 else None)
        if j in (0, r - 1):
            b.tag(tq, f"q{0 if j == 0 else 1}", l, p["s_q"])
        tqs = b.op({"Map": {"x": tq, "map": {"HeadSpread": {"dh": HD, "heads": NK, "l": L}}}})
        tsc = b.op({"GemmAT": {"a": tk, "x": tqs, "shift": p["sh_sc"], "relu": False}})
        hi = b.op({"CausalSoftmaxS": {"x": tsc, "len": "LEN", "l": L, "fine": False, "scale": p["t"]}})
        lo = b.op({"CausalSoftmaxS": {"x": tsc, "len": "LEN", "l": L, "fine": True, "scale": p["t"]}})
        c_hi = b.op({"GemmAT": {"a": tvt, "x": hi, "shift": p["sho"], "relu": False}})
        c_lo = b.op({"GemmAT": {"a": tvt, "x": lo, "shift": p["sho"] + 6, "relu": False}})
        cx = b.op({"Bias": {"x": b.op({"Add": {"a": c_hi, "b": c_lo, "relu": False}}), "w": 0, "row": one_ctx, "relu": False}})   # recentres the 2 floors
        cg = b.op({"Map": {"x": cx, "map": {"HeadGather": {"dh": HD, "heads": NK, "l": L}}}})
        if j in (0, r - 1):
            b.tag(cg, f"ctx{0 if j == 0 else 1}", l, p["s_ctx"])
        wo = w["o"].reshape(D, NK, r, HD)[:, :, j, :].reshape(D, nrows_kv)
        to, _ = b.gemm(wo, cg, p["s_ctx"], s_res)
        os_.append(to)
    to = os_[0]
    for t in os_[1:]:
        to = b.op({"Add": {"a": to, "b": t, "relu": False}})
    to = b.tag(_recentre(b, to, r, l, D), "o", l, s_res)
    y1 = b.tag(b.op({"Add": {"a": x, "b": to, "relu": False}}), "y1", l, s_res)
    # feed-forward
    n2 = b.tag(b.op({"RmsNorm": {"x": y1, "len": D, "group": D, "shift": p["rs_n2"]}}), "n2", l, s_n2)
    sp = math.sqrt(N)
    acc, nt = None, 0
    off = np.zeros(D)                                                  # real-unit offset of the MLP output caused by the floors (see below)
    for lo_ in range(0, FF, N):
        hi_ = min(FF, lo_ + N)
        tl = p["tiles"][nt]
        tg, s_g = b.gemm(w["g"][lo_:hi_], n2, s_n2, tl["s_g"])
        b.tag(tg, f"g{nt}", l, s_g, (lo_, hi_))
        lut = np.clip(np.rint((np.arange(-64, 64) * s_g) / (1 + np.exp(-np.arange(-64, 64) * s_g)) / tl["s_a"]), -64, 64)
        ta = b.op({"Lut": {"x": tg, "w": 0, "row": b.row(lut)}})
        b.tag(ta, f"a{nt}", l, tl["s_a"], (lo_, hi_))
        sh_m = max(0, round(math.log2(tl["s_sw"] / (tl["s_a"] * tl["s_u"]))))
        tu, s_u = b.gemm(w["u"][lo_:hi_], n2, s_n2, tl["s_sw"] / (tl["s_a"] * 2.0 ** sh_m))
        b.tag(tu, f"u{nt}", l, s_u, (lo_, hi_))
        ts = b.op({"MulT": {"a": ta, "b": tu, "shift": sh_m}})
        s_s = tl["s_a"] * s_u * 2.0 ** sh_m
        s_m = s_s
        delta = -0.5 * s_m * np.ones(hi_ - lo_)                        # the MulT output is floored: -1/2 LSB on every channel
        b.tag(ts, f"swi{nt}", l, s_s, (lo_, hi_))
        if w["hb"] is not None and hi_ - lo_ == N:
            sh_h = max(0, math.ceil(math.log2(max(tl["s_swH_min"] * sp / s_s, 1e-30))))
            ts = b.gemm_int("HB", lambda: ctx_["signs"], ts, sh_h)
            s_s = hadamard_scale(s_s, sh_h)
            b.tag(ts, f"swiH{nt}", l, s_s, (lo_, hi_))
            # the rotation carries the MulT offset to H.delta, and its own floor adds -1/2 LSB on every output
            delta = w["hb"][:hi_ - lo_, :hi_ - lo_] @ delta - 0.5 * s_s * np.ones(hi_ - lo_)
        td, _ = b.gemm(w["d"][:, lo_:hi_], ts, s_s, s_res)
        wint, sw_d = b.last_w
        off += (wint.astype(np.float64) * sw_d) @ delta - 0.5 * s_res    # + the floor of the down projection itself
        b.tag(td, f"d{nt}", l, s_res)
        acc = td if acc is None else b.op({"Add": {"a": acc, "b": td, "relu": False}})
        nt += 1
    # every floor above is a constant offset (the same for all tokens): cancelled exactly by the bias row, no operation added
    acc = b.tag(b.op({"Bias": {"x": acc, "w": 0, "row": b.row(np.clip(np.rint(-off / s_res), -64, 64)), "relu": False}}), "mlp", l, s_res)
    x2 = b.op({"Add": {"a": y1, "b": acc, "relu": False}})
    return b.tag(x2, "x", l, s_res)


def _recentre(b, t, count, l, nrows):
    """Each Gemm output is floored (bias -1/2 on every feature): the sum of `count` of them is recentred by round(count / 2); for an odd
    count the rounding alternates with the layer so that the residual stream does not drift."""
    c = count // 2 + (count % 2) * (l % 2)
    if c == 0:
        return t
    return b.op({"Bias": {"x": t, "w": 0, "row": b.const_row(c, nrows), "relu": False}})


def build(P, cfg, st, log=print, layers_per_job=None):
    """Statistics -> (jobs, plan). Each job is a Builder."""
    k0s, v0s = sink_kv(P, cfg)
    plans = [plan_layer(l, w, st, cfg, k0s[l], v0s[l]) for l, w in enumerate(P["layers"])]
    hb0 = P["layers"][0]["hb"]
    signs = np.rint(hb0 * math.sqrt(N)).astype(np.int64) if hb0 is not None else None

    def start(scale_in):
        b = Builder()
        m00, not0 = sink_blocks(b)
        ctx_ = {"rope": rope_blocks(cfg, b), "not0": not0, "signs": signs}
        c64 = b.op({"Lut": {"x": 0, "w": 0, "row": b.row(np.full(128, 64))}})     # a tensor of 64s, whatever the input
        ctx_["E"] = b.op({"MulW": {"x": c64, "w": m00, "shift": 0}})            # 64 at cell (0, 0), 0 elsewhere
        return b, ctx_

    if layers_per_job is None:                                         # as many layers as the bounds allow, from the cost of one layer
        b, ctx_ = start(1.0)
        n0, k0_ = len(b.ops), len(b.W)
        s_probe = max(plans[0]["res"], plans[0]["res_lb"], 1e-3)
        layer(b, P["layers"][0], cfg, plans[0], 0, 0, s_probe, k0s[0], v0s[0], ctx_)
        per_ops, per_blocks = len(b.ops) - n0, len(b.W) - k0_
        reserve_ops, reserve_blocks = 1, 1                             # rescale Gemm of the input and its block
        layers_per_job = max(1, min((OPS_MAX - n0 - reserve_ops) // per_ops, (BLOCKS_MAX - k0_ - reserve_blocks) // per_blocks))
        log(f"one layer = {per_ops} operations and {per_blocks} weight blocks (+{n0} operations, {k0_} blocks per job): {layers_per_job} layers per job")
    plan = {"n": N, "L": cfg.L, "D": cfg.D, "layers_per_job": layers_per_job, "prefix_id": PREFIX_ID, "layers": plans}
    need = lambda ls: max(max(plans[l]["res"], plans[l]["res_lb"]) for l in ls)
    jobs, s_in = [], None
    for first in range(0, cfg.NL, layers_per_job):
        group = range(first, min(cfg.NL, first + layers_per_job))
        if s_in is None:
            s_in = plan["s0"] = plans[0]["s_x"]                        # scale at which the client quantises the first input
        b, ctx_ = start(s_in)
        # the first operations rescale the input to the scale of the job (the column operations cannot read the input tensor itself)
        x_norm = None
        if first == 0:                                                 # the embedding keeps its own fine scale for the first normalisation
            x_norm = b.op({"Gemm": {"w": b.block("COPY", lambda: 32 * np.eye(N)), "x": 0, "shift": 5, "relu": False}})
        target = max(s_in, need(group))
        sh = max(0, math.floor(math.log2(63 * target / s_in)))
        wgt = max(1, min(63, int(math.floor(s_in / target * 2.0 ** sh))))
        x = b.op({"Gemm": {"w": b.block("RESCALE", lambda: wgt * np.eye(N)), "x": 0, "shift": sh, "relu": False}})
        s_job = s_in * 2.0 ** sh / wgt                                 # realised scale (>= target)
        assert s_job >= need(group) * (1 - 1e-9)
        for l in group:
            x = layer(b, P["layers"][l], cfg, plans[l], l, x, s_job, k0s[l], v0s[l], ctx_, x_norm if l == 0 else None)
        b.s_job = s_job
        jobs.append(b)
        s_in = s_job
        if len(b.ops) > OPS_MAX or len(b.W) > BLOCKS_MAX:
            raise SystemExit(f"job {len(jobs) - 1}: {len(b.ops)} operations, {len(b.W)} blocks: bounds exceeded (--layers-per-job)")
    plan["s_out"] = jobs[-1].s_job
    plan["job_scales"] = [b.s_job for b in jobs]
    return jobs, plan


# ---------------------------------------------------------------------------------------------------------------------------------
# integer execution (independent reference engine), client side, export
# ---------------------------------------------------------------------------------------------------------------------------------
def program_for(b, lr):
    return json.loads(json.dumps(b.ops).replace('"LEN"', str(lr)))


def enc_input(P, cfg, ids, s0):
    """Client: rotated embeddings of the tokens, features x tokens; column 0 (the sink) is zero: its keys and values are constants."""
    x = P["embed"][list(ids)].astype(np.float64)
    t = np.zeros((N, N), np.int64)
    t[:cfg.D, :len(ids)] = np.clip(np.rint(x.T / s0), -64, 64)
    t[:, 0] = 0
    return t


def run_int(jobs, plan, P, cfg, ids, Ws=None, keep=False):
    """The chained jobs on the reference engine: integer residual of the last token (D,) and the float embedding."""
    Ws = Ws or [b.weights() for b in jobs]
    x = enc_input(P, cfg, ids, plan["s0"])
    Ts = []
    for b, W in zip(jobs, Ws):
        T, _ = R.run_program(x, W, program_for(b, len(ids)))
        Ts.append(T)
        x = np.zeros((N, N), np.int64)
        x[:cfg.D, :cfg.L] = T[-1][:cfg.D, :cfg.L]
    last = Ts[-1][-1][:cfg.D, len(ids) - 1].astype(np.int64)
    emb = finish(P, cfg, last * plan["s_out"])
    return (emb, last, Ts) if keep else emb


def export(out, jobs, plan, cfg, name, extra=None):
    js = [{"ops_template": b.ops, "weights": [R.enc(w) for w in b.weights()]} for b in jobs]
    model = {"name": name + "-int8", "hf": name, "n": N, "L": cfg.L, "D": cfg.D, "jobs": js, "plan": plan, **(extra or {})}
    with open(out, "w") as f:
        json.dump(model, f, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o.item())
    return os.path.getsize(out)


def diag(jobs, plan, P, cfg, ids_list, quiet=False):
    """Cosine and saturation of the tagged tensors (integer * scale against the float tensor), per layer."""
    acc = {}
    for ids in ids_list:
        cap = {}
        forward(P, cfg, ids, cap=cap)
        _, _, Ts = run_int(jobs, plan, P, cfg, ids, keep=True)
        n = len(ids)
        for b, T in zip(jobs, Ts):
            for t, (name, l, sc, cols) in b.tags.items():
                f = np.asarray(cap[name.rstrip("0123456789") if cols else name][l], np.float64)[1:n, slice(*cols) if cols else slice(None)].T
                i = T[t][:f.shape[0], 1:n].astype(np.float64) * sc
                cos = float((f * i).sum() / (np.linalg.norm(f) * np.linalg.norm(i) + 1e-30))
                sat = float((np.abs(T[t][:f.shape[0], 1:n]) >= 64).mean())
                acc.setdefault((name, l), []).append((cos, sat))
    if quiet:
        return acc
    order = ["xin", "n1", "kk", "v", "q0", "q1", "ctx0", "ctx1", "o", "y1", "n2", "g", "a", "swi", "swiH", "mlp", "x"]
    key = lambda nm: order.index(nm) if nm in order else len(order) + hash(nm) % 100
    for (name, l), v in sorted(acc.items(), key=lambda kv: (kv[0][1], key(kv[0][0]))):
        print(f"  layer {l:2d} {name:5s}: cosine {np.mean([c for c, _ in v]):.4f}   saturated {100 * np.mean([s for _, s in v]):.2f} %")


def ids_of(tk, text, cfg):
    """Prefix token + text tokens (the tokenizer appends the end token); at most L tokens."""
    return [PREFIX_ID] + tk(text, truncation=True, max_length=cfg.L - 1)["input_ids"]


def main(out):
    t0 = time.time()
    tk, P0, cfg = load()
    cfg.check()
    print(f"{NAME}: {cfg.NL} layers, {cfg.NH} query heads / {cfg.NKV} key-value heads of {cfg.HD}, d = {cfg.D}, FFN {cfg.FF}")
    if "--texts" in sys.argv:
        texts = [t.strip() for t in open(arg("--texts", ""), encoding="utf-8") if t.strip()]
    else:
        from datasets import load_dataset
        ds = load_dataset("fancyzhx/ag_news")
        texts = list(ds["train"]["text"][:arg("--calib", 48, int)]) + list(ds["test"]["text"][100:100 + arg("--eval", 12, int)])
    ncal, nev = arg("--calib", 48, int), arg("--eval", 12, int)
    cal, ev = texts[:ncal], texts[ncal:ncal + nev]
    cal_ids = [ids_of(tk, t, cfg) for t in cal]
    P, st = prepare(P0, cfg, cal_ids)
    print(f"prepared (fold, rotation, smoothing, block Hadamard) and statistics of {len(cal_ids)} texts: {time.time() - t0:.0f} s")
    jobs, plan = build(P, cfg, st, layers_per_job=arg("--layers-per-job", None, int))
    for k, b in enumerate(jobs):
        print(f"job {k}: {len(b.ops)} operations, {len(b.W)} weight blocks, {len(b.tab)} table rows")
    ev_ids = [ids_of(tk, t, cfg) for t in ev]
    if "--diag" in sys.argv:
        diag(jobs, plan, P, cfg, ev_ids[:2])
        return
    Ws = [b.weights() for b in jobs]
    cos_f, cos_o = [], []
    for ids in ev_ids:
        f = forward(P, cfg, ids)
        o = forward(P0, cfg, ids)
        i = run_int(jobs, plan, P, cfg, ids, Ws)
        cos_f.append(float(f @ i))
        cos_o.append(float(o @ i))
    print(f"{len(ev_ids)} texts: cosine integer / float of the same prefixed text {np.mean(cos_f):.4f} (min {np.min(cos_f):.3f}); "
          f"transformed model against the original: {np.mean([float(forward(P0, cfg, i) @ forward(P, cfg, i)) for i in ev_ids[:3]]):.6f}; "
          f"{time.time() - t0:.0f} s")
    size = export(out, jobs, plan, cfg, NAME, {"report": {"cos_int_float": float(np.mean(cos_f))}, "tokenizer": NAME, "rot_seed": 0, "eps": cfg.eps, "theta": cfg.theta})
    print(f"written {out} ({size >> 20} MiB)")


if __name__ == "__main__":
    NAME = arg("--model", NAME)
    PCT = arg("--pct", PCT, float)
    main(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/tmp/qwen3-int8.json")
