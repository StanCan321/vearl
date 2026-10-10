#!/usr/bin/env python3
"""BERT-type EMBEDDING model (normalisation AFTER each block) converted into CHAINED integer VEARL programs. By default
BAAI/bge-base-en-v1.5 (MIT licence: 12 layers, 12 heads of 64, d = 768, FFN 3072 GELU, 110 M parameters; embedding = state of the
[CLS] token, normalised). The dimensions are read from the model configuration: the same code converts other BERTs.

Layout (like tools/quantize_minilm.py): features x tokens, <= L = 64 tokens, one text per job; the 12 heads in ONE product
(HeadSpread: 12 x 64 = 768 columns), Softmax + SoftmaxLo (probabilities in 1/4096), V.A, HeadGather; integer LayerNorm then gamma (Mul)
and beta (Bias); FFN cut into blocks of 1024 rows (GELU by table); the input embedding (word table + positions + LayerNorm) is
computed by the client. The layers are split into jobs of LAYERS_PER_JOB layers (bounds: 240 operations, 64 weight blocks per job);
the output of a job (residual stream D x L, verified) is the input of the next one; the last job returns the state of the [CLS] token (column 0).

    python3 quantize_bert.py OUT.json [--model BAAI/bge-base-en-v1.5] [--calib 200] [--eval 300] [--pct 99.9] [--diag] [--params P.npz]
"""
import json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vearl_ref as R

arg = lambda name, default, cast=str: cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
NAME = arg("--model", "BAAI/bge-base-en-v1.5")
N, L = 1024, 64
R.N = N
C_SOFTMAX = math.log(2) / 8
PCT = arg("--pct", 99.9, float)
LAYERS_PER_JOB = arg("--layers-per-job", 4, int)
FINE_ATTN = "--coarse-attn" not in sys.argv


def erf(x):
    import torch
    return torch.special.erf(torch.from_numpy(np.asarray(x, dtype=np.float64))).numpy()


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


class Cfg:
    def __init__(self, c):
        self.D, self.H, self.NL, self.FF = c.hidden_size, c.num_attention_heads, c.num_hidden_layers, c.intermediate_size
        self.DH = self.D // self.H
        self.eps = c.layer_norm_eps
        assert self.D <= N and self.H * L <= N, "dimensions out of the chain's bounds"


def load():
    from transformers import AutoTokenizer, AutoModel
    tk = AutoTokenizer.from_pretrained(NAME)
    m = AutoModel.from_pretrained(NAME).eval()
    return tk, {k: v.detach().numpy().astype(np.float64) for k, v in m.state_dict().items()}, Cfg(m.config)


def K(l, name):
    return f"encoder.layer.{l}.{name}"


def ln_f(x, g, b, eps):
    mu = x.mean(0, keepdims=True)
    var = x.var(0, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g[:, None] + b[:, None]


def embed(sd, cfg, ids):
    """Input embedding (client): words + positions + type 0, LayerNorm; output (D, Lr)."""
    x = sd["embeddings.word_embeddings.weight"][ids] + sd["embeddings.position_embeddings.weight"][:len(ids)] + sd["embeddings.token_type_embeddings.weight"][0]
    return ln_f(x.T, sd["embeddings.LayerNorm.weight"], sd["embeddings.LayerNorm.bias"], cfg.eps)


def float_forward(sd, cfg, ids):
    """EXACT floating-point model (formulas of BertModel), layout (features, tokens); [CLS] embedding and intermediates."""
    D, H, DH, FF = cfg.D, cfg.H, cfg.DH, cfg.FF
    x = embed(sd, cfg, ids)
    Lr = x.shape[1]
    inter = []
    for l in range(cfg.NL):
        P = lambda n: sd[K(l, n)]
        q = P("attention.self.query.weight") @ x + P("attention.self.query.bias")[:, None]
        k = P("attention.self.key.weight") @ x + P("attention.self.key.bias")[:, None]
        v = P("attention.self.value.weight") @ x + P("attention.self.value.bias")[:, None]
        ctx, scs = np.zeros((D, Lr)), []
        for h in range(H):
            s = slice(h * DH, (h + 1) * DH)
            sc = k[s].T @ q[s] / math.sqrt(DH)
            a = np.exp(sc - sc.max(0, keepdims=True))
            a /= a.sum(0, keepdims=True)
            ctx[s] = v[s] @ a
            scs.append(sc)
        wo = P("attention.output.dense.weight") @ ctx
        ao = wo + P("attention.output.dense.bias")[:, None]
        y1 = x + ao
        hh = ln_f(y1, P("attention.output.LayerNorm.weight"), P("attention.output.LayerNorm.bias"), cfg.eps)
        W1, W2 = P("intermediate.dense.weight"), P("output.dense.weight")
        w1x = W1 @ hh
        f1 = w1x + P("intermediate.dense.bias")[:, None]
        g = gelu(f1)
        f2parts = [W2[:, lo:lo + N] @ g[lo:lo + N] for lo in range(0, FF, N)]
        f2 = sum(f2parts) + P("output.dense.bias")[:, None]
        y2 = hh + f2
        xn = ln_f(y2, P("output.LayerNorm.weight"), P("output.LayerNorm.bias"), cfg.eps)
        inter.append(dict(x=x, q=q, k=k, v=v, sc=np.concatenate(scs, 1), ctx=ctx, wo=wo, ao=ao, y1=y1, h=hh, w1x=w1x, f1=f1, g=g,
                          f2parts=f2parts, f2=f2, y2=y2, xn=xn))
        x = xn
    return x[:, 0], inter


def pct(arrs):
    return float(np.percentile(np.abs(np.concatenate([np.asarray(a).ravel() for a in arrs])), PCT))


class Builder:
    """A job: operations + RECIPES of the weight blocks (param, slices, scale) and of the table rows (block 0)."""

    def __init__(self, D):
        self.D = D
        self.ops, self.wrec, self.trec, self.tags = [], [("table",)], [], {}

    def tag(self, t, name, l, scale, rows=None):
        self.tags[t] = (name, l, scale, rows or (0, self.D))
        return t

    def op(self, o):
        self.ops.append(o)
        return len(self.ops)

    def row(self, rec):
        self.trec.append(rec)
        return len(self.trec) - 1

    def gemm_scaled(self, sd, key, rs, cs, x, s_in, s_t):
        wf = sd[key][rs[0]:rs[1], cs[0]:cs[1]]
        sw_nat = np.abs(wf).max() / 63.0
        sh = math.floor(math.log2(s_t / (s_in * sw_nat)))
        if sh < 0:
            sh, s_t = 0, s_in * sw_nat
        sw = s_t / (s_in * 2.0 ** sh)
        self.wrec.append(("mat", key, rs, cs, sw))
        t = self.op({"Gemm": {"w": len(self.wrec) - 1, "x": x, "shift": sh, "relu": False}})
        return t, s_t, dict(w=len(self.wrec) - 1, sw=sw, sh=sh, s=s_t)

    def bias(self, x, key, lo, hi, s):
        return self.op({"Bias": {"x": x, "w": 0, "row": self.row(("vec", key, lo, hi, 1.0 / s, 64)), "relu": False}})

    def affine(self, sd, t_ln, gkey, bkey, s_out):
        D = self.D
        f = np.abs(sd[gkey]).max() / (16 * s_out)
        sh = max(0, math.floor(math.log2(64 / f)))
        fac = 2.0 ** sh / (16 * s_out)
        t = self.op({"Mul": {"x": t_ln, "w": 0, "row": self.row(("vec", gkey, 0, D, fac, 64)), "shift": sh}})
        return self.bias(t, bkey, 0, D, s_out), dict(sh=sh, fac=fac, s=s_out)


def layer(b, sd, cfg, calib, l, x, s_x, s_next, pl):
    D, H, DH, FF = cfg.D, cfg.H, cfg.DH, cfg.FF
    col = lambda key: [c[l][key] for c in calib]
    P = lambda n: K(l, n)
    sA = s_x
    s_B = max(pct(col("h")), pct(col("y2")), pct(col("f2")), max(pct([c[l]["f2parts"][k] for c in calib]) for k in range(len(calib[0][l]["f2parts"])))) / 63
    s_k = max(pct(col("k")), 1e-9) / 63
    s_v = max(pct(col("v")), 1e-9) / 63
    tk, s_k, pl["k"] = b.gemm_scaled(sd, P("attention.self.key.weight"), (0, D), (0, D), x, s_x, s_k)
    tk = b.tag(b.bias(tk, P("attention.self.key.bias"), 0, D, s_k), "k", l, s_k)
    tv, s_v, pl["v"] = b.gemm_scaled(sd, P("attention.self.value.weight"), (0, D), (0, D), x, s_x, s_v)
    tv = b.tag(b.bias(tv, P("attention.self.value.bias"), 0, D, s_v), "v", l, s_v)
    sq_min = max(pct(col("q")) / 63, 1e-9)
    sh_sc = max(0, math.floor(math.log2(math.sqrt(DH) * C_SOFTMAX / (s_k * sq_min))))
    s_q_t = math.sqrt(DH) * C_SOFTMAX * 2.0 ** (-sh_sc) / s_k
    tq, s_q, pl["q"] = b.gemm_scaled(sd, P("attention.self.query.weight"), (0, D), (0, D), x, s_x, s_q_t)
    tq = b.tag(b.bias(tq, P("attention.self.query.bias"), 0, D, s_q), "q", l, s_q)
    pl.update(sh_sc=sh_sc)
    tqs = b.op({"Map": {"x": tq, "map": {"HeadSpread": {"dh": DH, "heads": H, "l": L}}}})
    tsc = b.tag(b.op({"GemmAT": {"a": tk, "x": tqs, "shift": sh_sc, "relu": False}}), "sc", l, C_SOFTMAX, "sc")
    tsm = b.tag(b.op({"Softmax": {"x": tsc, "len": "LEN"}}), "a", l, 1 / 64, "sc")
    tvt = b.op({"Map": {"x": tv, "map": "Transpose"}})
    zo = pct(col("ctx")) * 64 / s_v
    sho = max(0, math.ceil(math.log2(max(zo, 1) / 63)))
    if FINE_ATTN:
        tlo = b.op({"SoftmaxLo": {"x": tsc, "len": "LEN"}})
        thi = b.op({"GemmAT": {"a": tvt, "x": tsm, "shift": sho, "relu": False}})
        tfi = b.op({"GemmAT": {"a": tvt, "x": tlo, "shift": sho + 6, "relu": False}})
        tcx = b.op({"Add": {"a": thi, "b": tfi, "relu": False}})
        tcx = b.op({"Bias": {"x": tcx, "w": 0, "row": b.row(("const", 1)), "relu": False}})    # recentres the two default roundings
    else:
        tcx = b.op({"GemmAT": {"a": tvt, "x": tsm, "shift": sho, "relu": False}})
    tg = b.op({"Map": {"x": tcx, "map": {"HeadGather": {"dh": DH, "heads": H, "l": L}}}})
    s_ctx = s_v / 64 * 2.0 ** sho
    b.tag(tg, "ctx", l, s_ctx)
    to, _, pl["o"] = b.gemm_scaled(sd, P("attention.output.dense.weight"), (0, D), (0, D), tg, s_ctx, sA)
    to = b.tag(b.bias(to, P("attention.output.dense.bias"), 0, D, sA), "ao", l, sA)
    ty1 = b.tag(b.op({"Add": {"a": x, "b": to, "relu": False}}), "y1", l, sA)
    tn1 = b.tag(b.op({"LayerNorm": {"x": ty1, "len": D}}), "n1", l, 1 / 16)
    th, pl["ln1"] = b.affine(sd, tn1, P("attention.output.LayerNorm.weight"), P("attention.output.LayerNorm.bias"), s_B)
    b.tag(th, "h", l, s_B)
    parts = []
    for lo in range(0, FF, N):
        hi = min(FF, lo + N)
        s1 = pct([c[l]["f1"][lo:hi] for c in calib] + [c[l]["w1x"][lo:hi] for c in calib]) / 63
        t1, s1, _ = b.gemm_scaled(sd, P("intermediate.dense.weight"), (lo, hi), (0, D), th, s_B, s1)
        t1 = b.tag(b.bias(t1, P("intermediate.dense.bias"), lo, hi, s1), "f1", l, s1, (lo, hi))
        sg = pct([c[l]["g"][lo:hi] for c in calib]) / 63
        lut = np.clip(np.round(gelu(np.arange(-64, 64) * s1) / sg), -64, 64)
        tl = b.tag(b.op({"Lut": {"x": t1, "w": 0, "row": b.row(("lut", lut, s1, sg))}}), "g", l, sg, (lo, hi))
        t2x, _, _ = b.gemm_scaled(sd, P("output.dense.weight"), (0, D), (lo, hi), tl, sg, s_B)
        parts.append(t2x)
    t2 = parts[0]
    for tp in parts[1:]:
        t2 = b.op({"Add": {"a": t2, "b": tp, "relu": False}})
    t2 = b.tag(b.bias(t2, P("output.dense.bias"), 0, D, s_B), "f2", l, s_B)
    ty2 = b.tag(b.op({"Add": {"a": th, "b": t2, "relu": False}}), "y2", l, s_B)
    tn2 = b.tag(b.op({"LayerNorm": {"x": ty2, "len": D}}), "n2", l, 1 / 16)
    xo, pl["ln2"] = b.affine(sd, tn2, P("output.LayerNorm.weight"), P("output.LayerNorm.bias"), s_next)
    b.tag(xo, "xn", l, s_next)
    pl.update(sA=sA, sB=s_B)
    return xo, s_next


def build(sd, cfg, calib):
    """calib: floating-point intermediates (one per text). Returns (list of Builders, one per job; plan)."""
    col = lambda key, l: [c[l][key] for c in calib]
    s_A = [max(pct(col("x", l)), pct(col("y1", l)), pct(col("ao", l)), pct(col("wo", l))) / 63 for l in range(cfg.NL)]
    s_out = pct(col("xn", cfg.NL - 1)) / 63
    plan = {"s0": s_A[0], "layers": [], "L": L, "D": cfg.D, "layers_per_job": LAYERS_PER_JOB}
    jobs, b, x, s_x = [], None, 0, s_A[0]
    for l in range(cfg.NL):
        if l % LAYERS_PER_JOB == 0:
            b = Builder(cfg.D)
            jobs.append(b)
            x = 0
        pl = {}
        x, s_x = layer(b, sd, cfg, calib, l, x, s_x, s_A[l + 1] if l + 1 < cfg.NL else s_out, pl)
        plan["layers"].append(pl)
    plan["s_out"] = s_x
    plan["job_scales"] = [s_A[l] for l in range(0, cfg.NL, LAYERS_PER_JOB)]
    return jobs, plan


def program_for(b, lr):
    return json.loads(json.dumps(b.ops).replace('"LEN"', str(lr)))


def materialize(b, params):
    tab = np.zeros((N, N), np.int64)
    for r, rec in enumerate(b.trec):
        if rec[0] == "lut":
            tab[r, :128] = rec[1]
        elif rec[0] == "const":
            tab[r, :] = rec[1]
        else:
            _, key, lo, hi, fac, lim = rec
            tab[r, :hi - lo] = np.clip(np.round(params[key][lo:hi] * fac), -lim, lim)
    W = []
    for rec in b.wrec:
        if rec[0] == "table":
            W.append(tab)
        else:
            _, key, rs, cs, sw = rec
            blk = np.zeros((N, N), np.int64)
            blk[:rs[1] - rs[0], :cs[1] - cs[0]] = np.clip(np.round(params[key][rs[0]:rs[1], cs[0]:cs[1]] / sw), -63, 63)
            W.append(blk)
    return W


def ids_of(tk, text):
    return tk(text, truncation=True, max_length=L)["input_ids"]


def enc_input(sd, cfg, ids, s0):
    x = embed(sd, cfg, ids)
    t = np.zeros((N, N), np.int64)
    t[:cfg.D, :x.shape[1]] = np.clip(np.round(x / s0), -64, 64)
    return t


def run_int(bs, plan, params, sd, cfg, ids, Ws=None, keep=False):
    """The chained jobs (independent reference engine): integer [CLS] embedding (D,); each job only receives the D x L region."""
    Ws = Ws or [materialize(b, params) for b in bs]
    x = enc_input(sd, cfg, ids, plan["s0"])
    Ts = []
    for b, W in zip(bs, Ws):
        T, _ = R.run_program(x, W, program_for(b, len(ids)))
        Ts.append(T)
        x = np.zeros((N, N), np.int64)
        x[:cfg.D, :L] = T[-1][:cfg.D, :L]
    out = Ts[-1][-1][:cfg.D, 0].astype(np.int64)
    return (out, Ts) if keep else out


def export(out, bs, plan, params, cfg, extra=None):
    jobs = [{"ops_template": b.ops, "weights": [R.enc(w) for w in materialize(b, params)]} for b in bs]
    model = {"name": NAME.split("/")[-1] + "-int8", "hf": NAME, "n": N, "L": L, "D": cfg.D, "tokenizer": NAME, "jobs": jobs, "plan": plan, **(extra or {})}
    json.dump(model, open(out, "w"))
    return os.path.getsize(out)


def diag(bs, plan, params, sd, cfg, tk, texts):
    """Cosine and saturation, per labelled tensor, between the dequantised integer and the float (real tokens)."""
    acc = {}
    for text in texts:
        ids = ids_of(tk, text)
        lr = len(ids)
        _, inter = float_forward(params, cfg, ids)
        _, Ts = run_int(bs, plan, params, sd, cfg, ids, keep=True)
        for b, T in zip(bs, Ts):
            for t, (name, l, sc, rows) in b.tags.items():
                if rows == "sc":
                    it = np.concatenate([T[t][:lr, h * L:h * L + lr] for h in range(cfg.H)], 1).astype(np.float64) * sc
                    s_ = inter[l]["sc"]
                    if name == "sc":
                        fk = s_
                    else:
                        fk = np.concatenate([np.exp(s_[:, h * lr:(h + 1) * lr] - s_[:, h * lr:(h + 1) * lr].max(0)) for h in range(cfg.H)], 1)
                        fk = np.concatenate([fk[:, h * lr:(h + 1) * lr] / fk[:, h * lr:(h + 1) * lr].sum(0) for h in range(cfg.H)], 1)
                else:
                    lo, hi = rows
                    it = T[t][:hi - lo, :lr].astype(np.float64) * sc
                    if name in ("n1", "n2"):
                        src = inter[l]["y1" if name == "n1" else "y2"]
                        fk = (src - src.mean(0)) / np.sqrt(src.var(0) + cfg.eps)
                    else:
                        fk = inter[l][name][lo:hi]
                c = float(np.sum(it * fk) / (np.linalg.norm(it) * np.linalg.norm(fk) + 1e-12))
                sat = float(np.mean(np.abs(T[t][:it.shape[0], :lr]) >= 64))
                acc.setdefault((l, name), []).append((c, sat))
    for (l, name), v in acc.items():
        if name in ("k", "q", "v", "sc", "a", "ctx", "ao", "y1", "n1", "h", "f1", "g", "f2", "y2", "n2", "xn"):
            print(f"  layer {l:2d} {name:4s}: cosine {np.mean([x[0] for x in v]):.4f}   saturated {100 * np.mean([x[1] for x in v]):.2f} %")


def calibrate(sd, cfg, tk, texts):
    return [float_forward(sd, cfg, ids_of(tk, t))[1] for t in texts]


def main(out):
    t0 = time.time()
    tk, sd, cfg = load()
    print(f"{NAME}: {cfg.NL} layers, {cfg.H} heads of {cfg.DH}, d = {cfg.D}, FFN {cfg.FF}")
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news")
    texts = ds["test"]["text"][:20]
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(NAME, device="cpu")
    st.max_seq_length = L
    ref = st.encode(list(texts), normalize_embeddings=True)
    mine = np.array([float_forward(sd, cfg, ids_of(tk, t))[0] for t in texts])
    mine /= np.linalg.norm(mine, axis=1, keepdims=True)
    print(f"reimplemented floating-point model vs sentence-transformers: mean cosine {np.mean(np.sum(ref * mine, 1)):.6f} (20 texts)")
    bs, plan = build(sd, cfg, calibrate(sd, cfg, tk, ds["train"]["text"][:arg("--calib", 200, int)]))
    for k, b in enumerate(bs):
        n_gemm = sum(1 for o in b.ops if next(iter(o)) in ("Gemm", "GemmAT"))
        print(f"job {k}: {len(b.ops)} operations of which {n_gemm} GEMM, {len(b.wrec)} weight blocks, {len(b.trec)} table rows")
    params = dict(sd)
    if "--params" in sys.argv:
        params.update(dict(np.load(arg("--params", ""))))
    if "--diag" in sys.argv:
        diag(bs, plan, params, sd, cfg, tk, ds["test"]["text"][100:103])
        return
    nev = arg("--eval", 30, int)
    Ws = [materialize(b, params) for b in bs]
    cos = []
    for t in ds["test"]["text"][100:100 + nev]:
        ids = ids_of(tk, t)
        f = float_forward(sd, cfg, ids)[0]
        i = run_int(bs, plan, params, sd, cfg, ids, Ws).astype(np.float64)
        cos.append(float(f @ i / (np.linalg.norm(f) * np.linalg.norm(i) + 1e-12)))
    print(f"{nev} texts: mean cosine integer / float embedding {np.mean(cos):.4f} (min {np.min(cos):.3f}) in {time.time() - t0:.0f} s")
    size = export(out, bs, plan, params, cfg, {"report": {"cos_mean_ptq": float(np.mean(cos))}})
    print(f"written {out} ({size >> 20} MiB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/tmp/bert-int8.json")
