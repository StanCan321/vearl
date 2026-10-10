#!/usr/bin/env python3
"""The REAL all-MiniLM-L6-v2 (sentence-transformers: 6 layers, 12 heads of 32, d = 384, FFN 1536 GELU, LayerNorm with gamma/beta,
mean over the tokens) converted into an integer and verifiable VEARL PROGRAM, on a chain n = 1024 (docs/12 section 4 quinquies).

Layout: features x tokens (one column = one token, <= L = 64 tokens, ONE text per job; the padding tokens are excluded
by the `len` of the softmax and of the final mean). The 12 heads are processed in ONE product: HeadSpread places head h of Q in the column block
h (K^T . spread_Q = scores of all the heads side by side, 12 x 64 = 768 columns), a single Softmax, V . A, then HeadGather
takes back for each row the block of its head. Integer LayerNorm (16 steps per standard deviation) then Mul (gamma, per row) and Bias (beta).
FFN 1536 > n: W1 in two row blocks (1024 + 512), GELU by table (Lut), W2 in two column blocks, sum by Add.
The input embedding (word table + positions + LayerNorm) is computed by the client (a table lookup); the rest is verified.

The program is entirely determined by a PLAN (scales and shifts, fixed) and floating-point PARAMETERS: `materialize(b, params, lr)`
rebuilds the weight blocks; tools/qat_minilm.py adjusts the parameters through an exact simulation of the program then re-exports.

    python3 quantize_minilm.py OUT.json [--calib 200] [--eval 300] [--pct 99.9] [--diag] [--params P.npz]
"""
import json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vearl_ref as R

N, L, D, H, DH, FF = 1024, 64, 384, 12, 32, 1536
R.N = N
C_SOFTMAX = math.log(2) / 8
arg = lambda name, default, cast=str: cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
PCT = arg("--pct", 99.9, float)
NAME = "sentence-transformers/all-MiniLM-L6-v2"


def erf(x):
    import torch
    return torch.special.erf(torch.from_numpy(np.asarray(x, dtype=np.float64))).numpy()


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def load():
    from transformers import AutoTokenizer, AutoModel
    tk = AutoTokenizer.from_pretrained(NAME)
    m = AutoModel.from_pretrained(NAME).eval()
    return tk, {k: v.detach().numpy().astype(np.float64) for k, v in m.state_dict().items()}


def K(l, name):
    return f"encoder.layer.{l}.{name}"


def ln_f(x, g, b, eps=1e-12):
    mu = x.mean(0, keepdims=True)
    var = x.var(0, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g[:, None] + b[:, None]


def embed(sd, ids):
    """Input embedding (client): words + positions + type 0, LayerNorm; output (D, Lr)."""
    x = sd["embeddings.word_embeddings.weight"][ids] + sd["embeddings.position_embeddings.weight"][:len(ids)] + sd["embeddings.token_type_embeddings.weight"][0]
    return ln_f(x.T, sd["embeddings.LayerNorm.weight"], sd["embeddings.LayerNorm.bias"])


def float_forward(sd, ids):
    """EXACT floating-point model (same formulas as BertModel), layout (features, tokens); also returns the intermediates."""
    x = embed(sd, ids)
    Lr = x.shape[1]
    inter = []
    for l in range(6):
        P = lambda n: sd[K(l, n)]
        q = P("attention.self.query.weight") @ x + P("attention.self.query.bias")[:, None]
        k = P("attention.self.key.weight") @ x + P("attention.self.key.bias")[:, None]
        v = P("attention.self.value.weight") @ x + P("attention.self.value.bias")[:, None]
        ctx = np.zeros((D, Lr))
        scs = []
        for h in range(H):
            s = slice(h * DH, (h + 1) * DH)
            sc = k[s].T @ q[s] / math.sqrt(DH)                                   # (key, query)
            a = np.exp(sc - sc.max(0, keepdims=True))
            a /= a.sum(0, keepdims=True)
            ctx[s] = v[s] @ a
            scs.append(sc)
        wo = P("attention.output.dense.weight") @ ctx
        ao = wo + P("attention.output.dense.bias")[:, None]
        y1 = x + ao
        hh = ln_f(y1, P("attention.output.LayerNorm.weight"), P("attention.output.LayerNorm.bias"))
        W1, W2 = P("intermediate.dense.weight"), P("output.dense.weight")
        w1x = W1 @ hh
        f1 = w1x + P("intermediate.dense.bias")[:, None]
        g = gelu(f1)
        f2a, f2b = W2[:, :1024] @ g[:1024], W2[:, 1024:] @ g[1024:]
        f2 = f2a + f2b + P("output.dense.bias")[:, None]
        y2 = hh + f2
        xn = ln_f(y2, P("output.LayerNorm.weight"), P("output.LayerNorm.bias"))
        inter.append(dict(x=x, q=q, k=k, v=v, sc=np.concatenate(scs, 1), ctx=ctx, wo=wo, ao=ao, y1=y1, h=hh, w1x=w1x, f1=f1, g=g,
                          f2a=f2a, f2b=f2b, f2=f2, y2=y2, xn=xn,
                          n1=(y1 - y1.mean(0)) / np.sqrt(y1.var(0) + 1e-12), n2=(y2 - y2.mean(0)) / np.sqrt(y2.var(0) + 1e-12)))
        x = xn
    return x.mean(1), inter


def pct(arrs):
    return float(np.percentile(np.abs(np.concatenate([a.ravel() for a in arrs])), PCT))


class Builder:
    """Operations + RECIPES of the weight blocks (param, slices, scale) and of the table rows (block 0)."""

    def __init__(self):
        self.ops, self.wrec, self.trec, self.tags = [], [("table",)], [], {}

    def tag(self, t, name, l, scale, rows=(0, D)):
        self.tags[t] = (name, l, scale, rows)
        return t

    def op(self, o):
        self.ops.append(o)
        return len(self.ops)

    def row(self, rec):
        self.trec.append(rec)
        return len(self.trec) - 1

    def gemm_scaled(self, sd, key, rs, cs, x, s_in, s_t):
        """Gemm whose output is at the TARGET scale s_t (weights at scale s_t / (s_in 2^sh), without clipping the weights)."""
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
        # +-64: the protocol only judges a fraud proof on a Bias if the bias row stays in [-64, 64]
        return self.op({"Bias": {"x": x, "w": 0, "row": self.row(("vec", key, lo, hi, 1.0 / s, 64)), "relu": False}})

    def affine(self, sd, t_ln, gkey, bkey, s_out):
        """Integer LayerNorm (unit 1/16) -> Mul (gamma, shift sh) -> Bias (beta), output at scale s_out."""
        # +-64: the workers' registry and the model loader only accept weights in [-64, 64], table included
        f = np.abs(sd[gkey]).max() / (16 * s_out)                  # g_int / 2^sh = gamma / (16 s_out)
        sh = max(0, math.floor(math.log2(64 / f)))
        fac = 2.0 ** sh / (16 * s_out)
        t = self.op({"Mul": {"x": t_ln, "w": 0, "row": self.row(("vec", gkey, 0, D, fac, 64)), "shift": sh}})
        return self.bias(t, bkey, 0, D, s_out), dict(sh=sh, fac=fac, s=s_out, g=gkey, b=bkey)


def build(sd, calib):
    """calib: list of floating-point intermediates (one per text). Returns (Builder, plan)."""
    b = Builder()
    col = lambda key, l: [c[l][key] for c in calib]
    s_A = [max(pct(col("x", l)), pct(col("y1", l)), pct(col("ao", l)), pct(col("wo", l))) / 63 for l in range(6)]
    plan = {"s0": s_A[0], "layers": []}
    x, s_x = 0, s_A[0]
    for l in range(6):
        P = lambda n: K(l, n)
        sA = s_A[l]
        s_B = max(pct(col("h", l)), pct(col("y2", l)), pct(col("f2", l)), pct(col("f2a", l)), pct(col("f2b", l))) / 63
        pl = {"sA": sA, "sB": s_B, "s_in": s_x}
        s_k = max(pct(col("k", l)), 1e-9) / 63
        s_v = max(pct(col("v", l)), 1e-9) / 63
        tk, s_k, pl["k"] = b.gemm_scaled(sd, P("attention.self.key.weight"), (0, D), (0, D), x, s_x, s_k)
        tk = b.tag(b.bias(tk, P("attention.self.key.bias"), 0, D, s_k), "k", l, s_k)
        tv, s_v, pl["v"] = b.gemm_scaled(sd, P("attention.self.value.weight"), (0, D), (0, D), x, s_x, s_v)
        tv = b.tag(b.bias(tv, P("attention.self.value.bias"), 0, D, s_v), "v", l, s_v)
        sq_min = max(pct(col("q", l)) / 63, 1e-9)
        sh_sc = max(0, math.floor(math.log2(math.sqrt(DH) * C_SOFTMAX / (s_k * sq_min))))
        s_q_t = math.sqrt(DH) * C_SOFTMAX * 2.0 ** (-sh_sc) / s_k
        tq, s_q, pl["q"] = b.gemm_scaled(sd, P("attention.self.query.weight"), (0, D), (0, D), x, s_x, s_q_t)
        tq = b.tag(b.bias(tq, P("attention.self.query.bias"), 0, D, s_q), "q", l, s_q)
        pl.update(sh_sc=sh_sc, temperature=s_k * s_q * 2.0 ** sh_sc / (math.sqrt(DH) * C_SOFTMAX))
        tqs = b.op({"Map": {"x": tq, "map": {"HeadSpread": {"dh": DH, "heads": H, "l": L}}}})
        tsc = b.tag(b.op({"GemmAT": {"a": tk, "x": tqs, "shift": sh_sc, "relu": False}}), "sc", l, C_SOFTMAX, "sc")
        tsm = b.tag(b.op({"Softmax": {"x": tsc, "len": "LEN"}}), "a", l, 1 / 64, "sc")
        tvt = b.op({"Map": {"x": tv, "map": "Transpose"}})
        zo = pct(col("ctx", l)) * 64 / s_v
        sho = max(0, math.ceil(math.log2(max(zo, 1) / 63)))
        tcx = b.op({"GemmAT": {"a": tvt, "x": tsm, "shift": sho, "relu": False}})
        tg = b.op({"Map": {"x": tcx, "map": {"HeadGather": {"dh": DH, "heads": H, "l": L}}}})
        s_ctx = s_v / 64 * 2.0 ** sho
        b.tag(tg, "ctx", l, s_ctx)
        pl.update(sho=sho, s_ctx=s_ctx)
        to, _, pl["o"] = b.gemm_scaled(sd, P("attention.output.dense.weight"), (0, D), (0, D), tg, s_ctx, sA)
        to = b.tag(b.bias(to, P("attention.output.dense.bias"), 0, D, sA), "ao", l, sA)
        ty1 = b.tag(b.op({"Add": {"a": x, "b": to, "relu": False}}), "y1", l, sA)
        tn1 = b.tag(b.op({"LayerNorm": {"x": ty1, "len": D}}), "n1", l, 1 / 16)
        th, pl["ln1"] = b.affine(sd, tn1, P("attention.output.LayerNorm.weight"), P("attention.output.LayerNorm.bias"), s_B)
        b.tag(th, "h", l, s_B)
        pl["ffn"] = []
        gs = []
        for lo, hi in ((0, 1024), (1024, 1536)):
            s1 = pct([c[l]["f1"][lo:hi] for c in calib] + [c[l]["w1x"][lo:hi] for c in calib]) / 63
            t1, s1, g1 = b.gemm_scaled(sd, P("intermediate.dense.weight"), (lo, hi), (0, D), th, s_B, s1)
            t1 = b.tag(b.bias(t1, P("intermediate.dense.bias"), lo, hi, s1), "f1", l, s1, (lo, hi))
            sg = pct([c[l]["g"][lo:hi] for c in calib]) / 63
            lut = np.clip(np.round(gelu(np.arange(-64, 64) * s1) / sg), -64, 64)
            r = b.row(("lut", lut))
            tl = b.tag(b.op({"Lut": {"x": t1, "w": 0, "row": r}}), "g", l, sg, (lo, hi))
            gs.append((tl, sg, lo, hi))
            pl["ffn"].append(dict(lo=lo, hi=hi, w1=g1, s1=s1, sg=sg, lut=lut.tolist()))
        pl["w2"] = []
        outs = []
        for (tl, sg, lo, hi) in gs:
            t2x, _, g2 = b.gemm_scaled(sd, P("output.dense.weight"), (0, D), (lo, hi), tl, sg, s_B)
            outs.append(t2x)
            pl["w2"].append(g2)
        t2 = b.op({"Add": {"a": outs[0], "b": outs[1], "relu": False}})
        t2 = b.tag(b.bias(t2, P("output.dense.bias"), 0, D, s_B), "f2", l, s_B)
        ty2 = b.tag(b.op({"Add": {"a": th, "b": t2, "relu": False}}), "y2", l, s_B)
        tn2 = b.tag(b.op({"LayerNorm": {"x": ty2, "len": D}}), "n2", l, 1 / 16)
        s_next = s_A[l + 1] if l < 5 else pct(col("xn", 5)) / 63
        x, pl["ln2"] = b.affine(sd, tn2, P("output.LayerNorm.weight"), P("output.LayerNorm.bias"), s_next)
        b.tag(x, "xn", l, s_next)
        s_x = s_next
        plan["layers"].append(pl)
    # mean over the real tokens: Transpose, Gemm(row of LEN ones) >> SHP, Transpose; "ones" weight specific to the length
    tt = b.op({"Map": {"x": x, "map": "Transpose"}})
    b.wrec.append(("ones",))
    b.op({"Gemm": {"w": len(b.wrec) - 1, "x": tt, "shift": "SHP", "relu": False}})
    b.op({"Map": {"x": len(b.ops), "map": "Transpose"}})
    plan["s_out_tok"] = s_x
    return b, plan


def shp_for(lr):
    """Shift of the sum of lr tokens (values <= 64): the sum stays within +-64."""
    return max(0, math.ceil(math.log2(max(lr, 1))))


def materialize(b, params, lr):
    """Concrete weight blocks for a text of lr tokens, from the floating-point parameters (dict name -> array)."""
    tab = np.zeros((N, N), np.int64)
    for r, rec in enumerate(b.trec):
        if rec[0] == "lut":
            tab[r, :128] = rec[1]
        else:
            _, key, lo, hi, fac, lim = rec
            tab[r, :hi - lo] = np.clip(np.round(params[key][lo:hi] * fac), -lim, lim)
    W = []
    for rec in b.wrec:
        if rec[0] == "table":
            W.append(tab)
        elif rec[0] == "ones":
            o = np.zeros((N, N), np.int64)
            o[0, :lr] = 1
            W.append(o)
        else:
            _, key, rs, cs, sw = rec
            blk = np.zeros((N, N), np.int64)
            blk[:rs[1] - rs[0], :cs[1] - cs[0]] = np.clip(np.round(params[key][rs[0]:rs[1], cs[0]:cs[1]] / sw), -63, 63)
            W.append(blk)
    return W


def program_for(b, lr):
    return json.loads(json.dumps(b.ops).replace('"LEN"', str(lr)).replace('"SHP"', str(shp_for(lr))))


def enc_input(sd, ids, s0):
    x = embed(sd, ids)
    t = np.zeros((N, N), np.int64)
    t[:D, :x.shape[1]] = np.clip(np.round(x / s0), -64, 64)
    return t


def ids_of(tk, text):
    return tk(text, truncation=True, max_length=L)["input_ids"]


def run_int(b, plan, params, sd, ids):
    """Integer program (independent reference engine) for a text: int8 embedding (D values) and tensors."""
    ops, W = program_for(b, len(ids)), materialize(b, params, len(ids))
    T, _ = R.run_program(enc_input(sd, ids, plan["s0"]), W, ops)
    return T[-1][:D, 0].astype(np.int64), T


def diag(sd, tk, b, plan, params, texts):
    """Cosine, for each labelled tensor, between the dequantised integer value and the float (real tokens only)."""
    acc = {}
    for text in texts:
        ids = ids_of(tk, text)
        lr = len(ids)
        _, inter = float_forward(params, ids)
        _, T = run_int(b, plan, params, sd, ids)
        for t, (name, l, sc, rows) in b.tags.items():
            if rows == "sc":
                it = np.concatenate([T[t][:lr, h * L:h * L + lr] for h in range(H)], 1).astype(np.float64) * sc
                s_ = inter[l]["sc"]
                if name == "sc":
                    fk = s_
                else:
                    fk = np.concatenate([np.exp(s_[:, h * lr:(h + 1) * lr] - s_[:, h * lr:(h + 1) * lr].max(0)) for h in range(H)], 1)
                    fk = np.concatenate([fk[:, h * lr:(h + 1) * lr] / fk[:, h * lr:(h + 1) * lr].sum(0) for h in range(H)], 1)
            else:
                lo, hi = rows
                it = T[t][:hi - lo, :lr].astype(np.float64) * sc
                fk = inter[l][name][lo:hi] if name not in ("n1", "n2") else inter[l][name]
            c = float(np.sum(it * fk) / (np.linalg.norm(it) * np.linalg.norm(fk) + 1e-12))
            sat = float(np.mean(np.abs(T[t][:it.shape[0], :lr]) >= 64))
            acc.setdefault((l, name), []).append((c, sat))
    order = {k: i for i, k in enumerate(dict.fromkeys((l, n) for (n, l, _, _) in b.tags.values()))}
    for (l, name), v in sorted(acc.items(), key=lambda kv: order[kv[0]]):
        print(f"  layer {l} {name:4s}: cosine {np.mean([x[0] for x in v]):.4f}   saturated {100 * np.mean([x[1] for x in v]):.2f} %")


def evaluate(b, plan, params, sd, tk, texts, labels, teacher=None):
    """Integer program against the ORIGINAL floating-point MiniLM (sd); optional `teacher` = reference embeddings already computed."""
    fl, it = [], []
    for k, text in enumerate(texts):
        ids = ids_of(tk, text)
        fl.append(teacher[k] if teacher is not None else float_forward(sd, ids)[0])
        it.append(run_int(b, plan, params, sd, ids)[0].astype(np.float64))
    fl, it = np.array(fl), np.array(it)
    nz = lambda E: E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    cos = np.sum(nz(fl) * nz(it), 1)

    def nn(E, y):
        S = nz(E) @ nz(E).T
        np.fill_diagonal(S, -2)
        return float((y[S.argmax(1)] == y).mean())

    return dict(cos_mean=float(cos.mean()), cos_min=float(cos.min()), nn_float=nn(fl, labels), nn_int=nn(it, labels), texts=len(texts)), it


def export(out, b, plan, params, extra=None):
    """Model for the client: operation template, weights (block of ones for a length of 1: the client replaces it), plan."""
    W = materialize(b, params, 1)
    model = {"name": "minilm-l6-int8", "n": N, "L": L, "tokenizer": NAME, "ops_template": b.ops, "weights": [R.enc(w) for w in W],
             "ones_index": [i for i, r in enumerate(b.wrec) if r[0] == "ones"][0], "plan": plan, **(extra or {})}
    json.dump(model, open(out, "w"))
    return os.path.getsize(out)


def calibrate(sd, tk, texts):
    return [float_forward(sd, ids_of(tk, t))[1] for t in texts]


def main(out):
    t0 = time.time()
    tk, sd = load()
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news")
    ncal, nev = arg("--calib", 200, int), arg("--eval", 300, int)
    tr = ds["train"]["text"][:ncal]
    te, te_y = ds["test"]["text"][:nev], np.array(ds["test"]["label"][:nev])
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(NAME, device="cpu")
    st.max_seq_length = L
    ref = st.encode(list(te[:50]), normalize_embeddings=True)
    mine = np.array([float_forward(sd, ids_of(tk, t))[0] for t in te[:50]])
    mine /= np.linalg.norm(mine, axis=1, keepdims=True)
    print(f"reimplemented floating-point model vs sentence-transformers: mean cosine {np.mean(np.sum(ref * mine, 1)):.6f} (50 texts)")
    b, plan = build(sd, calibrate(sd, tk, tr))
    params = dict(np.load(arg("--params", ""))) if "--params" in sys.argv else sd
    n_gemm = sum(1 for o in b.ops if next(iter(o)) in ("Gemm", "GemmAT"))
    print(f"program: {len(b.ops)} operations of which {n_gemm} GEMM 1024^3, {len(b.wrec)} weight blocks, {len(b.trec)} table rows")
    if "--diag" in sys.argv:
        diag(sd, tk, b, plan, params, te[:5])
        return
    t1 = time.time()
    rep, _ = evaluate(b, plan, params, sd, tk, te, te_y)
    print(f"{nev} AG News texts in {time.time() - t1:.0f} s: mean cosine integer program / floating-point MiniLM {rep['cos_mean']:.4f} (min {rep['cos_min']:.3f}); "
          f"nearest neighbour of the same category: float {100 * rep['nn_float']:.1f} %, integer {100 * rep['nn_int']:.1f} %")
    size = export(out, b, plan, params, {"report": rep})
    print(f"written {out} ({size >> 20} MiB) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/tmp/minilm-int8.json")
