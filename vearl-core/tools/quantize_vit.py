#!/usr/bin/env python3
"""High-resolution vision: DeiT-Tiny (facebook/deit-tiny-patch16-224, Apache 2.0: 12 layers, 3 heads of 64, d = 192, FFN 768 GELU,
LayerNorm BEFORE each block, 224 x 224 images, 1000 ImageNet categories) converted into TWO integer and verifiable VEARL programs,
chained (layers 0-5 then 6-11 + classifier), chain n = 1024.

Layout: features x tokens, 197 columns (CLS token in column 0, then the 196 16 x 16 squares of the image, row by row).
Input x0: rows 0..767 = the 768 normalised pixels of a square (order channel, y, x: that of the convolution weights), CLS column zero;
rows 768..964 = position marker (value 64 at row 768 + t for column t). The CHAIN computes the embedding: one Gemm for the
projection of the squares, one Gemm for position + CLS token + bias (weight block at columns 768..964), one Add. Everything is therefore verified.
Caution: HeadSpread (3 heads x 197 = 591 columns), a Softmax of length 197, V . A, HeadGather (like MiniLM).
Residual stream: one scale per layer; a Mul by a constant row changes it between two layers if needed.

    python3 quantize_vit.py OUT.json [--calib 64] [--eval 500] [--pct 99.99] [--diag] [--params P.npz]
"""
import json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vearl_ref as R

N, D, H, DH, FF, T, NL, PX = 1024, 192, 3, 64, 768, 197, 12, 768
POS0 = PX                                   # first row of the position marker in x0
R.N = N
C_SOFTMAX = math.log(2) / 8
arg = lambda name, default, cast=str: cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
PCT = arg("--pct", 99.99, float)
NAME = "facebook/deit-tiny-patch16-224"
FINE_ATTN = "--coarse-attn" not in sys.argv   # SoftmaxLo (attention in 1/4096); --coarse-attn: Softmax only (1/64)
FINE_BIAS = "--no-fine-bias" not in sys.argv
SPLIT = 6                                   # job A: layers 0..5; job B: layers 6..11 + classifier
DATA = os.path.expanduser("~/data/imagenette2-320")
# Imagenette -> ImageNet-1k indices (checked against id2label at load time)
SYNSETS = {"n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491, "n03028079": 497,
           "n03394916": 566, "n03417042": 569, "n03425413": 571, "n03445777": 574, "n03888257": 701}


def erf(x):
    import torch
    return torch.special.erf(torch.from_numpy(np.asarray(x, dtype=np.float64))).numpy()


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def load():
    from transformers import ViTForImageClassification
    m = ViTForImageClassification.from_pretrained(NAME).eval()
    proc = PREP
    sd = {k: v.detach().numpy().astype(np.float64) for k, v in m.state_dict().items()}
    return m, proc, sd, m.config.id2label


def K(l, name):
    return f"vit.layers.{l}.{name}"


# Imagewoof (10 dog breeds, never seen during training): GENERALISATION test on other categories
WOOF = os.path.expanduser("~/data/imagewoof2-320")
WOOF_SYNSETS = {"n02086240": 155, "n02087394": 159, "n02088364": 162, "n02089973": 167, "n02093754": 182,
                "n02096294": 193, "n02099601": 207, "n02105641": 229, "n02111889": 258, "n02115641": 273}


def images(split, per_class, offset=0, woof=False):
    """(paths, ImageNet labels): per_class images of each Imagenette (or Imagewoof) category, sorted (reproducible)."""
    paths, labels = [], []
    root, syns = (WOOF, WOOF_SYNSETS) if woof else (DATA, SYNSETS)
    for syn, idx in syns.items():
        files = sorted(os.listdir(f"{root}/{split}/{syn}"))[offset:offset + per_class]
        paths += [f"{root}/{split}/{syn}/{f}" for f in files]
        labels += [idx] * len(files)
    return paths, np.array(labels)


# Preprocessing (without torchvision): "deit" = protocol of the DeiT paper (short side at 256 bicubic, central crop 224,
# ImageNet mean/standard deviation); "hf" = configuration file of the repository (direct 224 x 224 resize, mean/standard deviation 0.5).
PREP = arg("--prep", "deit")


def pixels(proc, path):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if proc == "deit":
        w, h = img.size
        s = 256 / min(w, h)
        img = img.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BICUBIC)
        w, h = img.size
        l, t = (w - 224) // 2, (h - 224) // 2
        img = img.crop((l, t, l + 224, t + 224))
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    else:
        img = img.resize((224, 224), Image.BILINEAR)
        mean, std = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])
    a = np.asarray(img, dtype=np.float64) / 255.0
    return ((a - mean) / std).transpose(2, 0, 1)                                      # (3, 224, 224)


def patches(pix):
    """(768, 196): column p = square p (row py, column px), values in the order (channel, y, x)."""
    return pix.reshape(3, 14, 16, 14, 16).transpose(1, 3, 0, 2, 4).reshape(196, PX).T


def ln_f(x, g, b, eps=1e-12):
    mu = x.mean(0, keepdims=True)
    var = x.var(0, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g[:, None] + b[:, None]


def derived(sd):
    """Derived parameters: projection of the squares (192 x 768) and effective position (192 x 197: position + CLS / bias)."""
    out = dict(sd)
    out["patch_w"] = sd["vit.embeddings.patch_embeddings.projection.weight"].reshape(D, PX)
    pos = sd["vit.embeddings.position_embeddings"][0].T.copy()            # (192, 197)
    pos[:, 0] += sd["vit.embeddings.cls_token"][0, 0]
    pos[:, 1:] += sd["vit.embeddings.patch_embeddings.projection.bias"][:, None]
    out["pos_eff"] = pos
    return out


def float_forward(sd, pix):
    """EXACT floating-point model (formulas of ViTModel), layout (features, tokens); logits and intermediates."""
    P0 = patches(pix)
    x = np.concatenate([np.zeros((D, 1)), sd["patch_w"] @ P0], 1) + sd["pos_eff"]
    inter = [dict(pe=np.concatenate([np.zeros((D, 1)), sd["patch_w"] @ P0], 1), pos=sd["pos_eff"])]
    for l in range(NL):
        P = lambda n: sd[K(l, n)]
        h = ln_f(x, P("layernorm_before.weight"), P("layernorm_before.bias"))
        q = P("attention.q_proj.weight") @ h + P("attention.q_proj.bias")[:, None]
        k = P("attention.k_proj.weight") @ h + P("attention.k_proj.bias")[:, None]
        v = P("attention.v_proj.weight") @ h + P("attention.v_proj.bias")[:, None]
        ctx, scs = np.zeros((D, T)), []
        for hh in range(H):
            s = slice(hh * DH, (hh + 1) * DH)
            sc = k[s].T @ q[s] / math.sqrt(DH)
            a = np.exp(sc - sc.max(0, keepdims=True))
            a /= a.sum(0, keepdims=True)
            ctx[s] = v[s] @ a
            scs.append(sc)
        wo = P("attention.o_proj.weight") @ ctx
        ao = wo + P("attention.o_proj.bias")[:, None]
        y1 = x + ao
        h2 = ln_f(y1, P("layernorm_after.weight"), P("layernorm_after.bias"))
        w1x = P("mlp.fc1.weight") @ h2
        f1 = w1x + P("mlp.fc1.bias")[:, None]
        g = gelu(f1)
        w2x = P("mlp.fc2.weight") @ g
        f2 = w2x + P("mlp.fc2.bias")[:, None]
        xo = y1 + f2
        inter.append(dict(x=x, h=h, q=q, k=k, v=v, sc=np.concatenate(scs, 1), ctx=ctx, wo=wo, ao=ao, y1=y1, h2=h2, w1x=w1x, f1=f1, g=g,
                          w2x=w2x, f2=f2, xo=xo))
        x = xo
    xn = ln_f(x, sd["vit.layernorm.weight"], sd["vit.layernorm.bias"])
    logits = sd["classifier.weight"] @ xn[:, 0] + sd["classifier.bias"]
    inter.append(dict(x=x, xn=xn, zc=sd["classifier.weight"] @ xn, logits=logits))
    return logits, inter


def pct(arrs, p=None):
    return float(np.percentile(np.abs(np.concatenate([np.asarray(a).ravel() for a in arrs])), PCT if p is None else p))


class Builder:
    """A job: operations + RECIPES of the weight blocks (param, slices, scale, starting column) and of the table rows (block 0)."""

    def __init__(self):
        self.ops, self.wrec, self.trec, self.tags = [], [("table",)], [], {}

    def tag(self, t, name, l, scale, rows=(0, D), cols=T):
        self.tags[t] = (name, l, scale, rows, cols)
        return t

    def op(self, o):
        self.ops.append(o)
        return len(self.ops)

    def row(self, rec):
        self.trec.append(rec)
        return len(self.trec) - 1

    def gemm_scaled(self, sd, key, rs, cs, x, s_in, s_t, col0=0):
        """Gemm whose output is at the TARGET scale s_t; the weight block starts at column col0 (input rows col0..)."""
        wf = sd[key][rs[0]:rs[1], cs[0]:cs[1]]
        sw_nat = np.abs(wf).max() / 63.0
        sh = math.floor(math.log2(s_t / (s_in * sw_nat)))
        if sh < 0:
            sh, s_t = 0, s_in * sw_nat
        sw = s_t / (s_in * 2.0 ** sh)
        self.wrec.append(("mat", key, rs, cs, sw, col0))
        t = self.op({"Gemm": {"w": len(self.wrec) - 1, "x": x, "shift": sh, "relu": False}})
        return t, s_t, dict(w=len(self.wrec) - 1, sw=sw, sh=sh, s=s_t)

    def bias(self, x, key, lo, hi, s):
        # +-64: a fraud proof on a Bias is only judged if the row stays in [-64, 64]
        return self.op({"Bias": {"x": x, "w": 0, "row": self.row(("vec", key, lo, hi, 1.0 / s, 64)), "relu": False}})

    def affine(self, sd, t_ln, gkey, bkey, s_out):
        """Integer LayerNorm (unit 1/16) -> Mul (gamma, shift sh) -> Bias (beta), output at scale s_out."""
        f = np.abs(sd[gkey]).max() / (16 * s_out)
        sh = max(0, math.floor(math.log2(64 / f)))
        fac = 2.0 ** sh / (16 * s_out)
        t = self.op({"Mul": {"x": t_ln, "w": 0, "row": self.row(("vec", gkey, 0, D, fac, 64)), "shift": sh}})
        return self.bias(t, bkey, 0, D, s_out), dict(sh=sh, fac=fac, s=s_out)

    def rescale(self, x, s_from, s_to):
        """Changes the scale of the residual stream: Mul by a constant row c (<= 64) and shift 6; returns (tensor, exact scale)."""
        c = int(min(64, max(1, round(64 * s_from / s_to))))
        if c == 64:
            return x, s_from
        t = self.op({"Mul": {"x": x, "w": 0, "row": self.row(("const", c)), "shift": 6}})
        return t, s_from * 64 / c


def layer(b, sd, calib, l, x, s_x, s_next_target, pl):
    col = lambda key: [c[l + 1][key] for c in calib]
    P = lambda n: K(l, n)
    tn = b.tag(b.op({"LayerNorm": {"x": x, "len": D}}), "n1", l, 1 / 16)
    s_h = pct(col("h")) / 63
    th, pl["ln1"] = b.affine(sd, tn, P("layernorm_before.weight"), P("layernorm_before.bias"), s_h)
    b.tag(th, "h", l, s_h)
    s_k = max(pct(col("k")), 1e-9) / 63
    s_v = max(pct(col("v")), 1e-9) / 63
    tk, s_k, pl["k"] = b.gemm_scaled(sd, P("attention.k_proj.weight"), (0, D), (0, D), th, s_h, s_k)
    tk = b.tag(b.bias(tk, P("attention.k_proj.bias"), 0, D, s_k), "k", l, s_k)
    tv, s_v, pl["v"] = b.gemm_scaled(sd, P("attention.v_proj.weight"), (0, D), (0, D), th, s_h, s_v)
    tv = b.tag(b.bias(tv, P("attention.v_proj.bias"), 0, D, s_v), "v", l, s_v)
    sq_min = max(pct(col("q")) / 63, 1e-9)
    sh_sc = max(0, math.floor(math.log2(math.sqrt(DH) * C_SOFTMAX / (s_k * sq_min))))
    s_q_t = math.sqrt(DH) * C_SOFTMAX * 2.0 ** (-sh_sc) / s_k
    tq, s_q, pl["q"] = b.gemm_scaled(sd, P("attention.q_proj.weight"), (0, D), (0, D), th, s_h, s_q_t)
    tq = b.tag(b.bias(tq, P("attention.q_proj.bias"), 0, D, s_q), "q", l, s_q)
    pl.update(sh_sc=sh_sc, temperature=s_k * s_q * 2.0 ** sh_sc / (math.sqrt(DH) * C_SOFTMAX))
    tqs = b.op({"Map": {"x": tq, "map": {"HeadSpread": {"dh": DH, "heads": H, "l": T}}}})
    tsc = b.tag(b.op({"GemmAT": {"a": tk, "x": tqs, "shift": sh_sc, "relu": False}}), "sc", l, C_SOFTMAX, "sc")
    tsm = b.tag(b.op({"Softmax": {"x": tsc, "len": T}}), "a", l, 1 / 64, "sc")
    tvt = b.op({"Map": {"x": tv, "map": "Transpose"}})
    zo = pct(col("ctx")) * 64 / s_v
    sho = max(0, math.ceil(math.log2(max(zo, 1) / 63)))
    if FINE_ATTN:
        # probabilities in 1/4096: V.A_high >> sho  +  V.A_fine >> (sho + 6)   (SoftmaxLo: fine part, in [-32, 32])
        tlo = b.op({"SoftmaxLo": {"x": tsc, "len": T}})
        thi = b.op({"GemmAT": {"a": tvt, "x": tsm, "shift": sho, "relu": False}})
        tfi = b.op({"GemmAT": {"a": tvt, "x": tlo, "shift": sho + 6, "relu": False}})
        tcx = b.op({"Add": {"a": thi, "b": tfi, "relu": False}})
        if FINE_BIAS:
            # two default roundings (-1/2 unit each on average): +1 recentres the error
            tcx = b.op({"Bias": {"x": tcx, "w": 0, "row": b.row(("const", 1)), "relu": False}})
    else:
        tcx = b.op({"GemmAT": {"a": tvt, "x": tsm, "shift": sho, "relu": False}})
    tg = b.op({"Map": {"x": tcx, "map": {"HeadGather": {"dh": DH, "heads": H, "l": T}}}})
    s_ctx = s_v / 64 * 2.0 ** sho
    b.tag(tg, "ctx", l, s_ctx)
    pl.update(sho=sho, s_ctx=s_ctx)
    to, _, pl["o"] = b.gemm_scaled(sd, P("attention.o_proj.weight"), (0, D), (0, D), tg, s_ctx, s_x)
    to = b.tag(b.bias(to, P("attention.o_proj.bias"), 0, D, s_x), "ao", l, s_x)
    ty1 = b.tag(b.op({"Add": {"a": x, "b": to, "relu": False}}), "y1", l, s_x)
    tn2 = b.tag(b.op({"LayerNorm": {"x": ty1, "len": D}}), "n2", l, 1 / 16)
    s_h2 = pct(col("h2")) / 63
    th2, pl["ln2"] = b.affine(sd, tn2, P("layernorm_after.weight"), P("layernorm_after.bias"), s_h2)
    b.tag(th2, "h2", l, s_h2)
    s1 = pct(col("f1") + col("w1x")) / 63
    t1, s1, pl["w1"] = b.gemm_scaled(sd, P("mlp.fc1.weight"), (0, FF), (0, D), th2, s_h2, s1)
    t1 = b.tag(b.bias(t1, P("mlp.fc1.bias"), 0, FF, s1), "f1", l, s1, (0, FF))
    sg = pct(col("g")) / 63
    lut = np.clip(np.round(gelu(np.arange(-64, 64) * s1) / sg), -64, 64)
    tl = b.tag(b.op({"Lut": {"x": t1, "w": 0, "row": b.row(("lut", lut, s1, sg))}}), "g", l, sg, (0, FF))
    t2, _, pl["w2"] = b.gemm_scaled(sd, P("mlp.fc2.weight"), (0, D), (0, FF), tl, sg, s_x)
    t2 = b.tag(b.bias(t2, P("mlp.fc2.bias"), 0, D, s_x), "f2", l, s_x)
    tx = b.tag(b.op({"Add": {"a": ty1, "b": t2, "relu": False}}), "xo", l, s_x)
    pl.update(s_x=s_x, s_h=s_h, s_h2=s_h2, s1=s1, sg=sg)
    return b.rescale(tx, s_x, s_next_target)


def build(sd, calib):
    """calib: floating-point intermediates (one per image). Returns ([Builder job A, Builder job B], plan)."""
    s_res = []
    for l in range(NL):
        c = [ci[l + 1] for ci in calib]
        s_res.append(max(pct([x["x"] for x in c]), pct([x["y1"] for x in c]), pct([x["xo"] for x in c]),
                         pct([x["ao"] for x in c]), pct([x["f2"] for x in c])) / 63)
    s_p = pct([patches(ci[0]["pix"]) for ci in calib]) / 63
    plan = {"s_pix": s_p, "id_val": 64, "layers": [], "split": SPLIT}
    # ---- job A: embedding + layers 0..SPLIT-1
    bA = Builder()
    s_x = s_res[0]
    tpe, s_x, plan["patch"] = bA.gemm_scaled(sd, "patch_w", (0, D), (0, PX), 0, s_p, s_x)
    tpos, _, plan["pos"] = bA.gemm_scaled(sd, "pos_eff", (0, D), (0, T), 0, 1.0 / 64, s_x, col0=POS0)
    x = bA.tag(bA.op({"Add": {"a": tpe, "b": tpos, "relu": False}}), "x", 0, s_x)
    plan["s0"] = s_x
    b = bA
    for l in range(NL):
        if l == SPLIT:
            plan["s_mid"] = s_x
            b = Builder()
            # the protocol does not allow a per-column operation (LayerNorm) on the raw input of a job: identity copy first
            x = b.op({"Map": {"x": 0, "map": {"Shift": {"dr": 0, "dc": 0}}}})
        pl = {}
        target = s_res[l + 1] if l + 1 < NL else s_x
        if l + 1 < NL:
            target = max(target, s_x)          # the residual stream grows: the scale does not decrease
        x, s_x = layer(b, sd, calib, l, x, s_x, target, pl)
        plan["layers"].append(pl)
    bB = b
    # ---- epilogue (job B): final LayerNorm, classifier on all the columns (column 0 = CLS)
    tn = bB.tag(bB.op({"LayerNorm": {"x": x, "len": D}}), "nf", NL, 1 / 16)
    s_f = pct([ci[NL + 1]["xn"] for ci in calib]) / 63
    tf, plan["lnf"] = bB.affine(sd, tn, "vit.layernorm.weight", "vit.layernorm.bias", s_f)
    s_lg = pct([ci[NL + 1]["logits"] for ci in calib], 100) / 60          # maximum: the best scores must not saturate
    tc, s_lg, plan["cls"] = bB.gemm_scaled(sd, "classifier.weight", (0, 1000), (0, D), tf, s_f, s_lg)
    bB.bias(tc, "classifier.bias", 0, 1000, s_lg)
    plan["s_logit"] = s_lg
    return [bA, bB], plan


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
            _, key, rs, cs, sw, c0 = rec
            blk = np.zeros((N, N), np.int64)
            blk[:rs[1] - rs[0], c0:c0 + cs[1] - cs[0]] = np.clip(np.round(params[key][rs[0]:rs[1], cs[0]:cs[1]] / sw), -63, 63)
            W.append(blk)
    return W


def enc_input(pix, plan):
    """x0 of job A (integers): pixels of the squares (rows 0..767, columns 1..196) and position marker (rows 768.., value 64)."""
    t = np.zeros((N, N), np.int64)
    t[:PX, 1:T] = np.clip(np.round(patches(pix) / plan["s_pix"]), -64, 64)
    t[POS0 + np.arange(T), np.arange(T)] = plan["id_val"]
    return t


def run_int(bs, plan, params, pix, keep=False):
    """The two programs (independent reference engine): integer logits (1000,) and, if keep, all the tensors of the two jobs."""
    WA, WB = materialize(bs[0], params), materialize(bs[1], params)
    TA, _ = R.run_program(enc_input(pix, plan), WA, bs[0].ops)
    mid = np.zeros((N, N), np.int64)
    mid[:D, :T] = TA[-1][:D, :T]                  # the client only transmits the useful region of the residual stream
    TB, _ = R.run_program(mid, WB, bs[1].ops)
    logits = TB[-1][:1000, 0].astype(np.int64)
    return (logits, TA, TB) if keep else logits


def diag(bs, plan, params, sd, proc, paths):
    """Cosine and saturation rate, for each labelled tensor, between the dequantised integer value and the float."""
    acc = {}
    for p in paths:
        pix = pixels(proc, p)
        _, inter = float_forward(params, pix)
        _, TA, TB = run_int(bs, plan, params, pix, keep=True)
        for b, Ts in ((bs[0], TA), (bs[1], TB)):
            for t, (name, l, sc, rows, cols) in b.tags.items():
                if rows == "sc":
                    it = np.concatenate([Ts[t][:T, h * T:(h + 1) * T] for h in range(H)], 1).astype(np.float64) * sc
                    s_ = inter[l + 1]["sc"]
                    fk = s_ if name == "sc" else np.concatenate([np.exp(s_[:, h * T:(h + 1) * T] - s_[:, h * T:(h + 1) * T].max(0)) /
                                                                  np.exp(s_[:, h * T:(h + 1) * T] - s_[:, h * T:(h + 1) * T].max(0)).sum(0) for h in range(H)], 1)
                else:
                    lo, hi = rows
                    it = Ts[t][:hi - lo, :T].astype(np.float64) * sc
                    if name in ("n1", "n2", "nf"):
                        src = inter[l + 1]["x"] if name == "n1" else inter[l + 1]["y1"] if name == "n2" else inter[NL + 1]["x"]
                        fk = (src - src.mean(0)) / np.sqrt(src.var(0) + 1e-12)
                    elif name == "x" and l == 0 and b is bs[0]:
                        fk = inter[1]["x"]
                    else:
                        fk = inter[l + 1][name][lo:hi]
                c = float(np.sum(it * fk) / (np.linalg.norm(it) * np.linalg.norm(fk) + 1e-12))
                sat = float(np.mean(np.abs(Ts[t][:it.shape[0], :T]) >= 64))
                acc.setdefault((l, name), []).append((c, sat))
    for (l, name), v in acc.items():
        print(f"  layer {l:2d} {name:4s}: cosine {np.mean([x[0] for x in v]):.4f}   saturated {100 * np.mean([x[1] for x in v]):.2f} %")


def calibrate(sd, proc, paths):
    out = []
    for p in paths:
        pix = pixels(proc, p)
        _, inter = float_forward(sd, pix)
        inter[0]["pix"] = pix
        out.append(inter)
    return out


def export(out, bs, plan, params, extra=None):
    jobs = []
    for k, b in enumerate(bs):
        W = materialize(b, params)
        jobs.append({"ops": b.ops, "weights": [R.enc(w) for w in W]})
    model = {"name": "deit-tiny-int8", "hf": NAME, "n": N, "tokens": T, "classes": 1000, "jobs": jobs, "plan": plan, **(extra or {})}
    json.dump(model, open(out, "w"))
    return os.path.getsize(out)


def main(out):
    t0 = time.time()
    m, proc, sd0, id2label = load()
    for syn, idx in SYNSETS.items():
        print(f"  {syn} -> {idx} {id2label[idx]}")
    sd = derived(sd0)
    # 1) does the reimplemented floating-point model give the same logits as transformers?
    import torch
    paths, labels = images("val", 1)
    diffs = []
    for p in paths:
        pix = pixels(proc, p)
        mine, _ = float_forward(sd, pix)
        with torch.no_grad():
            ref = m(torch.tensor(pix[None], dtype=torch.float32)).logits[0].double().numpy()
        diffs.append(np.abs(mine - ref).max())
    print(f"reimplemented floating-point model vs transformers: max gap of the logits {max(diffs):.2e} ({len(paths)} images)")
    if "--prep-only" in sys.argv:
        ps, ys = images("val", arg("--eval", 200, int) // 10, offset=5)
        for mode in ("deit", "hf"):
            ok = 0
            for p_, y_ in zip(ps, ys):
                with torch.no_grad():
                    lg = m(torch.tensor(pixels(mode, p_)[None], dtype=torch.float32)).logits[0].numpy()
                ok += int(lg.argmax() == y_)
            print(f"preprocessing {mode:4s}: float top-1 (transformers) {100 * ok / len(ps):.1f} % on {len(ps)} Imagenette images, 1000 categories")
        return
    cal_paths, _ = images("train", arg("--calib", 64, int) // 10)
    calib = calibrate(sd, proc, cal_paths)
    bs, plan = build(sd, calib)
    for k, b in enumerate(bs):
        n_gemm = sum(1 for o in b.ops if next(iter(o)) in ("Gemm", "GemmAT"))
        print(f"job {'AB'[k]}: {len(b.ops)} operations of which {n_gemm} GEMM, {len(b.wrec)} weight blocks, {len(b.trec)} table rows")
    params = dict(sd)
    if "--params" in sys.argv:
        params.update(dict(np.load(arg("--params", ""))))
    if "--export-only" in sys.argv:
        print(f"written {out} ({export(out, bs, plan, params, {'report': json.load(open(arg('--report', ''))) if '--report' in sys.argv else {}}) >> 20} MiB)")
        return
    if "--diag" in sys.argv:
        diag(bs, plan, params, sd, proc, images("val", 1, offset=5)[0][:4])
        return
    ev_paths, ev_labels = images("val", arg("--eval", 100, int) // 10, offset=5)
    fl_ok = it_ok = agree = 0
    t1 = time.time()
    for p, y in zip(ev_paths, ev_labels):
        pix = pixels(proc, p)
        lf, _ = float_forward(sd, pix)
        li = run_int(bs, plan, params, pix)
        fl_ok += int(lf.argmax() == y)
        it_ok += int(li.argmax() == y)
        agree += int(lf.argmax() == li.argmax())
    n = len(ev_paths)
    rep = dict(images=n, top1_float=fl_ok / n, top1_int=it_ok / n, agree=agree / n)
    print(f"{n} Imagenette images (1000 categories) in {time.time() - t1:.0f} s: float top-1 {100 * fl_ok / n:.1f} %, "
          f"integer {100 * it_ok / n:.1f} %, same class {100 * agree / n:.1f} %")
    size = export(out, bs, plan, params, {"report": rep})
    print(f"written {out} ({size >> 20} MiB) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/tmp/vit-int8.json")
