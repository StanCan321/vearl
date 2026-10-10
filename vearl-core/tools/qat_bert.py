#!/usr/bin/env python3
"""QAT of a BERT embedding model (tools/quantize_bert.py, BGE-base by default) for the integer format of VEARL: the programs of the
chained jobs are interpreted in torch with EXACTLY the engine's integer arithmetic (integer products, arithmetic shift,
+-64 saturation, integer Softmax / SoftmaxLo and LayerNorm, GELU by table), roundings in "straight-through"; the scale plan is fixed,
only the floating-point parameters are adjusted, by distillation of the embeddings of the ORIGINAL floating-point model (cosine). Bit-for-bit identity
with the reference engine checked before and after. GPU if available (exact products: TF32 disabled, sums < 2^24).

    python3 qat_bert.py OUT.json [--train 40000] [--epochs 2] [--lr 2e-5] [--bs 64] [--eval 300]
"""
import math, os, sys, time
import numpy as np
import torch
import torch.utils.checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantize_bert as QB
import vearl_ref as R

arg = QB.arg
L = QB.L
DEV = torch.device("cuda" if torch.cuda.is_available() and "--cpu" not in sys.argv else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
EXP2 = torch.tensor(R.EXP2, dtype=torch.int64, device=DEV)
C = QB.C_SOFTMAX


def ste(x, q):
    return x + (q - x).detach()


def isqrt_t(v):
    r = torch.sqrt(v.double()).floor().long()
    for _ in range(2):
        r = torch.where(r * r > v, r - 1, r)
        r = torch.where((r + 1) * (r + 1) <= v, r + 1, r)
    return r


class Sim:
    def __init__(self, bs, plan, sd, cfg):
        self.bs, self.plan, self.cfg = bs, plan, cfg
        keys = sorted({r[1] for b in bs for r in b.wrec if r[0] == "mat"} | {r[1] for b in bs for r in b.trec if r[0] == "vec"})
        self.P = {k: torch.nn.Parameter(torch.tensor(sd[k], dtype=torch.float32, device=DEV)) for k in keys}
        D, H, DH = cfg.D, cfg.H, cfg.DH
        self.m_spread = torch.zeros(D, H * L, device=DEV)
        for h in range(H):
            self.m_spread[h * DH:(h + 1) * DH, h * L:(h + 1) * L] = 1
        self.m_gather = self.m_spread.view(D, H, L)[:, :, 0].clone()

    def params_np(self, base):
        out = dict(base)
        out.update({k: v.detach().double().cpu().numpy() for k, v in self.P.items()})
        return out

    def weight(self, b, w):
        _, key, rs, cs, sw = b.wrec[w]
        W = self.P[key][rs[0]:rs[1], cs[0]:cs[1]].double() / sw
        return torch.clamp(ste(W, torch.round(W)), -63, 63).float()

    def row(self, b, r, rows):
        rec = b.trec[r]
        if rec[0] == "const":
            return torch.full((rows,), float(rec[1]), device=DEV)
        _, key, lo, hi, fac, lim = rec
        v = self.P[key][lo:hi].double() * fac
        v = torch.clamp(ste(v, torch.round(v)), -lim, lim).float()
        assert hi - lo == rows
        return v

    def run(self, b, x0, valid):
        cfg = self.cfg
        B = x0.shape[0]
        Ts = [x0]
        sat = lambda t: torch.clamp(t, -64, 64)
        vk = valid[:, :, None]                                                 # (B, L, 1): real keys
        for op in b.ops:
            ((kind, p),) = op.items()
            if kind == "Gemm":
                z = torch.matmul(self.weight(b, p["w"]), Ts[p["x"]]) / 2.0 ** p["shift"]
                Ts.append(sat(ste(z, torch.floor(z))))
            elif kind == "GemmAT":
                z = torch.matmul(Ts[p["a"]].transpose(1, 2), Ts[p["x"]]) / 2.0 ** p["shift"]
                Ts.append(sat(ste(z, torch.floor(z))))
            elif kind == "Bias":
                x = Ts[p["x"]]
                Ts.append(sat(x + self.row(b, p["row"], x.shape[1])[None, :, None]))
            elif kind == "Mul":
                x = Ts[p["x"]]
                y = x * self.row(b, p["row"], x.shape[1])[None, :, None] / 2.0 ** p["shift"]
                Ts.append(sat(ste(y, torch.floor(y))))
            elif kind == "Add":
                Ts.append(sat(Ts[p["a"]] + Ts[p["b"]]))
            elif kind == "Lut":
                x = Ts[p["x"]]
                _, tab, s1, sg = b.trec[p["row"]]
                tab = torch.tensor(tab, dtype=torch.float32, device=DEV)
                q = tab[(torch.clamp(x, -64, 63) + 64).long()]
                approx = torch.clamp(0.5 * x * s1 * (1 + torch.erf(x * s1 / math.sqrt(2))) / sg, -64, 64)
                Ts.append(ste(approx, q))
            elif kind in ("Softmax", "SoftmaxLo"):
                x = Ts[p["x"]]                                                  # (B, L keys, H L)
                xi = x.detach().long()
                m = torch.where(vk, xi, torch.full_like(xi, -10 ** 9)).max(1, keepdim=True).values
                e = EXP2[torch.clamp(m - xi, 0, 128)] * vk
                s = e.sum(1, keepdim=True)
                hi = torch.div(e * 64 + torch.div(s, 2, rounding_mode="floor"), s, rounding_mode="floor")
                if kind == "Softmax":
                    approx = 64 * torch.softmax((x * C).masked_fill(~vk, -1e9), 1)
                    Ts.append(ste(approx, hi.float()))
                else:
                    fine = torch.div(e * 4096 + torch.div(s, 2, rounding_mode="floor"), s, rounding_mode="floor")
                    Ts.append(torch.clamp(fine - 64 * hi, -64, 64).float())
            elif kind == "LayerNorm":
                x = Ts[p["x"]]
                xi = x.detach().long()
                n = p["len"]
                mu = torch.div(xi.sum(1, keepdim=True) + n // 2, n, rounding_mode="floor")
                var = torch.div(((xi - mu) ** 2).sum(1, keepdim=True), n, rounding_mode="floor")
                std = torch.clamp(torch.div(isqrt_t(4 * (var + 1)) + 1, 2, rounding_mode="floor"), min=1)
                q = torch.clamp(torch.div((xi - mu) * 16 + torch.div(std, 2, rounding_mode="floor"), std, rounding_mode="floor"), -64, 64).float()
                xf = x - x.mean(1, keepdim=True)
                approx = torch.clamp(16 * xf / torch.sqrt((xf * xf).mean(1, keepdim=True) + 1), -64, 64)
                Ts.append(ste(approx, q))
            elif kind == "Map":
                x, m = Ts[p["x"]], p["map"]
                if m == "Transpose":
                    Ts.append(x.transpose(1, 2))
                elif "HeadSpread" in m:
                    Ts.append(x.repeat(1, 1, cfg.H) * self.m_spread[None])
                elif "HeadGather" in m:
                    Ts.append((x.view(B, cfg.D, cfg.H, L) * self.m_gather[None, :, :, None]).sum(2))
                else:
                    raise ValueError(m)
            else:
                raise ValueError(kind)
        return Ts[-1]

    def forward(self, x0, lr):
        valid = torch.arange(L, device=DEV)[None, :] < lr[:, None]
        x = x0
        for b in self.bs:
            if torch.is_grad_enabled():
                # recomputation on the way back (GPU memory: the tensors of a job are only kept during its own computation)
                x = torch.utils.checkpoint.checkpoint(self.run, b, x, valid, use_reentrant=False)[:, :self.cfg.D, :L]
            else:
                x = self.run(b, x, valid)[:, :self.cfg.D, :L]
        return x[:, :, 0]                                                        # state of the [CLS] token


def batch_inputs(sd, cfg, tk, texts, s0):
    ids = [QB.ids_of(tk, t) for t in texts]
    x = np.zeros((len(ids), cfg.D, L), np.float32)
    for i, t in enumerate(ids):
        e = QB.embed(sd, cfg, t)
        x[i, :, :e.shape[1]] = np.clip(np.round(e / s0), -64, 64)
    return torch.tensor(x, device=DEV), torch.tensor([len(t) for t in ids], device=DEV), ids


def check_exact(sim, sd, cfg, tk, texts, bs, plan):
    x, lr, ids = batch_inputs(sd, cfg, tk, texts, plan["s0"])
    with torch.no_grad():
        mine = sim.forward(x, lr).long().cpu().numpy()
    params = sim.params_np(sd)
    Ws = [QB.materialize(b, params) for b in bs]
    return sum(int(np.array_equal(mine[i], QB.run_int(bs, plan, params, sd, cfg, ids[i], Ws))) for i in range(len(texts)))


def embed_all(sim, sd, cfg, tk, texts, plan, bs_=128):
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), bs_):
            x, lr, _ = batch_inputs(sd, cfg, tk, texts[s:s + bs_], plan["s0"])
            out.append(sim.forward(x, lr).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def evaluate(sim, sd, cfg, tk, plan, st):
    """(1) integer / float embedding cosine (AG News test); (2) STS-B test: float vs integer Spearman; (3) AG News: nearest neighbour."""
    from datasets import load_dataset
    from scipy.stats import spearmanr
    nz = lambda E: E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    ag = load_dataset("fancyzhx/ag_news")["test"]
    te, y = list(ag["text"][:arg("--eval", 300, int)]), np.array(ag["label"][:arg("--eval", 300, int)])
    fl = st.encode(te, normalize_embeddings=True, batch_size=64)
    it = nz(embed_all(sim, sd, cfg, tk, te, plan))
    cos = np.sum(fl * it, 1)

    def nn(E):
        S = E @ E.T
        np.fill_diagonal(S, -2)
        return float((y[S.argmax(1)] == y).mean())
    sts = load_dataset("sentence-transformers/stsb")["test"]
    a, b_, g = list(sts["sentence1"]), list(sts["sentence2"]), np.array(sts["score"])
    fa, fb = st.encode(a, normalize_embeddings=True, batch_size=64), st.encode(b_, normalize_embeddings=True, batch_size=64)
    ia, ib = nz(embed_all(sim, sd, cfg, tk, a, plan)), nz(embed_all(sim, sd, cfg, tk, b_, plan))
    return dict(cos_mean=float(cos.mean()), cos_min=float(cos.min()), nn_float=nn(fl), nn_int=nn(it),
                sts_float=float(spearmanr(np.sum(fa * fb, 1), g).correlation), sts_int=float(spearmanr(np.sum(ia * ib, 1), g).correlation), sts_pairs=len(g))


def fmt(r):
    return (f"integer/float cosine {r['cos_mean']:.4f} (min {r['cos_min']:.3f}); STS-B Spearman float {r['sts_float']:.4f}, integer {r['sts_int']:.4f} "
            f"({r['sts_pairs']} pairs); AG News nearest neighbour float {100 * r['nn_float']:.1f} %, integer {100 * r['nn_int']:.1f} %")


def main(out):
    torch.manual_seed(0)
    t0 = time.time()
    tk, sd, cfg = QB.load()
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    ds = load_dataset("fancyzhx/ag_news")
    bs, plan = QB.build(sd, cfg, QB.calibrate(sd, cfg, tk, ds["train"]["text"][:arg("--calib", 200, int)]))
    sim = Sim(bs, plan, sd, cfg)
    print(f"device {DEV}; {len(bs)} jobs of {[len(b.ops) for b in bs]} operations", flush=True)
    same = check_exact(sim, sd, cfg, tk, ds["test"]["text"][:4], bs, plan)
    print(f"torch simulation == reference engine (before training): {same}/4 embeddings bit-for-bit identical", flush=True)
    st = SentenceTransformer(QB.NAME, device=str(DEV))
    st.max_seq_length = L
    rep0 = evaluate(sim, sd, cfg, tk, plan, st)
    print("before QAT: " + fmt(rep0), flush=True)
    ntr = arg("--train", 40000, int)
    tr = list(ds["train"]["text"][200:200 + ntr])
    sts_tr = load_dataset("sentence-transformers/stsb")["train"]
    tr += list(sts_tr["sentence1"]) + list(sts_tr["sentence2"])                  # short and varied sentences
    teach = torch.tensor(st.encode(tr, normalize_embeddings=True, batch_size=128), dtype=torch.float32, device=DEV)
    print(f"teacher: {len(tr)} floating-point embeddings in {time.time() - t0:.0f} s", flush=True)
    opt = torch.optim.AdamW(sim.P.values(), lr=arg("--lr", 2e-5, float), weight_decay=0.0)
    bsz, epochs = arg("--bs", 32, int), arg("--epochs", 2, int)
    steps = epochs * (len(tr) // bsz)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=arg("--lr", 2e-5, float), total_steps=steps, pct_start=0.1)
    k = 0
    for ep in range(epochs):
        perm = np.random.default_rng(ep).permutation(len(tr))
        for s in range(0, len(perm) - bsz + 1, bsz):
            idx = perm[s:s + bsz]
            x, lr, _ = batch_inputs(sd, cfg, tk, [tr[i] for i in idx], plan["s0"])
            e = sim.forward(x, lr)
            loss = (1 - torch.nn.functional.cosine_similarity(e, teach[idx], dim=1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            k += 1
            if k % 100 == 0:
                print(f"  step {k}/{steps}: cosine {1 - loss.item():.4f} ; {time.time() - t0:.0f} s", flush=True)
        print(f"epoch {ep + 1}: " + fmt(evaluate(sim, sd, cfg, tk, plan, st)), flush=True)
    params = sim.params_np(sd)
    np.savez(out.replace(".json", ".params.npz"), **{k_: params[k_] for k_ in sim.P})
    same = check_exact(sim, sd, cfg, tk, ds["test"]["text"][:4], bs, plan)
    print(f"torch simulation == reference engine (after training): {same}/4", flush=True)
    rep = evaluate(sim, sd, cfg, tk, plan, st)
    print("after QAT: " + fmt(rep), flush=True)
    size = QB.export(out, bs, plan, params, cfg, {"report": rep, "report_before_qat": rep0, "exact_checks": same})
    print(f"written {out} ({size >> 20} MiB) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/tmp/bge-qat.json")
