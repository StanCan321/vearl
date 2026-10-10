#!/usr/bin/env python3
"""QAT of MiniLM for the integer format of VEARL: the PROGRAM itself (operations of quantize_minilm.py) is interpreted in torch, with
EXACTLY the engine's integer arithmetic (integer products, arithmetic shift, +-64 saturation, integer softmax and LayerNorm by table,
GELU by table); the roundings and the integer operations let the gradient of their floating-point version through ("straight-through").
Only the floating-point parameters of the model (weights, biases, gamma/beta) are adjusted, the scale plan stays fixed; we distil the embeddings
of the ORIGINAL floating-point MiniLM (cosine). Before and after training, the simulation is compared with the reference engine (bit-for-bit identity).

    python3 qat_minilm.py OUT.json [--train 40000] [--epochs 1] [--lr 2e-5] [--bs 32] [--eval 300]
"""
import math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantize_minilm as QM
import vearl_ref as R

arg = QM.arg
L, D, H, DH = QM.L, QM.D, QM.H, QM.DH
EXP2 = torch.tensor(R.EXP2, dtype=torch.int64)
C = QM.C_SOFTMAX


def ste(x, q):
    """Value q (integer), gradient of x."""
    return x + (q - x).detach()


def isqrt_t(v):
    r = torch.sqrt(v.double()).floor().long()
    for _ in range(2):
        r = torch.where(r * r > v, r - 1, r)
        r = torch.where((r + 1) * (r + 1) <= v, r + 1, r)
    return r


class Sim:
    def __init__(self, b, plan, sd):
        self.b, self.plan = b, plan
        keys = sorted({r[1] for r in b.wrec if r[0] == "mat"} | {r[1] for r in b.trec if r[0] == "vec"})
        self.P = {k: torch.nn.Parameter(torch.tensor(sd[k], dtype=torch.float32)) for k in keys}
        self.m_spread = torch.zeros(D, H * L)
        for h in range(H):
            self.m_spread[h * DH:(h + 1) * DH, h * L:(h + 1) * L] = 1
        self.m_gather = self.m_spread.view(D, H, L)[:, :, 0].clone()             # (D, H): head of each row
        self.luts = {}

    def params_np(self):
        return {k: v.detach().double().numpy() for k, v in self.P.items()}

    def weight(self, w):
        _, key, rs, cs, sw = self.b.wrec[w]
        W = self.P[key][rs[0]:rs[1], cs[0]:cs[1]].double() / sw              # in float64, like the re-quantisation at export
        return torch.clamp(ste(W, torch.round(W)), -63, 63).float()

    def row(self, r):
        rec = self.b.trec[r]
        if rec[0] == "lut":
            return None
        _, key, lo, hi, fac, lim = rec
        v = self.P[key][lo:hi].double() * fac
        return torch.clamp(ste(v, torch.round(v)), -lim, lim).float()

    def forward(self, x0, lr, keep=False):
        """x0 (B, D, L) integers (floats), lr (B,) lengths -> integer embedding (B, D) (straight-through gradient)."""
        B = x0.shape[0]
        T = [x0]
        valid_k = (torch.arange(L)[None, :] < lr[:, None])                         # (B, L)
        sat = lambda t: torch.clamp(t, -64, 64)
        for op in self.b.ops:
            ((kind, p),) = op.items()
            if kind == "Gemm" and p["shift"] == "SHP":                              # sum of the real tokens (row of ones)
                t = T[p["x"]]                                                       # (B, L, D)
                s = (t * valid_k[:, :, None]).sum(1, keepdim=True)                  # (B, 1, D)
                shp = torch.tensor([QM.shp_for(int(v)) for v in lr], dtype=torch.float32)[:, None, None]
                y = s / 2.0 ** shp
                T.append(sat(ste(y, torch.floor(y))))
            elif kind == "Gemm":
                z = torch.matmul(self.weight(p["w"]), T[p["x"]]) / 2.0 ** p["shift"]
                T.append(sat(ste(z, torch.floor(z))))
            elif kind == "GemmAT":
                z = torch.matmul(T[p["a"]].transpose(1, 2), T[p["x"]]) / 2.0 ** p["shift"]
                T.append(sat(ste(z, torch.floor(z))))
            elif kind == "Bias":
                T.append(sat(T[p["x"]] + self.row(p["row"])[None, :, None]))
            elif kind == "Mul":
                y = T[p["x"]] * self.row(p["row"])[None, :, None] / 2.0 ** p["shift"]
                T.append(sat(ste(y, torch.floor(y))))
            elif kind == "Add":
                T.append(sat(T[p["a"]] + T[p["b"]]))
            elif kind == "Lut":
                x = T[p["x"]]
                rec = self.b.trec[p["row"]]
                tab = torch.tensor(rec[1], dtype=torch.float32)
                q = tab[(torch.clamp(x, -64, 63) + 64).long()]
                s1, sg = self.lut_scales(p["row"])
                approx = torch.clamp(0.5 * x * s1 * (1 + torch.erf(x * s1 / math.sqrt(2))) / sg, -64, 64)
                T.append(ste(approx, q))
            elif kind == "Softmax":
                x = T[p["x"]]                                                       # (B, L, HL): rows = keys
                v = valid_k[:, :, None]
                xi = x.detach().long()
                m = torch.where(v, xi, torch.full_like(xi, -10 ** 9)).max(1, keepdim=True).values
                e = EXP2[torch.clamp(m - xi, 0, 128)] * v
                s = e.sum(1, keepdim=True)
                q = torch.div(e * 64 + torch.div(s, 2, rounding_mode="floor"), s, rounding_mode="floor").float()
                approx = 64 * torch.softmax((x * C).masked_fill(~v, -1e9), 1)
                T.append(ste(approx, q))
            elif kind == "LayerNorm":
                x = T[p["x"]]                                                       # (B, D, L): normalisation over the rows
                xi = x.detach().long()
                n = p["len"]
                mu = torch.div(xi.sum(1, keepdim=True) + n // 2, n, rounding_mode="floor")
                var = torch.div(((xi - mu) ** 2).sum(1, keepdim=True), n, rounding_mode="floor")
                std = torch.clamp(torch.div(isqrt_t(4 * (var + 1)) + 1, 2, rounding_mode="floor"), min=1)
                q = torch.clamp(torch.div((xi - mu) * 16 + torch.div(std, 2, rounding_mode="floor"), std, rounding_mode="floor"), -64, 64).float()
                xf = x - x.mean(1, keepdim=True)
                approx = torch.clamp(16 * xf / torch.sqrt((xf * xf).mean(1, keepdim=True) + 1), -64, 64)
                T.append(ste(approx, q))
            elif kind == "Map":
                x, m = T[p["x"]], p["map"]
                if m == "Transpose":
                    T.append(x.transpose(1, 2))
                elif "HeadSpread" in m:
                    T.append(x.repeat(1, 1, H) * self.m_spread[None])
                elif "HeadGather" in m:
                    T.append((x.view(B, D, H, L) * self.m_gather[None, :, :, None]).sum(2))
                else:
                    raise ValueError(m)
            else:
                raise ValueError(kind)
        if keep:
            return T
        return T[-1][:, :, 0]                                                       # (B, D)

    def lut_scales(self, row):
        if row not in self.luts:
            for pl in self.plan["layers"]:
                for f in pl["ffn"]:
                    if f["lut"] == list(self.b.trec[row][1]):
                        self.luts[row] = (f["s1"], f["sg"])
            if row not in self.luts:
                raise KeyError(row)
        return self.luts[row]


def batch_inputs(sd, tk, texts, s0):
    ids = [QM.ids_of(tk, t) for t in texts]
    x = np.zeros((len(ids), D, L))
    for i, t in enumerate(ids):
        e = QM.embed(sd, t)
        x[i, :, :e.shape[1]] = np.clip(np.round(e / s0), -64, 64)
    return torch.tensor(x, dtype=torch.float32), torch.tensor([len(t) for t in ids]), ids


def check_exact(sim, sd, tk, texts, b, plan):
    """Do the simulation and the reference engine (exported program) give the SAME integer embedding?"""
    x, lr, ids = batch_inputs(sd, tk, texts, plan["s0"])
    with torch.no_grad():
        mine = sim.forward(x, lr).long().numpy()
    params = sim.params_np()
    same = sum(int(np.array_equal(mine[i], QM.run_int(b, plan, params, sd, ids[i])[0])) for i in range(len(texts)))
    return same


def debug_exact(sim, sd, tk, text, b, plan):
    """First operation where the simulation differs from the reference engine (same region)."""
    x, lr, ids = batch_inputs(sd, tk, [text], plan["s0"])
    with torch.no_grad():
        Ts = sim.forward(x, lr, keep=True)
    _, Tr = QM.run_int(b, plan, sim.params_np(), sd, ids[0])
    for k, (a, op) in enumerate(zip(Ts[1:], b.ops), start=1):
        a = a[0].long().numpy()
        r = Tr[k][:a.shape[0], :a.shape[1]].astype(np.int64)
        if not np.array_equal(a, r):
            d = np.argwhere(a != r)
            print(f"DIVERGENCE at operation {k} {op}: {len(d)} cells; first {d[0].tolist()} sim {a[tuple(d[0])]} ref {r[tuple(d[0])]} (length {len(ids[0])})")
            return k
    print("no divergence")
    return None


def main(out):
    torch.set_num_threads(os.cpu_count() or 8)
    torch.manual_seed(0)
    t0 = time.time()
    tk, sd = QM.load()
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    ds = load_dataset("fancyzhx/ag_news")
    ntr, nev = arg("--train", 40000, int), arg("--eval", 300, int)
    tr = ds["train"]["text"][200:200 + ntr]
    te, te_y = ds["test"]["text"][:nev], np.array(ds["test"]["label"][:nev])
    b, plan = QM.build(sd, QM.calibrate(sd, tk, ds["train"]["text"][:200]))
    sim = Sim(b, plan, sd)
    if "--debug" in sys.argv:
        debug_exact(sim, sd, tk, te[0], b, plan)
        return
    same = check_exact(sim, sd, tk, te[:8], b, plan)
    print(f"torch simulation == reference engine (before training): {same}/8 embeddings bit-for-bit identical", flush=True)
    rep0, _ = QM.evaluate(b, plan, sim.params_np(), sd, tk, te, te_y)
    print(f"before QAT: cosine {rep0['cos_mean']:.4f}; nearest neighbour float {100 * rep0['nn_float']:.1f} %, integer {100 * rep0['nn_int']:.1f} %", flush=True)
    st = SentenceTransformer(QM.NAME, device="cpu")
    st.max_seq_length = L
    teach = torch.tensor(st.encode(list(tr), batch_size=256, normalize_embeddings=True), dtype=torch.float32)
    print(f"teacher: {len(tr)} floating-point MiniLM embeddings in {time.time() - t0:.0f} s", flush=True)
    opt = torch.optim.AdamW(sim.P.values(), lr=arg("--lr", 2e-5, float), weight_decay=0.0)
    bs, epochs = arg("--bs", 32, int), arg("--epochs", 1, int)
    steps = epochs * (len(tr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=arg("--lr", 2e-5, float), total_steps=steps, pct_start=0.1)
    k = 0
    for ep in range(epochs):
        perm = np.random.default_rng(ep).permutation(len(tr))
        for s in range(0, len(perm) - bs + 1, bs):
            idx = perm[s:s + bs]
            x, lr, _ = batch_inputs(sd, tk, [tr[i] for i in idx], plan["s0"])
            e = sim.forward(x, lr)
            loss = (1 - torch.nn.functional.cosine_similarity(e, teach[idx], dim=1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            k += 1
            if k % 50 == 0:
                print(f"  step {k}/{steps}: loss {loss.item():.4f} (cosine {1 - loss.item():.4f}); {time.time() - t0:.0f} s", flush=True)
    same = check_exact(sim, sd, tk, te[:8], b, plan)
    print(f"torch simulation == reference engine (after training): {same}/8", flush=True)
    params = sim.params_np()
    np.savez(out.replace(".json", ".params.npz"), **params)
    rep, _ = QM.evaluate(b, plan, params, sd, tk, te, te_y)
    print(f"after QAT: cosine integer program / ORIGINAL floating-point MiniLM {rep['cos_mean']:.4f} (min {rep['cos_min']:.3f}); "
          f"nearest neighbour of the same category: float {100 * rep['nn_float']:.1f} %, integer {100 * rep['nn_int']:.1f} %", flush=True)
    size = QM.export(out, b, plan, params, {"report": rep, "report_before_qat": rep0})
    print(f"written {out} ({size >> 20} MiB) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/tmp/minilm-qat.json")
