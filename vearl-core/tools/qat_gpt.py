#!/usr/bin/env python3
"""QAT of the small language model (tools/quantize_gpt.py, TinyStories-33M by default) for the integer format of VEARL: the BODY
and HEAD programs are interpreted in torch with EXACTLY the engine's integer arithmetic (integer products, arithmetic shift,
+-64 saturation, CausalSoftmax high + fine, integer LayerNorm, GELU by table), roundings in "straight-through"; the scale plan is
fixed, only the floating-point parameters (transformed matrices and biases) are adjusted, by distillation of the next-token probabilities
of the ORIGINAL floating-point model (KL divergence, all the positions). Bit-for-bit identity with the reference engine checked before and after.

    python3 qat_gpt.py OUT.json [--train 100000] [--epochs 2] [--lr 3e-5] [--bs 32] [--eval 400]
"""
import math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantize_gpt as QG
import quantize_bert as QB
import vearl_ref as R

arg = QG.arg
L = QG.L
DEV = torch.device("cuda" if torch.cuda.is_available() and "--cpu" not in sys.argv else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
EXP2 = torch.tensor(R.EXP2, dtype=torch.int64, device=DEV)
C = QG.C_SOFTMAX


def ste(x, q):
    return x + (q - x).detach()


def isqrt_t(v):
    r = torch.sqrt(v.double()).floor().long()
    for _ in range(2):
        r = torch.where(r * r > v, r - 1, r)
        r = torch.where((r + 1) * (r + 1) <= v, r + 1, r)
    return r


def gelu_new_t(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def layernorm_int(x, n):
    """Integer LayerNorm of the engine over the first n rows (x: (B, n, cols)), with its differentiable approximation."""
    xi = x.detach().long()
    mu = torch.div(xi.sum(1, keepdim=True) + n // 2, n, rounding_mode="floor")
    var = torch.div(((xi - mu) ** 2).sum(1, keepdim=True), n, rounding_mode="floor")
    std = torch.clamp(torch.div(isqrt_t(4 * (var + 1)) + 1, 2, rounding_mode="floor"), min=1)
    q = torch.clamp(torch.div((xi - mu) * 16 + torch.div(std, 2, rounding_mode="floor"), std, rounding_mode="floor"), -64, 64).float()
    xf = x - x.mean(1, keepdim=True)
    approx = torch.clamp(16 * xf / torch.sqrt((xf * xf).mean(1, keepdim=True) + 1), -64, 64)
    return ste(approx, q)


class Sim:
    def __init__(self, body, head, plan, P, cfg):
        self.body, self.head, self.plan, self.cfg = body, head, plan, cfg
        keys = sorted({r[1] for b in (body, head) for r in b.wrec if r[0] == "mat"} | {r[1] for b in (body, head) for r in b.trec if r[0] == "vec"}
                      | {k for r in head.wrec if r[0] == "headblk" for k in r[1:3]})
        self.P = {k: torch.nn.Parameter(torch.tensor(P[k], dtype=torch.float32, device=DEV)) for k in keys}
        D, H, DH = cfg.D, cfg.H, cfg.DH
        self.m_spread = torch.zeros(D, H * L, device=DEV)
        for h in range(H):
            self.m_spread[h * DH:(h + 1) * DH, h * L:(h + 1) * L] = 1
        self.m_gather = self.m_spread.view(D, H, L)[:, :, 0].clone()
        q = torch.arange(H * L, device=DEV) % L
        self.causal = torch.arange(L, device=DEV)[:, None] <= q[None, :]          # (keys, (head, query)): key <= query
        # head: the Gemms of the vocabulary blocks ("headblk" weights), in order
        self.head_gemms = [o["Gemm"] for o in head.ops if "Gemm" in o and head.wrec[o["Gemm"]["w"]][0] == "headblk"]

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

    def run_body(self, x0, lr):
        b, cfg = self.body, self.cfg
        B = x0.shape[0]
        Ts = [x0]
        sat = lambda t: torch.clamp(t, -64, 64)
        vk = (torch.arange(L, device=DEV)[None, :] < lr[:, None])[:, :, None] & self.causal[None]      # (B, keys, H L)
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
                approx = torch.clamp(gelu_new_t(x * s1) / sg, -64, 64)
                Ts.append(ste(approx, q))
            elif kind == "CausalSoftmax":
                x = Ts[p["x"]]                                                  # (B, L keys, H L)
                xi = x.detach().long()
                m = torch.where(vk, xi, torch.full_like(xi, -10 ** 9)).max(1, keepdim=True).values
                e = EXP2[torch.clamp(m - xi, 0, 128)] * vk
                s = e.sum(1, keepdim=True)
                hi = torch.div(e * 64 + torch.div(s, 2, rounding_mode="floor"), s, rounding_mode="floor")
                if not p["fine"]:
                    approx = 64 * torch.softmax((x * C).masked_fill(~vk, -1e9), 1)
                    Ts.append(ste(approx, hi.float()))
                else:
                    fine = torch.div(e * 4096 + torch.div(s, 2, rounding_mode="floor"), s, rounding_mode="floor")
                    Ts.append(torch.clamp(fine - 64 * hi, -64, 64).float())
            elif kind == "LayerNorm":
                Ts.append(layernorm_int(Ts[p["x"]], p["len"]))
            elif kind == "Map":
                x, m = Ts[p["x"]], p["map"]
                if m == "Transpose":
                    Ts.append(x.transpose(1, 2))
                elif "HeadSpread" in m:
                    Ts.append(x.repeat(1, 1, cfg.H) * self.m_spread[None])
                elif "HeadGather" in m:
                    Ts.append((x.view(B, cfg.D, cfg.H, L) * self.m_gather[None, :, :, None]).sum(2))
                elif "Shift" in m and m["Shift"] == {"dr": 0, "dc": 0}:
                    Ts.append(x)
                else:
                    raise ValueError(m)
            else:
                raise ValueError(kind)
        return Ts[-1]

    def run_head(self, x):
        """INTEGER logits (B, V, columns): same per-column computation as the HEAD job (MARK witness rows: bias in the product; the
        final Shift / Add only arranges the blocks)."""
        n = layernorm_int(x, self.cfg.D)
        k = QG.N - self.cfg.D
        outs = []
        for g in self.head_gemms:
            _, key, bkey, rs, sw = self.head.wrec[g["w"]]
            W = self.P[key][rs[0]:rs[1]].double() / sw
            W = torch.clamp(ste(W, torch.round(W)), -63, 63).float()
            t = self.P[bkey][rs[0]:rs[1]].double() / (sw * QG.MARK / 16)
            t = torch.clamp(ste(t, torch.round(t)), -63 * k, 63 * k).float()
            z = (torch.matmul(W, n) + QG.MARK * t[None, :, None]) / 2.0 ** g["shift"]
            outs.append(torch.clamp(ste(z, torch.floor(z)), -64, 64))
        return torch.cat(outs, 1)[:, :self.cfg.V]

    def forward(self, x0, lr):
        return self.run_head(self.run_body(x0, lr))                                  # (B, V, L), unites s_lg


class Data:
    def __init__(self, tk, E, Pp):
        self.tk, self.E, self.Pp = tk, torch.tensor(E, dtype=torch.float32, device=DEV), torch.tensor(Pp, dtype=torch.float32, device=DEV)

    def batch(self, texts):
        ids = [QG.ids_of(self.tk, t) for t in texts]
        lr = torch.tensor([len(t) for t in ids], device=DEV)
        idm = torch.zeros(len(ids), L, dtype=torch.long, device=DEV)
        for i, t in enumerate(ids):
            idm[i, :len(t)] = torch.tensor(t, device=DEV)
        valid = torch.arange(L, device=DEV)[None, :] < lr[:, None]
        x = torch.clamp(self.E[idm] + self.Pp[None, :L], -64, 64) * valid[:, :, None]
        return x.transpose(1, 2).contiguous(), lr, idm, valid


def teacher_logits(m, idm, valid):
    with torch.no_grad():
        return m(input_ids=idm, attention_mask=valid.long()).logits.float()          # (B, L, V)


def check_exact(sim, data, texts, body, head, plan, P, E, Pp):
    """Integer logits of the LAST token: torch simulation == independent reference engine (vearl_ref, real programs)."""
    x, lr, idm, _ = data.batch(texts)
    with torch.no_grad():
        mine = sim.forward(x, lr).long().cpu().numpy()
    params = sim.params_np(P)
    Ws = (QG.materialize(body, params), QG.materialize(head, params))
    same = 0
    for i, t in enumerate(texts):
        ids = QG.ids_of(data.tk, t)
        ref = QG.run_int(body, head, plan, params, E, Pp, ids, Ws)
        same += int(np.array_equal(mine[i, :, len(ids) - 1], ref))
    return same


def evaluate(sim, m, data, texts, s_lg, bs=32):
    """Perplexity (integer and float) and next-token agreement (top-1) on story beginnings."""
    nll_i = nll_f = agree = tot = 0.0
    with torch.no_grad():
        for s in range(0, len(texts), bs):
            x, lr, idm, valid = data.batch(texts[s:s + bs])
            li = sim.forward(x, lr).transpose(1, 2) * s_lg                            # (B, L, V)
            lf = teacher_logits(m, idm, valid)
            tgt = idm[:, 1:]
            msk = valid[:, 1:]
            lpi = torch.log_softmax(li[:, :-1], -1).gather(-1, tgt[..., None])[..., 0]
            lpf = torch.log_softmax(lf[:, :-1], -1).gather(-1, tgt[..., None])[..., 0]
            nll_i -= float((lpi * msk).sum())
            nll_f -= float((lpf * msk).sum())
            agree += float(((li[:, :-1].argmax(-1) == lf[:, :-1].argmax(-1)) & msk).sum())
            tot += float(msk.sum())
    return dict(ppl_int=math.exp(nll_i / tot), ppl_float=math.exp(nll_f / tot), top1=agree / tot, positions=int(tot))


def fmt(r):
    return (f"integer perplexity {r['ppl_int']:.3f} (float {r['ppl_float']:.3f}); next token identical to the float "
            f"{100 * r['top1']:.1f} % ({r['positions']} positions)")


def main(out):
    torch.manual_seed(0)
    t0 = time.time()
    tk, m = QG.hf_load()
    cal, ev = QG.load_texts(arg("--calib", 64, int), arg("--eval", 400, int))
    cfg, P0, P, body, head, plan = QG.prepare(tk, m, cal)
    E, Pp = QG.embed_tables(P, plan)
    m = m.to(DEV)
    sim = Sim(body, head, plan, P, cfg)
    data = Data(tk, E, Pp)
    print(f"device {DEV}; BODY {len(body.ops)} operations, HEAD {len(head.ops)} operations", flush=True)
    same = check_exact(sim, data, ev[:4], body, head, plan, P, E, Pp)
    print(f"torch simulation == reference engine (before training): {same}/4 logits bit-for-bit identical", flush=True)
    rep0 = evaluate(sim, m, data, ev, plan["s_lg"])
    print("before QAT: " + fmt(rep0), flush=True)
    from datasets import load_dataset
    # the first N stories only, streamed: the whole train split is a 2 GB download that this run does not need
    import itertools
    stream = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    tr = [x["text"] for x in itertools.islice(stream, arg("--train", 100000, int))]
    opt = torch.optim.AdamW(sim.P.values(), lr=arg("--lr", 3e-5, float), weight_decay=0.0)
    bsz, epochs = arg("--bs", 32, int), arg("--epochs", 2, int)
    steps = epochs * (len(tr) // bsz)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=arg("--lr", 3e-5, float), total_steps=steps, pct_start=0.05)
    k = 0
    for ep in range(epochs):
        perm = np.random.default_rng(ep).permutation(len(tr))
        for s in range(0, len(perm) - bsz + 1, bsz):
            x, lr, idm, valid = data.batch([tr[i] for i in perm[s:s + bsz]])
            lt = teacher_logits(m, idm, valid)
            li = sim.forward(x, lr).transpose(1, 2) * plan["s_lg"]
            kl = (torch.softmax(lt, -1) * (torch.log_softmax(lt, -1) - torch.log_softmax(li, -1))).sum(-1)
            loss = (kl * valid).sum() / valid.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            k += 1
            if k % 200 == 0:
                print(f"  step {k}/{steps}: KL {loss.item():.4f} ; {time.time() - t0:.0f} s", flush=True)
        print(f"epoch {ep + 1}: " + fmt(evaluate(sim, m, data, ev, plan["s_lg"])), flush=True)
    params = sim.params_np(P)
    np.savez(out.replace(".json", ".params.npz"), **{k_: params[k_] for k_ in sim.P})
    same = check_exact(sim, data, ev[:8], body, head, plan, P, E, Pp)
    print(f"torch simulation == reference engine (after training): {same}/8", flush=True)
    rep = evaluate(sim, m, data, ev, plan["s_lg"])
    print("after QAT: " + fmt(rep), flush=True)
    size = QG.export(out, body, head, plan, params, E, Pp, {"report": rep, "report_before_qat": rep0, "exact_checks": same})
    print(f"written {out} ({size >> 20} MiB) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/tmp/gpt-qat.json")
