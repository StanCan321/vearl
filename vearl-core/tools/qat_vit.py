#!/usr/bin/env python3
"""QAT of DeiT-Tiny for the integer format of VEARL (tools/quantize_vit.py): the TWO programs (jobs A and B) are interpreted in torch
with EXACTLY the engine's integer arithmetic (integer products, arithmetic shift, +-64 saturation, integer Softmax / SoftmaxLo and LayerNorm,
GELU by table); roundings in "straight-through". The scale plan stays fixed; only the floating-point parameters (weights, biases,
gamma/beta, position) are adjusted, by distillation of the logits of the ORIGINAL floating-point DeiT (KL divergence) on the Imagenette images.
Before and after: bit-for-bit identity with the independent reference engine (vearl_ref) on a few images.

    python3 qat_vit.py OUT.json [--train 9469] [--epochs 3] [--lr 1e-4] [--bs 32] [--eval 500]
"""
import math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantize_vit as QV
import vearl_ref as R

arg = QV.arg
D, H, DH, T, PX, POS0 = QV.D, QV.H, QV.DH, QV.T, QV.PX, QV.POS0
DEV = torch.device("cuda" if torch.cuda.is_available() and "--cpu" not in sys.argv else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False          # EXACT products: integers representable in float32 (sums < 2^24)
torch.backends.cudnn.allow_tf32 = False
EXP2 = torch.tensor(R.EXP2, dtype=torch.int64, device=DEV)
C = QV.C_SOFTMAX


def ste(x, q):
    return x + (q - x).detach()


def isqrt_t(v):
    r = torch.sqrt(v.double()).floor().long()
    for _ in range(2):
        r = torch.where(r * r > v, r - 1, r)
        r = torch.where((r + 1) * (r + 1) <= v, r + 1, r)
    return r


class Sim:
    def __init__(self, bs, plan, sd):
        self.bs, self.plan = bs, plan
        keys = sorted({r[1] for b in bs for r in b.wrec if r[0] == "mat"} | {r[1] for b in bs for r in b.trec if r[0] == "vec"})
        self.P = {k: torch.nn.Parameter(torch.tensor(sd[k], dtype=torch.float32, device=DEV)) for k in keys}
        self.m_spread = torch.zeros(D, H * T, device=DEV)
        for h in range(H):
            self.m_spread[h * DH:(h + 1) * DH, h * T:(h + 1) * T] = 1
        self.m_gather = self.m_spread.view(D, H, T)[:, :, 0].clone()

    def params_np(self, base):
        out = dict(base)
        out.update({k: v.detach().double().cpu().numpy() for k, v in self.P.items()})
        return out

    def weight(self, b, w, n_in):
        _, key, rs, cs, sw, c0 = b.wrec[w]
        W = self.P[key][rs[0]:rs[1], cs[0]:cs[1]].double() / sw
        W = torch.clamp(ste(W, torch.round(W)), -63, 63).float()
        full = torch.zeros(W.shape[0], n_in, device=DEV)
        full[:, c0:c0 + W.shape[1]] = W
        return full

    def row(self, b, r, rows):
        rec = b.trec[r]
        if rec[0] == "const":
            return torch.full((rows,), float(rec[1]), device=DEV)
        _, key, lo, hi, fac, lim = rec
        v = self.P[key][lo:hi].double() * fac
        v = torch.clamp(ste(v, torch.round(v)), -lim, lim).float()
        if hi - lo < rows:
            v = torch.cat([v, torch.zeros(rows - (hi - lo), device=DEV)])
        return v[:rows]

    def run(self, b, x0, keep=False):
        B = x0.shape[0]
        Ts = [x0]
        sat = lambda t: torch.clamp(t, -64, 64)
        for op in b.ops:
            ((kind, p),) = op.items()
            if kind == "Gemm":
                x = Ts[p["x"]]
                z = torch.matmul(self.weight(b, p["w"], x.shape[1]), x) / 2.0 ** p["shift"]
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
                x = Ts[p["x"]]                                                  # (B, T keys, 3T): softmax over the rows
                assert p["len"] == x.shape[1]
                xi = x.detach().long()
                m = xi.max(1, keepdim=True).values
                e = EXP2[torch.clamp(m - xi, 0, 128)]
                s = e.sum(1, keepdim=True)
                hi = torch.div(e * 64 + torch.div(s, 2, rounding_mode="floor"), s, rounding_mode="floor")
                if kind == "Softmax":
                    Ts.append(ste(64 * torch.softmax(x * C, 1), hi.float()))
                else:
                    fine = torch.div(e * 4096 + torch.div(s, 2, rounding_mode="floor"), s, rounding_mode="floor")
                    Ts.append(torch.clamp(fine - 64 * hi, -64, 64).float())      # gradient carried by the high part
            elif kind == "LayerNorm":
                x = Ts[p["x"]]
                xi = x.detach().long()
                n = p["len"]
                assert n == x.shape[1]
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
                elif "Shift" in m and m["Shift"] == {"dr": 0, "dc": 0}:
                    Ts.append(x)
                elif "HeadSpread" in m:
                    Ts.append(x.repeat(1, 1, H) * self.m_spread[None])
                elif "HeadGather" in m:
                    Ts.append((x.view(B, D, H, T) * self.m_gather[None, :, :, None]).sum(2))
                else:
                    raise ValueError(m)
            else:
                raise ValueError(kind)
        return Ts if keep else Ts[-1]

    def forward(self, x0):
        """x0 (B, 965, T) integers (floats) -> integer logits (B, 1000); job B receives the useful region of the residual stream of job A."""
        mid = self.run(self.bs[0], x0)[:, :D, :T]
        return self.run(self.bs[1], mid)[:, :1000, 0]


def x0_batch(pix_list, plan):
    x = np.zeros((len(pix_list), POS0 + T, T), np.float32)
    for i, pix in enumerate(pix_list):
        x[i, :PX, 1:] = np.clip(np.round(QV.patches(pix) / plan["s_pix"]), -64, 64)
        x[i, POS0 + np.arange(T), np.arange(T)] = plan["id_val"]
    return torch.tensor(x, device=DEV)


def check_exact(sim, bs, plan, base, pix_list):
    with torch.no_grad():
        mine = sim.forward(x0_batch(pix_list, plan)).long().cpu().numpy()
    params = sim.params_np(base)
    return sum(int(np.array_equal(mine[i], QV.run_int(bs, plan, params, pix))) for i, pix in enumerate(pix_list))


def evaluate(sim, plan, pix_list, labels, bs_=64):
    ok = 0
    with torch.no_grad():
        for s in range(0, len(pix_list), bs_):
            lg = sim.forward(x0_batch(pix_list[s:s + bs_], plan))
            ok += int((lg.argmax(1).cpu().numpy() == labels[s:s + bs_]).sum())
    return ok / len(pix_list)


def main(out):
    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count() or 8)
    t0 = time.time()
    m, proc, sd0, _ = QV.load()
    sd = QV.derived(sd0)
    cal_paths, _ = QV.images("train", arg("--calib", 64, int) // 10)
    bs, plan = QV.build(sd, QV.calibrate(sd, proc, cal_paths))
    sim = Sim(bs, plan, sd)
    print(f"device {DEV}; jobs {len(bs[0].ops)} + {len(bs[1].ops)} operations", flush=True)
    ev_paths, ev_y = QV.images("val", arg("--eval", 500, int) // 10, offset=5)
    ev_pix = [QV.pixels(proc, p) for p in ev_paths]
    same = check_exact(sim, bs, plan, sd, ev_pix[:4])
    print(f"torch simulation == reference engine (before training): {same}/4 logits bit-for-bit identical", flush=True)
    with torch.no_grad():
        fl_ok = sum(int(m(torch.tensor(np.stack(ev_pix[s:s + 64]), dtype=torch.float32)).logits.argmax(1).numpy().__eq__(ev_y[s:s + 64]).sum())
                    for s in range(0, len(ev_pix), 64))
    acc0 = evaluate(sim, plan, ev_pix, ev_y)
    print(f"before QAT: integer top-1 {100 * acc0:.1f} %; float {100 * fl_ok / len(ev_pix):.1f} % ({len(ev_pix)} images, 1000 categories)", flush=True)
    # training data: Imagenette images (train) + horizontal flip; teacher = logits of the floating-point DeiT
    if arg("--data", "imagenette") == "coco":
        # COCO photos (everyday scenes, without labels: the floating-point teacher gives the targets): none of the
        # evaluation categories is seen during training -> the accuracy measured on Imagenette AND Imagewoof is a real generalisation
        d = os.path.expanduser("~/data/val2017")
        tr_paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jpg"))
    else:
        tr_paths, _ = QV.images("train", 10 ** 6)
    tr_paths = tr_paths[: arg("--train", len(tr_paths), int)]
    X, teach = [], []
    for s in range(0, len(tr_paths), 64):
        pix = [QV.pixels(proc, p) for p in tr_paths[s:s + 64]]
        pix += [p[:, :, ::-1].copy() for p in pix]
        with torch.no_grad():
            teach.append(m(torch.tensor(np.stack(pix), dtype=torch.float32)).logits.numpy())
        X.append(np.stack([np.clip(np.round(QV.patches(p) / plan["s_pix"]), -64, 64).astype(np.int8) for p in pix]))
    X, teach = np.concatenate(X), torch.tensor(np.concatenate(teach), device=DEV)
    print(f"{len(X)} training images (flips included) and teacher logits in {time.time() - t0:.0f} s", flush=True)
    ident = torch.zeros(T, T, device=DEV)
    ident[torch.arange(T), torch.arange(T)] = plan["id_val"]

    def batch(idx):
        x = torch.zeros(len(idx), POS0 + T, T, device=DEV)
        x[:, :PX, 1:] = torch.tensor(X[idx].astype(np.float32), device=DEV)
        x[:, POS0:, :] = ident
        return x

    opt = torch.optim.AdamW(sim.P.values(), lr=arg("--lr", 1e-4, float), weight_decay=0.0)
    bsz, epochs = arg("--bs", 32, int), arg("--epochs", 3, int)
    steps = epochs * (len(X) // bsz)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=arg("--lr", 1e-4, float), total_steps=steps, pct_start=0.1)
    s_lg, Tk = plan["s_logit"], arg("--temp", 1.0, float)
    k = 0
    for ep in range(epochs):
        perm = np.random.default_rng(ep).permutation(len(X))
        for s in range(0, len(perm) - bsz + 1, bsz):
            idx = perm[s:s + bsz]
            lg = sim.forward(batch(idx)) * s_lg
            loss = torch.nn.functional.kl_div(torch.log_softmax(lg / Tk, 1), torch.softmax(teach[idx] / Tk, 1), reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            k += 1
            if k % 50 == 0:
                print(f"  step {k}/{steps}: KL {loss.item():.4f} ; {time.time() - t0:.0f} s", flush=True)
        acc = evaluate(sim, plan, ev_pix, ev_y)
        print(f"epoch {ep + 1}: integer top-1 {100 * acc:.1f} %", flush=True)
    params = sim.params_np(sd)
    np.savez(out.replace(".json", ".params.npz"), **{k_: params[k_] for k_ in sim.P})
    same = check_exact(sim, bs, plan, sd, ev_pix[:4])
    print(f"torch simulation == reference engine (after training): {same}/4", flush=True)
    acc = evaluate(sim, plan, ev_pix, ev_y)
    rep = dict(images=len(ev_pix), top1_int=acc, top1_float=fl_ok / len(ev_pix), top1_int_before_qat=acc0, exact_checks=same)
    print(f"after QAT: integer top-1 {100 * acc:.1f} % (float {100 * fl_ok / len(ev_pix):.1f} %, before QAT {100 * acc0:.1f} %) on {len(ev_pix)} images", flush=True)
    size = QV.export(out, bs, plan, params, {"report": rep})
    print(f"written {out} ({size >> 20} MiB) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/tmp/vit-qat.json")
