#!/usr/bin/env python3
"""Feasibility probe (outside consensus): does a small GPT-type language model (normalisation BEFORE each block) support
the arithmetic of the chain (integer activations in [-64, 64], integer LayerNorm at 16 per standard deviation, attention scores at step
ln2/8 saturated at +-64, 8-bit weights per block)? SIMULATED quantisation (rounded floats), without training: we compare perplexity
and next-token agreement (top-1) with the floating-point model, with or without the following exact transformations:
  --center : zero-mean residual stream (the matrices that write into the stream are centred: LayerNorm = RMS normalisation)
  --rotate : orthogonal rotation of the residual stream (fixes the vector 1, hence keeps the zero mean): spreads out the outlier dimensions
  --kbar   : recentres the keys (b_k -= mean of the keys: scores shifted by a constant per query, softmax unchanged)

    python3 llm_probe.py gpt2 [--center --rotate --kbar] [--n 64]
"""
import math, sys
import numpy as np
import torch

arg = lambda name, default, cast=str: cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
NAME = sys.argv[1]
L = 64
NSEQ = arg("--n", 64, int)
PCT = arg("--pct", 99.9, float)
C_SM = math.log(2) / 8
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False


def load():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tk = AutoTokenizer.from_pretrained(NAME)
    m = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float64).eval().to(dev)
    return tk, m


class G:
    """Weights of a GPT (gpt2 / gpt_neo) in uniform form: y = W x + b (rows = outputs)."""

    def __init__(self, m):
        c = m.config
        sd = {k: v.detach().to(dev, torch.float64) for k, v in m.state_dict().items()}
        self.kind = c.model_type
        self.D = c.hidden_size
        self.H = c.num_attention_heads if hasattr(c, "num_attention_heads") else c.num_heads
        self.DH = self.D // self.H
        self.NL = c.num_hidden_layers if hasattr(c, "num_hidden_layers") else c.num_layers
        self.eps = c.layer_norm_epsilon
        p = "transformer."
        self.wte, self.wpe = sd[p + "wte.weight"], sd[p + "wpe.weight"]
        self.head = sd.get("lm_head.weight", self.wte)
        self.lnf = (sd[p + "ln_f.weight"], sd[p + "ln_f.bias"])
        self.layers = []
        D = self.D
        for l in range(self.NL):
            q = lambda n: sd[f"{p}h.{l}.{n}"]
            if self.kind == "gpt2":
                Wa, ba = q("attn.c_attn.weight").T, q("attn.c_attn.bias")
                Wq, Wk, Wv = Wa[:D], Wa[D:2 * D], Wa[2 * D:]
                bq, bk, bv = ba[:D], ba[D:2 * D], ba[2 * D:]
                Wo, bo = q("attn.c_proj.weight").T, q("attn.c_proj.bias")
                W1, b1 = q("mlp.c_fc.weight").T, q("mlp.c_fc.bias")
                W2, b2 = q("mlp.c_proj.weight").T, q("mlp.c_proj.bias")
                sc = 1 / math.sqrt(self.DH)
            else:
                z = torch.zeros(D, device=dev, dtype=torch.float64)
                Wq, Wk, Wv = q("attn.attention.q_proj.weight"), q("attn.attention.k_proj.weight"), q("attn.attention.v_proj.weight")
                bq, bk, bv = z, z.clone(), z.clone()
                Wo, bo = q("attn.attention.out_proj.weight"), q("attn.attention.out_proj.bias")
                W1, b1 = q("mlp.c_fc.weight"), q("mlp.c_fc.bias")
                W2, b2 = q("mlp.c_proj.weight"), q("mlp.c_proj.bias")
                sc = 1.0                                                  # GPT-Neo: scores not divided by sqrt(dh)
            self.layers.append(dict(ln1=(q("ln_1.weight"), q("ln_1.bias")), ln2=(q("ln_2.weight"), q("ln_2.bias")), Wq=Wq * sc, bq=bq * sc,
                                    Wk=Wk, bk=bk, Wv=Wv, bv=bv, Wo=Wo, bo=bo, W1=W1, b1=b1, W2=W2, b2=b2))
        self.gamma_folded = False

    def center(self):
        """Writers of the residual stream centred (zero-mean output): LayerNorm becomes an RMS normalisation (identical function)."""
        cen = lambda W: W - W.mean(0, keepdim=True)
        self.wte = self.wte - self.wte.mean(1, keepdim=True)
        self.wpe = self.wpe - self.wpe.mean(1, keepdim=True)
        for P in self.layers:
            P["Wo"], P["bo"] = cen(P["Wo"]), P["bo"] - P["bo"].mean()
            P["W2"], P["b2"] = cen(P["W2"]), P["b2"] - P["b2"].mean()

    def fold(self):
        """gamma and beta of the LayerNorms folded into the matrices that read them (x_norm -> W diag(g) x_norm + W beta + b)."""
        for P in self.layers:
            g, b = P["ln1"]
            for w, bb in (("Wq", "bq"), ("Wk", "bk"), ("Wv", "bv")):
                P[bb] = P[bb] + P[w] @ b
                P[w] = P[w] * g[None, :]
            g, b = P["ln2"]
            P["b1"] = P["b1"] + P["W1"] @ b
            P["W1"] = P["W1"] * g[None, :]
        g, b = self.lnf
        self.head_b = self.head @ b
        self.head = self.head * g[None, :]
        self.gamma_folded = True

    def rotate(self, seed=0):
        """Residual stream rotated by an orthogonal Q with Q 1 = 1 (zero mean kept): writers Q W, readers W Q^T."""
        assert self.gamma_folded
        D = self.D
        g = torch.Generator().manual_seed(seed)
        R, _ = torch.linalg.qr(torch.randn(D, D, generator=g, dtype=torch.float64))
        R = R.to(dev)
        e = torch.ones(D, device=dev, dtype=torch.float64) / math.sqrt(D)
        u = R @ e - e
        Hh = torch.eye(D, device=dev, dtype=torch.float64) - 2 * torch.outer(u, u) / (u @ u)
        Q = Hh @ R
        assert torch.allclose(Q @ e, e, atol=1e-9)
        self.wte, self.wpe = self.wte @ Q.T, self.wpe @ Q.T
        for P in self.layers:
            for w in ("Wq", "Wk", "Wv", "W1"):
                P[w] = P[w] @ Q.T
            P["Wo"], P["bo"] = Q @ P["Wo"], Q @ P["bo"]
            P["W2"], P["b2"] = Q @ P["W2"], Q @ P["b2"]
        self.head = self.head @ Q.T


def gelu_new(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def forward(g, ids, qz=None, stats=None):
    """ids (B, T); qz: dict of scales (simulated quantisation) or None (float). Returns the logits (B, T, V)."""
    B, T = ids.shape
    D, H, DH = g.D, g.H, g.DH

    def Q(name, x):
        if stats is not None:
            stats.setdefault(name, []).append(x.detach().abs().flatten()[::7].float().cpu())
        if qz is None:
            return x
        s = qz[name]
        return torch.clamp(torch.round(x / s), -64, 64) * s

    def lnorm(x, gb):
        mu = x.mean(-1, keepdim=True)
        if qz is None:
            y = (x - mu) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + g.eps)
        else:
            sd_ = torch.sqrt(x.var(-1, unbiased=False, keepdim=True)).clamp_min(1e-9)
            y = torch.clamp(torch.round(16 * (x - mu) / sd_), -64, 64) / 16
        if g.gamma_folded:
            return y
        return y * gb[0] + gb[1]

    x = Q("x0", g.wte[ids] + g.wpe[:T][None])
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=dev), 1)
    for l, P in enumerate(g.layers):
        h = lnorm(x, P["ln1"])
        if not g.gamma_folded:
            h = Q(f"h1.{l}", h)
        q = Q(f"q.{l}", h @ P["Wq"].T + P["bq"]).view(B, T, H, DH).transpose(1, 2)
        kl = h @ P["Wk"].T + P["bk"]
        if stats is not None:
            stats.setdefault(f"kmean.{l}", []).append(kl.mean((0, 1)))
        k = Q(f"k.{l}", kl).view(B, T, H, DH).transpose(1, 2)
        v = Q(f"v.{l}", h @ P["Wv"].T + P["bv"]).view(B, T, H, DH).transpose(1, 2)
        s = q @ k.transpose(-1, -2)
        if stats is not None:
            stats.setdefault(f"sc.{l}", []).append(s.masked_fill(mask, 0).detach().abs().flatten()[::7].float().cpu())
        if qz is not None:
            s = torch.clamp(torch.round(s / C_SM), -64, 64) * C_SM
        a = torch.softmax(s.masked_fill(mask, -1e9), -1)
        if qz is not None:
            a = torch.round(a * 4096) / 4096
        ctx = Q(f"ctx.{l}", (a @ v).transpose(1, 2).reshape(B, T, D))
        x = Q(f"x.{l}.a", x + Q(f"o.{l}", ctx @ P["Wo"].T + P["bo"]))
        h = lnorm(x, P["ln2"])
        if not g.gamma_folded:
            h = Q(f"h2.{l}", h)
        f = Q(f"f1.{l}", h @ P["W1"].T + P["b1"])
        f = Q(f"g.{l}", gelu_new(f))
        x = Q(f"x.{l}.b", x + Q(f"f2.{l}", f @ P["W2"].T + P["b2"]))
    h = lnorm(x, g.lnf)
    return h @ g.head.T + (g.head_b if g.gamma_folded else 0)


def kmax_offsets(g, ids, target=4.0):
    """Shift of the scores PER QUERY (softmax unchanged): b_k -= u, with q.u ~ max_k score(q, k) - target (least squares, per
    head): the maximum score of each query falls near target, under the saturation (64 steps of ln2/8 = 5.55)."""
    B, T = ids.shape
    D, H, DH = g.D, g.H, g.DH
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=dev), 1)
    x = g.wte[ids] + g.wpe[:T][None]
    for P in g.layers:
        mu = x.mean(-1, keepdim=True)
        h = (x - mu) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + g.eps)
        q = (h @ P["Wq"].T + P["bq"]).view(B, T, H, DH).transpose(1, 2)
        k = (h @ P["Wk"].T + P["bk"]).view(B, T, H, DH).transpose(1, 2)
        s = (q @ k.transpose(-1, -2)).masked_fill(mask, -1e9)
        m = s.max(-1).values                                               # (B, H, T)
        for hh in range(H):
            Qm = q[:, hh].reshape(-1, DH)
            u = torch.linalg.lstsq(Qm, (m[:, hh].reshape(-1) - target)[:, None]).solution[:, 0]
            P["bk"][hh * DH:(hh + 1) * DH] -= u
        k = (h @ P["Wk"].T + P["bk"]).view(B, T, H, DH).transpose(1, 2)
        v = (h @ P["Wv"].T + P["bv"]).view(B, T, H, DH).transpose(1, 2)
        a = torch.softmax((q @ k.transpose(-1, -2)).masked_fill(mask, -1e9), -1)
        x = x + (a @ v).transpose(1, 2).reshape(B, T, D) @ P["Wo"].T + P["bo"]
        mu = x.mean(-1, keepdim=True)
        h = (x - mu) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + g.eps)
        x = x + gelu_new(h @ P["W1"].T + P["b1"]) @ P["W2"].T + P["b2"]


def qweights(g):
    """8-bit weights: one scale per block of 1024 rows x 1024 columns (like the blocks of the chain), values in [-63, 63]."""
    def qw(W):
        W = W.clone()
        for r in range(0, W.shape[0], 1024):
            for c in range(0, W.shape[1], 1024):
                blk = W[r:r + 1024, c:c + 1024]
                s = blk.abs().max() / 63
                W[r:r + 1024, c:c + 1024] = torch.round(blk / s) * s
        return W
    for P in g.layers:
        for w in ("Wq", "Wk", "Wv", "Wo", "W1", "W2"):
            P[w] = qw(P[w])


def data(tk):
    from datasets import load_dataset
    if "TinyStories" in NAME:
        ds = load_dataset("roneneldan/TinyStories", split="validation")
        texts = ds["text"][:3000]
    else:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t) > 200]
    ids = tk("\n\n".join(texts), return_tensors="pt")["input_ids"][0]
    nseq = min(len(ids) // L, 2 * NSEQ)
    seqs = ids[:nseq * L].view(nseq, L)
    return seqs[:NSEQ].to(dev), seqs[NSEQ:2 * NSEQ].to(dev)          # calibration, evaluation


def ppl(logits, ids):
    lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None])[..., 0]
    return float(torch.exp(-lp.mean()))


def main():
    tk, m = load()
    g = G(m)
    print(f"{NAME} ({g.kind}): {g.NL} layers, d = {g.D}, {g.H} heads, vocabulary {g.head.shape[0]}, "
          f"{sum(p.numel() for p in m.parameters()) / 1e6:.0f} M parameters")
    cal, ev = data(tk)
    with torch.no_grad():
        ref = m(ev).logits.to(torch.float64)
        mine = forward(g, ev)
        print(f"floating-point reimplementation: max gap of the logits {float((ref - mine).abs().max()):.2e} ; perplexity {ppl(ref, ev):.2f}")
        if "--center" in sys.argv:
            g.center()
        if "--center" in sys.argv or "--rotate" in sys.argv:
            g.fold()
        if "--rotate" in sys.argv:
            g.rotate()
        if "--kbar" in sys.argv:
            st = {}
            forward(g, cal[:32], stats=st)
            for l, P in enumerate(g.layers):
                P["bk"] = P["bk"] - torch.stack(st[f"kmean.{l}"]).mean(0)
        if "--kmax" in sys.argv:
            kmax_offsets(g, cal[:32])
        chk = forward(g, ev)
        print(f"after exact transformations: max gap of the logits {float((ref - chk).abs().max()):.2e}")
        stats = {}
        for k in range(0, len(cal), 16):
            forward(g, cal[k:k + 16], stats=stats)
        qz = {k: float(np.percentile(torch.cat(v).numpy(), PCT)) / 63 for k, v in stats.items() if not k.startswith("kmean")}
        for k in ("x0", "x.0.a", f"x.{g.NL // 2}.b", f"x.{g.NL - 1}.b"):
            a = torch.cat(stats[k]).numpy()
            print(f"  {k:8s} : p50 {np.percentile(a, 50):8.3f}  p99.9 {np.percentile(a, 99.9):8.3f}  max {a.max():9.2f}   (step {qz[k]:.4f})")
        for l in (0, g.NL // 2, g.NL - 1):
            a = torch.cat(stats[f"sc.{l}"]).numpy()
            print(f"  scores layer {l}: p99 {np.percentile(a, 99):.2f}  p99.9 {np.percentile(a, 99.9):.2f}  max {a.max():.2f}  "
                  f"(saturated beyond {64 * C_SM:.2f}): {100 * np.mean(a > 64 * C_SM):.3f} % saturated")
        qweights(g)
        out = forward(g, ev, qz=qz)
        top_f, top_q = ref[:, :-1].argmax(-1), out[:, :-1].argmax(-1)
        print(f"SIMULATED INTEGER: perplexity {ppl(out, ev):.2f} (float {ppl(ref, ev):.2f}); next token identical to the float "
              f"{100 * float((top_f == top_q).float().mean()):.1f} %")
        lg = ref[:, :-1] - ref[:, :-1].mean(-1, keepdim=True)
        top2 = lg.topk(2, -1).values
        print(f"centred logits: max p50 {float(top2[..., 0].median()):.2f}, min p1 {float(lg.min(-1).values.quantile(0.01)):.2f} ; "
              f"top1-top2 gap < 0.25: {100 * float(((top2[..., 0] - top2[..., 1]) < 0.25).float().mean()):.1f} %")


if __name__ == "__main__":
    main()
