#!/usr/bin/env python3
"""Small text-embedding TRANSFORMER (1 layer, 1 head, d = 64, FFN 128, 32 tokens, output 128) trained by distillation from
all-MiniLM-L6-v2, with EXACTLY the topology that the chain knows how to execute and verify (docs/12 section 6): input projection, LayerNorm without affine,
Q / K / V, attention (scores K^T Q, softmax per column), output projection, residuals, ReLU FFN, final LayerNorm, sum over the tokens, projection.

    python3 train_tiny_transformer.py OUT.npz [--epochs 8]       # uses the teacher cache /tmp/teacher_minilm_120000.npz
Output: OUT.npz (weights, vocabulary, scale) and a floating-point quality report (nearest neighbour of the same category, AG News).
It is NOT MiniLM: a tiny transformer, trained on 120,000 texts, which shows that ATTENTION runs and is verified on the chain.
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embed_text import tokens

arg = lambda name, default, cast=str: cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
EPOCHS = arg("--epochs", 8, int)
V, L = 8000, 32
D, FF, DOUT, LAYERS = arg("--d", 64, int), arg("--ff", 128, int), arg("--dout", 128, int), arg("--layers", 1, int)
torch.manual_seed(7)
np.random.seed(7)
torch.set_num_threads(os.cpu_count() or 8)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Embedding(V + 2, D, padding_idx=0)          # 0 = PAD, 1 = UNK
        self.P = torch.nn.Parameter(torch.randn(L, D) * 0.1)
        lin = lambda a, b: torch.nn.Linear(a, b, bias=False)
        self.Win, self.Wout = lin(D, D), lin(D, DOUT)
        self.blocks = torch.nn.ModuleList([torch.nn.ModuleDict({k: v for k, v in dict(Wq=lin(D, D), Wk=lin(D, D), Wv=lin(D, D), Wo=lin(D, D), W1=lin(D, FF), W2=lin(FF, D)).items()}) for _ in range(LAYERS)])
        self.proj = lin(DOUT, 384)                                       # training only: teacher's space

    @staticmethod
    def ln(x, eps=1e-3):
        return (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + eps)

    def encode(self, tok):
        x0 = self.E(tok) + self.P                                        # (B, L, D)
        r0 = self.Win(x0)
        r = r0
        for blk in self.blocks:
            h = self.ln(r)
            q, k, v = blk["Wq"](h), blk["Wk"](h), blk["Wv"](h)
            a = torch.softmax(k @ q.transpose(1, 2) / D ** 0.5, dim=1)      # (B, key i, query j): softmax over the KEYS
            o = v.transpose(1, 2) @ a                                        # (B, D, L): V . A
            r = r + blk["Wo"](o.transpose(1, 2))
            h2 = self.ln(r)
            r = r + blk["W2"](torch.relu(blk["W1"](h2)))
        h3 = self.ln(r)
        return self.Wout(h3.sum(1))                                      # sum of the tokens then projection

    def forward(self, tok):
        return self.proj(self.encode(tok))


def encode_texts(texts, index):
    out = np.zeros((len(texts), L), dtype=np.int64)
    for r, t in enumerate(texts):
        ids = [index.get(w, 1) for w in tokens(t)][:L]
        out[r, :len(ids)] = ids
    return out


if __name__ == "__main__":
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news")
    tr_t, te_t = ds["train"]["text"][:120000], ds["test"]["text"]
    te_y = np.array(ds["test"]["label"])
    z = np.load("/tmp/teacher_minilm_120000.npz")
    Ttr, Tte = torch.from_numpy(z["tr"]), z["te"]
    cnt = {}
    for t in tr_t:
        for w in tokens(t):
            cnt[w] = cnt.get(w, 0) + 1
    vocab = [w for w, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:V]]
    index = {w: i + 2 for i, w in enumerate(vocab)}
    Xtr, Xte = torch.from_numpy(encode_texts(tr_t, index)), torch.from_numpy(encode_texts(te_t, index))
    net = Tiny()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    steps = EPOCHS * ((len(Xtr) + 255) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    t0 = time.time()
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(perm), 256):
            b = perm[s:s + 256]
            opt.zero_grad()
            (1 - torch.nn.functional.cosine_similarity(net(Xtr[b]), Ttr[b]).mean()).backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            cs = torch.nn.functional.cosine_similarity(net(Xte[:2000]), torch.from_numpy(Tte[:2000])).mean().item()
        print(f"epoch {ep + 1}/{EPOCHS}: student/teacher cosine (test) {cs:.4f} ({time.time() - t0:.0f} s)", flush=True)
    net.eval()
    M = 7168
    with torch.no_grad():
        Z = net.encode(Xte[:M]).numpy()

    def nn_purity(E, y):
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        S = E @ E.T
        np.fill_diagonal(S, -2)
        return float((y[S.argmax(1)] == y).mean())

    print(f"small transformer (float): nearest neighbour of the same category {100 * nn_purity(Z, te_y[:M]):.2f} % (MiniLM 85.7 %, bag-of-words student 82.3 %)")
    sd = {k: v.detach().numpy().astype(np.float64) for k, v in net.state_dict().items() if not k.startswith("proj")}
    sd["meta"] = np.array([D, FF, DOUT, LAYERS, L])
    np.savez(sys.argv[1], vocab=np.array(vocab), **{k.replace(".", "_"): v for k, v in sd.items()})
    json.dump({"vocab": vocab, "L": L, "V": V}, open(sys.argv[1].replace(".npz", ".vocab.json"), "w"))
    print("weights written:", sys.argv[1])
