#!/usr/bin/env python3
"""Experiment: does the quality of the embedding student depend on the size of the input vocabulary? (floating-point only, without quantisation)
    python3 tools/exp_vocab.py 1024 4096 8192     # uses the teacher cache /tmp/teacher_minilm_120000.npz (created by train_embed.py)"""
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embed_text import Featurizer
from datasets import load_dataset

ds = load_dataset("fancyzhx/ag_news")
tr_t, te_t = ds["train"]["text"][:120000], ds["test"]["text"]
te_y = np.array(ds["test"]["label"])
z = np.load("/tmp/teacher_minilm_120000.npz")
Ttr, Tte = z["tr"], z["te"]
torch.set_num_threads(os.cpu_count() or 8)


def nn_purity(E, y):
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    S = E @ E.T
    np.fill_diagonal(S, -2)
    return float((y[S.argmax(1)] == y).mean())


M = 7168
for size in map(int, sys.argv[1:]):
    t0 = time.time()
    f = Featurizer.fit(tr_t, size=size)
    Xtr, Xte = f.transform(tr_t).astype(np.float32), f.transform(te_t[:M]).astype(np.float32)
    net = torch.nn.Sequential(torch.nn.Dropout(0.15), torch.nn.Linear(size, 2048, bias=False), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(2048, 384, bias=False))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    X, Y = torch.from_numpy(Xtr), torch.from_numpy(Ttr.astype(np.float32))
    steps = 12 * ((len(X) + 255) // 256)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for ep in range(12):
        net.train()
        perm = torch.randperm(len(X))
        for s in range(0, len(perm), 256):
            b = perm[s:s + 256]
            opt.zero_grad()
            (1 - torch.nn.functional.cosine_similarity(net(X[b]), Y[b]).mean()).backward()
            opt.step()
            sched.step()
    net.eval()
    with torch.no_grad():
        E = net(torch.from_numpy(Xte)).numpy()
    cos = float(np.mean(np.sum(E * Tte[:M], 1) / (np.linalg.norm(E, axis=1) * np.linalg.norm(Tte[:M], axis=1) + 1e-9)))
    print(f"vocabulary {size:5d}: student/teacher cosine {cos:.3f} ; nearest neighbour of the same category {100 * nn_purity(E, te_y[:M]):.2f} % ({time.time() - t0:.0f} s)", flush=True)
