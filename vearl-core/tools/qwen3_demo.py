#!/usr/bin/env python3
"""Qwen3-Embedding-0.6B (0.44 B weights) executed and VERIFIED on a real n = 1024 chain by a worker: each text = 10 chained jobs.

    python3 tools/qwen3_demo.py /tmp/qwen3-int8.json [--texts 4] [--gpu]

Hand-written sentences (4 themes): the embeddings obtained through the chain are compared bit for bit with the independent reference engine; their
similarities show the topics. The worker registry must hold the ~530 weight blocks (cap raised to 1024).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from netlab import Lab
from qwen3_client import Qwen3EmbedClient

SENT = [
    ("sport", "the striker scored twice as the team won the championship final on saturday"),
    ("sport", "the coach said the players trained hard before the season opener against their rivals"),
    ("business", "shares fell sharply as oil prices rose and investors worried about inflation and interest rates"),
    ("business", "the company reported quarterly profits above analysts expectations and raised its forecast"),
    ("tech", "researchers released a new software update with improved security for the internet browser"),
    ("tech", "the chip maker unveiled a faster processor for computers and mobile phones"),
    ("world", "the president met foreign leaders to discuss the peace treaty and the border conflict"),
    ("world", "the government announced new sanctions after the election in the capital"),
]
MP = [a for a in sys.argv[1:] if not a.startswith("--") and not a.isdigit()][0]
NT = int(sys.argv[sys.argv.index("--texts") + 1]) if "--texts" in sys.argv else len(SENT)
extra = ["--worker"] + (["--gpu"] if "--gpu" in sys.argv else [])
lab = Lab({"W": dict(port=9861, seed=2, extra=extra, peers=[])}, n=1024, divisor=int(os.environ.get("DIVISOR", 64)),
          window=int(os.environ.get("WINDOW", 300)))
lab.start("W")
time.sleep(8)
W = lab.addrof("W")
try:
    c = Qwen3EmbedClient(MP, W, W, inflight=int(os.environ.get("INFLIGHT", 8)))
    sel = SENT[:NT]
    texts = [s for _, s in sel]
    t0 = time.time()
    last, E = c.embed_int(texts)
    dt = time.time() - t0
    same = 0
    for k, t in enumerate(texts):
        l2, e2 = c.expected(t)
        same += int(np.array_equal(last[k], l2))
    print(f"{len(texts)} texts embedded, VERIFIED and paid in {dt:.1f} s ({dt / len(texts):.1f} s/text, {len(c.m['jobs'])} jobs per text); "
          f"identical to the independent engine: {same}/{len(texts)}")
    S = E @ E.T
    np.fill_diagonal(S, -2)
    for i, (th, s) in enumerate(sel):
        j = int(S[i].argmax())
        print(f"  [{th:8s}] {s[:56]:56s} -> [{sel[j][0]:8s}] {sel[j][1][:38]}  (cos {S[i, j]:.2f})")
    for k, log in enumerate(c.logs[-3:]):
        print(f"job log {k}: " + log.strip().splitlines()[-1][:220])
finally:
    lab.stop_all()
