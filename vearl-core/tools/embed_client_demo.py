#!/usr/bin/env python3
"""Demonstration of the TEXT EMBEDDING CLIENT: HAND-WRITTEN sentences (outside the training set) are sent to a GPU worker of a real node,
embedded, verified, paid; the nearest neighbour of each sentence must be of the same theme.

    python3 tools/embed_client_demo.py MODEL.json          # after tools/train_embed.py; GPU required
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from netlab import Lab
from embed_client import EmbedClient, TransformerEmbedClient

SENT = [
    ("world", "the president met foreign leaders to discuss the peace treaty and the border conflict"),
    ("world", "the government announced new sanctions after the election in the capital"),
    ("sport", "the striker scored twice as the team won the championship final on saturday"),
    ("sport", "the coach said the players trained hard before the season opener against their rivals"),
    ("economy", "shares fell sharply as oil prices rose and investors worried about inflation and interest rates"),
    ("economy", "the company reported quarterly profits above analysts expectations and raised its forecast"),
    ("techno", "researchers released a new software update with improved security for the internet browser"),
    ("techno", "the chip maker unveiled a faster processor for computers and mobile phones"),
]
TR = "--transformer" in sys.argv
MP = [a for a in sys.argv[1:] if not a.startswith("--")][0]
lab = Lab({"W": dict(port=9851, seed=2, extra=["--worker"] + ([] if TR else ["--gpu"]), peers=[])}, n=128 if TR else 1024, divisor=int(os.environ.get("DIVISOR", 64)), window=int(os.environ.get("WINDOW", 120)))
lab.start("W")
time.sleep(8)
W = lab.addrof("W")
ec = TransformerEmbedClient(MP, MP.replace("-int8.json", "-float.npz"), W, W) if TR else EmbedClient(MP, W, W, "seed:4")
t0 = time.time()
E = ec.embed([s for _, s in SENT])
print(f"{len(E)} sentences -> int8 embeddings {E.shape}, verified and paid in {time.time() - t0:.1f} s")
print(ec.last_log.strip().splitlines()[-1][:220])
S = EmbedClient.similarity(E, E)
np.fill_diagonal(S, -2)
ok = 0
for i, (th, s) in enumerate(SENT):
    j = int(S[i].argmax())
    ok += SENT[j][0] == th
    print(f"  [{th:8s}] {s[:58]:58s} -> neighbour: [{SENT[j][0]:8s}] {SENT[j][1][:40]}  (cos {S[i, j]:.2f})")
print(f"nearest neighbour of the same theme: {ok}/{len(SENT)}")
lab.stop_all()
