#!/usr/bin/env python3
"""all-MiniLM-L6-v2 (converted to int8 and retrained for the format) executed and VERIFIED on a real n = 1024 chain by a GPU worker.

    python3 tools/minilm_demo.py /tmp/minilm-qat.json [--ag 64]      # after tools/qat_minilm.py; GPU required

1) 8 hand-written sentences (4 themes): embeddings obtained by the chain, bit-for-bit identical to the independent reference engine,
   compared with the ORIGINAL floating-point MiniLM (cosine), nearest neighbour of the same theme;
2) --ag N: N AG News texts (throughput, mean cosine, nearest neighbour of the same category).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from netlab import Lab
from minilm_client import MiniLMEmbedClient
import quantize_minilm as QM

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
MP = [a for a in sys.argv[1:] if not a.startswith("--")][0]
NAG = int(sys.argv[sys.argv.index("--ag") + 1]) if "--ag" in sys.argv else 0
nz = lambda E: np.asarray(E, dtype=np.float64) / (np.linalg.norm(np.asarray(E, dtype=np.float64), axis=1, keepdims=True) + 1e-12)

lab = Lab({"W": dict(port=9861, seed=2, extra=["--worker", "--gpu"], peers=[])}, n=1024, divisor=int(os.environ.get("DIVISOR", 64)), window=int(os.environ.get("WINDOW", 120)))
lab.start("W")
time.sleep(8)
W = lab.addrof("W")
c = MiniLMEmbedClient(MP, W, W, inflight=int(os.environ.get("INFLIGHT", 16)))
tk, sd = QM.load()

def run(texts, labels, title):
    t0 = time.time()
    E = c.embed(texts)
    dt = time.time() - t0
    same = sum(int(np.array_equal(E[k], c.expected(t))) for k, t in enumerate(texts[:16]))
    ref = np.array([QM.float_forward(sd, QM.ids_of(tk, t))[0] for t in texts])
    cos = np.sum(nz(E) * nz(ref), 1)
    S = nz(E) @ nz(E).T
    np.fill_diagonal(S, -2)
    Sf = nz(ref) @ nz(ref).T
    np.fill_diagonal(Sf, -2)
    lab_ = np.array(labels)
    print(f"[{title}] {len(texts)} texts embedded, VERIFIED and paid in {dt:.1f} s ({dt / len(texts):.1f} s/text); "
          f"identical to the independent reference engine: {same}/{min(16, len(texts))} ; cosine with the original floating-point MiniLM: mean {cos.mean():.3f}, min {cos.min():.3f}")
    print(f"[{title}] nearest neighbour of the same category: chain (int8) {np.mean(lab_[S.argmax(1)] == lab_):.1%}, floating-point MiniLM {np.mean(lab_[Sf.argmax(1)] == lab_):.1%}")
    return E, S

E, S = run([s for _, s in SENT], [t for t, _ in SENT], "phrases")
for i, (th, s) in enumerate(SENT):
    j = int(S[i].argmax())
    print(f"  [{th:8s}] {s[:56]:56s} -> neighbour: [{SENT[j][0]:8s}] {SENT[j][1][:38]}  (cos {S[i, j]:.2f})")
if NAG:
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news")
    run(list(ds["test"]["text"][1000:1000 + NAG]), list(ds["test"]["label"][1000:1000 + NAG]), "AG News")
print(c.logs[-1].strip().splitlines()[-1][:230])
lab.stop_all()
