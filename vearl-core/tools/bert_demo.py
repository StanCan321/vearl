#!/usr/bin/env python3
"""BGE-base (110 M parameters) executed and VERIFIED on a real n = 1024 chain by a GPU worker: each text = three chained jobs.

    python3 tools/bert_demo.py /tmp/bge-qat.json [--ag 32]

1) 8 hand-written sentences (4 themes): embeddings obtained by the chain, bit-for-bit identical to the independent reference engine,
   compared with the ORIGINAL floating-point BGE (cosine), nearest neighbour of the same theme; 2) --ag N: N AG News texts (throughput, cosine).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from netlab import Lab
from bert_client import BertEmbedClient

SENT = [
    ("world", "the president met foreign leaders to discuss the peace treaty and the border conflict"),
    ("world", "the government announced new sanctions after the election in the capital"),
    ("sport", "the striker scored twice as the team won the championship final on saturday"),
    ("sport", "the coach said the players trained hard before the season opener against their rivals"),
    ("business", "shares fell sharply as oil prices rose and investors worried about inflation and interest rates"),
    ("business", "the company reported quarterly profits above analysts expectations and raised its forecast"),
    ("tech", "researchers released a new software update with improved security for the internet browser"),
    ("tech", "the chip maker unveiled a faster processor for computers and mobile phones"),
]
MP = [a for a in sys.argv[1:] if not a.startswith("--") and not a.isdigit()][0]
NAG = int(sys.argv[sys.argv.index("--ag") + 1]) if "--ag" in sys.argv else 0
nz = lambda E: np.asarray(E, np.float64) / (np.linalg.norm(np.asarray(E, np.float64), axis=1, keepdims=True) + 1e-12)

lab = Lab({"W": dict(port=9861, seed=2, extra=["--worker", "--gpu"], peers=[])}, n=1024, divisor=int(os.environ.get("DIVISOR", 64)),
          window=int(os.environ.get("WINDOW", 300)))
lab.start("W")
time.sleep(8)
W = lab.addrof("W")
try:
    c = BertEmbedClient(MP, W, W, inflight=int(os.environ.get("INFLIGHT", 16)))
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(c.m["tokenizer"], device="cpu")
    st.max_seq_length = c.L

    def run(texts, labels, title):
        t0 = time.time()
        E = c.embed(texts)
        dt = time.time() - t0
        same = sum(int(np.array_equal(E[k], c.expected(t))) for k, t in enumerate(texts[:8]))
        ref = st.encode(list(texts), normalize_embeddings=True)
        cos = np.sum(nz(E) * ref, 1)
        S = nz(E) @ nz(E).T
        np.fill_diagonal(S, -2)
        lab_ = np.array(labels)
        print(f"[{title}] {len(texts)} texts embedded, VERIFIED and paid in {dt:.1f} s ({dt / len(texts):.2f} s/text, "
              f"{len(c.m['jobs'])} jobs per text); identical to the independent engine: {same}/{min(8, len(texts))} ; "
              f"cosine with the original floating-point BGE: mean {cos.mean():.3f}, min {cos.min():.3f} ; "
              f"nearest neighbour of the same category {100 * np.mean(lab_[S.argmax(1)] == lab_):.1f} %")
        return S

    S = run([s for _, s in SENT], [t for t, _ in SENT], "phrases")
    for i, (th, s) in enumerate(SENT):
        j = int(S[i].argmax())
        print(f"  [{th:8s}] {s[:56]:56s} -> [{SENT[j][0]:8s}] {SENT[j][1][:38]}  (cos {S[i, j]:.2f})")
    if NAG:
        from datasets import load_dataset
        ds = load_dataset("fancyzhx/ag_news")
        run(list(ds["test"]["text"][1000:1000 + NAG]), list(ds["test"]["label"][1000:1000 + NAG]), "AG News")
    for k, log in enumerate(c.logs[-3:]):
        print(f"job {k}: " + log.strip().splitlines()[-1][:200])
finally:
    lab.stop_all()
