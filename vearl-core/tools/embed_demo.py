#!/usr/bin/env python3
"""BULK TEXT EMBEDDINGS, verified, then SIMILARITY SEARCH by large-scale linear algebra, on an n=1024 chain with a
GPU worker of a real node.

  1. `vearl-cli infer`: NB batches of 1024 texts (AG News, MLP encoder distilled from MiniLM, int8) -> NB jobs; the weights are sent only ONCE
     (worker registry); each job is verified operation by operation (Freivalds + recomputation) before payment.
  2. The output of the chain is compared BIT FOR BIT with the independent Python reference engine (OUT.eval.npz).
  3. Quality: nearest neighbour of the same category (chain output) vs MiniLM teacher, floating-point student and TF-IDF.
  4. Similarity: S = Q * D^T (1024 queries x 7168 documents) computed by a `tiled` job (7 block products), verified, compared with numpy.

    python3 tools/embed_demo.py MODEL.json [NB=20]     # after cargo build --release --features gpu --bins; GPU required
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from netlab import Lab, run

MODEL = sys.argv[1]
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 20
EV = np.load(MODEL.replace(".json", ".eval.npz"))
DIM, N = 384, 1024

lab = Lab({"W": dict(port=9831, seed=2, extra=["--worker", "--gpu"], peers=[])}, n=N, divisor=int(os.environ.get("DIVISOR", 64)), window=int(os.environ.get("WINDOW", 120)))
lab.start("W")
time.sleep(8)
W = lab.addrof("W")

# ---------------- 1. bulk embeddings ----------------
t0 = time.time()
out = run("vearl-cli", "infer", W, W, "seed:4", "1", MODEL, "0", str(NB - 1), "/tmp/embed_out.i8", os.environ.get("INFLIGHT", "4"), timeout=7200)
print(out)
assert "INFER:" in out, "the batch did not complete"
wall = time.time() - t0
E = np.fromfile("/tmp/embed_out.i8", dtype=np.int8).reshape(-1, DIM)
print(f"{len(E)} embeddings of {DIM} dimensions received and verified in {wall:.1f} s = {len(E) / wall:.0f} embeddings/s (end to end, a single A10 worker)")
ref = EV["int8"].astype(np.int8)
M = len(ref)
print(f"network output == independent Python reference engine (bit for bit) on {M} test texts: {bool((E[:M] == ref).all())}")

# ---------------- 3. quality ----------------
y = EV["labels"]


def nn_same_class(X):
    X = X.astype(np.float64)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = X @ X.T
    np.fill_diagonal(S, -2)
    return float((y[S.argmax(1)] == y).mean())


print(f"nearest neighbour of the same category (AG News, {M} texts): int8 network {100 * nn_same_class(E[:M]):.2f} % | floating-point student {100 * nn_same_class(EV['student']):.2f} % | MiniLM teacher {100 * nn_same_class(EV['teacher']):.2f} %")

# ---------------- 4. large-scale similarity: S = Q * D^T by a tiled job ----------------
Ef = E[:M].astype(np.float64)
Ef /= np.linalg.norm(Ef, axis=1, keepdims=True) + 1e-9
scale = 64.0 / np.percentile(np.abs(Ef), 99.99)
Eq = np.clip(np.round(Ef * scale), -64, 64).astype(np.int8)          # normalised embeddings, quantised over [-64, 64]
NQ = 1024
BJ = -(-M // N)
A = np.zeros((N, N), np.int8)
A[:NQ, :DIM] = Eq[:NQ]
B = np.zeros((N, BJ * N), np.int8)
B[:DIM, :M] = Eq.T
A.tofile("/tmp/sim_A.i8")
B.tofile("/tmp/sim_B.i8")
t1 = time.time()
tout = run("vearl-cli", "tiled-run", W, W, "seed:4", "5", "/tmp/sim_A.i8", "/tmp/sim_B.i8", "1", "1", str(BJ), "/tmp/sim_C.i32", timeout=3600)
print(tout)
S = np.fromfile("/tmp/sim_C.i32", dtype=np.int32).reshape(N, BJ * N)[:NQ, :M].astype(np.int64)
exact = Eq[:NQ].astype(np.int64) @ Eq.astype(np.int64).T
print(f"similarity matrix {NQ} x {M} computed, verified and paid in {time.time() - t1:.1f} s; identical to numpy (exact): {bool((S == exact).all())}")
S[np.arange(NQ), np.arange(NQ)] = -10**9
nn = S.argmax(1)
print(f"nearest document search (via the network's verified matrix): same category for {100 * float((y[nn] == y[:NQ]).mean()):.2f} % of the {NQ} queries")
lab.stop_all()
