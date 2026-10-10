#!/usr/bin/env python3
"""PCA (principal component analysis) of the text embeddings, computed with VERIFIED JOBS of the network:
  1. Gram matrix G = X^T X of 20,480 embeddings (384 dimensions): K = 20,480 rows = 20 blocks -> 3 jobs (8 + 8 + 4 blocks), each one verified;
  2. covariance = (G - N mu mu^T) / N (exact algebra on the client side), eigenvalues/eigenvectors (384 x 384, local: small);
  3. projection X V_k onto the first k components, by a verified job (V_k quantised to int8);
  4. quality: nearest neighbour of the same category after projection to k dimensions, against the original 384 dimensions.

    python3 tools/pca_demo.py MODEL.json      # after tools/embed_demo.py (which writes /tmp/embed_out.i8); GPU required
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from netlab import Lab
from la_client import LA

MODEL = sys.argv[1]
EV = np.load(MODEL.replace(".json", ".eval.npz"))
E = np.fromfile("/tmp/embed_out.i8", dtype=np.int8).reshape(-1, 384)
N = len(E)
print(f"{N} int8 embeddings of 384 dimensions (verified output of embed_demo.py)")
lab = Lab({"W": dict(port=9841, seed=2, extra=["--worker", "--gpu"], peers=[])}, n=1024, divisor=int(os.environ.get("DIVISOR", 64)), window=int(os.environ.get("WINDOW", 120)))
lab.start("W")
time.sleep(8)
W = lab.addrof("W")
la = LA(W, W, "seed:4", escrow_per_product=5)

# ---- 1. Gram par jobs verifies ----
t0 = time.time()
G = la.gram(E)
exact = E.astype(np.int64).T @ E.astype(np.int64)
print(f"Gram 384 x 384 over {N} samples: {la.jobs} verified jobs in {time.time() - t0:.1f} s; identical to numpy (exact): {bool((G == exact).all())}")

# ---- 2. covariance et ACP (local, 384 x 384) ----
mu = E.astype(np.float64).mean(0)
cov = (G - N * np.outer(mu, mu)) / N
w, V = np.linalg.eigh(cov)
order = np.argsort(w)[::-1]
w, V = w[order], V[:, order]
ev = w / w.sum()
print("explained variance:" + ", ".join(f"{k} comp. == {100 * ev[:k].sum():.1f} %" for k in (2, 8, 32, 64)))

# ---- 3. projection by a verified job ----
Xc = E[:len(EV["labels"])]                                   # the test texts (known labels)
y = EV["labels"]
res = {}
for k in (8, 32, 64):
    Vk = V[:, :k]
    s = 63.0 / np.abs(Vk).max()
    Vq = np.clip(np.round(Vk * s), -63, 63).astype(np.int8)
    t1 = time.time()
    P = la.matmul(Xc, Vq)                                    # exact int32, verified
    P = P.astype(np.float64) / s - (mu @ Vk)                 # removes the mean (exact algebra) and the scale
    ref = (Xc.astype(np.float64) - mu) @ Vk
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    S = Pn @ Pn.T
    np.fill_diagonal(S, -2)
    res[k] = float((y[S.argmax(1)] == y).mean())
    print(f"projection to {k} dimensions by a verified job ({time.time() - t1:.1f} s): gap to the floating-point projection {np.abs(P - ref).max():.3f} ; nearest neighbour of the same category {100 * res[k]:.2f} %")
Xn = Xc.astype(np.float64) / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-9)
S = Xn @ Xn.T
np.fill_diagonal(S, -2)
print(f"reference, original 384 dimensions: {100 * float((y[S.argmax(1)] == y).mean()):.2f} %")
lab.stop_all()
