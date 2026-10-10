#!/usr/bin/env python3
"""LARGE-SCALE LINEAR ALGEBRA end to end on an n=1024 chain with a GPU worker: C = A*B with int8 matrices
(BI*1024) x (BK*1024) and (BK*1024) x (BJ*1024), cut into 1024 x 1024 blocks (`tiled` jobs). The client deposits A and B by rows,
submits, downloads and VERIFIES each block product (Merkle root + Freivalds), sums locally, then Accept; C is compared with an exact
reference product (torch float64: integers < 2^53).

    python3 tools/la_demo.py [BI BK BJ = 4 4 4]           # after cargo build --release --features gpu --bins; GPU required
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from netlab import Lab, run

BI, BK, BJ = (int(v) for v in sys.argv[1:4]) if len(sys.argv) >= 4 else (4, 4, 4)
N = 1024
rng = np.random.default_rng(11)
A = rng.integers(-64, 65, (BI * N, BK * N), dtype=np.int8)
B = rng.integers(-64, 65, (BK * N, BJ * N), dtype=np.int8)
A.tofile("/tmp/la_A.i8")
B.tofile("/tmp/la_B.i8")
gemm_gop = 2.0 * BI * N * BK * N * BJ * N / 1e9
print(f"A {A.shape}, B {B.shape}: {BI * BK * BJ} block products, {gemm_gop:.0f} GOP")

lab = Lab({"W": dict(port=9821, seed=2, extra=["--worker", "--gpu"] + os.environ.get("CHEAT", "").split(), peers=[])}, n=N, divisor=int(os.environ.get("DIVISOR", 64)), window=int(os.environ.get("WINDOW", 120)))
lab.start("W")
time.sleep(8)
W = lab.addrof("W")
t0 = time.time()
out = run("vearl-cli", "tiled-run", W, W, "seed:4", "5", "/tmp/la_A.i8", "/tmp/la_B.i8", str(BI), str(BK), str(BJ), "/tmp/la_C.i32", timeout=3600)
print(out)
wall = time.time() - t0
if os.environ.get("CHEAT"):
    print("cheating worker (", os.environ["CHEAT"], "): client output above; final status below")
    import time as _t
    for _ in range(60):
        st = run("vearl-cli", "tiled-status", W, out.split("job_id = ")[1].split()[0])
        if "Slashed" in st or "Settled" in st:
            break
        _t.sleep(2)
    print("job status:", st[:60])
    lab.stop_all()
    sys.exit(0)
C = np.fromfile("/tmp/la_C.i32", dtype=np.int32).reshape(BI * N, BJ * N)
import torch
torch.set_num_threads(os.cpu_count() or 8)
t1 = time.time()
ref = (torch.from_numpy(A.astype(np.float64)) @ torch.from_numpy(B.astype(np.float64))).numpy()
print(f"reference product (torch CPU float64, exact): {time.time() - t1:.1f} s; bit-for-bit identical to the network's verified result: {bool((ref == C).all())}")
print(f"total end to end: {wall:.1f} s for {gemm_gop:.0f} GOP = {gemm_gop / wall:.1f} GOP/s effective")
for _ in range(30):
    st = run("vearl-cli", "tiled-status", W, out.split("job_id = ")[1].split()[0])
    if "Settled" in st:
        break
    time.sleep(2)
print("job status:", st[:60])
for l in open(f"{lab.dir}/W.log", errors="replace"):
    if "[worker]" in l:
        print(l.strip())
lab.stop_all()
