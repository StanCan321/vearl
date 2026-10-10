#!/usr/bin/env python3
"""Sustained throughput: K MNIST jobs (1024 images each) submitted back to back to a GPU worker (n=1024 chain), then verified and paid.
Measures images/s end to end (submission -> Complete, then client verification), to be compared with the raw computation.
    python3 tools/mnist_throughput.py [K=8] [model]"""
import os, re, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from netlab import Lab, run

K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MODEL = sys.argv[2] if len(sys.argv) > 2 else "models/mnist-mlp.json"
lab = Lab({"W": dict(port=9811, seed=2, extra=["--worker", "--gpu"], peers=[])}, n=1024, divisor=64)
lab.start("W")
time.sleep(8)
W = lab.addrof("W")
t0 = time.time()
ids = []
for k in range(K):
    out = run("vearl-cli", "prog-submit", W, W, "seed:4", "20", "0", f"model:{MODEL}", timeout=300)
    ids.append(re.search(r"job_id = ([0-9a-f]{64})", out).group(1))
t_sub = time.time() - t0
pending = set(ids)
while pending and time.time() - t0 < 600:
    for i in list(pending):
        if "posted: 3" in run("vearl-cli", "prog-status", W, i):
            pending.discard(i)
    time.sleep(0.3)
t_done = time.time() - t0
t1 = time.time()
ok = 0
for i in ids:
    out = run("vearl-cli", "prog-finish", W, W, "seed:4", i, "0", f"model:{MODEL}", timeout=600)
    ok += "Settled" in out
t_ver = time.time() - t1
imgs = K * 1024
print(f"{K} jobs = {imgs} images: submission {t_sub:.1f} s; computation and publication by the worker {t_done:.1f} s in total ({imgs / t_done:.0f} images/s); sequential client verification {t_ver:.1f} s ({imgs / t_ver:.0f} images/s); {ok}/{K} jobs settled")
lab.stop_all()
