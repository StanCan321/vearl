#!/usr/bin/env python3
"""Demonstration of the Python SDK (tools/vearl_client.py) against REAL nodes: an honest worker (product paid) and a
cheating worker (fraud detected by Freivalds, fraud proof published, worker penalised)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from netlab import Lab
from vearl_client import Client

NODES = {
    "H": dict(port=9601, seed=2, extra=["--worker", "--ask", "2"], peers=[]),          # honest worker
    "T": dict(port=9602, seed=3, extra=["--worker", "--cheat"], peers=["H"]),          # cheating worker
}
lab = Lab(NODES)
for n in NODES:
    lab.start(n)
    time.sleep(0.4)
time.sleep(6)
c = Client(lab.addrof("H"), seed=4)
print("client account:", c.account())
rng = np.random.default_rng(1)
ok = True
for label, w in [("honest", "H"), ("cheater", "T")]:
    A, B = rng.integers(-64, 65, (128, 128)), rng.integers(-64, 65, (128, 128))
    job = c.submit_matmul(lab.addrof(w), A, B, escrow_vrl=5)
    c.wait_status(job, "Posted")
    res = c.result_and_settle(job, lab.addrof(w), A, B)
    time.sleep(8)
    st = c.job_status(job)
    good = (res is not None and bool((res == A @ B).all()) and st.startswith("Settled")) if label == "honest" else (res is None and st.startswith("Slashed"))
    print(f"worker {label}: result {'accepted (== A@B)' if res is not None else 'REFUSED, fraud proof published'}; final status {st} -> {'OK' if good else 'FAIL'}")
    ok &= good
lab.stop_all()
print("VERDICT:", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
