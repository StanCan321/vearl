#!/usr/bin/env python3
"""DATA AVAILABILITY of a program: a worker that WITHHOLDS its results (--no-serve, TEST) cannot be paid. The client (`vearl-cli infer`)
cannot verify anything; it then demands a column ON-CHAIN (Fetch); the worker does not serve it; after SERVE_SECS anyone triggers SlashUnserved:
bond shared, escrow returned. Then the honest case: same request, the worker serves, the client accepts.

    python3 tools/da_demo.py [model=models/digits-cnn-maxpool.json]
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from netlab import Lab, run

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/digits-cnn-maxpool.json"
os.environ["VEARL_SERVE_SECS"] = "15"            # the test bench geneses set serve_secs = 15
lab = Lab({"W": dict(port=9871, seed=2, extra=["--worker", "--no-serve"], peers=[])}, n=json.load(open(MODEL))["n"], divisor=64, window=120)
lab.start("W")
time.sleep(6)
W = lab.addrof("W")
print("worker that withholds the data:")
out = run("vearl-cli", "infer", W, W, "seed:4", "1", MODEL, "0", "0", "/tmp/da_out.i8", "1", timeout=900)
print("\n".join(l[:200] for l in out.splitlines() if "batch 0" in l or "INFER" in l))
print("block-wise product (worker that withholds the data):")
import numpy as np
n = json.load(open(MODEL))["n"]
rng = np.random.default_rng(3)
rng.integers(-64, 65, (n, n), dtype=np.int8).tofile("/tmp/da_A.i8")
rng.integers(-64, 65, (n, n), dtype=np.int8).tofile("/tmp/da_B.i8")
out = run("vearl-cli", "tiled-run", W, W, "seed:4", "2", "/tmp/da_A.i8", "/tmp/da_B.i8", "1", "1", "1", "/tmp/da_C.i32", timeout=900)
print(chr(10).join(l[:200] for l in out.splitlines() if "unavailable" in l or "Fetch" in l or "SlashUnserved" in l or "status" in l))
lab.stop_all()
