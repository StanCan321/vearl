#!/usr/bin/env python3
"""PyTorch CNN (convolutions + folded BatchNorm + max-pooling) converted into a VEARL program, executed on a REAL chain (n = 128) by a
worker: 64 batches of 2 handwritten digits = 128 images, each batch verified operation by operation before payment (`vearl-cli infer`).

    python3 tools/cnn_demo.py [model=models/digits-cnn-maxpool.json]
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from netlab import Lab, run

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/digits-cnn-maxpool.json"
M = json.load(open(MODEL))
NS, B, C = len(M["samples"]), M["batch"], M["classes"]
lab = Lab({"W": dict(port=9861, seed=2, extra=["--worker"], peers=[])}, n=M["n"], divisor=64, window=120)
lab.start("W")
time.sleep(6)
W = lab.addrof("W")
t0 = time.time()
out = run("vearl-cli", "infer", W, W, "seed:4", "1", MODEL, "0", str(NS - 1), "/tmp/cnn_out.i8", "8", timeout=1800)
print(out.splitlines()[-1][:230])
assert "INFER:" in out, out[-800:]
S = np.fromfile("/tmp/cnn_out.i8", dtype=np.int8).reshape(NS * B, C)
y = np.array([l for s in M["samples"] for l in s["labels"]])
print(f"{NS * B} images classified by the network (every operation verified before payment): {100 * float((S.argmax(1) == y).mean()):.2f} % correct, {time.time() - t0:.1f} s")
lab.stop_all()
