#!/usr/bin/env python3
"""TRANSFORMER (attention, LayerNorm, softmax, residuals, FFN) executed on a REAL chain and VERIFIED operation by operation: 256 AG News texts
-> 256 jobs (one 32-token text per job) -> int8 embeddings of 128 dimensions, compared BIT FOR BIT with the independent Python engine.

    python3 tools/transformer_demo.py MODEL.json          # tools/quantize_transformer.py (WEIGHTS.npz MODEL.json); chain n = 128
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import vearl_ref as R
from netlab import Lab, run

MODEL = sys.argv[1]
M = json.load(open(MODEL))
NS = len(M["samples"])
lab = Lab({"W": dict(port=9881, seed=2, extra=["--worker"], peers=[])}, n=M["n"], divisor=64, window=120)
lab.start("W")
time.sleep(6)
W = lab.addrof("W")
t0 = time.time()
out = run("vearl-cli", "infer", W, W, "seed:4", "1", MODEL, "0", str(NS - 1), "/tmp/tr_out.i8", os.environ.get("INFLIGHT", "16"), timeout=3600)
print(out.splitlines()[-1][:230])
assert "INFER:" in out, out[-800:]
E = np.fromfile("/tmp/tr_out.i8", dtype=np.int8).reshape(NS, M["classes"])
# reference: the independent engine executes the same program
R.N = M["n"]
Wt = [R.dec(w, M["n"]) for w in M["weights"]]
ok = 0
for s in range(NS):
    T, _ = R.run_program(R.dec(M["samples"][s]["x0"], M["n"]), Wt, M["ops"])
    ok += int(np.array_equal(T[-1][:M["classes"], 0], E[s]))
print(f"{NS} embeddings of {M['classes']} dimensions produced by the chain in {time.time() - t0:.1f} s; bit-for-bit identical to the Python reference engine: {ok}/{NS}")
ev = np.load(MODEL.replace(".json", ".eval.npz"))
y = ev["labels"][:NS]
En = E.astype(np.float64)
En /= np.linalg.norm(En, axis=1, keepdims=True) + 1e-9
S = En @ En.T
np.fill_diagonal(S, -2)
print(f"nearest neighbour of the same category among these {NS} texts (chain output): {100 * float((y[S.argmax(1)] == y).mean()):.1f} %")
lab.stop_all()
