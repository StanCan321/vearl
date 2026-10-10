#!/usr/bin/env python3
"""ResNet-20 CIFAR-10 (residual blocks, stride-2 1x1 shortcuts, folded BatchNorm, global mean) executed on a REAL n = 1024 chain by a GPU worker:
one image per job (76 operations of which 22 1024^3 GEMMs), each operation verified before payment; output compared BIT FOR BIT with the independent Python engine.

    python3 tools/resnet_demo.py MODEL.json          # tools/qat_graph.py resnet MODEL.pt OUT.json --samples 16; GPU required
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import vearl_ref as R
from netlab import Lab, run

MODEL = sys.argv[1]
M = json.load(open(MODEL))
NS = len(M["samples"])
lab = Lab({"W": dict(port=9891, seed=2, extra=["--worker", "--gpu"], peers=[])}, n=M["n"], divisor=64, window=120)
lab.start("W")
time.sleep(8)
W = lab.addrof("W")
t0 = time.time()
out = run("vearl-cli", "infer", W, W, "seed:4", "1", MODEL, "0", str(NS - 1), "/tmp/rn_out.i8", os.environ.get("INFLIGHT", "4"), timeout=7200)
open("/tmp/infer_log.txt", "w").write(out)                     # complete log (VEARL_TIMING=1: breakdown of the verification time)
print(out.splitlines()[-1][:250])
assert "INFER:" in out, out[-800:]
S = np.fromfile("/tmp/rn_out.i8", dtype=np.int8).reshape(NS * M["batch"], M["classes"])
y = np.array([l for s in M["samples"] for l in s["labels"]])
print(f"{NS} CIFAR-10 images classified by the network in {time.time() - t0:.1f} s: {100 * float((S.argmax(1) == y).mean()):.1f} % correct (every operation verified before payment)")
R.N = M["n"]
try:
    import torch
    R._mm = lambda a, b: torch.round(torch.from_numpy(np.asarray(a, dtype=np.float64)) @ torch.from_numpy(np.asarray(b, dtype=np.float64))).numpy().astype(np.int64)
except ImportError:
    pass
Wt = [R.dec(w, M["n"]) for w in M["weights"]]
ok = 0
for s in range(NS):
    T, _ = R.run_program(R.dec(M["samples"][s]["x0"], M["n"]), Wt, M["ops"])
    ok += int(np.array_equal(T[-1][:M["classes"], 0], S[s]))
print(f"network output == independent Python reference engine (bit for bit): {ok}/{NS}")
lab.stop_all()
