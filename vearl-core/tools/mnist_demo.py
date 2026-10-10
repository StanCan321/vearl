#!/usr/bin/env python3
"""Demonstration of REAL USEFULNESS: classify the first 1024 MNIST test images with an int8 MLP, on a chain of width
n=1024, executed by a GPU worker of a REAL node; each operation is verified (Freivalds + recomputation) before payment; then the
EMBEDDINGS (activations of hidden layer 2, published in the tensors of the program) are used for a search of neighbouring images.

    python3 tools/mnist_demo.py [model=models/mnist-mlp.json]     # after cargo build --release --features gpu --bins; GPU required
"""
import json, os, re, struct, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import light_client as L
from netlab import Lab, run

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/mnist-mlp.json"
MJ = json.load(open(MODEL))
NOPS = len(MJ["ops"])
NGEMM = sum(1 for o in MJ["ops"] if "Gemm" in o)
NODES = {"W": dict(port=9801, seed=2, extra=["--worker", "--gpu"], peers=[])}
DIV = int(os.environ.get("DIVISOR", 64))          # target: 64 = almost instantaneous blocks; ~2e9 = one block every ~30 s on an A10 (n=1024)
lab = Lab(NODES, n=1024, divisor=DIV, window=int(os.environ.get("WINDOW", 120)))
lab.start("W")
time.sleep(8)
W = lab.addrof("W")
print("chain n=1024 started; GPU worker online")
t0 = time.time()
out = run("vearl-cli", "prog-submit", W, W, "seed:4", "20", "0", f"model:{MODEL}", timeout=600)
jid = re.search(r"job_id = ([0-9a-f]{64})", out).group(1)
print(f"job submitted ({jid[:8]}): {out.splitlines()[0][:80]}")
while True:
    st = run("vearl-cli", "prog-status", W, jid)
    if f"posted: {NOPS}" in st:
        break
    time.sleep(0.5)
    assert time.time() - t0 < 900, st
t_compute = time.time() - t0
print(f"the worker computed and published the {NOPS} operations ({NGEMM} GEMM) in {t_compute:.1f} s (submission -> Complete)")
t1 = time.time()
fin = run("vearl-cli", "prog-finish", W, W, "seed:4", jid, "0", f"model:{MODEL}", timeout=1800)
print(fin.strip().replace("\n", "\n  "))
print(f"verification by the client (Freivalds on {NGEMM} GEMM + exact recomputation of the other operations) and payment: {time.time()-t1:.1f} s")

if NOPS != 3:
    lab.stop_all()
    sys.exit(0)
# ---- embeddings: tensor T_2 (output of operation 1) published by the worker and verified with the rest of the program ----
p = L.Peer(W)
p.s.settimeout(120)
p.send(struct.pack("<I", 34) + bytes.fromhex(jid) + bytes([1]))       # GetOp{job, op=1}
while True:
    r = L.Rd(p.recv())
    if r.u32() == 35:
        break
z = r.u8()
assert z == 0 or r.vec(lambda: r.u32())
x = np.frombuffer(bytes(r.take(r.u64())), dtype=np.int8).reshape(1024, 1024)
emb = x[:256, :].T.astype(np.float64)        # 1024 images x 256 dimensions
labels = np.array(json.load(open(MODEL))["samples"][0]["labels"])
e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
sim = e @ e.T
np.fill_diagonal(sim, -1)
nn = sim.argmax(1)
print(f"embeddings (256 dimensions, 1024 images, published and verified by the network): nearest neighbour of the same class == {100 * (labels[nn] == labels).mean():.2f} % (similar image search, without any label)")
lab.stop_all()
