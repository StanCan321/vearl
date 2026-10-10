#!/usr/bin/env python3
"""Profile of a MiniLM series on a real chain: complete log of `vearl-cli infer` with VEARL_TIMING=1 (fetching, roots,
verification per batch) and job timeline.

    python3 tools/minilm_profile.py /tmp/minilm-qat.json [N_TEXTS]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["VEARL_TIMING"] = "1"
from netlab import Lab
from minilm_client import MiniLMEmbedClient

MP = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
lab = Lab({"W": dict(port=9861, seed=2, extra=["--worker", "--gpu"], peers=[])}, n=1024, divisor=int(os.environ.get("DIVISOR", 64)), window=int(os.environ.get("WINDOW", 120)))
lab.start("W")
time.sleep(8)
W = lab.addrof("W")
c = MiniLMEmbedClient(MP, W, W, inflight=int(os.environ.get("INFLIGHT", 16)))
from datasets import load_dataset
texts = list(load_dataset("fancyzhx/ag_news")["test"]["text"][1000:1000 + N])
t0 = time.time()
c.embed(texts)
print(f"{N} texts in {time.time() - t0:.1f} s")
print(c.logs[-1])
print("worker log:", f"{lab.dir}/W.log")
lab.stop_all()
