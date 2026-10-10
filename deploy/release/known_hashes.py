#!/usr/bin/env python3
"""Official release hashes for the fee auditor (docs/17): BLAKE3 of node(.exe) and quantus-miner(.exe) of every platform of a release.

    python deploy/release/known_hashes.py --tool path/to/vearl-miner.exe --windows DIR_WINDOWS --linux DIR_LINUX \
        --out known-hashes.json [--merge old-known-hashes.json]

The beacons of the miners carry the hashes of their own executables; a hash that is not in this file means an unofficial build. Keep the
hashes of previous releases (--merge) as long as miners may still run them.
"""
import argparse
import json
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--tool", required=True)
ap.add_argument("--windows", required=True)
ap.add_argument("--linux", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--merge")
a = ap.parse_args()


def h(path):
    r = subprocess.run([a.tool, "hash", path], capture_output=True, text=True)
    if r.returncode != 0 or len(r.stdout.strip()) != 64:
        sys.exit(f"cannot hash {path}: {r.stderr.strip()}")
    return r.stdout.strip()


known = json.load(open(a.merge)) if a.merge and os.path.exists(a.merge) else {"node": [], "quantus": []}
for d, ext in ((a.windows, ".exe"), (a.linux, "")):
    for key, name in (("node", "node"), ("quantus", "quantus-miner")):
        v = h(os.path.join(d, name + ext))
        if v not in known[key]:
            known[key].append(v)
json.dump(known, open(a.out, "w"), indent=1)
print(json.dumps(known, indent=1))
