#!/usr/bin/env python3
"""Pins the Quantus miner builds that a release ships (docs/17-fee-protection.md).

    python deploy/release/pin_quantus.py --windows path/to/quantus-miner.exe --linux path/to/quantus-miner \
        --tool path/to/vearl-miner.exe

Writes vearl-core/quantus-miner.pins ("windows <blake3>" / "linux <blake3>"). The node and the launcher are compiled with that file
(include_str!) and refuse any other quantus-miner binary, so this MUST be run BEFORE the final build of node / vearl-miner, and the
SAME files must then be put in the release archives. Any later change of quantus-miner (even a rebuild) needs a new pin and a new
build of node, vearl-miner.
"""
import argparse
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--windows", required=True)
ap.add_argument("--linux", required=True)
ap.add_argument("--tool", required=True, help="any built vearl-miner (its `hash FILE` command prints the BLAKE3 hash)")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vearl-core", "quantus-miner.pins"))
a = ap.parse_args()


def h(path):
    r = subprocess.run([a.tool, "hash", path], capture_output=True, text=True)
    out = r.stdout.strip()
    if r.returncode != 0 or len(out) != 64:
        sys.exit(f"cannot hash {path}: {r.stderr.strip() or out}")
    return out


with open(a.out, "w", newline="\n") as f:
    f.write(f"windows {h(a.windows)}\nlinux {h(a.linux)}\n")
print(open(a.out).read(), end="")
