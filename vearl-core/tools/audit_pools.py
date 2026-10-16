#!/usr/bin/env python3
"""Audits the registered mining pools of the testnet rewards program (docs/16 section 13, docs/19 section 3 bis).

    python3 tools/audit_pools.py --spec pools.json --genesis genesis.json --blocks blocks.json --audit-bin /opt/vearl/bin/vearl-pool-audit

pools.json: {"pools": [{"address": "vrl1...", "name": "...", "url": "https://pool.example.org", "dir": "pool-example", "audit": "pool-example-audit.json"}]}
(paths relative to the file). For each pool with a `url` it downloads the signed documents and draws random shares (`fetch_pool_bundle.py`), runs
`vearl-pool-audit` against the chain blocks (`blocks.json` of the rewards run) and writes the report to `audit`; `testnet_rewards.py --pools`
reads the documents from `dir` and the report. A pool that cannot be reached keeps its previous documents and report. Exit status 0 even when a
pool is found fraudulent (that is a result, not an error). Standard library only.
"""
import argparse
import glob
import json
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--genesis", required=True)
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--audit-bin", required=True)
    ap.add_argument("--samples", default="16")
    ap.add_argument("--full", action="store_true", help="also download and verify the complete share lists")
    ap.add_argument("--last", default="0", help="only the last N epochs of each pool")
    ap.add_argument("--max-epochs", default="400", help="most epochs read from a pool in one run")
    ap.add_argument("--audit-arg", action="append", default=[], help="extra argument for vearl-pool-audit (repeat), e.g. --audit-arg=--verify-max --audit-arg=500")
    ap.add_argument("--receipts-dir", help="folder of pool-receipts*.jsonl files kept by miners (more evidence for the audit)")
    a = ap.parse_args()
    base = os.path.dirname(os.path.abspath(a.spec))
    here = os.path.dirname(os.path.abspath(__file__))
    with open(a.spec, encoding="utf-8") as f:
        pools = json.load(f)["pools"]
    for p in pools:
        if not p.get("url"):
            continue
        bundle = os.path.join(base, p["dir"])
        report = os.path.join(base, p.get("audit") or p["dir"] + "-audit.json")
        fetch = [sys.executable, os.path.join(here, "fetch_pool_bundle.py"), "--pool", p["url"], "--out", bundle, "--samples", a.samples, "--last", a.last if a.last != "0" else a.max_epochs]
        if a.full:
            fetch.append("--full")
        r = subprocess.run(fetch, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"{p.get('name', p['address'])}: cannot read the pool ({r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'error'}); documents and report left as they are")
            continue
        cmd = [a.audit_bin, "--bundle", bundle, "--genesis", a.genesis, "--pool", p["address"], "--blocks", a.blocks, "--out", report] + a.audit_arg
        if a.receipts_dir:
            for rf in sorted(glob.glob(os.path.join(a.receipts_dir, "*.jsonl"))):
                cmd += ["--receipts", rf]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout.strip())
        if r.returncode not in (0, 1, 2):
            # A pool must not be able to escape its audit by feeding it data that breaks it: a failed audit leaves a report that holds every hour
            # (the epochs are not listed as ok), instead of leaving the previous, favourable report in force.
            print(f"{p.get('name', p['address'])}: the audit program failed: {r.stderr.strip()[:300]}")
            with open(report, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"proto": "vpool-audit/1", "pool": p["address"], "verdict": "suspect", "epochs": {}, "stats": {},
                           "findings": [{"epoch": None, "severity": "suspect", "code": "audit_failed", "detail": "the audit program failed on this pool's data"}]}, f)


if __name__ == "__main__":
    sys.exit(main())
