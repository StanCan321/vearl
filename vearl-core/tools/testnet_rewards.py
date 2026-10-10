#!/usr/bin/env python3
"""Testnet rewards: scores the contributors of the public testnet from the blocks of the chain (rules: docs/19-testnet-rewards.md).

    python3 tools/testnet_rewards.py --node HOST:PORT --start 2026-10-23T16:00:00Z --out rewards-out [--dump-blocks]
    python3 tools/testnet_rewards.py --blocks-file rewards-out/blocks.json --start 2026-10-23T16:00:00Z --out rewards-check

--node reads every block through the public API of a node (/api/blocks) and, with --dump-blocks, saves them in OUT/blocks.json together
with their SHA-256: anybody can publish or download such a file and re-run the computation with --blocks-file to get exactly the same
result. Nothing else than the blocks is used, so the score of a miner cannot be inflated off-chain.

Score: 1 point per valid block mined to the address, 1.5 for a block that carried paid work (a job of a client), multiplied by 1.25 during
the first three days. Eligibility: at least 3 distinct active days (24-hour windows since the start), a minimum share of all the points,
a registered address if a registration list is given, and not on the exclusion list. The reward pool is shared between the eligible
addresses in proportion to their points. All parameters are printed in the output and can be changed with the options below.
Python 3 standard library only.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

DEFAULTS = {
    "epoch_days": 1,          # length of a scoring period, for the leaderboard (the mainnet is expected soon after the start: days, not weeks)
    "early_days": 3,          # the first days get a multiplier (the network is most fragile at the start)
    "early_mult": 1.25,
    "useful_mult": 1.5,       # a block that carried paid work counts more than a block of native work
    "min_active_days": 3,     # distinct 24-hour windows with at least one block
    "min_share": 0.00005,     # of all the points (0.005 %): no dust
    "pool_pct": 5.0,          # informational: percentage of the mainnet supply set aside by the team (docs/19)
}


def parse_time(s):
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp())


def fetch_blocks(node, timeout=30):
    """All the blocks of the chain, oldest first: {height, timestamp, miner, work}."""
    def get(path):
        for attempt in range(4):
            try:
                with urllib.request.urlopen(f"http://{node}{path}", timeout=timeout) as r:
                    return json.load(r)
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
    first = get("/api/blocks?limit=1")
    tip = first.get("tip", 0)
    out, start = [], tip - 1
    while start >= 0:
        page = get(f"/api/blocks?from={start}&limit=100").get("blocks", [])
        if not page:
            break
        for b in page:
            out.append({"height": b["height"], "timestamp": b["timestamp"], "miner": b["miner"], "work": b["work"]["type"]})
        start = min(b["height"] for b in page) - 1
    out.sort(key=lambda b: b["height"])
    return out


def data_hash(blocks):
    canon = json.dumps(blocks, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canon).hexdigest()


def score(blocks, start_ts, p, registered=None, excluded=()):
    """The leaderboard from a list of blocks. `registered`: set of addresses (None = no registration needed)."""
    epoch_secs = p["epoch_days"] * 86400
    miners = {}
    last_ts = start_ts
    epochs = {}
    for b in blocks:
        ts = b["timestamp"]
        if ts < start_ts:
            continue
        last_ts = max(last_ts, ts)
        ep = (ts - start_ts) // epoch_secs
        useful = b["work"] != "native"
        pts = (p["useful_mult"] if useful else 1.0) * (p["early_mult"] if ts - start_ts < p["early_days"] * 86400 else 1.0)
        m = miners.setdefault(b["miner"], {"address": b["miner"], "blocks": 0, "useful_blocks": 0, "points": 0.0, "days": set(), "per_epoch": {}})
        m["blocks"] += 1
        m["useful_blocks"] += 1 if useful else 0
        m["points"] += pts
        m["days"].add((ts - start_ts) // 86400)
        m["per_epoch"][ep] = m["per_epoch"].get(ep, 0.0) + pts
        epochs[ep] = epochs.get(ep, 0) + 1
    total = sum(m["points"] for m in miners.values())
    rows = []
    for m in miners.values():
        share = m["points"] / total if total else 0.0
        reasons = []
        if m["address"] in excluded:
            reasons.append("excluded by the team (see docs/19)")
        if registered is not None and m["address"] not in registered:
            reasons.append("address not registered")
        if len(m["days"]) < p["min_active_days"]:
            reasons.append(f"fewer than {p['min_active_days']} active days ({len(m['days'])})")
        if share < p["min_share"]:
            reasons.append(f"share below {p['min_share'] * 100:.3f} %")
        rows.append({"address": m["address"], "blocks": m["blocks"], "useful_blocks": m["useful_blocks"], "points": round(m["points"], 4),
                     "share": share, "active_days": len(m["days"]), "eligible": not reasons, "reasons": reasons,
                     "points_per_epoch": {str(k): round(v, 4) for k, v in sorted(m["per_epoch"].items())}})
    rows.sort(key=lambda r: (-r["points"], r["address"]))
    elig_total = sum(r["points"] for r in rows if r["eligible"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        r["pool_share"] = (r["points"] / elig_total) if r["eligible"] and elig_total else 0.0
    return {
        "totals": {"blocks": sum(r["blocks"] for r in rows), "miners": len(rows), "eligible": sum(1 for r in rows if r["eligible"]),
                   "points": round(total, 4), "eligible_points": round(elig_total, 4), "last_block_time": last_ts},
        "epochs": [{"index": k, "blocks": v} for k, v in sorted(epochs.items())],
        "miners": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--node", help="HOST:PORT of a node API")
    src.add_argument("--blocks-file", help="blocks.json written earlier by --dump-blocks")
    ap.add_argument("--start", required=True, help="start of the testnet, ISO UTC (e.g. 2026-10-23T16:00:00Z)")
    ap.add_argument("--end", help="ignore the blocks after this ISO UTC time (final snapshot)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--dump-blocks", action="store_true", help="save the blocks used in OUT/blocks.json (for reproduction)")
    ap.add_argument("--registered", help="JSON file made by verify_registration.py: only these addresses can be eligible")
    ap.add_argument("--exclude", help="text file, one address per line (disqualified, with a public reason in docs/19)")
    ap.add_argument("--pool-tokens", type=float, help="size of the reward pool in tokens: adds a token amount to every eligible row")
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args()
    p = {k: getattr(a, k) for k in DEFAULTS}
    start_ts = parse_time(a.start)
    if a.node:
        blocks = fetch_blocks(a.node)
    else:
        blocks = json.load(open(a.blocks_file))["blocks"]
    if a.end:
        end_ts = parse_time(a.end)
        blocks = [b for b in blocks if b["timestamp"] <= end_ts]
    registered = None
    if a.registered:
        registered = set(json.load(open(a.registered))["addresses"])
    excluded = set()
    if a.exclude:
        excluded = {l.strip() for l in open(a.exclude) if l.strip() and not l.startswith("#")}
    result = score(blocks, start_ts, p, registered, excluded)
    if a.pool_tokens:
        for r in result["miners"]:
            r["tokens"] = round(r["pool_share"] * a.pool_tokens, 8)
    result = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "start": a.start, "params": p,
              "data_hash": data_hash([b for b in blocks if b["timestamp"] >= start_ts]), **result}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "leaderboard.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=1)
    with open(os.path.join(a.out, "leaderboard.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["rank", "address", "blocks", "useful_blocks", "points", "share", "active_days", "eligible", "pool_share", "reasons"])
        for r in result["miners"]:
            w.writerow([r["rank"], r["address"], r["blocks"], r["useful_blocks"], r["points"], f"{r['share']:.6f}", r["active_days"], r["eligible"], f"{r['pool_share']:.6f}", "; ".join(r["reasons"])])
    if a.dump_blocks:
        with open(os.path.join(a.out, "blocks.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"data_hash": result["data_hash"], "blocks": blocks}, f, separators=(",", ":"))
    t = result["totals"]
    print(f"{t['blocks']} blocks, {t['miners']} miners, {t['eligible']} eligible; data hash {result['data_hash'][:16]}...; written to {a.out}")


if __name__ == "__main__":
    sys.exit(main())
