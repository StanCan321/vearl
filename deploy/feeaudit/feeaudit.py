#!/usr/bin/env python3
"""Fee auditor of the combined Vearl + Quantus miner (docs/17-fee-protection.md).

Reads the signed beacons collected by a seed node (`node --collect-fee-beacons`, local API /api/fee/beacons), then asks the PUBLIC API of
the Quantus pool for the workers of every operator and of the development address, and checks that the fee worker `vrl-fee-<id>` carries
about `fee_pct` percent of the Quantus work of the operator's worker `vrl-<id>` (both are time slices of the same GPU).

    feeaudit.py --beacons http://127.0.0.1:8545/api/fee/beacons --out report.json --public compliance.json --state state.json

Statuses: ok, warming_up (the fee miner only starts after ~80 s of Quantus time and the pool averages over hours), low, no_fee (violation),
wrong_fee_pct (the beacon itself declares another fee), unofficial_build (hashes not in --known), not_mining_quantus, offline, unverifiable.
The public file lists only Vearl addresses and statuses (never Quantus addresses). What to DO with the result (listing, refusing project
services such as the faucet or the free compute gateway) is a policy decision of the project: this tool only measures.
"""
import argparse
import json
import sys
import time
import urllib.request

DEV_ADDRESS = "qzm7tR2aEJEdMFajwDspjiWefz2F4iDzsys32gE1DK8Dxc59P"
POOL_API = "https://quanpool.com/api/miner/mainnet"
EXPECTED_FEE_PCT = 10
GRACE_S = 3 * 3600          # a new beacon is not judged before 3 hours
ACTIVE_S = 2 * 3600         # a beacon older than this means the miner is offline
OK_RATIO = 0.5              # fee share >= 50 % of the announced fee (5 % of the Quantus work for a 10 % fee) passes
LOW_RATIO = 0.2


def get_json(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": "vearl-feeaudit/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def worker_rate(w):
    """Time-averaged work rate of a pool worker: the 6 h average, else 24 h, else 1 h."""
    for k in ("hashrate_6h", "hashrate_24h", "hashrate_1h"):
        v = w.get(k)
        if v:
            return float(v)
    return 0.0


def classify(b, user_workers, fee_workers, first_seen, now, known=None, expected_fee=EXPECTED_FEE_PCT):
    """One beacon -> (status, detail). `user_workers`/`fee_workers` are lists of pool workers, or None when the pool could not be read."""
    wid = b["worker_id"]
    if b.get("age_s", 0) > ACTIVE_S:
        return "offline", "no beacon for more than 2 hours"
    if b.get("fee_pct") != expected_fee:
        return "wrong_fee_pct", f"the miner declares a fee of {b.get('fee_pct')} % instead of {expected_fee} %"
    if known is not None and (b.get("node_hash") not in known.get("node", []) or (b.get("quantus_hash") not in known.get("quantus", []))):
        return "unofficial_build", "the hash of the node or of the Quantus miner is not an official release hash"
    if b.get("share_pct", 0) == 0:
        return "not_mining_quantus", "share_pct is 0"
    if user_workers is None or fee_workers is None:
        return "unverifiable", "the pool API could not be read"
    uw = next((w for w in user_workers if w.get("name") == f"vrl-{wid}"), None)
    fw = next((w for w in fee_workers if w.get("name") == f"vrl-fee-{wid}"), None)
    ur = worker_rate(uw) if uw else 0.0
    fr = worker_rate(fw) if fw else 0.0
    if ur <= 0:
        return "not_mining_quantus", f"no work from worker vrl-{wid} on the declared Quantus address"
    young = now - first_seen < GRACE_S
    ratio = fr / (fr + ur)
    expected = expected_fee / 100.0
    detail = f"fee share of the Quantus work {ratio * 100:.1f} % (announced {expected_fee} %)"
    if fr <= 0:
        return ("warming_up", detail) if young else ("no_fee", detail + ": no fee worker on the development address")
    if ratio >= OK_RATIO * expected:
        return "ok", detail
    if young:
        return "warming_up", detail
    return ("low" if ratio >= LOW_RATIO * expected else "no_fee"), detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beacons", required=True, help="URL of /api/fee/beacons on the seed (loopback) or a JSON file")
    ap.add_argument("--pool-api", default=POOL_API)
    ap.add_argument("--dev-address", default=DEV_ADDRESS)
    ap.add_argument("--known", help="JSON {node: [blake3...], quantus: [blake3...]} of official release hashes")
    ap.add_argument("--state", default="feeaudit-state.json", help="first-seen times (kept between runs)")
    ap.add_argument("--out", default="fee-report.json")
    ap.add_argument("--public", default="compliance.json")
    ap.add_argument("--delay", type=float, default=0.4, help="seconds between pool API requests")
    a = ap.parse_args()
    now = int(time.time())
    data = json.load(open(a.beacons)) if not a.beacons.startswith("http") else get_json(a.beacons)
    known = json.load(open(a.known)) if a.known else None
    try:
        state = json.load(open(a.state))
    except Exception:
        state = {}
    try:
        dev = get_json(f"{a.pool_api}/{a.dev_address}").get("workers", [])
    except Exception as e:
        print("cannot read the development address on the pool:", e, file=sys.stderr)
        dev = None
    cache, rows = {}, []
    for b in data.get("beacons", []):
        acc = b["account"]
        state.setdefault(acc, now)
        uw = None
        if dev is not None:
            q = b["quantus"]
            if q not in cache:
                time.sleep(a.delay)
                try:
                    cache[q] = get_json(f"{a.pool_api}/{q}").get("workers", [])
                except Exception:
                    cache[q] = None
            uw = cache[q]
        status, detail = classify(b, uw, dev, state[acc], now, known)
        rows.append({"vearl_address": b["vearl_address"], "worker_id": b["worker_id"], "status": status, "detail": detail, "version": b["version"], "age_s": b["age_s"]})
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    json.dump(state, open(a.state, "w"))
    json.dump({"generated": now, "counts": counts, "miners": rows}, open(a.out, "w"), indent=1)
    json.dump({"generated": now, "counts": counts, "miners": {r["vearl_address"]: r["status"] for r in rows}}, open(a.public, "w"), indent=1)
    print(json.dumps({"generated": now, "beacons": len(rows), "counts": counts}))
    return 1 if counts.get("no_fee") or counts.get("wrong_fee_pct") or counts.get("unofficial_build") else 0


if __name__ == "__main__":
    sys.exit(main())
