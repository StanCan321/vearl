#!/usr/bin/env python3
"""Points of the miners of a pool (docs/16 section 13, docs/19 section 3 bis).

A pool is the `miner` of its blocks, so the chain does not say who among its miners earned the points of a block. The pool publishes a signed
document for every hour (`/api/points/epoch/N`): the work (share weight) of each miner, the blocks it found and the Merkle root of the list of
shares. This module reads those documents, checks them, and shares the points of the pool's blocks between its miners in proportion to their
verified work:

    points of the pool's blocks of a scoring day  x  (work of the miner in that day / work of all the pool's miners in that day)

Only miners that speak vpool/2 (they hold signed receipts, so they can prove what the pool owes them) take part. An hour that the audit
(`vearl-pool-audit`, report `audit.json`) found fraudulent counts for nothing; an hour marked `suspect` is held until a person has looked at it;
a pool with no audit report yet is held as well (`unaudited='hold'`, or `'count'` to accept the documents once they are signed and consistent).
Needs `pip install cryptography` (Ed25519 signatures). Standard library otherwise.
"""
import hashlib
import json
import os

DOC_DOMAIN = b"vearl/pool-points/v1\0"
CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def canonical(obj):
    """What the pool signs and hashes: keys sorted, no spaces, integers only (docs/16 section 13)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _without_sig(doc):
    return {k: v for k, v in doc.items() if k != "sig"}


def doc_hash(doc):
    return hashlib.sha256(canonical(_without_sig(doc)).encode()).hexdigest()


def pool_key(address):
    """Public key (bytes) of a vrl1 address."""
    sys_path_fix()
    from verify_registration import bech32m_decode
    hrp, pk = bech32m_decode(address)
    if hrp != "vrl" or len(pk) != 32:
        raise ValueError("not a classic vrl1 address")
    return pk


def sys_path_fix():
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)


def _polymod(values):
    gen = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for v in values:
        top = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ v
        for i in range(5):
            chk ^= gen[i] if (top >> i) & 1 else 0
    return chk


def address_of(pk, hrp="vrl"):
    """vrl1... address of a 32-byte public key (Bech32m), the inverse of pool_key (used by the tests and the tools)."""
    acc, bits, vals = 0, 0, []
    for b in pk:
        acc = (acc << 8) | b
        bits += 8
        while bits >= 5:
            bits -= 5
            vals.append((acc >> bits) & 31)
    if bits:
        vals.append((acc << (5 - bits)) & 31)
    exp = [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]
    mod = _polymod(exp + vals + [0] * 6) ^ 0x2BC830A3
    chk = [(mod >> 5 * (5 - i)) & 31 for i in range(6)]
    return hrp + "1" + "".join(CHARSET[v] for v in vals + chk)


def verify_doc(doc, address, allowed_epoch_secs=(3600,)):
    """None if the document is signed by the pool `address` and arithmetically consistent, otherwise the reason.

    `allowed_epoch_secs`: the epoch lengths the program counts (one hour on the public testnet). A pool that chose its own windows could move work
    between scoring days."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    if doc.get("proto") != "vpool-points/1":
        return "unknown document type"
    if doc.get("pool") != address:
        return "document of another pool"
    try:
        pk = pool_key(address)
        sig = bytes.fromhex(doc.get("sig", ""))
        Ed25519PublicKey.from_public_bytes(pk).verify(sig, DOC_DOMAIN + canonical(_without_sig(doc)).encode())
    except (InvalidSignature, ValueError, TypeError):
        return "bad signature"
    secs = doc.get("epoch_secs")
    e = doc.get("epoch")
    if isinstance(secs, bool) or not isinstance(secs, int) or secs not in allowed_epoch_secs:
        return "epoch length not supported"
    if isinstance(e, bool) or not isinstance(e, int) or e < 0 or doc.get("start") != e * secs or doc.get("end") != (e + 1) * secs:
        return "epoch window inconsistent"
    try:
        miners = doc["miners"]
        if len({m["address"] for m in miners}) != len(miners):
            return "a miner is listed twice"
        for m in miners:
            # plain non-negative integers only: a negative weight would shrink the total and make the others' shares exceed the points to share
            if not (isinstance(m["address"], str) and isinstance(m["weight"], str) and isinstance(m["v2_weight"], str) and m["weight"].isdigit() and m["v2_weight"].isdigit()):
                return "a weight is not a plain integer"
            for k in ("shares", "v2_shares"):
                if isinstance(m[k], bool) or not isinstance(m[k], int) or m[k] < 0:
                    return "a share count is not a plain integer"
        tw = sum(int(m["weight"]) for m in miners)
        ts = sum(m["shares"] for m in miners)
        if not isinstance(doc["total_weight"], str) or not doc["total_weight"].isdigit() or int(doc["total_weight"]) != tw or doc["total_shares"] != ts or doc["shares_count"] != ts:
            return "totals do not add up"
        if any(int(m["v2_weight"]) > int(m["weight"]) or m["v2_shares"] > m["shares"] for m in miners):
            return "vpool/2 figures exceed the totals"
    except (KeyError, ValueError, TypeError):
        return "unreadable miners list"
    return None


def load_docs(directory, address, allowed_epoch_secs=(3600,)):
    """Reads and checks the `epoch-N.json` documents of a folder: ({epoch: document}, [problems])."""
    docs, problems = {}, []
    for name in sorted(os.listdir(directory)):
        if not (name.startswith("epoch-") and name.endswith(".json")):
            continue
        try:
            with open(os.path.join(directory, name), encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError):
            problems.append(f"{name}: unreadable")
            continue
        why = verify_doc(doc, address, allowed_epoch_secs)
        if why:
            problems.append(f"{name}: {why}")
            continue
        docs[doc["epoch"]] = doc
    prev = None
    for e in sorted(docs):
        if prev is not None and docs[e].get("prev") != doc_hash(docs[prev]):
            problems.append(f"epoch {e}: does not commit to the previous document we hold (missing or rewritten document)")
        prev = e
    return docs, problems


def load_audit(path):
    """({epoch: 'ok'|'suspect'|'fraud'}, verdict) from an audit.json written by vearl-pool-audit, or ({}, None)."""
    if not path or not os.path.exists(path):
        return {}, None
    with open(path, encoding="utf-8") as f:
        a = json.load(f)
    return {int(k): v for k, v in a.get("epochs", {}).items()}, a.get("verdict")


def load_pools(spec_path):
    """pools.json: {"pools": [{"address": "vrl1...", "name": "...", "dir": "documents folder", "audit": "audit.json"}]} (paths relative to the file)."""
    base = os.path.dirname(os.path.abspath(spec_path))
    out = {}
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    for p in spec["pools"]:
        directory = os.path.join(base, p["dir"])
        docs, problems = load_docs(directory, p["address"], tuple(p.get("epoch_secs", [3600])))
        epochs, verdict = load_audit(os.path.join(base, p["audit"]) if p.get("audit") else None)
        out[p["address"]] = {"address": p["address"], "name": p.get("name", p["address"][:12]), "docs": docs, "problems": problems, "audit": epochs, "verdict": verdict}
    return out


def pools_hash(pools):
    """Fingerprint of everything the pool side of a result was computed from."""
    h = hashlib.sha256()
    for a in sorted(pools):
        p = pools[a]
        h.update(canonical({"a": a, "docs": {str(e): doc_hash(d) for e, d in sorted(p["docs"].items())}, "audit": {str(e): v for e, v in sorted(p["audit"].items())}}).encode())
    return h.hexdigest()


def allocate(pool, pot, start_ts, epoch_secs, unaudited="hold"):
    """Shares the pool's points between its miners.

    pot: {scoring period index: points the pool's blocks earned in it}; a document belongs to the period of its start time.
    Returns ({address: {"points", "periods": {index: points}, "days": {24-hour windows with work}}}, report) where report counts hours and points by outcome."""
    weights, active, report = {}, {}, {"epochs_counted": 0, "epochs_held": 0, "epochs_void": 0, "points_distributed": 0.0, "points_unattributed": 0.0}
    for e, doc in sorted(pool["docs"].items()):
        if doc["start"] < start_ts:
            continue
        status = pool["audit"].get(e)
        if status is None:
            status = "ok" if unaudited == "count" else "held"
        elif status == "suspect":
            status = "held"
        if status == "fraud":
            report["epochs_void"] += 1
            continue
        if status == "held":
            report["epochs_held"] += 1
            continue
        report["epochs_counted"] += 1
        period = (doc["start"] - start_ts) // epoch_secs
        for m in doc["miners"]:
            w = int(m["v2_weight"])
            if w > 0:
                weights.setdefault(period, {})
                weights[period][m["address"]] = weights[period].get(m["address"], 0) + w
                active.setdefault(m["address"], set()).add((doc["start"] - start_ts) // 86400)
    out = {}
    for period, points in sorted(pot.items()):
        ws = weights.get(period, {})
        total = sum(ws.values())
        if total == 0:
            report["points_unattributed"] += points
            continue
        for addr, w in ws.items():
            r = out.setdefault(addr, {"points": 0.0, "periods": {}, "days": active.get(addr, set())})
            share = points * w / total
            r["points"] += share
            r["periods"][period] = r["periods"].get(period, 0.0) + share
        report["points_distributed"] += points
    return out, report
