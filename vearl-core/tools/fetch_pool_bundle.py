#!/usr/bin/env python3
"""Downloads what a pool published, in the layout that `vearl-pool-audit` reads (docs/16 section 13).

    python3 tools/fetch_pool_bundle.py --pool https://pool.example.org --out bundle [--samples 16] [--full] [--last 48]

For every published epoch it saves the signed document (`epoch-N.json`). Only AFTER all the documents are on disk does it choose, at random, the
shares to examine (`secrets`, not predictable by the pool) and download them with their Merkle paths (`proofs-N.json`); the pool has already
committed to the list of shares through the Merkle root of the document, so it cannot know which shares will be asked for, and cannot
change them. `--full` also downloads the complete list of shares of each epoch (`shares-N.json`): the audit then recomputes every
miner's weight exactly. Documents already present are not downloaded again. The time at which each document was first seen is kept in
`seen.json` (the audit holds a document that appeared long after its epoch ended).

The pool is NOT trusted: everything it sends is checked before it is used as a file name or a number, answers are size-limited, and the number of
epochs read in one run is bounded. Standard library only.
"""
import argparse
import json
import os
import secrets
import sys
import time
import urllib.request

MAX_RESPONSE = 48 << 20      # bytes accepted for one answer (a complete share list is the largest: about 60 bytes a share)
MAX_EPOCHS = 400             # epochs read in one run
MAX_EPOCH_NUMBER = 1 << 40   # an epoch is a Unix time divided by its length: far below this
MAX_SHARES_PER_EPOCH = 5_000_000


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A pool that is not trusted must not be able to make this machine read another address (its own loopback API, for example) and have the
    answer saved in a folder that is published."""

    def redirect_request(self, *args, **kwargs):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def get(base, path, timeout=60, limit=MAX_RESPONSE):
    last = None
    for attempt in range(4):
        try:
            with _OPENER.open(base.rstrip("/") + path, timeout=timeout) as r:
                data = r.read(limit + 1)
            if len(data) > limit:
                raise ValueError(f"answer larger than {limit} bytes")
            return json.loads(data)
        except Exception as e:  # network error, HTTP error, bad JSON, too large: retry, then give up
            last = e
            time.sleep(1 + attempt)
    raise SystemExit(f"cannot read {path} from {base}: {last}")


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


def is_int(v, lo=0, hi=None):
    """A real integer (not a bool, not a float or a string) in [lo, hi]."""
    return isinstance(v, int) and not isinstance(v, bool) and v >= lo and (hi is None or v <= hi)


def is_hex(v, n):
    return isinstance(v, str) and len(v) == n and all(c in "0123456789abcdef" for c in v)


def clean_index(index):
    """The epoch numbers of a pool's index, or exit: they become file names and URLs, so only plain integers are accepted."""
    if not isinstance(index, dict) or not isinstance(index.get("epochs"), list):
        raise SystemExit("the pool's index is malformed")
    out = []
    for e in index["epochs"]:
        n = e.get("epoch") if isinstance(e, dict) else None
        if not is_int(n, 0, MAX_EPOCH_NUMBER):
            raise SystemExit(f"the pool's index holds an invalid epoch number: {str(n)[:40]!r}")
        out.append(n)
    if len(set(out)) != len(out):
        raise SystemExit("the pool's index lists an epoch twice")
    return sorted(out)


def clean_proof(p, epoch, count):
    """One share proof from the pool, checked before it is written to disk."""
    ok = isinstance(p, dict) and is_int(p.get("index"), 0, count - 1) and isinstance(p.get("record"), str) and 0 < len(p["record"]) <= 400 \
        and all(c in "0123456789abcdef" for c in p["record"]) and isinstance(p.get("path"), list) and len(p["path"]) <= 40 \
        and all(is_hex(x, 64) for x in p["path"])
    if not ok:
        raise SystemExit(f"epoch {epoch}: the pool sent a malformed share proof")
    return {"index": p["index"], "record": p["record"], "path": p["path"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, help="base URL of the pool's statistics server (the one that serves /api/points/...)")
    ap.add_argument("--out", required=True, help="output folder (the bundle)")
    ap.add_argument("--samples", type=int, default=16, help="shares drawn at random per epoch (default 16)")
    ap.add_argument("--full", action="store_true", help="also download the complete list of shares of every epoch")
    ap.add_argument("--last", type=int, default=0, help="only the last N epochs (default: all, at most %d)" % MAX_EPOCHS)
    a = ap.parse_args()
    if not a.pool.startswith(("http://", "https://")):
        raise SystemExit("--pool must be an http:// or https:// URL")
    os.makedirs(a.out, exist_ok=True)
    index = get(a.pool, "/api/points/index")
    epochs = clean_index(index)
    if a.last:
        epochs = epochs[-a.last:]
    epochs = epochs[-MAX_EPOCHS:]
    write_json(os.path.join(a.out, "index.json"), {"epochs": epochs})
    seen_path = os.path.join(a.out, "seen.json")
    try:
        with open(seen_path, encoding="utf-8") as f:
            seen = {str(k): int(v) for k, v in json.load(f).items()}
    except (OSError, ValueError, TypeError, AttributeError):
        seen = {}
    # 1. the documents first: the pool commits before it knows what will be asked
    docs = {}
    for e in epochs:
        path = os.path.join(a.out, f"epoch-{e}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                docs[e] = json.load(f)
            continue
        doc = get(a.pool, f"/api/points/epoch/{e}")
        if not isinstance(doc, dict):
            raise SystemExit(f"epoch {e}: the pool sent a document that is not an object")
        docs[e] = doc
        write_json(path, doc)
        seen[str(e)] = int(time.time())     # first time we hold it: a document that shows up long after its epoch is held by the audit
    write_json(seen_path, seen)
    # 2. then the shares to examine
    rnd = secrets.SystemRandom()
    for e in epochs:
        count = docs[e].get("shares_count", 0)
        if not is_int(count, 0, MAX_SHARES_PER_EPOCH):
            raise SystemExit(f"epoch {e}: implausible number of shares in the document")
        if a.full:
            path = os.path.join(a.out, f"shares-{e}.json")
            if not os.path.exists(path):
                recs = []
                while len(recs) < count:
                    page = get(a.pool, f"/api/points/shares/{e}?from={len(recs)}&limit=20000", timeout=300)
                    chunk = page.get("records") if isinstance(page, dict) else None
                    if not isinstance(chunk, list) or not chunk or len(recs) + len(chunk) > count or page.get("from") != len(recs) \
                            or not all(isinstance(x, str) and len(x) <= 400 and all(c in "0123456789abcdef" for c in x) for x in chunk):
                        raise SystemExit(f"epoch {e}: the pool sent a malformed share list")
                    recs += chunk
                write_json(path, {"records": recs})
        path = os.path.join(a.out, f"proofs-{e}.json")
        if count and a.samples and not os.path.exists(path) and not (a.full and os.path.exists(os.path.join(a.out, f"shares-{e}.json"))):
            proofs = []
            for i in sorted(rnd.sample(range(count), min(a.samples, count))):
                proofs.append(clean_proof(get(a.pool, f"/api/points/share/{e}/{i}", limit=1 << 16), e, count))
            write_json(path, {"proofs": proofs})
    print(f"{len(epochs)} epoch(s) saved in {a.out}")


if __name__ == "__main__":
    sys.exit(main())
