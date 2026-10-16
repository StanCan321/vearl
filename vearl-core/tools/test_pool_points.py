"""Tests of pool_points.py and of the pool side of testnet_rewards.py.   python3 -m unittest tools/test_pool_points.py   (from vearl-core/)

The folder tests/vectors/pool-points holds documents signed by the Rust pool itself (tests/pool_points.rs writes them): reading and verifying them
here proves that the Python tools and the pool agree on the canonical form that is signed and hashed."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool_points as pp  # noqa: E402
import testnet_rewards as tr  # noqa: E402

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402
from cryptography.hazmat.primitives import serialization  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "vectors", "pool-points")
START = tr.parse_time("2026-10-23T16:00:00Z")
DAY = 86400
P = dict(tr.DEFAULTS)


def raw_public(sk):
    return sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def miner_key(n):
    return pp.address_of(bytes([n]) * 32)


def make_doc(sk, epoch, miners, prev="00" * 32, secs=3600, blocks=()):
    """A document like the pool's, signed here. miners: {address: (weight, v2_weight)}."""
    address = pp.address_of(raw_public(sk))
    ms = [{"address": a, "shares": 5, "v2_shares": 5 if v2 else 0, "v2_weight": str(v2), "weight": str(w)} for a, (w, v2) in sorted(miners.items())]
    doc = {"proto": "vpool-points/1", "pool": address, "pool_key": raw_public(sk).hex(), "genesis": "11" * 32, "epoch": epoch, "epoch_secs": secs,
           "start": epoch * secs, "end": (epoch + 1) * secs, "prev": prev, "fee_bps": 100, "blocks": list(blocks), "jobs": [], "miners": ms,
           "total_weight": str(sum(w for w, _ in miners.values())), "total_shares": 5 * len(ms), "shares_count": 5 * len(ms), "shares_root": "22" * 32}
    doc["sig"] = sk.sign(pp.DOC_DOMAIN + pp.canonical(doc).encode()).hex()
    return doc


def chain_of_docs(sk, specs, secs=3600):
    """specs: list of (epoch, miners). Documents chained by `prev`, as the pool does."""
    docs, prev = {}, "00" * 32
    for e, miners in specs:
        d = make_doc(sk, e, miners, prev, secs)
        docs[e] = d
        prev = pp.doc_hash(d)
    return docs


def pool_entry(sk, docs, audit=None, verdict=None):
    return {"address": pp.address_of(raw_public(sk)), "name": "test pool", "docs": docs, "problems": [], "audit": audit or {}, "verdict": verdict}


class InteropWithTheRustPool(unittest.TestCase):
    def setUp(self):
        if not os.path.isdir(FIXTURE):
            self.skipTest("no fixture")
        with open(os.path.join(FIXTURE, "pool.txt")) as f:
            self.pool = f.read().strip()

    def test_documents_signed_by_rust_verify_and_chain_in_python(self):
        docs, problems = pp.load_docs(FIXTURE, self.pool, (4,))
        self.assertEqual(problems, [])
        self.assertGreaterEqual(len(docs), 3)
        es = sorted(docs)
        for a, b in zip(es, es[1:]):
            self.assertEqual(docs[b]["prev"], pp.doc_hash(docs[a]))

    def test_a_tampered_rust_document_is_refused(self):
        docs, _ = pp.load_docs(FIXTURE, self.pool, (4,))
        e = sorted(docs)[1]
        d = json.loads(json.dumps(docs[e]))
        d["miners"][0]["v2_weight"] = str(int(d["miners"][0]["v2_weight"]) + 1)
        self.assertEqual(pp.verify_doc(d, self.pool, (4,)), "bad signature")
        self.assertEqual(pp.verify_doc(docs[e], pp.address_of(b"\x07" * 32), (4,)), "document of another pool")

    def test_allocation_of_the_pool_blocks_follows_the_work(self):
        docs, _ = pp.load_docs(FIXTURE, self.pool, (4,))
        with open(os.path.join(FIXTURE, "blocks.json")) as f:
            blocks = json.load(f)["blocks"]
        start = min(d["start"] for d in docs.values()) - 10
        r = tr.score(blocks, start, P, pools={self.pool: pool_entry_from(docs, self.pool)}, unaudited="count")
        rows = {m["address"]: m for m in r["miners"]}
        self.assertNotIn(self.pool, rows, "the pool's own address earns nothing")
        w = {}
        for d in docs.values():
            for m in d["miners"]:
                w[m["address"]] = w.get(m["address"], 0) + int(m["v2_weight"])
        total_w = sum(w.values())
        pot = sum(1.25 for b in blocks if b["miner"] == self.pool and b["timestamp"] >= start)  # first days of the program: x1.25
        self.assertAlmostEqual(sum(m["points"] for m in rows.values()), pot, places=2)
        for a, x in w.items():
            self.assertAlmostEqual(rows[a]["points"], pot * x / total_w, delta=0.01 * pot)
        self.assertEqual(r["pools"][0]["blocks"], sum(1 for b in blocks if b["miner"] == self.pool))


def pool_entry_from(docs, address):
    return {"address": address, "name": "rust pool", "docs": docs, "problems": [], "audit": {}, "verdict": None}


class Addresses(unittest.TestCase):
    def test_roundtrip(self):
        for n in (0, 7, 255):
            pk = bytes([n]) * 32
            self.assertEqual(pp.pool_key(pp.address_of(pk)), pk)
        self.assertTrue(pp.address_of(b"\x01" * 32).startswith("vrl1"))


class Verify(unittest.TestCase):
    def setUp(self):
        self.sk = Ed25519PrivateKey.generate()
        self.addr = pp.address_of(raw_public(self.sk))

    def test_a_signed_document_passes_and_every_edit_fails(self):
        d = make_doc(self.sk, 5, {miner_key(1): (100, 100)})
        self.assertIsNone(pp.verify_doc(d, self.addr))
        for field, value in (("epoch", 6), ("fee_bps", 5), ("prev", "11" * 32), ("shares_root", "33" * 32)):
            e = dict(d)
            e[field] = value
            self.assertIsNotNone(pp.verify_doc(e, self.addr), field)

    def test_arithmetic_is_checked_even_when_signed(self):
        d = make_doc(self.sk, 5, {miner_key(1): (100, 100)})
        d["total_weight"] = "999"
        d["sig"] = self.sk.sign(pp.DOC_DOMAIN + pp.canonical({k: v for k, v in d.items() if k != "sig"}).encode()).hex()
        self.assertEqual(pp.verify_doc(d, self.addr), "totals do not add up")

    def test_a_broken_chain_is_reported(self):
        docs = chain_of_docs(self.sk, [(1, {miner_key(1): (10, 10)}), (2, {miner_key(1): (10, 10)}), (3, {miner_key(1): (10, 10)})])
        os_dir = self.tmp()
        for e, d in docs.items():
            if e == 2:
                d = make_doc(self.sk, 2, {miner_key(1): (99, 99)}, docs[1]["prev"])  # rewritten after epoch 3 committed to the old one
            with open(os.path.join(os_dir, f"epoch-{e}.json"), "w") as f:
                json.dump(d, f)
        loaded, problems = pp.load_docs(os_dir, self.addr)
        self.assertEqual(len(loaded), 3)
        self.assertTrue(any("epoch 3" in p for p in problems), problems)

    def tmp(self):
        import tempfile
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        return d


class Allocation(unittest.TestCase):
    def setUp(self):
        self.sk = Ed25519PrivateKey.generate()
        self.addr = pp.address_of(raw_public(self.sk))
        self.a, self.b, self.c = miner_key(1), miner_key(2), miner_key(3)

    def day(self, d, hour=0):
        return (START + d * DAY + hour * 3600) // 3600

    def blocks(self, n, day=1, miner=None):
        return [{"height": i, "timestamp": START + day * DAY + 100 + i, "miner": miner or self.addr, "work": "native"} for i in range(n)]

    def test_points_follow_the_verified_work_of_the_day(self):
        docs = chain_of_docs(self.sk, [(self.day(5), {self.a: (300, 300), self.b: (100, 100)}), (self.day(5, 1), {self.a: (100, 100), self.b: (100, 100)})])
        r = tr.score(self.blocks(8, day=5), START, P, pools={self.addr: pool_entry(self.sk, docs, {e: "ok" for e in docs})})
        rows = {m["address"]: m for m in r["miners"]}
        self.assertEqual(sorted(rows), sorted([self.a, self.b]))
        self.assertAlmostEqual(rows[self.a]["points"], 8 * 400 / 600, places=3)   # no early bonus on day 5
        self.assertAlmostEqual(rows[self.b]["points"], 8 * 200 / 600, places=3)
        self.assertEqual(rows[self.a]["blocks"], 0)
        self.assertEqual(rows[self.a]["pools"], [self.addr])
        self.assertEqual(rows[self.a]["active_days"], 1)
        self.assertNotIn(self.addr, rows)

    def test_early_period_multiplier_and_useful_blocks_apply_to_the_pot(self):
        docs = chain_of_docs(self.sk, [(self.day(0), {self.a: (50, 50)})])
        blocks = self.blocks(2, day=0) + [{"height": 9, "timestamp": START + 500, "miner": self.addr, "work": "job"}]
        r = tr.score(blocks, START, P, pools={self.addr: pool_entry(self.sk, docs, {e: "ok" for e in docs})})
        self.assertAlmostEqual(r["miners"][0]["points"], 1.25 * 2 + 1.5 * 1.25, places=3)

    def test_miners_without_vpool2_receipts_earn_nothing(self):
        docs = chain_of_docs(self.sk, [(self.day(5), {self.a: (100, 100), self.b: (900, 0)})])
        r = tr.score(self.blocks(10, day=5), START, P, pools={self.addr: pool_entry(self.sk, docs, {e: "ok" for e in docs})})
        rows = {m["address"]: m for m in r["miners"]}
        self.assertEqual(list(rows), [self.a])
        self.assertAlmostEqual(rows[self.a]["points"], 10.0, places=3)

    def test_fraud_voids_the_hour_and_suspect_holds_it(self):
        docs = chain_of_docs(self.sk, [(self.day(5, 0), {self.a: (100, 100)}), (self.day(5, 1), {self.b: (100, 100)}), (self.day(5, 2), {self.c: (100, 100)})])
        e0, e1, e2 = sorted(docs)
        audit = {e0: "ok", e1: "fraud", e2: "suspect"}
        r = tr.score(self.blocks(6, day=5), START, P, pools={self.addr: pool_entry(self.sk, docs, audit, "fraud")})
        rows = {m["address"]: m for m in r["miners"]}
        self.assertEqual(list(rows), [self.a], "the fraudulent hour and the held hour earn nothing")
        self.assertAlmostEqual(rows[self.a]["points"], 6.0, places=3)
        rep = r["pools"][0]
        self.assertEqual((rep["epochs_counted"], rep["epochs_void"], rep["epochs_held"]), (1, 1, 1))
        self.assertEqual(rep["audit_verdict"], "fraud")

    def test_without_an_audit_the_hours_are_held_unless_asked(self):
        docs = chain_of_docs(self.sk, [(self.day(5), {self.a: (100, 100)})])
        pools = {self.addr: pool_entry(self.sk, docs)}
        self.assertEqual(tr.score(self.blocks(4, day=5), START, P, pools=pools)["miners"], [])
        held = tr.score(self.blocks(4, day=5), START, P, pools=pools)["pools"][0]
        self.assertEqual(held["epochs_held"], 1)
        self.assertAlmostEqual(tr.score(self.blocks(4, day=5), START, P, pools=pools, unaudited="count")["miners"][0]["points"], 4.0, places=3)

    def test_points_with_no_work_behind_them_are_not_invented(self):
        docs = chain_of_docs(self.sk, [(self.day(6), {self.a: (100, 100)})])
        r = tr.score(self.blocks(3, day=5), START, P, pools={self.addr: pool_entry(self.sk, docs, {e: "ok" for e in docs})})
        self.assertEqual(r["miners"], [])
        self.assertAlmostEqual(r["pools"][0]["points_unattributed"], 3.0, places=3)

    def test_solo_blocks_and_pool_points_add_up_for_the_same_address(self):
        docs = chain_of_docs(self.sk, [(self.day(5), {self.a: (100, 100)})])
        blocks = self.blocks(4, day=5) + self.blocks(2, day=5, miner=self.a)
        r = tr.score(blocks, START, P, pools={self.addr: pool_entry(self.sk, docs, {e: "ok" for e in docs})})
        row = [m for m in r["miners"] if m["address"] == self.a][0]
        self.assertAlmostEqual(row["points"], 6.0, places=3)
        self.assertAlmostEqual(row["pool_points"], 4.0, places=3)
        self.assertEqual(row["blocks"], 2)

    def test_registration_is_still_required_for_pool_miners(self):
        docs = chain_of_docs(self.sk, [(self.day(5), {self.a: (100, 100), self.b: (100, 100)})])
        r = tr.score(self.blocks(40, day=5), START, P, registered={self.a}, pools={self.addr: pool_entry(self.sk, docs, {e: "ok" for e in docs})})
        rows = {m["address"]: m for m in r["miners"]}
        self.assertIn("address not registered", rows[self.b]["reasons"])
        self.assertNotIn("address not registered", rows[self.a]["reasons"])


class HostileDocuments(unittest.TestCase):
    """A registered pool signs its documents itself: whatever it signs must not let it create points or move work between days."""

    def setUp(self):
        self.sk = Ed25519PrivateKey.generate()
        self.addr = pp.address_of(raw_public(self.sk))

    def resign(self, d):
        d = {k: v for k, v in d.items() if k != "sig"}
        d["sig"] = self.sk.sign(pp.DOC_DOMAIN + pp.canonical(d).encode()).hex()
        return d

    def test_negative_or_odd_weights_are_refused_even_when_signed(self):
        for field, value in (("weight", "-5"), ("v2_weight", "-5"), ("weight", "1e3"), ("v2_weight", " 7"), ("weight", 7), ("shares", -1), ("shares", True), ("shares", 1.5)):
            d = make_doc(self.sk, 5, {miner_key(1): (100, 100), miner_key(2): (100, 100)})
            d["miners"][1][field] = value
            self.assertIsNotNone(pp.verify_doc(self.resign(d), self.addr), (field, value))

    def test_epoch_length_is_fixed_by_the_program_not_by_the_pool(self):
        for secs in (1, 60, 86400, 7200, 3601, 0, -3600, True, "3600", 3600.0):
            d = make_doc(self.sk, 5, {miner_key(1): (1, 1)})
            d["epoch_secs"] = secs
            self.assertIsNotNone(pp.verify_doc(self.resign(d), self.addr), secs)
        self.assertIsNone(pp.verify_doc(make_doc(self.sk, 5, {miner_key(1): (1, 1)}), self.addr))

    def test_a_pool_cannot_create_points_only_share_the_pot(self):
        # one real miner, one made-up miner with a huge weight: the points handed out never exceed what the pool's blocks earned
        docs = chain_of_docs(self.sk, [(5 * 24, {miner_key(1): (1, 1), miner_key(2): (10 ** 40, 10 ** 40)})])
        entry = pool_entry(self.sk, docs, {e: "ok" for e in docs})
        blocks = [{"height": i, "timestamp": START + 5 * DAY + 100, "miner": self.addr, "work": "native"} for i in range(7)]
        r = tr.score(blocks, START, P, pools={self.addr: entry})
        self.assertLessEqual(sum(m["points"] for m in r["miners"]), 7.0 + 1e-9)

    def test_a_document_cannot_be_two_things_at_once(self):
        d = make_doc(self.sk, 5, {miner_key(1): (1, 1)})
        self.assertIsNotNone(pp.verify_doc(self.resign(dict(d, proto="vpool-points/2")), self.addr))
        self.assertIsNotNone(pp.verify_doc(self.resign({k: v for k, v in d.items() if k != "miners"}), self.addr))
        self.assertIsNotNone(pp.verify_doc(self.resign(dict(d, miners=[d["miners"][0], d["miners"][0]])), self.addr))


class Fingerprint(unittest.TestCase):
    def test_hash_changes_with_documents_and_audit(self):
        sk = Ed25519PrivateKey.generate()
        docs = chain_of_docs(sk, [(1, {miner_key(1): (1, 1)})])
        a = pool_entry(sk, docs, {1: "ok"})
        b = pool_entry(sk, docs, {1: "fraud"})
        self.assertNotEqual(pp.pools_hash({a["address"]: a}), pp.pools_hash({b["address"]: b}))
        self.assertEqual(pp.pools_hash({a["address"]: a}), pp.pools_hash({a["address"]: a}))


if __name__ == "__main__":
    unittest.main()
