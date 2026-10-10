"""Tests of testnet_rewards.py and verify_registration.py.   python3 -m unittest tools/test_testnet_rewards.py   (from vearl-core/)"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import testnet_rewards as tr  # noqa: E402
import verify_registration as vr  # noqa: E402

START = tr.parse_time("2026-10-23T16:00:00Z")
DAY = 86400
P = dict(tr.DEFAULTS)


def block(h, ts, miner, work="native"):
    return {"height": h, "timestamp": ts, "miner": miner, "work": work}


def week_of_blocks(miner, first_day, days, per_day=2, work="native"):
    out = []
    for d in range(first_day, first_day + days):
        for k in range(per_day):
            out.append(block(0, START + d * DAY + k * 60, miner, work))
    return out


class ScoreTests(unittest.TestCase):
    def test_points_early_multiplier_and_useful_blocks(self):
        blocks = [block(1, START + 10, "a"), block(2, START + 20, "a", "job"), block(3, START + 4 * DAY + 5, "a")]
        r = tr.score(blocks, START, P)
        a = r["miners"][0]
        # first three days: native 1 * 1.25, useful 1.5 * 1.25; day 4 (no longer early): 1
        self.assertAlmostEqual(a["points"], 1.25 + 1.875 + 1.0)
        self.assertEqual((a["blocks"], a["useful_blocks"]), (3, 1))

    def test_blocks_before_the_start_are_ignored(self):
        r = tr.score([block(1, START - 5, "a"), block(2, START + 5, "b")], START, P)
        self.assertEqual([m["address"] for m in r["miners"]], ["b"])

    def test_eligibility_needs_active_days_and_a_minimum_share(self):
        blocks = week_of_blocks("steady", 0, 8) + week_of_blocks("burst", 0, 1, per_day=50)
        r = {m["address"]: m for m in tr.score(blocks, START, P)["miners"]}
        self.assertTrue(r["steady"]["eligible"])
        self.assertFalse(r["burst"]["eligible"])
        self.assertTrue(any("active days" in x for x in r["burst"]["reasons"]))
        self.assertEqual(P["min_active_days"], 3)
        dust = week_of_blocks("whale", 0, 8, per_day=40000) + week_of_blocks("dust", 0, 8, per_day=1)
        r = {m["address"]: m for m in tr.score(dust, START, P)["miners"]}
        self.assertFalse(r["dust"]["eligible"])
        self.assertTrue(any("share below" in x for x in r["dust"]["reasons"]))

    def test_pool_is_shared_between_eligible_addresses_only(self):
        blocks = week_of_blocks("a", 0, 8, per_day=3) + week_of_blocks("b", 0, 8, per_day=1) + week_of_blocks("short", 0, 2, per_day=5)
        res = tr.score(blocks, START, P)
        eligible = [m for m in res["miners"] if m["eligible"]]
        self.assertEqual({m["address"] for m in eligible}, {"a", "b"})
        self.assertAlmostEqual(sum(m["pool_share"] for m in res["miners"]), 1.0)
        a, b = (next(m for m in eligible if m["address"] == x) for x in "ab")
        self.assertAlmostEqual(a["pool_share"] / b["pool_share"], 3.0, places=6)

    def test_registration_and_exclusion_lists(self):
        blocks = week_of_blocks("a", 0, 8) + week_of_blocks("b", 0, 8) + week_of_blocks("c", 0, 8)
        res = tr.score(blocks, START, P, registered={"a", "b"}, excluded={"b"})
        by = {m["address"]: m for m in res["miners"]}
        self.assertTrue(by["a"]["eligible"])
        self.assertIn("excluded by the team (see docs/19)", by["b"]["reasons"])
        self.assertIn("address not registered", by["c"]["reasons"])

    def test_splitting_power_between_addresses_does_not_pay_more(self):
        one = tr.score(week_of_blocks("x", 0, 8, per_day=4), START, P)["miners"]
        split = tr.score(week_of_blocks("y1", 0, 8, per_day=2) + week_of_blocks("y2", 0, 8, per_day=2), START, P)["miners"]
        # the same total power gives the same total points, however many addresses it is spread over
        self.assertAlmostEqual(sum(m["points"] for m in split), one[0]["points"])
        self.assertTrue(all(m["eligible"] for m in split))

    def test_result_is_deterministic_and_the_data_hash_changes_with_the_data(self):
        blocks = week_of_blocks("a", 0, 8) + week_of_blocks("b", 0, 8)
        self.assertEqual(tr.score(blocks, START, P), tr.score(list(reversed(blocks)), START, P))
        self.assertNotEqual(tr.data_hash(blocks), tr.data_hash(blocks[:-1]))

    def test_command_line_reproduces_from_a_dump(self):
        blocks = [dict(b, height=i) for i, b in enumerate(week_of_blocks("a", 0, 8) + week_of_blocks("b", 0, 9))]
        with tempfile.TemporaryDirectory() as d:
            dump = os.path.join(d, "blocks.json")
            json.dump({"blocks": blocks}, open(dump, "w"))
            outs = []
            for name in ("one", "two"):
                sys.argv = ["x", "--blocks-file", dump, "--start", "2026-10-23T16:00:00Z", "--out", os.path.join(d, name), "--pool-tokens", "1000"]
                tr.main()
                outs.append(json.load(open(os.path.join(d, name, "leaderboard.json"))))
            for o in outs:
                o.pop("generated_at")
            self.assertEqual(outs[0], outs[1])
            self.assertAlmostEqual(sum(m.get("tokens", 0) for m in outs[0]["miners"]), 1000.0, places=4)


class FetchTests(unittest.TestCase):
    def test_paging_over_more_than_one_page_of_blocks(self):
        import threading
        import urllib.parse
        from http.server import BaseHTTPRequestHandler, HTTPServer
        N = 250
        chain = [{"height": h, "timestamp": START + h * 30, "miner": "m" + str(h % 3), "work": {"type": "native"}} for h in range(N)]

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                limit = min(int(q.get("limit", ["20"])[0]), 100)
                start = min(int(q.get("from", [str(N - 1)])[0]), N - 1)
                blocks = [chain[h] for h in range(start, max(start - limit, -1), -1)]
                body = json.dumps({"tip": N, "blocks": blocks}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            got = tr.fetch_blocks(f"127.0.0.1:{srv.server_port}")
        finally:
            srv.shutdown()
        self.assertEqual([b["height"] for b in got], list(range(N)))
        self.assertEqual(got[7]["miner"], "m1")


# a registration made by the real wallet (vearl-wallet sign) with a throw-away key
ADDR = "vrl158v7z57nwtvf05znhf98lnjv0xdwtwr5aq7ss7txnjwuy3lz4d4qh9y83e"
MSG = f"Vearl testnet rewards registration. Address: {ADDR}. Contact: tester#0001. I accept the terms in docs/19."
SIG = "c38c882801f9b1e4e30a28f22c584bef2a3f975b5fbf5ebadfeae89fe201286e46444b3397c20c741eb06f36cde4f8f3b7558fef63aa57d078933b06bbab1f0f"


class RegistrationTests(unittest.TestCase):
    def test_signature_made_by_the_rust_wallet_verifies(self):
        self.assertIsNone(vr.verify_signature(ADDR, MSG, SIG))

    def test_wrong_message_wrong_signature_and_bad_address_are_refused(self):
        self.assertIsNotNone(vr.verify_signature(ADDR, MSG + " ", SIG))
        self.assertIsNotNone(vr.verify_signature(ADDR, MSG, SIG[:-2] + "00"))
        self.assertIsNotNone(vr.verify_signature(ADDR[:-1] + "q", MSG, SIG))   # checksum
        self.assertIsNotNone(vr.verify_signature(ADDR, MSG, "zz"))

    def test_registration_file(self):
        items = [
            {"address": ADDR, "contact": "tester#0001", "message": MSG, "signature": SIG},
            {"address": ADDR, "contact": "tester#0001", "message": MSG, "signature": SIG},                       # sent twice: one entry
            {"address": ADDR, "contact": "x", "message": "hello", "signature": SIG},                              # wrong form
            {"address": "vrl1other", "contact": "x", "message": MSG, "signature": SIG},                           # names another address
        ]
        with tempfile.TemporaryDirectory() as d:
            src, out = os.path.join(d, "r.json"), os.path.join(d, "o.json")
            json.dump(items, open(src, "w"))
            sys.argv = ["x", src, "--out", out]
            vr.main()
            res = json.load(open(out))
        self.assertEqual(res["addresses"], [ADDR])
        self.assertEqual(len(res["rejected"]), 2)


if __name__ == "__main__":
    unittest.main()
