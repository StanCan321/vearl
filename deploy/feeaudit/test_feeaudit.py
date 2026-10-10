import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feeaudit as fa  # noqa: E402

NOW = 1_800_000_000
OLD = NOW - 10 * 3600


def beacon(**kw):
    b = {"worker_id": "abcd1234", "fee_pct": 10, "share_pct": 80, "age_s": 60, "node_hash": "n" * 64, "quantus_hash": "q" * 64}
    b.update(kw)
    return b


def workers(user_rate, fee_rate):
    u = [{"name": "vrl-abcd1234", "hashrate_6h": user_rate}] if user_rate is not None else []
    f = [{"name": "vrl-fee-abcd1234", "hashrate_6h": fee_rate}] if fee_rate is not None else []
    return u, f


class Classify(unittest.TestCase):
    def test_ok_when_the_fee_worker_carries_about_ten_percent(self):
        u, f = workers(90.0, 10.0)
        self.assertEqual(fa.classify(beacon(), u, f, OLD, NOW)[0], "ok")
        u, f = workers(94.0, 6.0)  # 6 % of the work: above half of the announced fee
        self.assertEqual(fa.classify(beacon(), u, f, OLD, NOW)[0], "ok")

    def test_no_fee_worker_is_a_violation_after_the_grace_period_only(self):
        u, f = workers(100.0, None)
        self.assertEqual(fa.classify(beacon(), u, f, OLD, NOW)[0], "no_fee")
        self.assertEqual(fa.classify(beacon(), u, f, NOW - 600, NOW)[0], "warming_up")

    def test_low_and_no_fee_thresholds(self):
        u, f = workers(97.0, 3.0)   # 3 %: between 20 % and 50 % of the announced fee
        self.assertEqual(fa.classify(beacon(), u, f, OLD, NOW)[0], "low")
        u, f = workers(99.5, 0.5)   # 0.5 %: practically no fee
        self.assertEqual(fa.classify(beacon(), u, f, OLD, NOW)[0], "no_fee")

    def test_a_beacon_that_declares_another_fee_is_flagged(self):
        u, f = workers(90.0, 10.0)
        self.assertEqual(fa.classify(beacon(fee_pct=0), u, f, OLD, NOW)[0], "wrong_fee_pct")

    def test_unofficial_builds_are_flagged_when_hashes_are_known(self):
        u, f = workers(90.0, 10.0)
        known = {"node": ["n" * 64], "quantus": ["q" * 64]}
        self.assertEqual(fa.classify(beacon(), u, f, OLD, NOW, known)[0], "ok")
        self.assertEqual(fa.classify(beacon(node_hash="x" * 64), u, f, OLD, NOW, known)[0], "unofficial_build")
        self.assertEqual(fa.classify(beacon(quantus_hash="x" * 64), u, f, OLD, NOW, known)[0], "unofficial_build")

    def test_other_states(self):
        u, f = workers(None, 10.0)
        self.assertEqual(fa.classify(beacon(), u, f, OLD, NOW)[0], "not_mining_quantus")
        self.assertEqual(fa.classify(beacon(share_pct=0), u, f, OLD, NOW)[0], "not_mining_quantus")
        self.assertEqual(fa.classify(beacon(age_s=3 * 3600), u, f, OLD, NOW)[0], "offline")
        self.assertEqual(fa.classify(beacon(), None, f, OLD, NOW)[0], "unverifiable")
        self.assertEqual(fa.classify(beacon(), u, None, OLD, NOW)[0], "unverifiable")

    def test_rate_falls_back_to_other_windows(self):
        self.assertEqual(fa.worker_rate({"hashrate_6h": 0, "hashrate_24h": 5.0}), 5.0)
        self.assertEqual(fa.worker_rate({}), 0.0)


if __name__ == "__main__":
    unittest.main()
