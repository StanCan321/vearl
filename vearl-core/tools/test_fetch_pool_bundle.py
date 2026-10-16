"""Tests of fetch_pool_bundle.py and audit_pools.py against a pool that is NOT honest.   python3 -m unittest tools/test_fetch_pool_bundle.py   (from vearl-core/)

The pool is a small local HTTP server whose answers are chosen by each test. Nothing it sends may become a file outside the output folder, a number
that is not an integer, or an unbounded download."""
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fetch_pool_bundle as fb  # noqa: E402
import pool_points as pp  # noqa: E402


class Pool:
    """Serves answers[path] as JSON (or as raw bytes)."""

    def __init__(self, answers):
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                a = answers.get(self.path)
                if isinstance(a, tuple) and a[0] == "redirect":
                    self.send_response(302)
                    self.send_header("Location", a[1])
                    self.end_headers()
                    return
                if a is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = a if isinstance(a, bytes) else json.dumps(a).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d" % self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def run_fetch(url, out, *extra):
    return subprocess.run([sys.executable, os.path.join(HERE, "fetch_pool_bundle.py"), "--pool", url, "--out", out, "--samples", "2", *extra],
                          capture_output=True, text=True, timeout=120)


def files_under(path):
    return sorted(os.path.relpath(os.path.join(d, f), path) for d, _, fs in os.walk(path) for f in fs)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


class Hostile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        self.out = os.path.join(self.tmp, "bundle")

    def serve(self, answers):
        p = Pool(answers)
        self.addCleanup(p.close)
        return p

    def test_epoch_numbers_that_are_not_plain_integers_never_reach_the_disk(self):
        for bad in ("../../escape", "..\\..\\escape", 1.5, True, None, -1, 2 ** 70, "7", [7], {"a": 1}):
            p = self.serve({"/api/points/index": {"epochs": [{"epoch": bad}]}})
            r = run_fetch(p.url, self.out)
            self.assertNotEqual(r.returncode, 0, bad)
            self.assertFalse(any("escape" in f for f in files_under(self.tmp)), bad)
            self.assertFalse(os.path.exists(os.path.join(self.tmp, "escape.json")))

    def test_a_malformed_index_or_a_repeated_epoch_is_refused(self):
        for index in ([], {"epochs": "x"}, {"epochs": [1, 2]}, {"epochs": [{"epoch": 3}, {"epoch": 3}]}):
            p = self.serve({"/api/points/index": index})
            self.assertNotEqual(run_fetch(p.url, self.out).returncode, 0, index)

    def test_an_oversized_answer_is_refused(self):
        p = self.serve({"/x": b'{"a": "' + b"z" * 5000 + b'"}'})
        with self.assertRaises(SystemExit):
            fb.get(p.url, "/x", 5, 1000)
        self.assertEqual(fb.get(p.url, "/x", 5, 100000)["a"][:3], "zzz")

    def test_a_document_that_is_not_an_object_or_has_an_absurd_share_count_is_refused(self):
        p = self.serve({"/api/points/index": {"epochs": [{"epoch": 5}]}, "/api/points/epoch/5": [1, 2, 3]})
        self.assertNotEqual(run_fetch(p.url, self.out).returncode, 0)
        p = self.serve({"/api/points/index": {"epochs": [{"epoch": 5}]}, "/api/points/epoch/5": {"shares_count": 10 ** 12}})
        self.assertNotEqual(run_fetch(p.url, os.path.join(self.tmp, "b2")).returncode, 0)

    def test_malformed_share_proofs_and_lists_are_refused(self):
        base = {"/api/points/index": {"epochs": [{"epoch": 5}]}, "/api/points/epoch/5": {"shares_count": 4}}
        good = {"epoch": 5, "count": 4, "record": "ab" * 57, "path": ["cd" * 32, "ef" * 32]}
        for name, change in (("index out of range", {"index": 9}), ("record not hex", {"record": "zz"}), ("path entry not hash", {"path": ["no"]}),
                             ("path too long", {"path": ["cd" * 32] * 50}), ("index a string", {"index": "1"})):
            answers = dict(base)
            for i in range(4):
                answers["/api/points/share/5/%d" % i] = dict(good, **{"index": i, **change})
            p = self.serve(answers)
            out = os.path.join(self.tmp, "b-" + name.replace(" ", "-"))
            self.assertNotEqual(run_fetch(p.url, out).returncode, 0, name)
            self.assertFalse(os.path.exists(os.path.join(out, "proofs-5.json")), name)
        p = self.serve(dict(base, **{"/api/points/shares/5": {"records": ["not hex"]}}))
        self.assertNotEqual(run_fetch(p.url, os.path.join(self.tmp, "full"), "--full").returncode, 0)

    def test_a_redirect_to_another_address_is_not_followed(self):
        # a hostile pool points the document at this machine's own (private) API: the answer must never be read, let alone saved
        secret = self.serve({"/api/status": {"internal": "do not publish"}})
        p = self.serve({"/api/points/index": {"epochs": [{"epoch": 5}]}, "/api/points/epoch/5": ("redirect", secret.url + "/api/status")})
        r = run_fetch(p.url, self.out)
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(os.path.exists(os.path.join(self.out, "epoch-5.json")))
        for f in os.listdir(self.out):
            with open(os.path.join(self.out, f), encoding="utf-8") as fh:
                self.assertNotIn("internal", fh.read())

    def test_the_share_list_is_read_page_by_page_and_a_page_out_of_order_is_refused(self):
        recs = ["ab" * 57] * 5
        base = {"/api/points/index": {"epochs": [{"epoch": 5}]}, "/api/points/epoch/5": {"shares_count": 5}}
        ok = self.serve(dict(base, **{"/api/points/shares/5?from=0&limit=20000": {"count": 5, "from": 0, "records": recs}}))
        r = run_fetch(ok.url, os.path.join(self.tmp, "ok"), "--full")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(read_json(os.path.join(self.tmp, "ok", "shares-5.json"))["records"]), 5)
        for name, page in (("wrong offset", {"count": 5, "from": 3, "records": recs}), ("too many", {"count": 5, "from": 0, "records": recs * 2}), ("empty", {"count": 5, "from": 0, "records": []})):
            bad = self.serve(dict(base, **{"/api/points/shares/5?from=0&limit=20000": page}))
            out = os.path.join(self.tmp, "bad-" + name.replace(" ", "-"))
            self.assertNotEqual(run_fetch(bad.url, out, "--full").returncode, 0, name)
            self.assertFalse(os.path.exists(os.path.join(out, "shares-5.json")), name)

    def test_a_correct_pool_is_saved_and_the_time_of_first_sight_is_kept(self):
        answers = {"/api/points/index": {"epochs": [{"epoch": 5}, {"epoch": 6}]}}
        for e in (5, 6):
            answers["/api/points/epoch/%d" % e] = {"shares_count": 2, "epoch": e}
            for i in range(2):
                answers["/api/points/share/%d/%d" % (e, i)] = {"epoch": e, "index": i, "count": 2, "record": "ab" * 57, "path": ["cd" * 32]}
        p = self.serve(answers)
        r = run_fetch(p.url, self.out)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(sorted(os.listdir(self.out)), ["epoch-5.json", "epoch-6.json", "index.json", "proofs-5.json", "proofs-6.json", "seen.json"])
        seen = read_json(os.path.join(self.out, "seen.json"))
        self.assertEqual(sorted(seen), ["5", "6"])
        r = run_fetch(p.url, self.out)       # again: nothing is downloaded again and the first-seen times stay
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(read_json(os.path.join(self.out, "seen.json")), seen)


class AuditFailure(unittest.TestCase):
    def test_a_failing_audit_program_leaves_a_report_that_holds_every_hour(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        p = Pool({"/api/points/index": {"epochs": [{"epoch": 5}]}, "/api/points/epoch/5": {"shares_count": 0, "epoch": 5}})
        self.addCleanup(p.close)
        crash = os.path.join(tmp, "crash.py")
        with open(crash, "w") as f:
            f.write("import sys\nsys.exit(101)\n")
        if os.name == "nt":
            wrapper = os.path.join(tmp, "audit.bat")
            text = '@echo off\r\n"%s" "%s"\r\nexit /b %%errorlevel%%\r\n' % (sys.executable, crash)
        else:
            wrapper = os.path.join(tmp, "audit.sh")
            text = '#!/bin/sh\nexec "%s" "%s"\n' % (sys.executable, crash)
        with open(wrapper, "w", newline="") as f:
            f.write(text)
        os.chmod(wrapper, 0o755)
        spec = os.path.join(tmp, "pools.json")
        write_json(spec, {"pools": [{"address": "vrl1xxx", "name": "p", "url": p.url, "dir": "d", "audit": "a.json"}]})
        # an old report that said everything was fine is in place: it must be replaced
        write_json(os.path.join(tmp, "a.json"), {"verdict": "ok", "epochs": {"5": "ok"}})
        r = subprocess.run([sys.executable, os.path.join(HERE, "audit_pools.py"), "--spec", spec, "--genesis", "g", "--blocks", "b", "--audit-bin", wrapper],
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        rep = read_json(os.path.join(tmp, "a.json"))
        self.assertEqual(rep["verdict"], "suspect")
        self.assertEqual(rep["epochs"], {})
        self.assertEqual(rep["findings"][0]["code"], "audit_failed")
        self.assertEqual(pp.load_audit(os.path.join(tmp, "a.json")), ({}, "suspect"))   # no epoch is listed as ok: every hour of the pool is held


if __name__ == "__main__":
    unittest.main()
