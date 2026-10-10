#!/usr/bin/env python3
"""VERIFIED IMAGE CLASSIFICATION client (DeiT-Tiny, 224 x 224, 1000 ImageNet categories; tools/quantize_vit.py + tools/qat_vit.py).

One image = two chained jobs: job A (embedding + layers 0-5) then job B (layers 6-11 + classifier), each one verified operation
by operation before payment. The client prepares the input (cutting into 196 squares of 16 x 16, position marker), retrieves the useful region
of the VERIFIED residual stream of job A (192 x 197) and gives it to job B. Both series go out entirely (jobs in flight in parallel).

    c = VitClient("/tmp/vit-qat.json", node, worker)
    top = c.classify(["cat.jpg", "truck.jpg"])      # [(ImageNet index, name, integer logits)]
"""
import json, os, subprocess, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantize_vit as QV
import vearl_ref as R


class VitClient:
    def __init__(self, model_path, node, worker, key="seed:4", escrow_per_gemm=1, cli=None, inflight=16):
        self.m = json.load(open(model_path))
        self.plan = self.m["plan"]
        self.node, self.worker, self.key, self.esc, self.inflight = node, worker, key, escrow_per_gemm, inflight
        self.cli = cli or os.path.join(os.environ.get("B", "target/release"), "vearl-cli")
        self.n = self.m["n"]
        R.N = self.n
        self.logs = []
        from transformers import ViTForImageClassification
        self.labels = ViTForImageClassification.from_pretrained(QV.NAME).config.id2label

    def _series(self, job, x0s, classes, batch, tag):
        """One `vearl-cli infer` series of job `job` on the inputs x0s; returns the outputs (classes x batch) of each sample."""
        J = self.m["jobs"][job]
        model = {"name": f"deit-tiny-int8-{tag}", "n": self.n, "ops": J["ops"], "weights": J["weights"], "classes": classes, "batch": batch,
                 "samples": [{"x0": R.enc(x), "labels": [0]} for x in x0s]}
        with tempfile.TemporaryDirectory() as d:
            mp, op = f"{d}/m.json", f"{d}/o.i8"
            json.dump(model, open(mp, "w"))
            r = subprocess.run([self.cli, "infer", self.node, self.worker, self.key, str(self.esc), mp, "0", str(len(x0s) - 1), op, str(self.inflight)],
                               capture_output=True, text=True, timeout=7200)
            log = r.stdout + r.stderr
            self.logs.append(log)
            if r.returncode != 0 or not os.path.exists(op) or "0 refused" not in log:
                raise RuntimeError(f"job {tag}: inference refused or fraud detected:\n" + log[-1500:])
            # infer writes, per sample, out[c][b] for b < batch then c < classes: order (batch, classes)
            return np.fromfile(op, dtype=np.int8).reshape(len(x0s), batch, classes)

    def x0(self, path):
        from PIL import Image  # noqa: F401  (QV.pixels opens the image)
        return QV.enc_input(QV.pixels(QV.PREP, path), self.plan)

    def classify(self, paths):
        xa = [self.x0(p) for p in paths]
        mid = self._series(0, xa, QV.D, QV.T, "A")                      # (images, 197 tokens, 192 features)
        xb = []
        for k in range(len(paths)):
            t = np.zeros((self.n, self.n), np.int64)
            t[:QV.D, :QV.T] = mid[k].T.astype(np.int64)
            xb.append(t)
        lg = self._series(1, xb, 1000, 1, "B")[:, 0, :].astype(np.int64)   # (images, 1000)
        return [(int(l.argmax()), self.labels[int(l.argmax())], l) for l in lg]

    def expected(self, path):
        """Expected logits, computed by the INDEPENDENT reference engine (without a network): to compare bit for bit."""
        W = [[R.dec(w, self.n).astype(np.int64) for w in J["weights"]] for J in self.m["jobs"]]
        TA, _ = R.run_program(self.x0(path), W[0], self.m["jobs"][0]["ops"])
        mid = np.zeros((self.n, self.n), np.int64)
        mid[:QV.D, :QV.T] = TA[-1][:QV.D, :QV.T]
        TB, _ = R.run_program(mid, W[1], self.m["jobs"][1]["ops"])
        return TB[-1][:1000, 0].astype(np.int64)
