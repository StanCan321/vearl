#!/usr/bin/env python3
"""Client of verified EMBEDDINGS for a BERT converted by tools/quantize_bert.py (BGE-base by default, 110 M parameters, 12 layers).

A text (<= 64 tokens) = CHAINED jobs (4 layers each); each job is verified operation by operation before payment, its verified
output (residual stream D x L) is the input of the next one; the last returns the state of the [CLS] token (the embedding). Each sample carries its own
program (length of the text: Softmax / SoftmaxLo); all the texts of a step go out in ONE series.

    c = BertEmbedClient("/tmp/bge-qat.json", node, worker)
    E = c.embed(["the market fell", "the team won"])      # (2, 768) int8, to be normalised for the cosine
"""
import json, os, subprocess, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantize_bert as QB
import vearl_ref as R


class BertEmbedClient:
    def __init__(self, model_path, node, worker, key="seed:4", escrow_per_gemm=1, cli=None, inflight=16):
        self.m = json.load(open(model_path))
        self.plan, self.n, self.D, self.L = self.m["plan"], self.m["n"], self.m["D"], self.m["L"]
        from transformers import AutoTokenizer, AutoModel
        self.tk = AutoTokenizer.from_pretrained(self.m["tokenizer"])
        mdl = AutoModel.from_pretrained(self.m["tokenizer"])
        self.cfg = QB.Cfg(mdl.config)
        self.sd = {k: v.detach().numpy().astype(np.float64) for k, v in mdl.state_dict().items() if k.startswith("embeddings.")}
        self.node, self.worker, self.key, self.esc, self.inflight = node, worker, key, escrow_per_gemm, inflight
        self.cli = cli or os.path.join(os.environ.get("B", "target/release"), "vearl-cli")
        R.N = self.n
        self.logs = []

    def ops_for(self, job, lr):
        return json.loads(json.dumps(self.m["jobs"][job]["ops_template"]).replace('"LEN"', str(lr)))

    def _series(self, job, x0s, lens, classes, batch):
        lmax = max(lens)
        samples = []
        for x, lr in zip(x0s, lens):
            s = {"x0": R.enc(x), "labels": [0]}
            if lr != lmax:
                s["ops"] = self.ops_for(job, lr)
            samples.append(s)
        model = {"name": f"{self.m['name']}-job{job}", "n": self.n, "ops": self.ops_for(job, lmax), "weights": self.m["jobs"][job]["weights"],
                 "classes": classes, "batch": batch, "samples": samples}
        with tempfile.TemporaryDirectory() as d:
            mp, op = f"{d}/m.json", f"{d}/o.i8"
            json.dump(model, open(mp, "w"))
            r = subprocess.run([self.cli, "infer", self.node, self.worker, self.key, str(self.esc), mp, "0", str(len(x0s) - 1), op, str(self.inflight)],
                               capture_output=True, text=True, timeout=7200)
            log = r.stdout + r.stderr
            self.logs.append(log)
            if r.returncode != 0 or not os.path.exists(op) or "0 refused" not in log:
                raise RuntimeError(f"job {job}: inference refused or fraud detected:\n" + log[-1500:])
            return np.fromfile(op, dtype=np.int8).reshape(len(x0s), batch, classes)      # (texts, columns, rows)

    def embed(self, texts):
        ids = [QB.ids_of(self.tk, t) for t in texts]
        lens = [len(t) for t in ids]
        xs = [QB.enc_input(self.sd, self.cfg, t, self.plan["s0"]) for t in ids]
        nj = len(self.m["jobs"])
        for j in range(nj):
            last = j == nj - 1
            out = self._series(j, xs, lens, self.D, 1 if last else self.L)
            if last:
                return out[:, 0, :].astype(np.int8)                                       # column 0 = [CLS]
            xs = []
            for k in range(len(texts)):
                t = np.zeros((self.n, self.n), np.int64)
                t[:self.D, :self.L] = out[k].T.astype(np.int64)
                xs.append(t)

    def expected(self, text):
        """Expected embedding, by the INDEPENDENT reference engine (without a network): to compare bit for bit."""
        ids = QB.ids_of(self.tk, text)
        x = QB.enc_input(self.sd, self.cfg, ids, self.plan["s0"])
        for j, J in enumerate(self.m["jobs"]):
            W = [R.dec(w, self.n).astype(np.int64) for w in J["weights"]]
            T, _ = R.run_program(x, W, self.ops_for(j, len(ids)))
            x = np.zeros((self.n, self.n), np.int64)
            x[:self.D, :self.L] = T[-1][:self.D, :self.L]
        return x[:self.D, 0].astype(np.int8)
