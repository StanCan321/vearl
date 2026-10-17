#!/usr/bin/env python3
"""Client of verified EMBEDDINGS for Qwen3-Embedding-0.6B converted by tools/quantize_qwen3.py.

A text (<= 127 tokens + the prefix token) = 10 CHAINED jobs (3 layers each); each job is verified operation by operation before payment (`vearl-cli
infer`), its verified output (residual stream D x L) is the input of the next one; the last one returns the integer state of the last token, from which the
client computes the embedding (RMS of the state, inverse rotation, final gain, L2 normalisation: off-chain, like the token table and the rotation of the
input). All the texts of a step go out in ONE series (each sample carries the program of its own length).

    c = Qwen3EmbedClient("/tmp/qwen3-int8.json", node, worker)
    E = c.embed(["the market fell", "the team won"])       # (2, 1024) float, unit norm (embedding of the PREFIXED text, see docs/12 section 4 nonies)
    c.expected("the market fell")                           # the same, by the independent reference engine, without a network (bit for bit)

    python3 tools/qwen3_client.py MODEL.json[.xz] NODE WORKER KEY "text 1" "text 2" ...      # prints the embeddings; the cost is escrowed per Gemm

The workers must hold ~530 weight blocks of 1 MiB in their registry (raised cap: 1024).
"""
import json, os, subprocess, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantize_qwen3 as Q
import vearl_ref as R


class _RotatedTable:
    """Rows of the word table in the rotated basis of the programs (x @ rot), computed on demand."""

    def __init__(self, table, rot):
        self.table, self.rot = table, rot

    def __getitem__(self, ids):
        return self.table[ids].astype(np.float64) @ self.rot


class Qwen3EmbedClient:
    def __init__(self, model_path, node, worker, key="seed:4", escrow_per_gemm=1, cli=None, inflight=8):
        opener = __import__("lzma").open if str(model_path).endswith(".xz") else open
        with opener(model_path, "rt", encoding="utf-8") as f:
            self.m = json.load(f)
        self.plan, self.n, self.D, self.L = self.m["plan"], self.m["n"], self.m["D"], self.m["L"]
        Q.set_n(self.n)
        Q.PREFIX_ID = self.plan["prefix_id"]
        name = self.m.get("tokenizer", self.m["hf"])
        from transformers import AutoTokenizer, AutoModel
        self.tk = AutoTokenizer.from_pretrained(name)
        mdl = AutoModel.from_pretrained(name, dtype="float32").eval()
        self.cfg = Q.Cfg.from_hf(mdl.config)
        rot = Q.hadamard(self.cfg.D, self.m.get("rot_seed", 0))
        table = mdl.embed_tokens.weight.detach().numpy().copy()
        self.P = {"embed": _RotatedTable(table, rot), "final": mdl.norm.weight.detach().numpy().astype(np.float64), "rot": rot}
        del mdl
        self.node, self.worker, self.key, self.esc, self.inflight = node, worker, key, escrow_per_gemm, inflight
        self.cli = cli or os.path.join(os.environ.get("B", "target/release"), "vearl-cli")
        self.logs = []

    def ops_for(self, job, lr):
        return json.loads(json.dumps(self.m["jobs"][job]["ops_template"]).replace('"LEN"', str(lr)))

    def _series(self, job, x0s, lens, last):
        lmax = max(lens)
        samples = []
        for x, lr in zip(x0s, lens):
            s = {"x0": R.enc(x), "labels": [0]}
            if lr != lmax:
                s["ops"] = self.ops_for(job, lr)
            samples.append(s)
        model = {"name": f"{self.m['name']}-job{job}", "n": self.n, "ops": self.ops_for(job, lmax), "weights": self.m["jobs"][job]["weights"],
                 "classes": self.D, "batch": self.L, "samples": samples}
        with tempfile.TemporaryDirectory() as d:
            mp, op = f"{d}/m.json", f"{d}/o.i8"
            json.dump(model, open(mp, "w"))
            r = subprocess.run([self.cli, "infer", self.node, self.worker, self.key, str(self.esc), mp, "0", str(len(x0s) - 1), op, str(self.inflight)],
                               capture_output=True, text=True, timeout=14400)
            log = r.stdout + r.stderr
            self.logs.append(log)
            if r.returncode != 0 or not os.path.exists(op) or "0 refused" not in log:
                raise RuntimeError(f"job {job}: inference refused or fraud detected:\n" + log[-1500:])
            return np.fromfile(op, dtype=np.int8).reshape(len(x0s), self.L, self.D)          # (texts, columns = tokens, rows = features)

    def embed_int(self, texts):
        """(integer state of the last token (D,) per text, float embeddings)."""
        ids = [Q.ids_of(self.tk, t, self.cfg) for t in texts]
        lens = [len(t) for t in ids]
        xs = [Q.enc_input(self.P, self.cfg, t, self.plan["s0"]) for t in ids]
        nj = len(self.m["jobs"])
        for j in range(nj):
            out = self._series(j, xs, lens, j == nj - 1)
            if j == nj - 1:
                last = np.stack([out[k, lens[k] - 1].astype(np.int64) for k in range(len(texts))])
                return last, np.stack([Q.finish(self.P, self.cfg, v * self.plan["s_out"]) for v in last])
            xs = []
            for k in range(len(texts)):
                t = np.zeros((self.n, self.n), np.int64)
                t[:self.D, :self.L] = out[k].T.astype(np.int64)
                xs.append(t)

    def embed(self, texts):
        return self.embed_int(texts)[1]

    def expected(self, text):
        """(integer last-token state, float embedding) by the INDEPENDENT reference engine, without a network."""
        ids = Q.ids_of(self.tk, text, self.cfg)
        x = Q.enc_input(self.P, self.cfg, ids, self.plan["s0"])
        for j, J in enumerate(self.m["jobs"]):
            W = [R.dec(w, self.n).astype(np.int64) for w in J["weights"]]
            T, _ = R.run_program(x, W, self.ops_for(j, len(ids)))
            x = np.zeros((self.n, self.n), np.int64)
            x[:self.D, :self.L] = T[-1][:self.D, :self.L]
        last = x[:self.D, len(ids) - 1].astype(np.int64)
        return last, Q.finish(self.P, self.cfg, last * self.plan["s_out"])


if __name__ == "__main__":
    if len(sys.argv) < 6:
        sys.exit(__doc__)
    model, node, worker, key, *texts = sys.argv[1:]
    c = Qwen3EmbedClient(model, node, worker, key)
    E = c.embed(texts)
    for t, e in zip(texts, E):
        print(f"{t[:60]!r}: first values {np.round(e[:6], 4).tolist()} (norm {np.linalg.norm(e):.3f}, 1024 dimensions)")
    if len(texts) > 1:
        print("cosine similarities:")
        print(np.round(E @ E.T, 3))
    print(c.logs[-1].strip().splitlines()[-1])
