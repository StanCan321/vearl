#!/usr/bin/env python3
"""Client of verified MiniLM EMBEDDINGS (all-MiniLM-L6-v2 converted by tools/quantize_minilm.py, retrained by tools/qat_minilm.py).

One text per job (<= 64 WordPiece tokens); the client computes the input embedding (table lookup + LayerNorm), the worker executes the
195 operations (61 mined 1024^3 GEMMs, 12 attention heads, affine LayerNorms, GELU, mean), the client verifies EVERY operation before
paying. The program depends on the length of the text (len of the softmax, row of ones of the mean): each sample carries its own and all
the texts go out in ONE series (jobs in flight in parallel, common weight blocks sent once).

    c = MiniLMEmbedClient("/tmp/minilm-qat.json", node, worker)
    E = c.embed(["the market fell", "the team won"])        # (2, 384) int8, to be normalised for the cosine
"""
import base64, json, os, subprocess, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantize_minilm as QM
import vearl_ref as R


class MiniLMEmbedClient:
    def __init__(self, model_path, node, worker, key="seed:4", escrow_per_gemm=1, cli=None, inflight=16):
        self.m = json.load(open(model_path))
        from transformers import AutoTokenizer, AutoModel
        self.tk = AutoTokenizer.from_pretrained(self.m["tokenizer"])
        emb = AutoModel.from_pretrained(self.m["tokenizer"]).state_dict()
        self.sd = {k: v.detach().numpy().astype(np.float64) for k, v in emb.items() if k.startswith("embeddings.")}
        self.node, self.worker, self.key, self.esc, self.inflight = node, worker, key, escrow_per_gemm, inflight
        self.cli = cli or os.path.join(os.environ.get("B", "target/release"), "vearl-cli")
        self.n = self.m["n"]
        R.N = self.n
        self.logs = []

    def ids(self, text):
        return QM.ids_of(self.tk, text)

    def ops_for(self, lr):
        return json.loads(json.dumps(self.m["ops_template"]).replace('"LEN"', str(lr)).replace('"SHP"', str(QM.shp_for(lr))))

    def ones_for(self, lr):
        ones = np.zeros((self.n, self.n), np.int64)
        ones[0, :lr] = 1
        return R.enc(ones)

    def model_for(self, lr, samples):
        ws = list(self.m["weights"])
        ws[self.m["ones_index"]] = self.ones_for(lr)
        return {"name": f"minilm-l6-int8-len{lr}", "n": self.n, "ops": self.ops_for(lr), "weights": ws, "classes": QM.D, "batch": 1,
                "input_scale": self.m["plan"]["s0"], "samples": samples}

    def embed(self, texts, series=64):
        """ONE series per packet of `series` texts: each sample carries its program (length) and its mean vector; the
        49 other weight blocks are shared (the worker's registry only receives the new mean vectors, 1 MiB each)."""
        if len(texts) > series:
            return np.concatenate([self.embed(texts[k:k + series], series) for k in range(0, len(texts), series)])
        ids = [self.ids(t) for t in texts]
        lmax = max(len(t) for t in ids)
        samples = []
        for t in ids:
            s = {"x0": R.enc(QM.enc_input(self.sd, t, self.m["plan"]["s0"])), "labels": [0]}
            if len(t) != lmax:
                s["ops"] = self.ops_for(len(t))
                s["weights"] = {str(self.m["ones_index"]): self.ones_for(len(t))}
            samples.append(s)
        with tempfile.TemporaryDirectory() as d:
            mp, op = f"{d}/m.json", f"{d}/o.i8"
            json.dump(self.model_for(lmax, samples), open(mp, "w"))
            r = subprocess.run([self.cli, "infer", self.node, self.worker, self.key, str(self.esc), mp, "0", str(len(texts) - 1), op, str(self.inflight)],
                               capture_output=True, text=True, timeout=7200)
            log = r.stdout + r.stderr
            self.logs.append(log)
            if r.returncode != 0 or not os.path.exists(op) or "0 refused" not in log:
                raise RuntimeError("inference refused or fraud detected:\n" + log[-1500:])
            return np.fromfile(op, dtype=np.int8).reshape(len(texts), QM.D)

    def expected(self, text):
        """Expected embedding, computed by the INDEPENDENT reference engine (without a network): to compare bit for bit."""
        ids = self.ids(text)
        mf = self.model_for(len(ids), [])
        W = [R.dec(w, self.n).astype(np.int64) for w in mf["weights"]]
        T, _ = R.run_program(QM.enc_input(self.sd, ids, self.m["plan"]["s0"]), W, mf["ops"])
        return T[-1][:QM.D, 0].astype(np.int8)
