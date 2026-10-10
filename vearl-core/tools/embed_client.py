#!/usr/bin/env python3
"""TEXT EMBEDDING CLIENT for VEARL: you give texts, you receive their embeddings (int8, 384 dimensions), computed by a
GPU worker and VERIFIED operation by operation before payment (`vearl-cli infer`). The model weights are sent only once to the
worker's registry.

    from embed_client import EmbedClient
    ec = EmbedClient("/tmp/embed-agnews.json", node="127.0.0.1:9831", worker="127.0.0.1:9831", key="seed:4")
    E = ec.embed(["the stock market fell", "the team won the final"])         # (2, 384) int8
    sims = ec.similarity(E[:1], E)                                              # normalised dot products (numpy)

    python3 tools/embed_client.py MODEL.json NODE WORKER KEY "text 1" "text 2" ...

Model: the output of tools/train_embed.py (MODEL.json + MODEL.featurizer.json). The tokenisation and the vocabulary (1,024 words) are those
of the training (tools/embed_text.py): it is the CLIENT that applies them, off-chain; what the network executes and verifies starts at
the integer T_0 = round(64 * features).
"""
import json, os, subprocess, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import vearl_ref as R
from embed_text import Featurizer

N, DIM = 1024, 384


class EmbedClient:
    def __init__(self, model_path, node, worker, key="seed:4", escrow_per_gemm=1, cli="target/release/vearl-cli", inflight=8):
        self.model = json.load(open(model_path))
        self.model["samples"] = []
        self.feat = Featurizer.load(model_path.replace(".json", ".featurizer.json"))
        self.batch = self.model["batch"]                       # texts per job (1024, or 256 for a vocabulary of 4096: wide input)
        self.nin = -(-len(self.feat.vocab) // N)
        self.node, self.worker, self.key, self.esc, self.cli, self.inflight = node, worker, key, escrow_per_gemm, cli, inflight
        R.N = N

    def embed(self, texts):
        """int8 (len(texts), 384). Each batch of 1,024 texts is a job; the jobs are verified and paid one by one."""
        x = self.feat.transform(list(texts))
        samples = []
        B = self.batch
        for s in range(0, len(x), B):
            xb = x[s:s + B]
            t = np.zeros((N, N), np.int64)
            for b in range(self.nin):                          # feature block b: columns [b*B, (b+1)*B) (Shift op)
                seg = xb[:, b * N:(b + 1) * N]
                t[:seg.shape[1], b * B:b * B + len(xb)] = np.round(seg.T * 64).astype(np.int64)
            samples.append({"x0": R.enc(t), "labels": [0] * len(xb)})
        m = dict(self.model, samples=samples)
        with tempfile.TemporaryDirectory() as d:
            mp, out = f"{d}/m.json", f"{d}/out.i8"
            json.dump(m, open(mp, "w"))
            r = subprocess.run([self.cli, "infer", self.node, self.worker, self.key, str(self.esc), mp, "0", str(len(samples) - 1), out, str(self.inflight)], capture_output=True, text=True)
            log = r.stdout + r.stderr
            if r.returncode != 0 or not os.path.exists(out) or "0 refused" not in log:
                raise RuntimeError("inference refused or fraud detected:\n" + log[-1500:])
            self.last_log = log
            E = np.fromfile(out, dtype=np.int8).reshape(-1, DIM)
        # each batch is complete (1,024 columns); we only keep the requested texts
        keep = np.concatenate([np.arange(s * B, s * B + min(B, len(x) - s * B)) for s in range(len(samples))])
        return E[keep]

    @staticmethod
    def similarity(Q, D):
        Q, D = Q.astype(np.float64), D.astype(np.float64)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
        D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
        return Q @ D.T


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print(__doc__)
        sys.exit(2)
    ec = EmbedClient(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    texts = sys.argv[5:]
    E = ec.embed(texts)
    print(f"{len(E)} embeddings of {E.shape[1]} dimensions, verified by the network:")
    S = ec.similarity(E, E)
    for i, t in enumerate(texts):
        print(f"  [{i}] {t[:60]!r}: nearest other text == {int(np.argsort(-S[i])[1]) if len(texts) > 1 else '-'}")


class TransformerEmbedClient:
    """Embeddings by the verified SMALL TRANSFORMER (tools/train_tiny_transformer.py + quantize_transformer.py): one text (32 tokens) per job, 128-dimension int8 embedding.
    The lookup of the token embedding table (E + positions) is done by the client, off-chain; the network executes and
    verifies all the rest (projection, attention, softmax, LayerNorm, residuals, FFN, summation).
        tc = TransformerEmbedClient("models/tiny-transformer-int8.json", "models/tiny-transformer-float.npz", node, worker)
        E = tc.embed(["the market fell", "the team won"])            # (2, 128) int8"""

    def __init__(self, model_path, weights_npz, node, worker, key="seed:4", escrow_per_gemm=1, cli="target/release/vearl-cli", inflight=16):
        import quantize_transformer as QT
        self.QT, self.model = QT, json.load(open(model_path))
        self.model["samples"] = []
        self.P = dict(np.load(weights_npz))
        QT.configure(self.P)
        vocab = json.load(open(weights_npz.replace(".npz", ".vocab.json")))["vocab"]
        self.index = {w: i + 2 for i, w in enumerate(vocab)}
        self.s0 = self.model["input_scale"]
        self.node, self.worker, self.key, self.esc, self.cli, self.inflight = node, worker, key, escrow_per_gemm, cli, inflight
        R.N = self.model["n"]

    def embed(self, texts):
        from embed_text import tokens
        samples = []
        for t in texts:
            ids = [self.index.get(w, 1) for w in tokens(t)][:self.QT.L]
            tok = np.zeros(self.QT.L, dtype=np.int64)
            tok[:len(ids)] = ids
            samples.append({"x0": R.enc(self.QT.enc_input(self.P, tok, self.s0)), "labels": [0]})
        with tempfile.TemporaryDirectory() as d:
            mp, out = f"{d}/m.json", f"{d}/out.i8"
            json.dump(dict(self.model, samples=samples), open(mp, "w"))
            r = subprocess.run([self.cli, "infer", self.node, self.worker, self.key, str(self.esc), mp, "0", str(len(samples) - 1), out, str(self.inflight)], capture_output=True, text=True)
            log = r.stdout + r.stderr
            if r.returncode != 0 or not os.path.exists(out) or "0 refused" not in log:
                raise RuntimeError("inference refused or fraud detected:\n" + log[-1500:])
            self.last_log = log
            return np.fromfile(out, dtype=np.int8).reshape(-1, self.model["classes"])
