#!/usr/bin/env python3
"""TEXT feature extraction -> VEARL input tensor (one column per text, 1024 rows = 1024 words of the vocabulary).

The same code serves training (train_embed.py) and the client: two different implementations of the tokenisation would give different
inputs and therefore different embeddings. Feature of a word j in a text: min(1, (1 + ln tf) * idf_j / scale),
then integer round(64 x) in [0, 64] (same convention as quantize.py). No float enters the CONSENSUS: this step is
done by the client, off-chain; what the network verifies starts at the integer T_0.
"""
import json, re
import numpy as np

TOKEN = re.compile(r"[a-z][a-z]+")


def tokens(text):
    return TOKEN.findall(text.lower())


class Featurizer:
    def __init__(self, vocab, idf, scale):
        self.vocab, self.idf, self.scale = list(vocab), np.asarray(idf, dtype=np.float64), float(scale)
        self.index = {w: i for i, w in enumerate(self.vocab)}
        self.bigrams = any(" " in w for w in self.vocab)

    @staticmethod
    def fit(texts, size=1024, min_df=5, n_bi=0):
        """size words in total; n_bi of them are BIGRAMS (two consecutive words of the vocabulary, separated by a space)."""
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        df = {}
        for t in texts:
            for w in set(tokens(t)):
                df[w] = df.get(w, 0) + 1
        cand = [(c, w) for w, c in df.items() if c >= min_df and w not in ENGLISH_STOP_WORDS]
        cand.sort(key=lambda cw: (-cw[0], cw[1]))
        vocab = [w for _, w in cand[:size - n_bi]]
        if n_bi:
            uni = set(vocab)
            bdf = {}
            for t in texts:
                seq = [w for w in tokens(t) if w in uni]
                for b in set(f"{a} {c}" for a, c in zip(seq, seq[1:])):
                    bdf[b] = bdf.get(b, 0) + 1
            bc = sorted(((c, b) for b, c in bdf.items() if c >= min_df), key=lambda cb: (-cb[0], cb[1]))[:n_bi]
            vocab += [b for _, b in bc]
            df.update({b: c for c, b in bc})
        N = len(texts)
        idf = [np.log((1 + N) / (1 + df[w])) + 1 for w in vocab]
        f = Featurizer(vocab, idf, 1.0)
        raw = f.transform(texts[:20000], scale=False)
        f.scale = float(np.percentile(raw[raw > 0], 99.5))
        return f

    def transform(self, texts, scale=True):
        """(len(texts), len(vocab)) float64 in [0, 1] (after division by `scale`)."""
        out = np.zeros((len(texts), len(self.vocab)))
        for r, t in enumerate(texts):
            tf = {}
            seq = []
            for w in tokens(t):
                j = self.index.get(w)
                if j is not None:
                    tf[j] = tf.get(j, 0) + 1
                    seq.append(w)
            if self.bigrams:
                for a, c in zip(seq, seq[1:]):
                    j = self.index.get(f"{a} {c}")
                    if j is not None:
                        tf[j] = tf.get(j, 0) + 1
            for j, c in tf.items():
                out[r, j] = (1 + np.log(c)) * self.idf[j]
        return np.minimum(out / self.scale, 1.0) if scale else out

    def save(self, path):
        json.dump({"vocab": self.vocab, "idf": list(map(float, self.idf)), "scale": self.scale}, open(path, "w"))

    @staticmethod
    def load(path):
        d = json.load(open(path))
        return Featurizer(d["vocab"], d["idf"], d["scale"])
