#!/usr/bin/env python3
"""VERIFIED TEXT GENERATION client for a small GPT converted by tools/quantize_gpt.py / qat_gpt.py (TinyStories-33M).

One generation step = two series of jobs, each VERIFIED operation by operation before payment:
  BODY: one job per text in progress (<= 64 tokens, causal attention) -> final residual stream (D x L);
  HEAD: ONE job for up to 20 texts -> integer logits of the 50,257 vocabulary words for the last token of each text.
The client chooses the next word (greedy decoding: maximal integer logit, in case of a tie the smallest identifier), appends it and starts again.
The input embedding (integer tables of the model file) and this choice are the only local computations.

    c = GptClient("/tmp/gpt-qat.json", node, worker)
    texts = c.generate(["Once upon a time, there was a little cat"], max_new=40)
"""
import base64, json, os, subprocess, sys, tempfile, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vearl_ref as R


def _dec(s, shape):
    return np.frombuffer(base64.b64decode(s), dtype=np.int8).reshape(shape).astype(np.int64)


class GptClient:
    def __init__(self, model_path, node, worker, key="seed:4", escrow_per_gemm=1, cli=None, inflight=16):
        self.m = json.load(open(model_path))
        self.plan, self.n, self.D, self.L, self.V = self.m["plan"], self.m["n"], self.m["D"], self.m["L"], self.m["V"]
        self.E, self.Pp = _dec(self.m["embed"], self.m["embed_shape"]), _dec(self.m["pos"], self.m["pos_shape"])
        from transformers import AutoTokenizer
        self.tk = AutoTokenizer.from_pretrained(self.m["tokenizer"])
        self.eos = self.tk.eos_token_id
        self.node, self.worker, self.key, self.esc, self.inflight = node, worker, key, escrow_per_gemm, inflight
        self.cli = cli or os.path.join(os.environ.get("B", "target/release"), "vearl-cli")
        R.N = self.n
        self.logs, self.times = [], []

    # ---- programs and inputs
    def body_ops(self, lr):
        return json.loads(json.dumps(self.m["body"]["ops_template"]).replace('"LEN"', str(lr)))

    def body_input(self, ids):
        t = np.zeros((self.n, self.n), np.int64)
        t[:self.D, :len(ids)] = np.clip(self.E[ids] + self.Pp[:len(ids)], -64, 64).T
        return t

    def head_input(self, states):
        t = np.zeros((self.n, self.n), np.int64)
        for s, v in enumerate(states):
            t[:self.D, self.plan["CB"] + s] = v
            t[self.D:, self.plan["CB"] + s] = self.plan["MARK"]                # witness rows: vocabulary biases in the product
        return t

    def logits_of(self, out_cols, s):
        """out_cols: (columns, rows) of the HEAD output; integer logits (V,) of text s."""
        S = self.plan["S"]
        return np.concatenate([out_cols[b * S + s] for b in range(self.plan["NB"])])[:self.V].astype(np.int64)

    # ---- execution on the chain
    def _infer(self, name, ops, weights, samples, classes, batch):
        model = {"name": name, "n": self.n, "ops": ops, "weights": weights, "classes": classes, "batch": batch, "samples": samples}
        with tempfile.TemporaryDirectory() as d:
            mp, op = f"{d}/m.json", f"{d}/o.i8"
            json.dump(model, open(mp, "w"))
            r = subprocess.run([self.cli, "infer", self.node, self.worker, self.key, str(self.esc), mp, "0", str(len(samples) - 1), op, str(self.inflight)],
                               capture_output=True, text=True, timeout=7200)
            log = r.stdout + r.stderr
            self.logs.append(log)
            if r.returncode != 0 or not os.path.exists(op) or "0 refused" not in log:
                raise RuntimeError(f"{name}: inference refused or fraud detected:\n" + log[-1500:])
            return np.fromfile(op, dtype=np.int8).reshape(len(samples), batch, classes)

    def step(self, seqs):
        """One step for texts (lists of identifiers, <= 20): VERIFIED integer logits of the next token, (len(seqs), V)."""
        lmax = max(len(s) for s in seqs)
        samples = []
        for s in seqs:
            smp = {"x0": R.enc(self.body_input(s)), "labels": [0]}
            if len(s) != lmax:
                smp["ops"] = self.body_ops(len(s))
            samples.append(smp)
        t0 = time.time()
        out = self._infer(f"{self.m['name']}-body", self.body_ops(lmax), self.m["body"]["weights"], samples, self.D, self.L)
        t1 = time.time()
        states = [out[k, len(s) - 1, :].astype(np.int64) for k, s in enumerate(seqs)]
        S, NB = self.plan["S"], self.plan["NB"]
        head = self._infer(f"{self.m['name']}-head", self.m["head"]["ops"], self.m["head"]["weights"],
                           [{"x0": R.enc(self.head_input(states)), "labels": [0]}], self.n, NB * S)
        self.times.append((t1 - t0, time.time() - t1, len(seqs)))
        return np.stack([self.logits_of(head[0], k) for k in range(len(seqs))])

    def generate(self, prompts, max_new=40):
        """Verified greedy decoding. Returns (complete texts, identifiers, integer logits of the last step of each text)."""
        seqs = [self.tk(p)["input_ids"][:self.L - 1] for p in prompts]
        done = [False] * len(seqs)
        last = [None] * len(seqs)
        for _ in range(max_new):
            act = [k for k in range(len(seqs)) if not done[k]]
            if not act:
                break
            for g in range(0, len(act), self.plan["S"]):
                grp = act[g:g + self.plan["S"]]
                lg = self.step([seqs[k] for k in grp])
                for k, row in zip(grp, lg):
                    last[k] = (list(seqs[k]), row)
                    nxt = int(np.argmax(row))                                   # in case of a tie: the smallest identifier
                    seqs[k] = seqs[k] + [nxt]
                    if nxt == self.eos or len(seqs[k]) >= self.L:
                        done[k] = True
        texts = [self.tk.decode([t for t in s if t != self.eos]) for s in seqs]
        return texts, seqs, last

    # ---- independent reference engine (without a network)
    def expected_logits(self, ids):
        """Expected integer logits of the next token, by vearl_ref (real programs, same placement as on the chain)."""
        Wb = [R.dec(w, self.n).astype(np.int64) for w in self.m["body"]["weights"]]
        T, _ = R.run_program(self.body_input(ids), Wb, self.body_ops(len(ids)))
        st = T[-1][:self.D, len(ids) - 1].astype(np.int64)
        Wh = [R.dec(w, self.n).astype(np.int64) for w in self.m["head"]["weights"]]
        Th, _ = R.run_program(self.head_input([st]), Wh, self.m["head"]["ops"])
        return self.logits_of(Th[-1].T, 0)
