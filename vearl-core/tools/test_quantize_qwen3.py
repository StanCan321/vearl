"""Tests of quantize_qwen3.py on a tiny random Qwen3-like model with the tensor size n = 128 (same code paths as the 0.6 B model at
n = 1024: grouped-query attention with two query heads per key/value head, query tensor cut in two passes, three feed-forward tiles,
sink token, RoPE tables, block Hadamard).   python3 -m unittest tools/test_quantize_qwen3.py   (from vearl-core/)

  * the floating-point reimplementation agrees with the transformers Qwen3 model (skipped if transformers is not installed);
  * the weight transformations (gain folding, rotation, smoothing, block Hadamard) leave the embedding unchanged (1e-9);
  * the programs respect the protocol bounds and pass the independent validator (prog_ref.py);
  * the integer chain approximates the float embedding of the same text;
  * the Rust engine (`vearl-model dump`) computes exactly the tensors of the Python reference engine, job by job (skipped if the
    binary is not built)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import quantize_qwen3 as Q  # noqa: E402
    import vearl_ref as R  # noqa: E402
    import prog_ref  # noqa: E402
except ImportError as e:  # the CI installs only a few packages: these tests need numpy, blake3 and cryptography
    raise unittest.SkipTest(f"missing dependency: {e}")

N_TEST = 128
VOCAB = 300
PREFIX = 7
HERE = os.path.dirname(os.path.abspath(__file__))


def tiny_sd(cfg, seed=0):
    rng = np.random.default_rng(seed)
    g = lambda *s, sc=1.0: rng.standard_normal(s) * sc
    sd = {"embed_tokens.weight": g(VOCAB, cfg.D, sc=0.7), "norm.weight": 1 + 0.2 * g(cfg.D)}
    for l in range(cfg.NL):
        p = f"layers.{l}."
        sd[p + "input_layernorm.weight"] = 1 + 0.3 * g(cfg.D)
        sd[p + "post_attention_layernorm.weight"] = 1 + 0.3 * g(cfg.D)
        sd[p + "self_attn.q_proj.weight"] = g(cfg.NH * cfg.HD, cfg.D, sc=cfg.D ** -0.5)
        sd[p + "self_attn.k_proj.weight"] = g(cfg.NKV * cfg.HD, cfg.D, sc=cfg.D ** -0.5)
        sd[p + "self_attn.v_proj.weight"] = g(cfg.NKV * cfg.HD, cfg.D, sc=cfg.D ** -0.5)
        sd[p + "self_attn.o_proj.weight"] = g(cfg.D, cfg.NH * cfg.HD, sc=(cfg.NH * cfg.HD) ** -0.5)
        sd[p + "self_attn.q_norm.weight"] = 1 + 0.3 * g(cfg.HD)
        sd[p + "self_attn.k_norm.weight"] = 1 + 0.3 * g(cfg.HD)
        sd[p + "mlp.gate_proj.weight"] = g(cfg.FF, cfg.D, sc=cfg.D ** -0.5)
        sd[p + "mlp.up_proj.weight"] = g(cfg.FF, cfg.D, sc=cfg.D ** -0.5)
        sd[p + "mlp.down_proj.weight"] = g(cfg.D, cfg.FF, sc=cfg.FF ** -0.5)
    return sd


def rand_ids(rng, lo=5, hi=14):
    return [PREFIX] + [int(v) for v in rng.integers(0, VOCAB, rng.integers(lo, hi))]


def find_model_binary():
    for rel in ("../target/release/vearl-model.exe", "../target/release/vearl-model", "../target/debug/vearl-model.exe", "../target/debug/vearl-model"):
        p = os.path.join(HERE, rel)
        if os.path.exists(p):
            return p
    return None


class Qwen3Export(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = (Q.N, Q.PREFIX_ID)
        Q.set_n(N_TEST)
        Q.PREFIX_ID = PREFIX
        cls.cfg = Q.Cfg(D=128, FF=384, NH=16, NKV=8, HD=16, NL=4)
        cls.cfg.check()
        cls.sd = tiny_sd(cls.cfg)
        cls.P0 = Q.params_from_sd(cls.sd, cls.cfg)
        rng = np.random.default_rng(1)
        cls.cal = [rand_ids(rng) for _ in range(20)]
        cls.P, cls.st = Q.prepare(cls.P0, cls.cfg, cls.cal)
        cls.jobs, cls.plan = Q.build(cls.P, cls.cfg, cls.st, log=lambda *a: None)
        cls.ev = [rand_ids(rng) for _ in range(6)]

    @classmethod
    def tearDownClass(cls):
        Q.set_n(cls.saved[0])
        Q.PREFIX_ID = cls.saved[1]

    def test_float_model_matches_transformers(self):
        try:
            import torch
            from transformers import Qwen3Config, Qwen3Model
        except Exception as e:                                        # pragma: no cover
            self.skipTest(f"transformers unavailable: {e}")
        c = self.cfg
        hf = Qwen3Config(vocab_size=VOCAB, hidden_size=c.D, intermediate_size=c.FF, num_hidden_layers=c.NL, num_attention_heads=c.NH,
                         num_key_value_heads=c.NKV, head_dim=c.HD, rms_norm_eps=c.eps, rope_theta=c.theta, tie_word_embeddings=True,
                         max_position_embeddings=64, attention_bias=False, use_sliding_window=False)
        model = Qwen3Model(hf).eval()
        sd = {k: torch.from_numpy(np.asarray(v, np.float32)) for k, v in self.sd.items()}
        model.load_state_dict(sd, strict=True)
        ids = self.ev[0]
        with torch.no_grad():
            h = model(input_ids=torch.tensor([ids])).last_hidden_state[0, -1].double().numpy()
        want = h / np.linalg.norm(h)
        P32 = Q.params_from_sd(self.sd, c, dtype=np.float32)
        got = Q.forward(P32, c, ids)
        self.assertGreater(float(want @ got), 1 - 1e-6)

    def test_transformations_are_exact(self):
        for ids in self.cal[:4] + self.ev:
            a, b = Q.forward(self.P0, self.cfg, ids), Q.forward(self.P, self.cfg, ids)
            self.assertLess(float(np.abs(a - b).max()), 1e-9)

    def test_percentile_tracker(self):
        rng = np.random.default_rng(3)
        t = Q.Top()
        chunks = [rng.standard_normal(70000) * (1 + i) for i in range(5)]
        for c in chunks:
            t.add(c)
        allv = np.abs(np.concatenate(chunks))
        for p in (99.0, 99.9, 99.99):
            self.assertAlmostEqual(t.pct(p), float(np.percentile(allv, p)), delta=0.02 * float(np.percentile(allv, p)))

    def test_programs_respect_the_protocol_bounds(self):
        self.assertEqual(self.plan["layers_per_job"], 3)
        self.assertEqual(len(self.jobs), 2)
        for b in self.jobs:
            W = b.weights()
            self.assertLessEqual(len(b.ops), 240)
            self.assertLessEqual(len(W), 64)
            self.assertLessEqual(max(int(np.abs(w).max()) for w in W), 64)
            prog_ref.validate_ops(Q.program_for(b, 9), len(W), N_TEST)
            kinds = {next(iter(o)) for o in b.ops}
            self.assertTrue({"RmsNorm", "MulT", "MulW", "CausalSoftmaxS", "Lut", "Mul", "Map", "Gemm", "GemmAT"} <= kinds)

    def test_integer_chain_approximates_the_float_embedding(self):
        Ws = [b.weights() for b in self.jobs]
        cos = []
        for ids in self.ev:
            f = Q.forward(self.P, self.cfg, ids)
            i = Q.run_int(self.jobs, self.plan, self.P, self.cfg, ids, Ws)
            self.assertAlmostEqual(float(np.linalg.norm(i)), 1.0, places=9)
            cos.append(float(f @ i))
        self.assertGreater(float(np.mean(cos)), 0.93)
        self.assertGreater(min(cos), 0.88)

    def test_sink_columns_are_constants_and_never_leak(self):
        # whatever the sink column of the input contains, the embedding of the integer chain does not change
        ids = self.ev[1]
        Ws = [b.weights() for b in self.jobs]
        a = Q.run_int(self.jobs, self.plan, self.P, self.cfg, ids, Ws)
        orig = Q.enc_input

        def noisy(P, cfg, ids_, s0):
            t = orig(P, cfg, ids_, s0)
            t[:cfg.D, 0] = np.random.default_rng(5).integers(-64, 65, cfg.D)
            return t
        Q.enc_input = noisy
        try:
            b = Q.run_int(self.jobs, self.plan, self.P, self.cfg, ids, Ws)
        finally:
            Q.enc_input = orig
        self.assertTrue(np.array_equal(a, b))

    def test_export_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "m.json")
            size = Q.export(out, self.jobs, self.plan, self.cfg, "tiny/qwen3")
            self.assertGreater(size, 100000)
            m = json.load(open(out))
        self.assertEqual((m["n"], m["D"], m["L"], len(m["jobs"])), (N_TEST, 128, 16, len(self.jobs)))
        self.assertEqual(m["plan"]["prefix_id"], PREFIX)
        for j, b in zip(m["jobs"], self.jobs):
            self.assertEqual(j["ops_template"], b.ops)
            for enc, w in zip(j["weights"], b.weights()):
                self.assertTrue(np.array_equal(R.dec(enc, N_TEST), w))

    def test_rust_engine_agrees_job_by_job(self):
        exe = find_model_binary()
        if exe is None:
            self.skipTest("vearl-model not built")
        ids = self.ev[2]
        Ws = [b.weights() for b in self.jobs]
        _, _, Ts = Q.run_int(self.jobs, self.plan, self.P, self.cfg, ids, Ws, keep=True)
        x = Q.enc_input(self.P, self.cfg, ids, self.plan["s0"])
        for b, W, T in zip(self.jobs, Ws, Ts):
            ops = Q.program_for(b, len(ids))
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump({"n": N_TEST, "ops": ops, "weights": [R.enc(w) for w in W], "x0": R.enc(x)}, f)
                path = f.name
            try:
                r = subprocess.run([exe, "dump", path], capture_output=True, text=True)
            finally:
                os.unlink(path)
            self.assertEqual(r.returncode, 0, r.stderr[-800:])
            got = [R.dec(s, N_TEST) for s in json.loads(r.stdout)["tensors"]]
            self.assertEqual(len(got), len(T))
            for k, (a, c) in enumerate(zip(got, T)):
                self.assertTrue(np.array_equal(a, c), f"tensor {k} differs ({ops[k - 1] if k else 'input'})")
            x = np.zeros((N_TEST, N_TEST), np.int64)
            x[:self.cfg.D, :self.cfg.L] = T[-1][:self.cfg.D, :self.cfg.L]


if __name__ == "__main__":
    unittest.main()
