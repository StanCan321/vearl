#!/usr/bin/env python3
"""Small LANGUAGE MODEL (TinyStories-33M) executed and VERIFIED on a real n = 1024 chain by a GPU worker: each generated word =
one BODY job per story + one common HEAD job, verified before payment.

    python3 tools/gpt_demo.py /tmp/gpt-qat.json [--new 40] [--out /tmp/gpt_stories.json]

Prints the stories, the time per step, compares with the ORIGINAL floating-point model (same beginning, greedy decoding) and checks bit for bit the
logits of a few steps against the independent reference engine.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from netlab import Lab
from gpt_client import GptClient

PROMPTS = [
    "Once upon a time, there was a little cat named Tom.",
    "Lily and Ben went to the park to play.",
    "One day, a small bird found a shiny key.",
    "Sam wanted to build a big tower with his blocks.",
    "The sun was hot, so Mia and her mom went to the beach.",
    "There was a friendly dragon who lived in a cave.",
    "Tim had a red ball. He loved to",
    "Anna saw a sad puppy in the rain.",
]
MP = [a for a in sys.argv[1:] if not a.startswith("--") and not a.isdigit()][0]
NEW = int(sys.argv[sys.argv.index("--new") + 1]) if "--new" in sys.argv else 40
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "/tmp/gpt_stories.json"

lab = Lab({"W": dict(port=9861, seed=2, extra=["--worker", "--gpu"], peers=[])}, n=1024, divisor=int(os.environ.get("DIVISOR", 64)),
          window=int(os.environ.get("WINDOW", 300)))
lab.start("W")
time.sleep(8)
W = lab.addrof("W")
try:
    c = GptClient(MP, W, W, inflight=int(os.environ.get("INFLIGHT", 16)))
    t0 = time.time()
    texts, seqs, last = c.generate(PROMPTS, max_new=NEW)
    dt = time.time() - t0
    ntok = sum(len(s) - len(c.tk(p)["input_ids"]) for s, p in zip(seqs, PROMPTS))
    body_t = sum(t[0] for t in c.times)
    head_t = sum(t[1] for t in c.times)
    print(f"{len(PROMPTS)} stories, {ntok} words generated in {dt:.0f} s ({len(c.times)} steps; BODY jobs {body_t:.0f} s, HEAD jobs {head_t:.0f} s): "
          f"{dt / len(c.times):.1f} s per step for {len(PROMPTS)} stories,{dt / max(ntok, 1):.2f} s per word on average")
    # ORIGINAL floating-point model, same beginning, greedy decoding
    import torch
    from transformers import AutoModelForCausalLM
    fm = AutoModelForCausalLM.from_pretrained(c.m["hf"]).eval()
    same_tok = tot_tok = 0
    ref_texts = []
    for p, s in zip(PROMPTS, seqs):
        ids = c.tk(p)["input_ids"]
        with torch.no_grad():
            g = fm.generate(torch.tensor([ids]), max_new_tokens=len(s) - len(ids), do_sample=False, pad_token_id=c.eos)[0].tolist()
        ref_texts.append(c.tk.decode([t for t in g if t != c.eos]))
        k = len(ids)
        while k < min(len(s), len(g)) and s[k] == g[k]:
            k += 1
        same_tok += k - len(ids)
        tot_tok += len(s) - len(ids)
    print(f"same sequence of words as the original floating-point model (before the first difference): {same_tok}/{tot_tok} words")
    exact = 0
    for ids, row in last:
        exact += int(np.array_equal(row, c.expected_logits(ids)))
    print(f"logits of the last step bit-for-bit identical to the independent engine: {exact}/{len(last)}")
    for p, t, r in zip(PROMPTS, texts, ref_texts):
        print(f"\n[chain]    {t}\n[float]  {r}")
    json.dump({"model": c.m["name"], "prompts": PROMPTS, "chain": texts, "float": ref_texts, "seconds": dt, "steps": len(c.times),
               "tokens": ntok, "exact_last_step": exact, "same_prefix_tokens": [same_tok, tot_tok], "report": c.m.get("report")},
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    for k, log in enumerate(c.logs[-2:]):
        print(f"job {k}: " + log.strip().splitlines()[-1][:200])
finally:
    lab.stop_all()
