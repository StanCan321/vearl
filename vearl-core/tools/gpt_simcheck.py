#!/usr/bin/env python3
"""Diagnostic: compares, operation by operation, the torch simulation of qat_gpt.py and the reference engine (vearl_ref) on a text."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import quantize_gpt as QG
import quantize_bert as QB
import vearl_ref as R
import qat_gpt as QT

tk, m = QG.hf_load()
cal, ev = QG.load_texts(64, 4)
cfg, P0, P, body, head, plan = QG.prepare(tk, m, cal)
E, Pp = QG.embed_tables(P, plan)
sim = QT.Sim(body, head, plan, P, cfg)
data = QT.Data(tk, E, Pp)
text = ev[0]
ids = QG.ids_of(tk, text)
lr = len(ids)
x, lrt, _, _ = data.batch([text])
# simulation: keep all the tensors
Ts_sim = []
orig_append = list.append


class Rec(list):
    def append(self, v):
        Ts_sim.append(v.detach())
        super().append(v)


# re-execution of run_body with an instrumented list
import types
src_run = QT.Sim.run_body


def run_body_rec(self, x0, lr_):
    Ts_sim.clear()
    return src_run(self, x0, lr_)


with torch.no_grad():
    # copy of run_body where Ts = [x0] is replaced by Rec([x0])
    import inspect
    code = inspect.getsource(QT.Sim.run_body).replace("Ts = [x0]", "Ts = Rec([x0])")
    code = "\n".join(l[4:] for l in code.splitlines())
    ns = dict(QT.__dict__)
    ns["Rec"] = Rec
    exec(code, ns)
    out_sim = ns["run_body"](sim, x, lrt)
Wb = QG.materialize(body, sim.params_np(P))
T, _ = R.run_program(QG.enc_input(E, Pp, ids), Wb, QB.program_for(body, lr))
print(f"input: {np.array_equal(x[0].long().cpu().numpy()[:, :lr], T[0][:cfg.D, :lr])}")
for k, op in enumerate(body.ops):
    s = Ts_sim[k].long().cpu().numpy()[0]
    r = T[k + 1][:s.shape[0], :s.shape[1]]
    kind = next(iter(op))
    if kind in ("CausalSoftmax",) or (kind == "GemmAT" and s.shape[1] > 64):
        cols = [h * 64 + q for h in range(cfg.H) for q in range(lr)]
        s, r = s[:lr][:, cols], r[:lr][:, cols]
    else:
        s, r = s[:, :lr], r[:, :lr]
    d = int(np.sum(s != r))
    if d:
        print(f"op {k} {op}: {d} cells different out of {s.size} ; example sim {s[s != r][:5]} ref {r[s != r][:5]}")
        break
else:
    print("BODY identical")
st = T[-1][:cfg.D, lr - 1].astype(np.int64)
ref = QG.logits_of(QG.run_head(head, QG.materialize(head, sim.params_np(P)), [st])[-1], plan, 0)
with torch.no_grad():
    hs = sim.run_head(torch.tensor(T[-1][:cfg.D, :64][None], dtype=torch.float32, device=QT.DEV)).long().cpu().numpy()[0][:, lr - 1]
print(f"HEAD: {int(np.sum(hs != ref))} logits different out of {ref.size}")
