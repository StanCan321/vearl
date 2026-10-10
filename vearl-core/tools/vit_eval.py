#!/usr/bin/env python3
"""Accuracy of the INTEGER DeiT-Tiny (plan of quantize_vit.py + parameters trained by qat_vit.py) against the original floating-point DeiT, on
Imagenette (categories seen in training) and Imagewoof (10 dog breeds NEVER seen: generalisation). Bit-exact simulation
(checked against the reference engine on a few images).

    python3 vit_eval.py PARAMS.npz [--n 50]      (n images per category)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import quantize_vit as QV
import qat_vit as QA

m, proc, sd0, id2label = QV.load()
sd = QV.derived(sd0)
for syn, idx in QV.WOOF_SYNSETS.items():
    print(f"  {syn} -> {idx} {id2label[idx]}")
cal, _ = QV.images("train", QV.arg("--calib", 64, int) // 10)
bs, plan = QV.build(sd, QV.calibrate(sd, proc, cal))
params = dict(sd)
params.update(dict(np.load(sys.argv[1])))
sim = QA.Sim(bs, plan, params)
per = QV.arg("--n", 50, int)
for name, woof, off in (("Imagenette", False, 5), ("Imagewoof (categories never seen)", True, 0)):
    paths, y = QV.images("val", per, offset=off, woof=woof)
    pix = [QV.pixels(proc, p) for p in paths]
    if not woof:
        print(f"simulation == reference engine: {QA.check_exact(sim, bs, plan, params, pix[:3])}/3")
    with torch.no_grad():
        fl = np.concatenate([m(torch.tensor(np.stack(pix[s:s + 64]), dtype=torch.float32)).logits.argmax(1).numpy() for s in range(0, len(pix), 64)])
        it = np.concatenate([sim.forward(QA.x0_batch(pix[s:s + 64], plan)).argmax(1).cpu().numpy() for s in range(0, len(pix), 64)])
    print(f"{name}: {len(y)} images, top-1 over 1000 categories: float {100 * np.mean(fl == y):.1f} %, integer {100 * np.mean(it == y):.1f} % ; "
          f"same prediction as the float {100 * np.mean(fl == it):.1f} %")
