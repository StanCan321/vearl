#!/usr/bin/env python3
"""Thumbnails of the CIFAR-10 images that ResNet-20 classified on the testnet, rebuilt from the model's own input tensors
(x0: 3 channel rows x 1024 pixel columns, int8), plus the predictions of that verified run.

    python3 make_cifar_sprite.py MODEL.json OUTPUT.i8 OUT_DIR    -> cifar-sprite.png (N tiles of 32x32), cifar-run.json
"""
import base64, json, sys
import numpy as np
from PIL import Image

model, outp, out_dir = sys.argv[1:4]
C = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
m = json.load(open(model))
n = m["n"]
scores = np.fromfile(outp, dtype=np.int8).reshape(-1, 10)
tiles, run = [], []
for k, s in enumerate(m["samples"][:len(scores)]):
    x0 = np.frombuffer(base64.b64decode(s["x0"]), dtype=np.int8).reshape(n, n).astype(np.float64)
    img = x0[:3, :1024].reshape(3, 32, 32).transpose(1, 2, 0)
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    img = np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1)
    tiles.append((img * 255).round().astype(np.uint8))
    p, l = int(scores[k].argmax()), int(s["labels"][0])
    run.append({"label": C[l], "pred": C[p]})
sprite = np.concatenate(tiles, axis=1)
Image.fromarray(sprite).save(f"{out_dir}/cifar-sprite.png", optimize=True)
json.dump(run, open(f"{out_dir}/cifar-run.json", "w"))
print(len(run), "images,", sum(r["label"] == r["pred"] for r in run), "correct")
