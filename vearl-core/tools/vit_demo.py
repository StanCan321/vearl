#!/usr/bin/env python3
"""DeiT-Tiny (224 x 224, 1000 ImageNet categories) executed and VERIFIED on a real n = 1024 chain by a GPU worker: each image = two
chained jobs, each operation verified before payment.

    python3 tools/vit_demo.py /tmp/vit-qat.json [--n 40] [--photos FOLDER]

Without --photos: N Imagenette validation images never seen in training nor in evaluation (4 per category); top-1 accuracy over
1000 categories. With --photos: all the images of the folder (predictions displayed, no label).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from netlab import Lab
from vit_client import VitClient
import quantize_vit as QV

MP = [a for a in sys.argv[1:] if not a.startswith("--") and not a.isdigit()][0]
NIMG = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 40
PHOTOS = sys.argv[sys.argv.index("--photos") + 1] if "--photos" in sys.argv else None

if PHOTOS:
    paths = sorted(os.path.join(PHOTOS, f) for f in os.listdir(PHOTOS) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
    labels = None
else:
    paths, labels = QV.images("val", max(1, NIMG // 10), offset=200)

lab = Lab({"W": dict(port=9861, seed=2, extra=["--worker", "--gpu"], peers=[])}, n=1024, divisor=int(os.environ.get("DIVISOR", 64)),
          window=int(os.environ.get("WINDOW", 240)))
lab.start("W")
time.sleep(8)
W = lab.addrof("W")
try:
    c = VitClient(MP, W, W, inflight=int(os.environ.get("INFLIGHT", 16)))
    t0 = time.time()
    res = c.classify(paths)
    dt = time.time() - t0
    same = sum(int(np.array_equal(res[k][2], c.expected(p))) for k, p in enumerate(paths[:8]))
    print(f"{len(paths)} images 224 x 224 classified, VERIFIED and paid in {dt:.1f} s ({dt / len(paths):.2f} s/image, two jobs per image); "
          f"identical to the independent reference engine: {same}/{min(8, len(paths))}")
    if labels is not None:
        ok = sum(int(r[0] == y) for r, y in zip(res, labels))
        print(f"top-1 over 1000 categories: {ok}/{len(paths)} = {100 * ok / len(paths):.1f} %")
    for p, r in list(zip(paths, res))[:16]:
        print(f"  {os.path.basename(p):28s} -> {r[1]}")
    if "--out" in sys.argv:
        import json
        rows = [{"file": os.path.basename(p), "pred": int(r[0]), "label": r[1], "logits": [int(v) for v in r[2]]} for p, r in zip(paths, res)]
        exact = {os.path.basename(p): bool(np.array_equal(res[k][2], c.expected(p))) for k, p in enumerate(paths)}
        for r in rows:
            r["bit_exact"] = exact[r["file"]]
        json.dump({"seconds": dt, "images": len(paths), "rows": rows}, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)
        print(f"results written; identical to the independent engine: {sum(exact.values())}/{len(exact)}")
    for tag, log in zip("AB", c.logs):
        print(f"job {tag}: " + log.strip().splitlines()[-1][:220])
finally:
    lab.stop_all()
