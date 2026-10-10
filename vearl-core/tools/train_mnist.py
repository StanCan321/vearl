#!/usr/bin/env python3
"""MNIST MLP (784 -> configurable hidden layers -> 10, no bias, ReLU), trained with torch (multi-core CPU), quantised to int8
(post-training, common module quantize.py) and exported as a VEARL PROGRAM for a chain of width n=1024: ONE job classifies
1024 images at a time (one image per column). A hidden layer wider than n is cut into blocks (docs/11).
Float / int8 accuracy report on the 10,000 test images (independent reference engine).

    python3 train_mnist.py OUT.json [--hidden 512,256] [--epochs 10] [--batch 128] [--lr 1e-3]     # pip install torch (CPU)
The activations of the last hidden layer are the EMBEDDINGS of the images: they are part of the published tensors.
"""
import json, os, sys, urllib.request
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantize as Q

N = 1024
arg = lambda name, default, cast=str: cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
hid = [int(v) for v in arg("--hidden", "512,256").split(",")]
epochs, BS, LR = arg("--epochs", 10, int), arg("--batch", 128, int), arg("--lr", 1e-3, float)
sizes = [784] + hid + [10]
torch.manual_seed(3)
np.random.seed(3)

# ---------------- data ----------------
if not os.path.exists("/tmp/mnist.npz"):
    urllib.request.urlretrieve("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", "/tmp/mnist.npz")
d = np.load("/tmp/mnist.npz")
xtr, ytr = d["x_train"].reshape(-1, 784).astype(np.float32) / 255.0, d["y_train"]
xte, yte = d["x_test"].reshape(-1, 784).astype(np.float32) / 255.0, d["y_test"]

# ---------------- floating-point training (torch) ----------------
layers = [torch.nn.Linear(a, b, bias=False) for a, b in zip(sizes[:-1], sizes[1:])]
model_t = torch.nn.Sequential(*[m for i, l in enumerate(layers) for m in ([l, torch.nn.ReLU()] if i < len(layers) - 1 else [l])])
opt = torch.optim.Adam(model_t.parameters(), lr=LR)
Xtr, Ytr, Xte, Yte = map(torch.from_numpy, (xtr, ytr.astype(np.int64), xte, yte.astype(np.int64)))
for ep in range(epochs):
    perm = torch.randperm(len(Xtr))
    for s in range(0, len(perm), BS):
        b = perm[s:s + BS]
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model_t(Xtr[b]), Ytr[b]).backward()
        opt.step()
    with torch.no_grad():
        acc = (model_t(Xte).argmax(1) == Yte).float().mean().item()
    print(f"epoch {ep + 1}/{epochs}: float test accuracy {acc * 100:.2f} %", flush=True)
Ws = [l.weight.detach().numpy().astype(np.float64) for l in layers]     # (output, input)
facc = acc

# ---------------- int8 quantisation (cut into blocks if a layer exceeds n) ----------------
model, rep = Q.quantize_mlp(Ws, xte.astype(np.float64), N, yte, batch=1024)
model["name"] = "mnist-mlp-int8" + ("" if len(model["weights"]) == 3 else "-" + "x".join(map(str, hid)))
rep.update(float_test_acc=float(facc), test_images=int(len(yte)))
out = sys.argv[1]
json.dump(model, open(out, "w"))
json.dump(rep, open(out.replace(".json", ".report.json"), "w"), indent=1)
print("model written:", out, os.path.getsize(out) // 1024, "KiB")
