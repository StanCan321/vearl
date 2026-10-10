#!/usr/bin/env python3
"""Quantisation-aware fine-tuning (QAT) of a branched network, by EXACT SIMULATION of the integer program in torch:
input round(64 x), products on integers, arithmetic shift (floor) then saturation to +-64, integer biases, ReLU, saturated Add, rounded
mean. The roundings are differentiable by "straight-through". The scales (shifts, output scale of each layer) are those
of the quantize_graph plan; only the floating-point weights and biases (BatchNorm already folded) are adjusted. Then we re-quantise normally.

    python3 qat_graph.py resnet MODEL.pt OUT.json [--epochs 4] [--lr 0.002]
"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantize_graph as QG


def ste(fn):
    return lambda x: x + (fn(x) - x).detach()


round_ste, floor_ste = ste(torch.round), ste(torch.floor)


def sim_forward(nodes, plan, params, x):
    """x (B, c, h, w) real in [0, 1] -> real logits; params[name] = (W, b) floating-point tensors (trainable)."""
    v = {}
    for nd in nodes:
        op, name = nd["op"], nd["name"]
        if op == "input":
            v[name] = torch.clamp(round_ste(x * 64), 0, 64)
            continue
        a = v[nd["src"][0]]
        if op in ("conv", "linear"):
            W, b = params[name]
            p = plan[name]
            Wq = torch.clamp(round_ste(W / p["sw"]), -63, 63)
            if op == "conv":
                z = F.conv2d(a, Wq, None, nd["stride"], nd["pad"])
            else:
                z = a.reshape(len(a), -1) @ Wq.reshape(Wq.shape[0], -1).T
            y = torch.clamp(floor_ste(z / 2.0 ** p["sh"]), -64, 64)
            bi = torch.clamp(round_ste(b / p["s_out"]), -64, 64)
            y = y + (bi[None, :, None, None] if op == "conv" else bi[None, :])
            v[name] = torch.clamp(y, 0 if nd["relu"] else -64, 64)
        elif op == "pool":
            k = nd["k"]
            v[name] = F.max_pool2d(a, k) if nd["mode"] == "max" else floor_ste(F.avg_pool2d(a, k) + 0.5)
        elif op == "flatten":
            v[name] = a.reshape(len(a), -1)
        elif op == "add":
            v[name] = torch.clamp(a + v[nd["src"][1]], 0 if nd["relu"] else -64, 64)
    last = nodes[-1]["name"]
    return v[last] * plan[last]["s_out"]


def qat(nodes, plan, xtr, ytr, xte, yte, epochs=4, lr=0.002, bs=128):
    torch.set_num_threads(os.cpu_count() or 8)
    params = {nd["name"]: (torch.nn.Parameter(torch.tensor(nd["W"], dtype=torch.float32)), torch.nn.Parameter(torch.tensor(nd["b"], dtype=torch.float32)))
              for nd in nodes if nd["op"] in ("conv", "linear")}
    flat = [t for pr in params.values() for t in pr]
    opt = torch.optim.SGD(flat, lr=lr, momentum=0.9)
    Xtr, Ytr = torch.from_numpy(xtr.astype(np.float32)), torch.from_numpy(ytr.astype(np.int64))
    Xte, Yte = torch.from_numpy(xte.astype(np.float32)), torch.from_numpy(yte.astype(np.int64))

    def acc():
        with torch.no_grad():
            return float(np.mean([(sim_forward(nodes, plan, params, Xte[i:i + 500]).argmax(1) == Yte[i:i + 500]).float().mean().item() for i in range(0, len(Xte), 500)]))

    print(f"simulated accuracy before fine-tuning: {100 * acc():.2f} % ({len(Xte)} images)", flush=True)
    steps = epochs * ((len(Xtr) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            x = Xtr[b]
            flip = torch.rand(len(b)) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(3), x)
            opt.zero_grad()
            F.cross_entropy(sim_forward(nodes, plan, params, x), Ytr[b]).backward()
            opt.step()
            sched.step()
        print(f"epoch {ep + 1}/{epochs}: simulated accuracy {100 * acc():.2f} % ({time.time() - t0:.0f} s)", flush=True)
    out = []
    for nd in nodes:
        nd = dict(nd)
        if nd["op"] in ("conv", "linear"):
            nd["W"], nd["b"] = params[nd["name"]][0].detach().numpy().astype(np.float64), params[nd["name"]][1].detach().numpy().astype(np.float64)
        out.append(nd)
    return out


def main():
    a = sys.argv[1:]
    sd = torch.load(a[1], weights_only=True, map_location="cpu")
    nodes = QG.resnet_nodes(sd)
    from train_resnet_cifar import load_cifar
    xtr, ytr, xte, yte = load_cifar()
    epochs = int(a[a.index("--epochs") + 1]) if "--epochs" in a else 4
    lr = float(a[a.index("--lr") + 1]) if "--lr" in a else 0.002
    ncal = int(a[a.index("--calib") + 1]) if "--calib" in a else 500
    n = 1024
    model, rep, info = QG.quantize_graph(nodes, xte[:ncal].astype(np.float64), n, yte[:ncal], shape=(3, 32, 32), return_info=True)
    print("before QAT:", {k: v for k, v in rep.items() if k in ("float_acc", "int8_acc")})
    # the fine-tuning is done on the training set; the simulated evaluation on 2000 test images different from the calibration
    nodes2 = qat(nodes, info["plan"], xtr, ytr, xte[ncal:ncal + 2000], yte[ncal:ncal + 2000], epochs, lr)
    model, rep2 = QG.quantize_graph(nodes2, xte[:ncal].astype(np.float64), n, yte[:ncal], shape=(3, 32, 32), max_samples=int(a[a.index("--samples") + 1]) if "--samples" in a else 4)
    json.dump(model, open(a[2], "w"))
    json.dump(dict(before=rep, after=rep2), open(a[2].replace(".json", ".report.json"), "w"), indent=1)
    print("model written:", a[2], os.path.getsize(a[2]) // 1024, "KiB")


if __name__ == "__main__":
    main()
