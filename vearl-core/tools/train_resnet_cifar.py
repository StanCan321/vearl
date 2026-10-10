#!/usr/bin/env python3
"""CIFAR-10 ResNet (depth 6n+2: n = 3 -> ResNet-20) trained on CPU, in the FORM that the chain knows how to execute (docs/12 section 7):
3x3 "same" convolutions, identity shortcuts or stride-2 1x1 convolution, BatchNorm (folded at import), ReLU, global mean, dense layer.
Input: pixels in [0, 1] WITHOUT normalisation (the integer input of the chain is round(64 * x)).

    python3 train_resnet_cifar.py OUT.pt [--n 3] [--epochs 15] [--width 16]          # CIFAR-10 must be in /tmp/cifar-10-batches-py
"""
import os, pickle, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

arg = lambda name, default, cast=str: cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
NB, EPOCHS, WIDTH = arg("--n", 3, int), arg("--epochs", 15, int), arg("--width", 16, int)


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, 0, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + (x if self.short is None else self.short(x)))


class ResNet(nn.Module):
    def __init__(self, nb=NB, w=WIDTH):
        super().__init__()
        self.conv = nn.Conv2d(3, w, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(w)
        blocks, cin = [], w
        for si, cout in enumerate([w, 2 * w, 4 * w]):
            for b in range(nb):
                blocks.append(Block(cin, cout, 2 if (b == 0 and si > 0) else 1))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(cin, 10)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.blocks(x)
        x = F.avg_pool2d(x, 8).flatten(1)
        return self.fc(x)


def load_cifar(d="/tmp/cifar-10-batches-py"):
    xs, ys = [], []
    for i in range(1, 6):
        b = pickle.load(open(f"{d}/data_batch_{i}", "rb"), encoding="bytes")
        xs.append(b[b"data"])
        ys += b[b"labels"]
    t = pickle.load(open(f"{d}/test_batch", "rb"), encoding="bytes")
    f = lambda a: a.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    return f(np.concatenate(xs)), np.array(ys), f(t[b"data"]), np.array(t[b"labels"])


if __name__ == "__main__":
    torch.manual_seed(1)
    torch.set_num_threads(os.cpu_count() or 8)
    xtr, ytr, xte, yte = load_cifar()
    Xtr, Ytr, Xte, Yte = map(torch.from_numpy, (xtr, ytr.astype(np.int64), xte, yte.astype(np.int64)))
    net = ResNet().to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    BS = 256
    steps = EPOCHS * ((len(Xtr) + BS - 1) // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.2, total_steps=steps, pct_start=0.25)
    t0 = time.time()
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(perm), BS):
            b = perm[s:s + BS]
            x = Xtr[b]
            flip = torch.rand(len(b)) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(3), x)
            dx, dy = np.random.randint(-3, 4, 2)                          # small random translation (per batch)
            x = torch.roll(x, (int(dy), int(dx)), (2, 3))
            opt.zero_grad()
            F.cross_entropy(net(x.contiguous(memory_format=torch.channels_last)), Ytr[b]).backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            acc = np.mean([(net(Xte[i:i + 1000]).argmax(1) == Yte[i:i + 1000]).float().mean().item() for i in range(0, 10000, 1000)])
        print(f"epoch {ep + 1}/{EPOCHS}: test accuracy {100 * acc:.2f} % ({time.time() - t0:.0f} s)", flush=True)
        torch.save(net.state_dict(), sys.argv[1])
