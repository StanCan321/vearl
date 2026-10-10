#!/usr/bin/env python3
"""Verifiable TEXT EMBEDDING model: an MLP encoder (1024 words -> hidden layer -> 384 dimensions) DISTILLED from
sentence-transformers/all-MiniLM-L6-v2 (the "teacher"), then quantised to int8 and exported as a VEARL program.

What it is, honestly: a STUDENT much smaller and weaker than MiniLM (bag of words, no attention), chosen because it
is expressed with the verifiable operations of the protocol. The report measures what is lost (student/teacher cosine, accuracy of
nearest-neighbour classification); MiniLM itself is NOT executed by the network (docs/12).

    python3 train_embed.py OUT.json [--hidden 2048] [--epochs 30] [--train 120000]      # pip install sentence-transformers datasets torch
Outputs: OUT.json (program + sample batches: 7 test batches then training batches), OUT.featurizer.json (vocabulary),
OUT.eval.npz (embeddings of the teacher / of the floating-point student, labels, to evaluate the network's output), OUT.report.json.
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantize as Q
import vearl_ref as R
from embed_text import Featurizer

arg = lambda name, default, cast=str: cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
HID, EPOCHS, NTRAIN = arg("--hidden", 2048, int), arg("--epochs", 12, int), arg("--train", 120000, int)
DROP = arg("--dropout", 0.3, float)
NBI = arg("--bigrams", 0, int)                            # number of bigrams among the 1,024 features
BIAS = "--no-bias" not in sys.argv                        # bias (Bias op): enabled by default
ACT = arg("--act", "gelu")                                 # relu | gelu | silu | tanh | sigmoid (Lut op except relu)
N, DIM = 1024, 384
VOCAB = arg("--vocab", 1024, int)                          # input features: > 1024 => WIDE INPUT (Shift op), fewer samples per job
NIN = -(-VOCAB // N)
BATCH = N // NIN
torch.manual_seed(5)
np.random.seed(5)
torch.set_num_threads(os.cpu_count() or 8)
out = sys.argv[1]

# ---------------- data ----------------
from datasets import load_dataset
ds = load_dataset("fancyzhx/ag_news")
tr_t, tr_y = ds["train"]["text"][:NTRAIN], np.array(ds["train"]["label"][:NTRAIN])
te_t, te_y = ds["test"]["text"], np.array(ds["test"]["label"])
print(f"AG News: {len(tr_t)} training texts, {len(te_t)} test texts, 4 categories", flush=True)

feat = Featurizer.fit(tr_t, size=VOCAB, n_bi=NBI)
Xtr, Xte = feat.transform(tr_t), feat.transform(te_t)
print(f"vocabulary {len(feat.vocab)} words; scale {feat.scale:.2f}; non-zero inputs per text: {(Xtr > 0).sum(1).mean():.1f}", flush=True)

# ---------------- teacher: MiniLM (normalised 384 embeddings) ----------------
from sentence_transformers import SentenceTransformer
t0 = time.time()
cache = f"/tmp/teacher_minilm_{NTRAIN}.npz"          # the teacher's embeddings are cached (5 min of CPU)
if os.path.exists(cache):
    z = np.load(cache)
    Ttr, Tte = z["tr"], z["te"]
else:
    teacher = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    Ttr = teacher.encode(tr_t, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    Tte = teacher.encode(te_t, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    np.savez(cache, tr=Ttr, te=Tte)
print(f"teacher MiniLM: {len(tr_t) + len(te_t)} texts encoded in {time.time() - t0:.0f} s", flush=True)

# ---------------- student (no bias, ReLU): distillation by cosine ----------------
l1, l2 = torch.nn.Linear(VOCAB, HID, bias=BIAS), torch.nn.Linear(HID, DIM, bias=BIAS)
act_mod = {"relu": torch.nn.ReLU(), "gelu": torch.nn.GELU(), "silu": torch.nn.SiLU(), "tanh": torch.nn.Tanh(), "sigmoid": torch.nn.Sigmoid()}[ACT]
net = torch.nn.Sequential(torch.nn.Dropout(DROP / 2), l1, act_mod, torch.nn.Dropout(DROP), l2)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
Xt, Yt, Xe, Ye = map(lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)), (Xtr, Ttr, Xte, Tte))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * ((len(Xt) + 255) // 256))
for ep in range(EPOCHS):
    perm = torch.randperm(len(Xt))
    for s in range(0, len(perm), 256):
        b = perm[s:s + 256]
        opt.zero_grad()
        loss = 1 - torch.nn.functional.cosine_similarity(net(Xt[b]), Yt[b]).mean()
        loss.backward()
        opt.step()
        sched.step()
    if ep % 2 == 1 or ep == EPOCHS - 1:
        net.eval()
        with torch.no_grad():
            cs = torch.nn.functional.cosine_similarity(net(Xe), Ye).mean().item()
        net.train()
        print(f"epoch {ep + 1}/{EPOCHS}: student/teacher cosine (test) {cs:.4f}", flush=True)
Ws = [l1.weight.detach().numpy().astype(np.float64), l2.weight.detach().numpy().astype(np.float64)]
Bs = [l1.bias.detach().numpy().astype(np.float64), l2.bias.detach().numpy().astype(np.float64)] if BIAS else None
net.eval()
with torch.no_grad():
    Ste = net(Xe).numpy()

# ---------------- int8 quantisation (hidden layer cut into blocks of 1024) ----------------
model, rep = Q.quantize_mlp(Ws, Xte[:2048].astype(np.float64), N, None, batch=BATCH, bs=Bs, act=ACT)
model["name"] = f"embed-agnews-int8-{HID}-{ACT}" + (f"-v{VOCAB}" if VOCAB != 1024 else "") + ("" if BIAS else "-nobias") + (f"-bi{NBI}" if NBI else "")
model["classes"], model["batch"] = DIM, BATCH


def enc_batch(x):
    t = np.zeros((N, N), np.int64)
    for b in range(NIN):
        seg = x[:, b * N:(b + 1) * N]
        t[:seg.shape[1], b * BATCH:b * BATCH + len(x)] = np.round(seg.T * 64).astype(np.int64)
    return t


# sample batches: 7 test batches (7168 texts) then the following training batches ("bulk" embeddings)
n_test_batches = len(te_t) // BATCH
batches_x = [Xte[i * BATCH:(i + 1) * BATCH] for i in range(n_test_batches)]
labels_b = [te_y[i * BATCH:(i + 1) * BATCH] for i in range(n_test_batches)]
n_train_batches = arg("--train-batches", 13 if NIN == 1 else 8, int)
for i in range(n_train_batches):
    batches_x.append(Xtr[i * BATCH:(i + 1) * BATCH])
    labels_b.append(tr_y[i * BATCH:(i + 1) * BATCH])
model["samples"] = [{"x0": R.enc(enc_batch(x)), "labels": [int(v) for v in y]} for x, y in zip(batches_x, labels_b)]
json.dump(model, open(out, "w"))

# int8 fidelity (independent reference engine) on the test batches, and downstream evaluation
q = []
W = [R.dec(w, N) for w in model["weights"]]
for x in batches_x[:n_test_batches]:
    T, _ = R.run_program(enc_batch(x), W, model["ops"])
    q.append(T[-1][:DIM, :len(x)].T.astype(np.float64))
Qte = np.concatenate(q)
M = len(Qte)


def nn_purity(E, y):
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    S = E @ E.T
    np.fill_diagonal(S, -2)
    return float((y[S.argmax(1)] == y).mean())


cos = lambda a, b: float(((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)).mean())
yy = te_y[:M]
report = dict(
    act=ACT, bias=BIAS, clipped_biases=rep.get("clipped_biases", 0), shifts=rep["shifts"], ops=rep["ops"], weights=rep["weights"], hidden=HID, test_texts=int(M),
    cos_student_float_vs_teacher=cos(Ste[:M], Tte[:M]), cos_int8_vs_float_student=cos(Qte, Ste[:M]), cos_int8_vs_teacher=cos(Qte, Tte[:M]),
    nn_same_class={"teacher_minilm": nn_purity(Tte[:M], yy), "student_float": nn_purity(Ste[:M], yy), "student_int8": nn_purity(Qte, yy),
                   "tfidf_1024_baseline": nn_purity(Xte[:M], yy)},
)
json.dump(report, open(out.replace(".json", ".report.json"), "w"), indent=1)
feat.save(out.replace(".json", ".featurizer.json"))
np.savez_compressed(out.replace(".json", ".eval.npz"), teacher=Tte[:M].astype(np.float32), student=Ste[:M].astype(np.float32), int8=Qte.astype(np.int8), labels=yy)
print(json.dumps(report, indent=1))
print("model written:", out, os.path.getsize(out) // 1024, "Kio")
