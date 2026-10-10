# 11 — Verifiable inference on GPU: real measurements, cost, and an honest comparison with Pearl

**Status: testnet, a single machine (A10), not audited.** This document reports what was **measured**, not what is hoped for. Points that were not measured are marked `[NOT MEASURED]`.

## 1. The measured use case: classifying MNIST images with an int8 MLP

Model: perceptron 784 → 512 → 256 → 10 (ReLU, no biases), trained in numpy (`tools/train_mnist.py`), post-training quantized to int8 (weights in [−63, 63], activations in [0, 64], shifts calibrated on 2 training batches). Run as a **tensor program** (3 GEMMs) on a chain of width **n = 1024**: **one job classifies 1024 images at a time**, one image per column.

| measurement | value |
|---|---|
| float accuracy (10,000 test images) | **98.20 %** |
| int8 accuracy, integer program (independent reference engine, 10,000 images) | **98.12 %** (loss 0.08 points, prediction agreement 99.80 %) |
| same program run by the Rust node then by a GPU worker of the network, on the first 1024 images | 1006/1024 = **98.24 %**, identical to the independent engine |
| embeddings (activations of hidden layer 2, 256 dimensions): nearest neighbour of the same class, with no label at all | **96.97 %** |

Limits of this result: a single dataset, a simple model, per-tensor quantization, no biases or batch normalization; 10,000 images give a margin of about ±0.25 points. "Loss of 0.08 points" means "not measurable here", not "zero".

## 2. Time and throughput (chain n = 1024, one worker on an A10 GPU, one client, a single process of each)

| measurement | value |
|---|---|
| one job (submission → 3 operations published) | **4.3 s** |
| verification by the client (Freivalds on 3 GEMMs 1024×1024, exact recomputation, Accept) | **2.2 s** |
| 8 jobs in a row = 8,192 images, worker side | **8.0 s in total, ≈ 1,000 images/s** |
| client verification of these 8 jobs, sequential | 16.0 s (≈ 510 images/s) |
| raw computation of the same program on CPU (without any proof) | 0.10 s per batch of 1024 |

Where the time goes: the computation itself is negligible; the cost is that of the **Merkle commitments** (a tree of 1,048,576 leaves per GEMM: 0.38 s sequentially, **0.10 s** after parallelisation, `par_map`), of the transfer of tensors of several MiB, and of the **block latency**. These times are measured with almost instantaneous blocks (easy target). (A later redesign of the commitment per row cut these costs further: `docs/12` §4 quater.)

**Latency with real blocks of ~30 s** (`DIVISOR=2000000000 WINDOW=300 python3 tools/mnist_demo.py`, target set for ≈ 1 block every 30 s on the A10 at n = 1024; 6 blocks in ≈ 4 min, hence ≈ 35-40 s on average, exponential variance): submission → the 3 operations published in **106 s** (≈ 3 blocks: submission, claim, publication), verification by the client ≈ 1 s of computation, then **one more block** to include the `Accept` (the client does not wait for the challenge window): **of the order of 2.5 to 3 minutes end to end**, and **up to 5 minutes more** (300 s default window) if nobody accepts and the settlement goes through `Settle`. An inference job is therefore an **asynchronous service on the scale of minutes**, not seconds.

**What this honestly says about competitiveness**: for a small model like this one, the network is **much slower and more expensive** than a direct computation on a GPU (this MLP costs ≈ 1.07 MFLOP per image: an A10-class GPU — ≈ 10² TFLOPS at peak — could classify millions per second, **an estimate by calculation, not measured**; here ≈ 1,000 per second per worker, i.e. several orders of magnitude less). Verification has a **fixed overhead** that is only amortised on computations much heavier than this MLP. One therefore **cannot** defend a price or speed advantage on this kind of workload. `[NOT MEASURED]`: the break-even point for larger models.

## 3. What one gets in exchange: verifiability, and a usefulness *tied to consensus*

- **The result is verified before payment**: each operation is checked (Freivalds for the GEMMs, exact recomputation for the rest); a wrong operation is provable with a single-entry proof and the worker loses its bond.
- **The intermediate tensors are published and verified**: the **embeddings** (similarity search, indexing) are a free by-product of the same job.
- **The consensus work is the client's computation**: an `Op`-class block is mined *with the client's matrices* (weights and activations committed by Merkle roots); in the throughput test, 21 blocks out of 87 (24 %) carried a client's GEMM, the rest being security work (Native) during idle time. With sustained demand, a worker mines on the GEMMs of its jobs first.

## 4. Comparison with Pearl (PRL) — what is sourced, what is design

Facts about Pearl: those of `docs/01` (third-party sources, a single empirical study for the central point, to be cross-checked): "NoisyGEMM" proof of work; **the verifier does not check where A and B come from**; a miner that runs the pipeline with random matrices is accepted; the dominant software contains no inference code; the study concludes "zero useful AI compute" on the observed network.

| | Pearl (according to `docs/01`) | VEARL (testnet) |
|---|---|---|
| origin of the work's matrices | chosen by the miner, not checked | **committed by a client** (Merkle roots in the job's transaction) for job blocks; the protocol's native matrices otherwise |
| link usefulness ↔ reward | "an economic property, not a cryptographic guarantee" | usefulness *rebate* **bounded by the amount burned** at job creation: a job's extra reward cannot exceed what its client destroyed (anti wash trading) |
| is the client's result verified? | no (validity is GEMM correctness) | **yes**: Freivalds, single-entry fraud proofs, worker bond |
| compute types | inference through a vLLM miner (repository) | matrix products, tiled jobs, multi-layer inference, tensor programs (convolutions, residuals, transformer), level 2 (Monte Carlo, rendering) |
| confidentiality | transparent | **also transparent** (nothing is confidential: `docs/09`, A17) |
| status | mainnet since April 2026 (third-party sources) | **testnet**, one machine, no external user |

**What can be stated, and what cannot.** One can state a difference of **design**: in VEARL, the work of a job block *is* the client's computation and is verified; the usefulness gap measured at Pearl comes precisely from the absence of this link. One **cannot** state that VEARL produces "more useful compute" than Pearl: there is no external client, no volume, no network; real usefulness depends on a **demand** that does not exist yet, and the measured throughput (§2) is far below a GPU used directly.

## 5. Current technical limits of inference

- **Size**: `n ≤ 1024` (an `n×n` tensor), at most 240 operations and 64 weight blocks per program (§5 bis): wide MLPs or small CNNs, **not** a large language model (a 4096×4096 int8 layer is already 16 blocks of 1 MiB). The input and output must fit in a tensor. (Later work reached models of 33 M to 110 M parameters by chaining jobs: `docs/12`.)
- **Quantization**: 8-bit integers in [−64, 64], power-of-two shifts; the loss is to be measured model by model. (Biases and batch-norm folding were added afterwards: `docs/12`.)
- **Latency**: ≈ 3 minutes end to end with 30 s blocks (§2, measured), plus the challenge window (300 s default) if the client does not accept itself.
- **The node computes under a lock**; a very heavy job briefly blocks the node `[TO IMPROVE]`.
- **No pricing at real cost**: the minimum escrow is the same whatever the job.

## 5 bis. Models wider than the chain: splitting into blocks (measured)

A layer wider than `n` is split **at the program level**, with no new operation: each output block `i` is the sum over `k` of the products `W_ik · X_k` (one Gemm per product, each requantized with the same shift), combined by `Add`s. The limit of a program rises from 16 operations / 8 weights to **64 operations / 32 weight blocks** (at that time; now 240 / 64); the weights are deposited at the worker **in batches** (`PutProgMore`, one message ≤ 16 MiB). Semantics: those of the *program* — the partial products are requantized separately (a margin of `⌈log2 #products⌉` bits so that the sum fits in [−64, 64]), which does not exactly equal a monolithic product; the calibration measures the cost.

| measurement | value |
|---|---|
| model | MNIST MLP **784 → 4096 → 4096 → 10** (≈ 20 million parameters), trained with torch (CPU, < 2 min) |
| program | **39 operations including 24 Gemms**, 24 weight blocks (24 MiB), chain n = 1024 |
| accuracy, 10,000 test images | float **97.96 %**; int8 (program split into blocks) **97.97 %** |
| on the network (GPU worker), first 1024 images | 1002/1024 = **97.85 %**, identical to the Rust node (`vearl-model eval`) |
| worker: submission → 39 operations published | **11.3 s** (against 4.3 s for the 3-GEMM MLP) |
| client: verification of the 39 operations (Freivalds on 24 GEMMs) + payment | **4.6 s** (7.9 s before the download and root checking were parallelised, 8 operations at a time) |

Two honest remarks: (1) this wider model **does not classify better** than the small MLP (MNIST is saturated around 98 %) — the point is to show that the mechanism scales, not an accuracy gain; (2) on a smaller test (hidden layer of 200 > n = 128, 497 images), the splitting cost **0.40 points** (98.99 % → 98.59 %) because of the partial requantizations: the cost is to be measured model by model. Limits: the input and output (classes) must fit in a single tensor (≤ n); no biases; MLP only in the conversion tool (`tools/quantize.py`).

## 6. Bringing your own model

`tools/quantize.py mlp weights.npz calib.npy --labels labels.npy --out model.json --n 1024` converts a float MLP (matrices `W0, W1, …` without biases, exportable from PyTorch with `Linear(bias=False)`) into an int8 VEARL program: per-tensor weight quantization, **automatic calibration of the shifts** (99.9th percentile), float/int8 accuracy report, exported input sample. Verified (`quantize.py selftest`) on a 64-48-10 MLP trained on the fly: 98.59 % in float as in int8 (497 images), and the Rust node (`vearl-model eval`) finds the same accuracy on the first batch. The file is submitted with `vearl-cli prog-submit … model:model.json` on a chain of the same width `n`. Limits: MLP only in this tool (CNNs go through `train_digits.py` as an example), widths ≤ `n`, ≤ 8 layers, no biases. (Later converters cover PyTorch MLPs and CNNs with biases and BatchNorm, ONNX, ResNet, transformers: `docs/12`.)

## 7. Reproduce

```bash
cd vearl-core && cargo build --release --features gpu --bins
python3 tools/train_mnist.py models/mnist-mlp.json --epochs 10      # ~8 min CPU; downloads MNIST; writes the model and its accuracy report
./target/release/vearl-model eval models/mnist-mlp.json            # accuracy of the integer program on the 1024 exported images
python3 tools/mnist_demo.py                                         # n=1024 chain on GPU: job, verification, payment, embeddings
python3 tools/train_mnist.py models/mnist-mlp-4096.json --hidden 4096,4096 --epochs 6 --batch 256 --lr 5e-4   # torch CPU; hidden layer split into blocks
python3 tools/mnist_demo.py models/mnist-mlp-4096.json             # 39 operations on GPU
python3 tools/mnist_throughput.py 8                                 # sustained throughput (8 jobs)
./target/release/proofbench                                         # where the time of a job goes (Merkle trees, Freivalds)
```
