# 05 — Compute types: what fits in, how it is verified, and in what order

Labels: `[DONE]` implemented and tested · `[PLAN]` designed, not done yet · `[RESEARCH]` needs unresolved work · `[LIT]` idea from the literature, to be re-read before relying on it.
Question asked: is this the right time to integrate inference, science and graphics? **Short answer: yes for the "quantized inference" family (the infrastructure is ready and part of it already works); not yet for science and graphics, which require a second verification framework and a sandbox.**

## 1. The criterion that decides everything: how is it verified?

VEARL has three levels of guarantee, which do not mix:

| level | verification | duplex (mines at the same time) | guarantee |
|---|---|---|---|
| **1 — verifiable GEMM** | Freivalds `O(n²)` + element-level fraud proofs (`O(n)` on-chain) | **yes**: the noisy GEMM is the consensus work | strong, cheap, a single wrong entry is provable |
| **2 — sampling / replication** | recomputation of randomly drawn pieces by other workers, committed seeds | **no** | probabilistic; does not find an isolated error in a large result |
| **3 — not verifiable at low cost** | — | no | none without trust (reputation only) |

**Economic consequence** (docs/04): a level-1 job is *subsidised* by the mining attempts it produces; a level-2 or level-3 job **produces none** (the GPU does not run a noisy GEMM). The worker therefore gives up ~`r` $/GPU-h of attempts ($0.165 at the model's equilibrium): these jobs must be paid at **full cost**, hence much more expensive for the client than level 1.

## 2. Category by category

### Inference (level 1 when everything reduces to GEMMs)

| workload | status | how | what is missing |
|---|---|---|---|
| **embeddings / similarity search** | **[DONE]** (dense form) | a base `W` (rows) × queries `X` (columns) = 1 GEMM; the top-k is done on the client side on `Z` | sizes > `n`: **tiled GEMM [DONE]** (up to 8×8×8 blocks; **measured at 8192³ = 1.1 TOP, verified end to end in ~13 s**); bulk text embeddings: **[DONE]** with a distilled encoder (`docs/12`), not with a full transformer `[RESEARCH]` |
| **classification** | **[DONE]** (dense heads / MLP) | last layer = logits; argmax on the client side | — |
| **small models** | **[DONE]** for dense networks (≤ 8 layers, ReLU + saturation) | multi-layer inference jobs, `docs/02` | CNNs and transformers: P10b/c |
| **batch inference** | **[DONE]** | the batch = the columns of `X` (n = 128 samples per job) | larger batches: P10a |
| **vision** (CNN) | **[DONE, toy]** (`prog.rs`: conv through `im2col`, average pooling, residuals; remaining: real networks by blocks, max-pooling, normalisation) | convolution = GEMM on unfolded patches (`im2col`); the unfolding is a *copy* verifiable by an element-level proof (one cell of the unfolded matrix = one cell of the previous activation, 2 Merkle openings); pooling and residuals = element-wise operations | rectangular tiled GEMM, `im2col`/`add`/`pool` operations with fraud proofs |
| **speech → text** | **[PLAN, partial]** | the **encoder** (convolutions + attention) is dominated by GEMMs; the **decoding** (beam search, autoregressive) is not | encoder: P10b/c; decoder: client side or level 2 |
| **AI agents** | **[NO]** | autoregressive loop with sampling, tool calls, external I/O: non-deterministic and with no cheaply verifiable structure | only an **isolated forward pass** (= batch inference) is verifiable; the agent itself stays outside VEARL |

**What must be built to go beyond dense networks** — a small "verifiable tensor VM" (P10):
1. **Tiled / rectangular GEMM**: any `M×K×N` = `n×n` blocks (each block is a mined unit) + additions verified element by element. (P10a)
2. **Element-wise and per-row operations, each with its fraud proof**: activation/requantisation `[DONE]`, addition (residuals), copy/unfolding (`im2col`, transposition), pooling, and — for transformers — `softmax` and layer normalisation in **integer arithmetic** (approximations exist in the integer-quantisation literature, e.g. I-BERT `[LIT, to be re-read before use]`). One proof per row (`O(n)`), not per element. (P10b)
3. **Import of quantized models** (int8 ONNX subset) + **conformance**: output bit-identical to an independent reference engine, and a **report of the accuracy loss** from quantisation (it exists: this is not the same model as in float). (P10c)
4. **Frozen integer semantics**: all the operations defined without ambiguity (rounding, saturation), otherwise two honest workers diverge and a fraud proof becomes disputable.

### Science

| workload | level | how it is verified | verdict |
|---|---|---|---|
| **Monte Carlo** | 2 | every independent batch is derived from a committed seed (`H(seed, index)`, counter-based generator); other workers **recompute randomly drawn batches**; the aggregate is a verifiable sum | **[PLAN]** feasible, but needs the level-2 framework; floats: integer accumulators or a documented statistical tolerance |
| **numerical optimisation** | 1 or 2 | gradient descent dominated by GEMMs: each iteration is a chained level-1 job (like the layers); otherwise replayed checkpoints | **[PLAN, partial]** after P10a |
| **scientific linear algebra** | 1 | integer/quantized matrix products: already covered; LU, decompositions: non-GEMM operations to define | GEMM **[DONE]**, the rest **[PLAN]** |
| **simulations (molecular dynamics, particles)** | 3 | chaotic systems + floats: two honest GPUs diverge, so **replication proves nothing**; only a deterministic fixed-point execution with checkpoints and **bisection** (Truebit style `[LIT]`) is verifiable | **[RESEARCH]**; out of scope in the short term |

### Graphics

| workload | level | how it is verified | verdict |
|---|---|---|---|
| **3D rendering / ray tracing / image generation** | 2 | image split into tiles; other workers **recompute randomly drawn tiles** with the same scene and seed | **[PLAN]**; requires a **deterministic** rendering engine frozen by the protocol (floats differ from one GPU/driver to another: fixed point, or comparison with a tolerance that weakens verification) and a **sandbox**: a client's code is never executed |
| **particle simulation** | 3 | like molecular dynamics | **[RESEARCH]** |

## 3. The sandbox, a condition of level 2

Level 1 only handles matrices: no client code is executed. From level 2 on, **job types frozen by the protocol** (rendering, Monte Carlo) are needed, executed in an isolated environment (microVM/container with no network, quotas, GPU reset between jobs). This is a major attack surface: it must not be opened before level 1 is solid and audited.

## 4. Proposed order

| step | content | depends on |
|---|---|---|
| **P8** (in progress) | wallet, JSON API, explorer, testnet packaging | — |
| **P10a** | **[DONE]** tiled GEMM, rectangular shapes (`tiled.rs`) | current market |
| **P10b** | **[DONE]** addition (residuals), unfolding/transposition, average pooling | P10a |
| **P10c** | **[DONE]** integer `softmax` and layer normalisation, `GemmAT`, transformer encoder block | P10b |
| **P10d** | **[DONE]** int8 model import (JSON file, `vearl-model`), bit-for-bit conformance with an independent engine, accuracy report on a real CNN. **[DONE]** networks wider than `n` by blocks *inside* a program (784→4096→4096→10 verified); biases (`Bias`), BatchNorm folding, import of a PyTorch MLP from a `state_dict`, per-worker model registry, serial execution (`vearl-cli infer`): `docs/12`. **[DONE]** import of a PyTorch CNN ("same" convolutions, BatchNorm, max/average pooling: `tools/quantize_cnn.py`). **[TO DO]** ONNX beyond sequential graphs, convolutions with stride > 1, per-channel / per-group quantisation | P10c |
| **P10f** | **level-2** framework: **[DONE]** jobs frozen by the protocol (no arbitrary code, hence no sandbox), sampling audits with a one-chunk fraud proof, **Monte Carlo** (fixed-point integrand machine) and **tiled rendering** (distance-field ray marching, integers) — see `docs/10`. **[TO DO]** k-of-n replication, public audit randomness, payment of third-party auditors, GPU | P10c |
| **P10e** | deterministic compute with checkpoints and bisection (simulations) | `[RESEARCH]` |

## 5. Why not everything now
1. Every new compute type **adds a surface to audit** (integer semantics, per-operation fraud proofs, sandbox). Adding them before level 1 is stabilised multiplies the risks with no security gain.
2. Level 2 **breaks the central property** (the useful work is also the consensus work): it must be designed as a service paid at full cost, with a weaker guarantee displayed clearly, not as a transparent extension.
3. Floats are not deterministic across hardware: half of the science and graphics workloads first require choosing between fixed point and tolerance.

## 6. Import of quantized models (P10d) — measured state

**What exists.** A model is a JSON file `{n, ops, weights[base64], samples[…]}` (`vearl-core/src/model.rs`). `vearl-model info|eval|dump` inspects it, runs it with the consensus's integer semantics, or dumps all its tensors. `vearl-cli prog-submit … model:FILE` submits it like any verifiable program (same market, same fraud proofs). New `Flatten` operation (conv → dense).

**Bit-for-bit conformance `[PROVEN BY TEST, not by formal proof]`.** `vearl-core/tools/vearl_ref.py` is a second engine (numpy + Python integers, exp2 table recomputed in 90-digit decimal, `math.isqrt`) that shares no code with the node. Fuzz: **1,997 random programs** (Gemm, GemmAT, Add, Transpose, Im2col, Pool, Flatten, Softmax, LayerNorm; ≈ 9,000 operations), **0 divergence**, compared tensor by tensor. The fuzz did find an error… in the reference engine (exp2 table: `2^-1·65536` computed as 32767.999…) and not in the node — that is the role of a second engine. Limit: two implementations written by the same person can share the same misreading of the specification; an independent second implementer remains desirable (`[AUDIT ACTION]`).

**Accuracy loss, measured** (`tools/train_digits.py`, sklearn *digits* 8×8, 396 never-seen test images, CNN conv3×3(8)→pool→conv3×3(16)→pool→dense(10), no biases, int8 weights ∈ [-63,63] per tensor, shifts calibrated on 150 pairs of training images, *post-training* quantisation):

| | test accuracy |
|---|---|
| float (float64) | 96.97 % |
| int8, integer VEARL program | **96.97 %** (loss 0.00 points) |
| float/int8 prediction agreement | 98.99 % |

The Rust node and the reference engine give exactly the same predictions. The shipped model (`models/digits-cnn.json`, 12 image pairs) runs as an end-to-end network job (honest → paid; falsified GEMM → proven and slashed: `tests/net.rs`).

**Honesty about the scope.** Small problem (8×8 digits), a single dataset, 396 images: the 95 % confidence interval on 97 % is about ±1.7 points, so "zero loss" means "not measurable here", not "always zero". No biases, no batch normalisation, activations and weights on ±64 (≈ 6 bits), n=128 only: a real vision network or an LLM requires splitting by blocks inside a program and careful calibration (per-channel quantisation, LayerNorm deviations when the variance is low: already documented limit). These points remained to be done at that stage (later work is in docs/12).

### Wider networks: the chain's `n` parameter

The model format accepts `n` a multiple of 128 up to 1024 (equal to the target chain's `n`) and the whole protocol (programs, Merkle, Freivalds, fraud proofs, GPU miner) is generic in `n`. **Verified** (`tests/bign.rs`, `tests/gpu.rs`): on chains n=256, n=512 and n=1024 (the GPU kernels: 256 and 512), the demonstration CNN runs, is paid, and a falsified GEMM is proven and slashed; the GPU kernel is bit-for-bit identical to the CPU at n=256 and n=512 with the safe parameters (rank 63, fold_k 64).

**Honest cost.** `n` is a genesis parameter: the whole chain has the same width (no mixed sizes). The cost of a mining attempt and the size of proof openings grow ∝ n, the weight matrix weighs n² bytes (512 → 256 KB, 1024 → 1 MiB), and the `MAX_OPS=16` / `MAX_WEIGHTS=8` limits per program remain (they were later raised, see docs/07 §12). A real vision network or an LLM therefore needed either a much larger `n` or a splitting by blocks *inside* a program (done afterwards: docs/12); kernel performance at n=512 had not been measured.
