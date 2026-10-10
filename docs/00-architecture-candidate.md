# VEARL — architecture file (v0, draft)

> **Update:** this document is the starting point. The research on Pearl is in `01-state-of-the-art-pearl.md`, and the design that was kept (Fee-Bound Useful Work, implemented and tested) is in `02-design-vearl-v1.md`. In case of contradiction, **02 is authoritative**.

Status: **working draft**. Every claim is labelled:
`[LIT]` known in the literature · `[MEASURED]` measured by our prototypes · `[HYP]` hypothesis · `[EXP]` experimental · `[RESEARCH NEEDED]` unresolved.
No source below has been checked to the letter yet: references marked "to be verified" must be re-read before any citation in a whitepaper.

## 1. Central question and unsolved problem

A PoW consensus requires work that is (a) costly, (b) not precomputable, (c) verifiable in O(small), and (d) **free of any exploitable structure** that would allow cheating. A useful client computation is usually the opposite: structured, chosen by a third party, sometimes predictable. Hence:

- **The same computation cannot be both chosen by a client and serve as a safe proof of work**, except through a dedicated cryptographic construction. `[RESEARCH NEEDED]`
- Lead `[LIT]`, to be verified: proofs of useful work built from noisy matrix multiplications (2025 work of the "Proofs of Useful Work from matrix multiplication" type; the Pearl/cuPOW project mentioned in the brief seems to belong to this family). To be read in the primary source before relying on it.

## 2. Candidate architecture: two separate paths (the simplest that holds)

**Path 1 — Native Work (consensus security).** Classic GPU PoW, with a hard kernel to specialize (memory bandwidth + GEMM), not useful in itself. It is the security base even when the market is empty. `[HYP]` that this is enough for security at launch.

**Path 2 — Compute Market (usefulness).** A market of GPU jobs with *separate settlement*: escrow deposit → execution → verification → payment. Verification uses a method adapted to the job type (see §4). The fee token is the same, but **consensus does not depend on path 2**.

**Convergence path (research)**: gradually replace Native Work by useful noisy GEMM. Only after a formal demonstration. `[RESEARCH NEEDED]`

Why this choice: the P0 simulation shows (§3) that in a time-sharing model clients do not contribute to security; decoupling the two avoids tying consensus safety to an uncertain demand.

## 3. What our prototypes established

### P1 — real GPU (NVIDIA A10, OVH VPS) `[MEASURED]`
| metric | value |
|---|---|
| Copy bandwidth | 473 GB/s (79 % of the theoretical 600) |
| FP32 FMA | 28.8 TFLOPS |
| INT32 xorshift | ~11.9 Tops/s |
| SGEMM FP32 8192³ | 13.8 TFLOPS |
| FP16 tensor-core GEMM | 73.4 TFLOPS |

Consequence: on this GPU, tensor cores are ~5× faster than FP32 SGEMM. A consensus kernel based on FP16 GEMM favours recent GPUs; **hardware centralization risk** (and a potential ASIC advantage on a frozen GEMM kernel). `[HYP]` to be tested on other GPUs.

### P0 — economic simulation (`proto0_econ/sim.py`) `[HYP]` (arbitrary parameters)
- Security (cost of a 51 % attack) ≈ emission × token price, **independent of client demand** under time-sharing.
- With decreasing emission + 1 % tail emission, the attack cost falls by a factor of ~15 over 8 years if the token price does not compensate.
- Empty market: the system stays viable only thanks to emission → the token must have a value for security to exist.
- At no point does the model guarantee miner profitability; never promise it.

## 4. Verification by job type (plan)

| job | method kept (v0) | verification cost | status |
|---|---|---|---|
| GEMM / linear algebra | **Freivalds** test (probabilistic, O(n²) vs O(n³)) `[LIT]` | low | to prototype (P2) |
| Deterministic AI inference | k-of-n replication + spot-check, Merkle commitments of activations | k× the computation | `[EXP]` GPU non-determinism to handle |
| Rendering / Monte Carlo | replication + committed seeds | k× | `[EXP]` |
| ZK proofs | native verification of the proof `[LIT]` | low | depends on the proof system |
| Everything else | optimistic + fraud proofs (challenge) | low if honest | complex, outside v1 |

Limit: Freivalds proves that **C = A·B** is correct (with error probability ≤ 2⁻ᵏ), not that *a given GPU* computed it. Nothing in this file proves "this GPU computed X" without an additional assumption (replication, TEE, or a complete ZK proof).

## 5. What is decided / undecided

Decided (v0): two separate paths; escrow market; no arbitrary client code (jobs = standardized types in a sandboxed runtime); **audited-only** crypto primitives, never home-made ones.

Undecided: UTXO vs accounts model; private layer (Zcash-like/Halo2 `[LIT]`, or CryptoNote-like `[LIT]`); stack (Rust recommended for the node, CUDA for the kernels, libp2p/QUIC `[HYP]`); final supply and emission curve (the P0 values are **provisional**).

## 6. Roadmap

| proto | content | status |
|---|---|---|
| P0 | economic simulation | done (v0) |
| P1 | GPU benchmark | done (A10) |
| P2 | local runtime + Freivalds on GEMM | **next** |
| P3 | job distribution (2 nodes) | to do |
| P4 | proof system (choice after reading the literature) | to do |
| P5+ | testnet, audits, public testnet; mainnet **only** if the properties are demonstrated | to do |

## 7. Open questions for the project owner
1. Final name/ticker? (VEARL = provisional; check trademark/ticker collisions.)
2. Do you want privacy from the testnet on (high cost) or in phase 2?
3. Target GPUs: NVIDIA only at first, or multi-vendor (ROCm) from P1?
4. Legal framework: issuing/selling a token is regulated depending on the jurisdiction; to be handled before any public launch.
