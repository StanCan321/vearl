# Vearl

[![CI](https://github.com/vearl-verified-compute/vearl/actions/workflows/ci.yml/badge.svg)](https://github.com/vearl-verified-compute/vearl/actions/workflows/ci.yml) · [Latest release](https://github.com/vearl-verified-compute/vearl/releases/latest) · [Website](https://vearl.org) · [Security policy](SECURITY.md)

**A proof-of-useful-work blockchain for GPUs.** Miners do not burn energy on pointless hashes: the work that secures the chain is an
int8 matrix product that can be the product a paying client asked for (AI inference, rendering, linear algebra). Everyone can check it at
almost no cost, so a result is audited before any miner is paid.

> **Testnet, not audited, no value.** The public testnet starts on **October 23, 2026 at 16:00 UTC**. Its tokens are worth nothing, the
> network can be reset, and no part of this code has been reviewed by a third party. Do not put real funds on it.
> Website: <https://vearl.org> · Explorer: <https://vearl.org/explorer/> · X: [@vearl_org](https://x.com/vearl_org)

## How it works

- **The work.** Each block is won by a lottery over int8 tile products (NoisyGEMM-T, `docs/02` and `docs/07`). A tile is checked by
  recomputing it; a whole product by a Merkle opening and a Freivalds test; a wrong entry by a fraud proof of one cell.
- **The market.** A client locks an escrow and posts a job. A worker bids, claims it, publishes the result with a bond. The client or
  anyone can prove a mistake and the bond is forfeited. Data availability is enforced on chain. (`docs/07`, `docs/10`)
- **Verifiable AI.** Quantized networks (MLPs, CNNs, ResNet-20, transformers, MiniLM, DeiT-Tiny, a small language model) are expressed
  as programs of verifiable operations. Each runs on the chain and is checked bit for bit against an independent engine. (`docs/12`)
- **Two implementations.** The consensus is also implemented independently in Python (`vearl-core/tools/`) and cross-checked against
  frozen test vectors. (`docs/09`)

Claims in the documents are labelled `[PROVEN]`, `[HYPOTHESIS]`, `[RESEARCH]` or `[AUDIT ACTION]`. Nothing here claims "no possible
attack", "anonymity" or "ASIC resistance".

## Repository layout

| path | content |
|---|---|
| `vearl-core/src/` | the Rust implementation: consensus (`work`, `chain`, `state`), verifiable compute (`mlp`, `tiled`, `prog`, `l2`, `render2`, `render3`), network (`net`, `secure`, `api`, `pool`), wallet, miner (`dual`, `feeguard`) |
| `vearl-core/src/bin/` | the programs: `node`, `vearl-miner`, `vearl-cli`, `vearl-wallet`, `vearl-genesis`, `vearl-faucet`, `vearl-pool`, `vearl-poolminer`, the test-vector generators |
| `vearl-core/gpu/` | CUDA kernels (tensor cores, batched mining, rendering), bit-for-bit identical to the CPU code |
| `vearl-core/tests/` | integration tests and the frozen consensus vectors in `tests/vectors/` |
| `vearl-core/tools/` | independent Python implementations, model quantization and training scripts, demos, endurance tests |
| `vearl-core/models/` | small demonstration models (see its README) |
| `docs/` | specification, design, threat model, economics, guides: see the table below |
| `site/` | the website, the explorer, the job page and the faucet page (static files) |
| `deploy/` | install scripts, systemd units, nginx configuration, genesis, release packaging, job gateway, monitoring |
| `proto0_econ/`, `proto1_gpu/` | the first economic simulation and GPU benchmarks that led to the design |
| `brand/` | logo, mark and banners |

## Build and test

```bash
cd vearl-core
cargo build --release                       # CPU build (add --features gpu with the CUDA toolkit installed: docs/15)
cargo test --release                        # about 200 tests: protocol, market, network, fuzzing, frozen vectors
cargo run --release --bin devnet            # a complete local demonstration (nodes, market, fraud proof)
```

More demos (inference, rendering, linear algebra, endurance, chain split and merge) are in `docs/18-demos-and-models.md`.

## Run a miner or a node

**Download:** ready-to-run Windows and Linux builds (miner, node, wallet, command-line tools) are attached to each
[release](https://github.com/vearl-verified-compute/vearl/releases), with their SHA-256 checksums. You need an NVIDIA GPU of the Ampere
generation or newer (RTX 30xx and later) and the NVIDIA driver; without a GPU the miner runs on the CPU, slowly. To build from source
instead, see the section above.

`vearl-miner` is the one program a miner opens: it creates a wallet, joins the testnet and mines Vearl with the whole GPU. An optional
add-on lets the same GPU also mine Quantus (QTC); it is off unless you install it and say yes (`docs/15` section 4 bis, `docs/17`).
Running a testnet from scratch (genesis, seed node, faucet, explorer) is in `docs/06`; pools are in `docs/16`.

## Documentation

| document | content |
|---|---|
| `docs/00`, `docs/01` | the candidate architecture and the state of the art (Pearl) |
| `docs/02` | **design v1** (authoritative): work, market, security, roadmap, GPU performance |
| `docs/03` | attack analysis of the work (a low-rank flaw was found and fixed) |
| `docs/04`, `docs/05` | economics and difficulty retargeting; compute types and what is verifiable |
| `docs/06` | **running the testnet**: deployment, wallet, faucet, launch checklist |
| `docs/07` | **normative specification**: work layer, chain, market, byte by byte, frozen vectors |
| `docs/08` | P2P network specification and chain selection |
| `docs/09` | **threat model and audit file**: covered, partial and open items with reproducible evidence |
| `docs/10` | level 2: Monte Carlo and 3D rendering, audits by sampling |
| `docs/11`, `docs/12` | measured verifiable inference; embeddings, serial inference and large-scale linear algebra |
| `docs/13`, `docs/14` | consensus (reorganizations, ASERT, attack simulations); chain cryptography and light clients |
| `docs/15`, `docs/16`, `docs/17` | miner guide; pool protocol; protection of the development fee of the optional add-on |
| `docs/18` | demos, models and tools |
| `docs/19` | **testnet contributors program**: how contributions are counted and what a contributor must do (a plan, not a promise) |

## Security, contributing, license

- Report vulnerabilities privately to security@vearl.org (`SECURITY.md`).
- Contributions are welcome: see `CONTRIBUTING.md` (everything in English, tests with every change, consensus changes need a discussion first).
- License: Apache License 2.0 (`LICENSE`, `NOTICE`). Third-party models, data and assets keep their own terms: `THIRD-PARTY-NOTICES.md`.
