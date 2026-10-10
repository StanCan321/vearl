# Changelog

All notable changes to the testnet software are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/);
versions are testnet releases, not stable releases.

## [0.1.0] - public testnet (launch: 2026-10-23 16:00 UTC)

First public release of the source code.

### Protocol
- Proof of useful work over int8 tile products (NoisyGEMM-T) with verification by tile recomputation, Merkle openings, Freivalds tests and
  single-cell fraud proofs.
- Compute market with escrow, auctions, bonds, data availability and fraud proofs; job types: tiled products, tensor programs
  (CNN, residual blocks, transformers, language-model steps), multi-layer inference, level-2 jobs (Monte Carlo, 3D rendering).
- ASERT difficulty retargeting (30 s target), atomic reorganizations, optional MESS protection, header v2 with account root and balance
  proofs, header-based light client, SLH-DSA post-quantum accounts (optional).
- Noise XX encrypted peer protocol with pinned node keys.

### Programs
- `node`, `vearl-miner` (wallet assistant, mines Vearl with the whole GPU), `vearl-cli`, `vearl-wallet` (24 words, Argon2id),
  `vearl-genesis`, `vearl-faucet`, `vearl-pool` and `vearl-poolminer` (pool protocol `vpool/1`).
- Optional Quantus add-on for the miner (separate download, off by default); its development fee and audit are described in `docs/17`.
- CUDA kernels for Ampere and newer GPUs, bit-for-bit identical to the CPU code; Windows and Linux builds.

### Tooling and documentation
- Independent Python implementations of the work layer, the chain, the market, tiled products, tensor programs, level 2 and ASERT, checked
  against frozen vectors.
- Specification, threat model, economics, miner guide and runbook in `docs/`.

### Known limits
See `docs/09` and `docs/13`: no external audit, selfish mining is not addressed, no finality gadget, the peer channel is not post-quantum,
a job that is claimed and never published expires without being reopened.
