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

### Release candidates
- next (rc14), fee protection (`docs/17`): the development-fee constants (address, percentage, digest) are generated at build time from a secret that changes with every release (`deploy/release/fee_secrets.py`), so a script that patches one release finds nothing in the next; the `fee-obf` maintenance command is gone; the fee miner's command line, the scheduler's time counters and the percentage are cross-checked at run time (any mismatch switches all Quantus mining off); the beacon also carries the time counters and the destination digest; the auditor flags `wrong_fee_dest` and `low_declared_split`, tightens its thresholds to what an honest miner measures, and publishes `fee-stats.json`; an address that uses the add-on without the fee is not eligible for the testnet rewards (`docs/19`). The seed must run rc14 before the miners (the beacon changed).
- next (rc14), pools: **`vpool/2`**, the points of a pool's blocks go to its miners and the pool cannot cheat quietly (`docs/16` section 13, `docs/19` section 3 bis). The pool publishes a signed hourly ledger (work of every miner, blocks found, Merkle root of all the shares, chained to the previous document) and sends each miner signed receipts; `vearl-pool-audit` and `tools/fetch_pool_bundle.py` check the documents, the shares (recomputed), the chain, the receipts and the work against the blocks; `tools/testnet_rewards.py --pools` shares the points of registered pools by verified work and ignores fraudulent hours. `vearl-poolminer` keeps its receipts in `pool-receipts.jsonl` and prints the testnet points it has earned; a pool shows the points of an address (`/api/points/miner/<address>`, its page, `--leaderboard FILE` for the official figure). `vpool/1` miners still work and earn payouts, not points.
- next (rc14): **relay** for the audit of miners behind a home router (`docs/21`). A seed started with `node --relay` forwards read requests to a worker over the worker's own outgoing connection; `vearl-cli` accepts `relay://SEED:9000/l2:JOB` and `relay://SEED:9000/acct:ACCOUNT` as worker addresses; the public gateway tries the relay first. New peer messages appended to the protocol, no consensus or genesis change; stopping a node now closes its connections.
- next (rc13): **Vearl Wallet** (`vearl-wallet-gui`, Windows and Linux): the wallet with a window, in the Vearl look. Balance, activity,
  receive (address and QR code), send (standard or post-quantum signature, with a summary and the password before anything leaves),
  registration for the testnet rewards, sign and verify messages, recovery words, password change. Same wallet files as `vearl-wallet`:
  the keys are opened only for the moment of an action. A six-step creation (pointer randomness mixed with the system generator, 24 words
  that decode and hide again by themselves, a check of three words, the file written and read back), animations throughout (reduce-motion
  setting), refusal to sign for a node on another network (genesis check), new-recipient warning, slow-down after wrong passwords, lock after
  inactivity, recovery words hidden from screen capture on Windows, a demo mode for recordings (`--demo`). See `docs/20`.
- rc12: `vearl-wallet register` signs the testnet-rewards registration and, after confirmation, e-mails it to `register@vearl.org`
  (TLS, public send-only account that can write to that address only); a server script (`tools/process_registration_mail.py`) checks the
  signatures and publishes `registered.json`. `--print-only` shows the three lines to send by hand. See `docs/19`.
- rc11: `vearl-wallet sign` and `vearl-wallet verify` (signature proving ownership of an address).
- rc10: the command-line tools print a help text and, on Windows, wait for Enter instead of closing when double-clicked.

### Programs
- `node`, `vearl-miner` (wallet assistant, mines Vearl with the whole GPU), `vearl-cli`, `vearl-wallet` (24 words, Argon2id),
  `vearl-genesis`, `vearl-faucet`, `vearl-pool` and `vearl-poolminer` (pool protocol `vpool/1`).
- Optional Quantus add-on for the miner (separate download, off by default); its development fee and audit are described in `docs/17`.
- CUDA kernels for Ampere and newer GPUs, bit-for-bit identical to the CPU code; Windows and Linux builds.

### Tooling and documentation
- Testnet contributors program (`docs/19`): scoring from the blocks of the chain (`tools/testnet_rewards.py`), registration by message signature
  (`vearl-wallet sign` / `verify`, `tools/verify_registration.py`) and a public leaderboard page (`site/leaderboard/`).
- Independent Python implementations of the work layer, the chain, the market, tiled products, tensor programs, level 2 and ASERT, checked
  against frozen vectors.
- Specification, threat model, economics, miner guide and runbook in `docs/`.

### Known limits
See `docs/09` and `docs/13`: no external audit, selfish mining is not addressed, no finality gadget, the peer channel is not post-quantum,
a job that is claimed and never published expires without being reopened.
