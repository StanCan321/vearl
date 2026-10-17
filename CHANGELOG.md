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
- **HiveOS package** (`deploy/hiveos/`, file `vearl-hiveos-0.1.0-rc16.2.tar.gz` attached to the rc16.2 release): a custom-miner archive for mining rigs (flight sheet, one-time wallet set-up with `setup.sh`, settings kept in `/hive-config/vearl`, speed and temperatures reported to HiveOS, `gpus=` and `max_jobs=` options). Same programs as the Linux release. Tested with a simulated Hive environment, not yet on a real HiveOS rig. See `docs/15` section 4 quinquies.
- rc16.2: **registration for the rewards by hand works.** The Copy button of the wallet window (Rewards page) gave the three lines without their labels (`address:`, `message:`, `signature:`), and the server script dropped such mails: someone whose network blocks the automatic e-mail (port 465) and who sent the lines by hand was not registered. The button now copies the labelled lines, and `tools/process_registration_mail.py` also accepts the three lines without labels (recognised by their shape; the signature is still verified) and a message cut into several lines by a mail program. Mails dropped before were processed again on the server. Programs other than the wallet window are the same as in rc16.1.
- rc16.1: **mining rigs: every GPU works.** Up to rc16 only the first GPU of a machine mined: the launcher started the Quantus miner with `--gpu-devices 1`, and the Vearl mining loop used GPU 0 only (reported on Discord: a rig of 4 GPUs, one at 100 %, three idle). Now `vearl-miner`, `node` and `vearl-poolminer --gpu` use every NVIDIA GPU of compute capability 8.0 or newer: one worker per GPU, the batches of sigmas of a block search are handed out in turn (a faster card does more of them, models can be mixed), a pool job's nonce range is shared in interleaved batches, and the Quantus miner is started without `--gpu-devices` (one worker per CUDA GPU). `CUDA_VISIBLE_DEVICES` restricts the GPUs for everything, `VEARL_GPUS` (CUDA indices) for Vearl mining only. The CUDA code keeps one context per device. No consensus or protocol change: rc16 and rc16.1 nodes interoperate, the genesis file and its hash do not change, and no seed upgrade is needed. Paid jobs still run on one GPU. Not yet measured on a real multi-GPU rig. See `docs/15` section 4 quater.
- next (rc16): **jobs find a worker more often.** Public level-2 jobs (Monte Carlo, rendering): a claim now lasts 30 minutes (15 x `serve_secs`); if the worker has not published by then, any other worker can claim the job again, and the first one loses its bond, instead of the job waiting for its deadline (consensus rule, `docs/10`; the genesis file and its hash do not change; every node must run rc16). Tensor-program jobs: `vearl-cli infer NODE WORKER1,WORKER2,...` deposits the model with several workers, which share the jobs (each takes at most 16 at a time, drops a claim that lost the race at once, and the client finds the worker that serves each result).
- next (rc15): **block time 10 s** (was 30 s), so that a verified job is paid in about 70 s instead of about 4 minutes (measured: 7 blocks per job, 225 s with 29-s blocks, 37 s with 5.5-s blocks). The emission per year does not change: `R0` is 6.66666666 VRL per block (20 VRL per 30 s, rounded down), `HALVING` 3,153,600 blocks (365 days), `TAIL` 0.16666666 VRL; `EPOCH_LEN` 300, `LWMA_N` 90, `SNAP_EVERY` 48 and the default pool confirmations 30 keep the same durations. The dispute window stays 300 s and 10 blocks, whichever is longer. The published genesis (hash `46c57f50…`) is unchanged; ASERT corrects its initial target (calibrated for 30 s) within the first hour. `vearl-cli` job deadlines are expressed in seconds. Test vectors regenerated and re-validated by the independent Python implementations; simulations of `docs/13` re-run for 10-s blocks.
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
