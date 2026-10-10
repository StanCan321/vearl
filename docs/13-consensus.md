# 13 — Consensus: rules, hardening and simulations

Labels: `[IMPL]` implemented and tested · `[SIM]` measured by simulation (a model, not a proof) · `[LIT]` known from the literature · `[HYP]` undemonstrated hypothesis. **Testnet, no external audit.**

## 1. Current consensus rules `[IMPL]`

| Rule | Detail | Where |
|---|---|---|
| Chain selection | greatest **cumulative work** (`block_work = 2^64 / (target+1)`), not the longest | `chain.rs`, `net.rs` |
| Difficulty | **ASERT** (aserti3-2d, half-life 900 s) by default for new genesis files (§5); LWMA-1 (window 30, times bounded to [1, 6T], change ×0.5–×2) remains available; integers only | `chain.rs::asert_core`, `lwma_next_target` |
| Timestamp | strictly increasing; local (non-consensus): ≤ clock + 60 s | `chain.rs` |
| **Block size** | **≤ 256 transactions** (`MAX_BLOCK_TXS`) **and ≤ 4 MiB of serialized transactions** (`MAX_BLOCK_TX_BYTES`): a block always fits in a P2P message (16 MiB) with its tile proof; 256 post-quantum transactions (≈ 2 MiB) pass, very heavy row-wise fraud proofs do not | `chain.rs`, Python mirrors |
| Anti-DoS filter | the announced jackpot must be under the target before any costly verification | `chain.rs` |

## 2. Reorganizations `[IMPL]`

**Before**: on a fork, the node re-downloaded and replayed the **peer's whole chain from genesis** (O(height) cost per fork, and a peer could announce fake work to trigger this cost).

**Now**:
0. **Synchronisation batches bounded in bytes** (8 MiB, at least one block): before, a batch of 64 heavy blocks could exceed the maximum message size and block synchronisation; the requester now continues until an **empty** batch instead of assuming that an incomplete batch marks the end (test: catching up 40 blocks served one by one).
1. **State snapshots** every 16 blocks (the last 64 + the genesis, `SNAP_EVERY`, `SNAP_KEEP`); `Chain::rewind_to(h)` restores the exact state at height `h` by replaying at most 15 blocks (test: state identical to a rebuilt chain).
2. **Search for the common ancestor** (`SyncMode::Locate`): we step back by 8, 16, 32… blocks until a batch connects to our chain, then we download **only the diverging branch** (network test: a 10-block fork resolved without replaying the genesis).
3. **Atomic reorganization** (`Chain::try_reorg`): we replace `fork_h..` with the candidate branch; if a block is invalid, if the final work is insufficient or if MESS refuses, **the original chain is restored** (tests: state, tip and supply invariant unchanged). An invalid block gets the peer banned; insufficient work is not a fault.

## 3. MESS: braking of deep reorganizations `[IMPL, local rule, OPTIONAL]` `[HYP]`

Inspired by Ethereum Classic's **ECIP-1100 (MESS)**, adopted after 51 % attacks on a low-hashrate chain [LIT]. **Disabled by default**; activation: `node --mess` (or `VEARL_MESS=1`).

A competing chain is only adopted if its work since the common ancestor exceeds `f(t)` times that of ours, where `t` = time elapsed **on our chain** since that ancestor; `f` goes from 1 to 31 (cubic smoothing, very close to ECIP-1100's cosine curve) between `t = 0` and `t = 25,132 s ≈ 7 h` (ECIP-1100's scale and ceiling). Integers only (`mess_factor_milli`). Examples: 3 min → +0.5 % of work required, 10 min → +5 %, 4 h → ×19.

**Why optional — measured.** With a first, more aggressive setting (ceiling reached in 2.1 h), `tools/fork_test.py` showed that two groups of **equal power** separated for only 90 s (≈ 160 fast blocks in this test) **no longer re-merged**: each demanded about 5 % more work from the other, which neither obtained. With ECIP-1100's scale the same partition re-merges (13 s, verdict OK, with `VEARL_MESS=1`); without MESS, 3 s. But a partition of **several tens of minutes** between comparable halves would still not heal on its own. On a young network, with few nodes and probable outages, **liveness** comes first: MESS is an operator's choice, to be enabled when the network is wide enough for long partitions to be improbable.

**What it brings when enabled**: an attacker who rents power to redo a private branch must, once our chain has been settled for a long time, have `f(t)` times more work than the honest miners, not just more (§4 A).

**What it costs**: a subjective rule (two nodes can diverge if each holds a branch for a long time); a long partition requires manual recovery (restart without `--mess`, or choose a branch); no protection against selfish mining or against a short attack.

## 4. Simulations `[SIM]` — `python3 tools/consensus_sim.py all` (fixed seed)

Model: exponential solve times, zero latency, blocks of equal work. **These figures bound orders of magnitude, they do not prove security.**

### A. Private branch, 4 h horizon — probability of replacing ≥ d honest blocks

| q (attacker's share) | d ≥ 6, plain work → MESS | d ≥ 30 | d ≥ 100 |
|---|---|---|---|
| 0.30 | 0.090 → 0.087 | 0.003 → 0 | 0 → 0 |
| 0.40 | 0.385 → 0.349 | 0.100 → 0.001 | 0.001 → 0 |
| 0.45 | 0.664 → 0.542 | 0.405 → **0.004** | 0.146 → **0** |
| 0.55 | 1.000 → 0.871 | 0.998 → **0.092** | 0.997 → **0** |
| 0.70 | 1.000 → 0.999 | 1.000 → 0.778 | 1.000 → **0** |

(ECIP-1100 parameters. The first, more aggressive setting also blocked reorganizations of 30 blocks at q = 0.70, at the price of the liveness measured above.)

Reading: MESS does not remove the short attack (the factor stays close to 1 early on), it weakens it; it **forbids the deep reorganization** as long as `q/(1−q) < f(t)`. The `d ≥ 100`, q = 0.90 cells are a horizon artefact (fewer than 100 honest blocks in 4 h).

### B. On/off miner against LWMA-1
A reactive rule ("I bring power when the target is easy") barely disturbs the block time (**+1 %**). A **periodic** miner (30 active blocks, 60 inactive) lengthens it by **+5 % (X = 2), +11 % (X = 5), +27 % (X = 20)**: LWMA-30 reacts with a delay [LIT]. **Fixed by the switch to ASERT (§5)**: same scenario, −0.1 %.

### C. Timestamps pushed to +60 s
Target drift of the order of ±10 % (standard deviation ≈ 0.2 over 4,000 blocks), real block time ≈ 30.3–30.4 s for q from 0.10 to 0.55: the floor (strict growth, bounded solve time) and the ceiling (+60 s) limit the effect.

### D. Selfish mining (Eyal–Sirer)
Attacker revenue (γ = 0): 0.274 at q = 0.30 (fair), 0.327 at 0.33, **0.387 at 0.36**, 0.484 at 0.40; with γ = 0.5 the gain appears from about 0.25. **VEARL does not prevent it** (like any Nakamoto chain); MESS does not change this.

## 5. ASERT difficulty rule `[IMPL]` `[LIT]` `[SIM]`

**Why.** LWMA-1 lets itself be slowed by a periodic miner (§4 B: +28 % at X = 20). Bitcoin Cash experienced exactly this problem (power oscillations between chains) and solved it in November 2020 with **ASERT** (*absolutely scheduled exponentially rising targets*, aserti3-2d variant) [LIT]: the target only depends on the gap between real time and the ideal schedule since an anchor block, `target = anchor · 2^((Δt − T·(Δh+1)) / half_life)`, with no sliding window — so there is no "memory" that an intermittent miner could exploit.

**What is implemented.** `chain.rs::asert_core` reproduces **exactly** the integer formula of the specification (16.16 exponent truncated toward zero, cubic polynomial for `2^x`, shifts), then VEARL's bounds `[2^16, 2^64 − 1]`. Anchor = the **first block** (its target, parent assumed on time): the schedule starts from the first mined block, not from the genesis date. Half-life **900 s = 30 blocks**. `Retarget::Asert` is the default rule (`vearl-genesis --retarget Asert|Lwma|Ema|Fixed`); existing chains keep their rule (it is in the genesis hash, the new variant is added at the end of the enumeration).

**How it is verified.** (1) `tools/asert_ref.py`: a Python implementation written from the published specification, **29/29 targets identical to the official vectors** of the BCHN repository (`qa-assets/test_vectors/aserti3-2d`, series 1, 3, 5 and 8: limit target, minimum target, half-life jumps, pseudo-random times); (2) the same implementation against the Rust node: **50,000/50,000** identical targets on random inputs (including overflows and floors), replayed by `tools/audit_all.py`; (3) the Python light client (`tools/light_client.py`) re-validated a real 85-block ASERT chain served by a node; (4) Rust tests: stable target on time, division/multiplication by 2 after a half-life ahead/behind, bounds, replay after rollback.

**Choice of the half-life** (`python3 tools/consensus_sim.py daa`, 4 × 3,000 blocks, same model as §4):

| rule | constant power | periodic X = 5 | periodic X = 20 | 100 blocks after a ×10 jump | 100 blocks after a ÷10 jump |
|---|---|---|---|---|---|
| LWMA-1 (N = 30) | 30.3 s (+1.0 %) | 33.1 s (+10.5 %) | 38.3 s (**+27.8 %**) | 20.7 s | 58.6 s |
| ASERT 600 s | 30.0 s | 30.0 s | 30.0 s (−0.1 %) | 11.4 s | 52.2 s |
| **ASERT 900 s (chosen)** | 30.0 s | 30.0 s | 30.0 s (−0.1 %) | 7.8 s | 62.1 s |
| ASERT 1,800 s | 30.0 s | 30.0 s | 30.0 s | 4.7 s | 86.9 s |
| ASERT 3,600 s | 30.0 s | 29.9 s | 29.8 s | 3.7 s | 120.9 s |

**Accepted trade-off.** ASERT neutralises the periodic miner whatever the half-life; its counterpart is the reaction to a **lasting** power jump: after a ×10, the next 100 blocks come out at 7.8 s on average (LWMA: 20.7 s), i.e. a temporarily faster emission — which the absolute schedule then **catches up** (LWMA does not). 900 s keeps a reaction to a drop close to LWMA's (62 s against 59 s). A timestamp advance of `MAX_FUTURE = 60 s` only makes the next block 2^(60/900) ≈ 4.7 % easier, and the gap is caught up at the next block (the schedule is absolute). Simple simulation model (exponential times, zero latency): orders of magnitude, not proof.

## 6. What is NOT done `[HYP]`
- **No finality** (no BFT-type gadget): reorganizations remain possible, only braked.
- Header v2 added the roots (`a_root`, `b_root`, `accounts_root`) that give a light client balance proofs (see docs/14); absence proofs and the choice between several peers are still missing.
- No compensation of fees/rewards between branches, no mempool by fee priority.
- The fairness of the NoisyGEMM lottery and the security of the low-rank noise are not formally proven (docs/03, docs/02 §4).
- None of these rules has been reviewed by a third party.
