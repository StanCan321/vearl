# 07 — Normative specification of the "work" layer (v1, block header v2)

**Status: testnet prototype, not audited.** This document describes, byte by byte, what a second implementer must reproduce to produce and validate the same blocks as the reference node (Rust). It covers the **work layer** (matrices, noise, attempt, jackpot, target), the **identifiers** (sigma, header) and the **Merkle commitments**. §10 adds the **Native + transfers block chain** (validation, rewards, LWMA-1). §11 adds the **compute market** (jobs, auctions, bonds, useful-work blocks, data, fraud). The P2P network and chain selection are in `docs/08`. §12 adds the **tensor programs** (CNN, transformer, 4 fraud-proof types). §13 adds the **tiled matrix products** (`TiledTx`, large-scale linear algebra). Not covered: the historical multi-layer inference jobs (`MlpTx`, **obsolete**: replaced by the programs of §12). Consensus hardening (transaction cap, snapshots, reorganizations, MESS braking) is in `docs/13`; post-quantum signatures and the accounts root are in `docs/14`.

**How it is verified.** (replay: `vearl-vectors > v.json && python3 tools/spec_ref.py v.json`, prerequisite `pip install blake3 numpy`) `vearl-vectors` produces test vectors (`vearl-core/tests/vectors/consensus-work-v1.json`, frozen: the `vectors_gate` test fails if the code deviates). `vearl-core/tools/spec_ref.py` is an implementation **written from this text**, without reading the Rust code: it reproduces the 40 checks (Merkle, native matrices, noise, C′, jackpots for 3 parameter sets, decoding, class encodings, sigma, header, target). Limit: the same person wrote the code and this specification; a truly independent implementer remains to be found `[AUDIT ACTION]`.

Convention: `||` = concatenation; integers are **little-endian** (LE) unless stated; `u32`/`u64` unsigned; "bytes" = a raw byte string.

## 1. Primitives

The only hash primitive: **BLAKE3** (32 bytes). No home-made primitive.

- `blake3(m)`: plain hash of `m`.
- `H(domain, p1, p2, …)`: `blake3` in **derive_key** mode with the context string `domain` (UTF-8), applied to `p1 || p2 || …`. 32-byte output.
- `XOF(domain, m, L)`: same derive_key mode, extended output (XOF) of `L` bytes for the message `m`.
- Signatures: Ed25519 (strict verification). Outside this work specification.

## 2. Structure encoding (`bincode` 1.x, default options)

Fixed-size integers in LE; `[u8; 32]` arrays with no length prefix; **enumerations: `u32` LE tag** then the fields in declaration order.

`WorkClass`:

| variant | tag | fields |
|---|---|---|
| `Native` | 0 | — |
| `Job(id)` | 1 | `id`: 32 bytes |
| `Layer{job, layer}` | 2 | `job` 32 B, `layer` `u8` |
| `Product{job, i, j, k}` | 3 | `job` 32 B, `i`,`j`,`k` `u8` |
| `Op{job, op}` | 4 | `job` 32 B, `op` `u8` |

`Header` **v2** (in this order; see `docs/13` for the why and `docs/14` for the cryptography): `version u32` (=2), `height u64`, `prev [32]`, `timestamp u64`, `miner [32]` (Ed25519 **or** post-quantum public key, depending on the length of the transactions' signature, `docs/14`), `class WorkClass`, `extra_nonce u64`, `ti u32`, `tj u32`, `jackpot [32]`, `tx_root [32]`, `target u64`, `a_root [32]`, `b_root [32]`, `accounts_root [32]`.

- `hash(header) = H("vearl/header/v1", bincode(header))` (the domain stays `v1`: only the header's content changes).
- `sigma = H("vearl/sigma/v1", prev, miner, bincode(class), extra_nonce (u64 LE), tx_root, timestamp (u64 LE))`.
- `tx_root`: `[0; 32]` if there is no transaction; otherwise the Merkle root (§3) whose leaf `i` is `leaf_hash(i, id_tx_i)`. A block cannot exceed `MAX_BLOCK_TXS = 256` transactions (consensus rule, `docs/13`).
- **`a_root`, `b_root`** (new): Merkle roots by rows of `A` and by columns of `B` **of this block's work** (native matrices of the current epoch, or the roots committed by the job, depending on the class). A client that knows these two roots can verify the tile proof (`build_proof`/`verify_tile_proof`) **without holding the complete state** — this is the only addition that serves a light client on the work itself.
- **`accounts_root`** (new): Merkle root of the accounts **after** applying the block's transactions and the reward (§4 quater). It serves to prove a balance through a Merkle path, without replaying the chain.

### 4 quater. Accounts root (`accounts_root`)

Leaves, in increasing public-key order, keeping only **live** accounts (non-zero balance or nonce):
- leaf `0`: `leaf_hash(0, "count" || number_of_live_accounts (u64 LE))`.
- leaf `i+1` (for the `i`-th live account, `i` starting at 0): `leaf_hash(i+1, public_key || balance (u64 LE) || nonce (u64 LE))`.

`accounts_root` is the Merkle root (§3) of these leaves. An **account proof** is `(index i, number of accounts N, path of length ⌈log2(N+1)⌉)`; it proves `(key, balance, nonce)` against a published root, exactly like a matrix row or column proof.

## 3. Merkle tree

- `leaf_hash(i, data) = H("vearl/merkle/leaf/v1", i (u64 LE), data)`.
- Node: `H("vearl/merkle/node/v1", left, right)`.
- The number of leaves is padded with **zero** 32-byte leaves up to the next power of 2 (at least 1). Depth `d = ⌈log2(number of leaves)⌉`; a leaf's path lists the siblings from bottom to top. Verification requires `len(path) = d` and `index < 2^d`.
- Commitment of a matrix `M` (n×n, signed 8-bit integers, row-major storage): **by rows** — leaf `i` = `leaf_hash(i, row_i)` — or **by columns** — leaf `j` = `leaf_hash(j, column_j)`; values are written as bytes (two's complement).
- In a job: `a_root` = root by rows of `A`, `b_root` = root by columns of `B`.
- **Integer matrix** `M` (n×n, 32-bit integers: result `C` of a job, `Z` of a program GEMM, tiled products and sums) — **commitment v2**: root **by rows**, leaf `i` = `leaf_hash(i, row_i)` where the row is written as little-endian `i32` (4n bytes), depth `⌈log2 n⌉`. An **opening** of an entry `(i, j)` gives the **whole row `i`** and its path; the value is `row[j]`. (v1 committed one leaf per entry, `leaf_hash(i·n + j, value)`, depth `⌈log2 n²⌉`: n² hash calls instead of n, i.e. ≈ 100× slower at n = 1024 — 396 ms against 4 ms on one core; measured, `docs/12` §4 quater.)

## 4. Parameters and native matrices

`WorkParams = {n, tile, rank, fold_k}`. Validity rules (enforced at genesis; from the attack analysis `docs/03`):

1. `tile = 16`; 2. `n` a multiple of 128; 3. `1 ≤ rank ≤ 63`; 4. `fold_k` a multiple of 32, dividing `n`; 5. `fold_k ≤ 1.05 · rank` (otherwise low-rank inputs make an attempt up to `fold_k/rank` times cheaper: flaw fixed, see `docs/03`). Testnet: `{128, 16, 63, 64}`.

Entries satisfy `|x| ≤ 64` (signed 8-bit integers).

Native matrices ("Native Work", no usefulness claimed) for epoch `e`: `bytes = XOF("vearl/native/v1", e (u64 LE), 2·n²)`; `A[i][j] = (bytes[i·n + j] mod 129) − 64`; `B[i][j] = (bytes[n² + i·n + j] mod 129) − 64`.

## 5. Low-rank noise

`stream = XOF("vearl/noise/v1", sigma, 4·n·rank)`, each byte `b → (b mod 3) − 1 ∈ {−1, 0, 1}`. Consecutive split:

- `EL`: `n × rank`, `EL[i][t] = stream[i·rank + t]`
- `ER`: `rank × n`, `ER[t][j] = stream[n·rank + t·n + j]`
- `FL`: `n × rank`, `FL[k][t] = stream[2·n·rank + k·rank + t]`
- `FR`: `rank × n`, `FR[t][j] = stream[3·n·rank + t·n + j]`

`A′ = A + EL·ER`, `B′ = B + FL·FR`, in exact integer arithmetic (the coefficients of `A′`, `B′` fit in signed 8 bits because `|A| ≤ 64` and `|EL·ER| ≤ rank ≤ 63`).

## 6. Attempt (one per 16×16 tile) and jackpot

For tile `(ti, tj)` (`0 ≤ ti, tj < n/16`), rows `x ∈ [0,16)`, columns `y ∈ [0,16)`:

```
acc[x][y] = 0   (i32, arithmetic modulo 2^32)
fold[0..64] = 0 (u32)
for s from 0 to n/fold_k − 1:
    k0 = s·fold_k
    acc[x][y] += Σ_{k=k0}^{k0+fold_k−1} A′[16·ti + x][k] · B′[k][16·tj + y]      (modulo 2^32)
    rot = (s mod 31) + 1
    for i = 16·x + y from 0 to 255:
        fold[i mod 64] ^= rotl32( u32(acc[x][y]), rot )
jackpot = blake3( "vearl/jackpot/v2" (16 bytes) || sigma || ti (u32 LE) || tj (u32 LE) || fold[0..64] (each word u32 LE) )
```

(plain `blake3`, not derive_key: a 312-byte message.) The attempt **wins** if `meets_target(jackpot, target)`:
the **big-endian** `u64` of the first 8 bytes of the jackpot `< target`.

A valid block contains in its header `(ti, tj, jackpot)` and a proof: the 16 rows of `A` (indices `16·ti…`) and the 16 columns of `B` (indices `16·tj…`) with their Merkle paths against the `a_root`/`b_root` of the work class; the validator derives the noise from `sigma`, recomputes **a single tile** (§6) and checks equality of the jackpot. Cost: O(16²·n) instead of O(n³).

## 7. Decoding (useful result)

The miner obtains `C′ = A′·B′` (n×n, i32). The useful product is:
`C = C′ − A·(FL·FR) − (EL·ER)·B − (EL·ER)·(FL·FR)`, modulo 2^32, i.e. exactly `A·B`. Overhead cost O(n²·rank).

## 8. What this specification does not say (to do)

Multi-layer and tiled inference jobs (`docs/05`; §11 covers simple products and §12 the tensor programs), reorganizations tested by a second implementation (the network and chain selection are described in `docs/08`). Each requires the same treatment: normative text + frozen vectors + an independent implementation. `[AUDIT ACTION]`

## 9. Security assumptions and limits (honest reminder)

- `[HYPOTHESIS]`: the XOR-rotation folding of the transcript has no formal analysis; the argument is empirical (`docs/03`, `attackreport`).
- `[PROVEN BY COMPUTATION]` only: the correctness of the decoding, CPU/GPU bit-for-bit determinism, the verification of a tile.
- The specification does not claim that the work is "ASIC resistant" nor that no attack exists.

## 10. Chain: blocks, transfers, rewards, difficulty (v1, Native Work)

Scope: a chain of **Native work blocks** carrying **transfers**. The other work classes (jobs, layers, products, operations) and the market transactions follow the same block rules but add state rules not yet specified (§8).

### 10.1 Genesis and constants

Initial state: balances = genesis allocation, nonces = 0. `prev` of block 0 = 32 zero bytes; "tip timestamp" before block 0 = genesis timestamp; initial target = genesis parameter. v1 constants: `BLOCK_TIME = 10` s, `EPOCH_LEN = 300`, `R0 = 666,666,666` units (1 VRL = 10⁸; 20 VRL per 30 s, rounded down), `HALVING = 3,153,600` blocks (365 days), `TAIL = 16,666,666` units, `LWMA_N = 90`, `MAX_FUTURE = 60` s. (Until 2026-10-17 the block time was 30 s, `EPOCH_LEN` 100, `R0` 20 VRL, `HALVING` 1,051,200, `TAIL` 0.5 VRL and `LWMA_N` 30: the same durations and the same emission per year.)

### 10.2 Transfer transaction

`Tx = from[32] || nonce u64 || fee u64 || TxKind`; `Transfer` = `u32` tag **0**, `to[32]`, `amount u64`. `SignedTx = Tx || sig` where `sig` is a byte vector encoded as `length u64 (=64) || 64 bytes`. The Ed25519 signature covers `bincode(Tx)` (key `from`). `id(tx) = H("vearl/tx/v1", bincode(SignedTx))`. Application, in order: check the signature; require `nonce = account nonce`, then increment it; debit `fee` (sufficient balance) and credit it to the **block's miner**; debit `amount` (sufficient balance) and credit it to `to`. Any error invalidates **the whole block**.

### 10.3 Validation of a block (order of the checks)

1. `version = 1`; 2. `height` = number of previous blocks; 3. `prev` = hash of the previous header; 4. `timestamp` **strictly** greater than the parent's; 5. `target` = expected target (§10.5); 6. announced `jackpot` `< target` (anti-DoS filter); 7. `tx_root` = root of the transactions (§2); 8. **work**: for the `Native` class, `A`,`B` = native matrices of epoch `height / EPOCH_LEN` (§4); recompute `sigma` (§2), the tile `(ti, tj)` (§6) and require equality with `jackpot`, then `jackpot < target`; 9. apply the transactions in order (§10.2); 10. pay the reward (§10.4). A rule outside consensus but enforced by nodes: refuse a timestamp `> local clock + MAX_FUTURE`.

### 10.4 Reward and monetary supply

`base_reward(h) = max(R0 >> min(h / HALVING, 63), TAIL)`, credited to the miner **after** the block's transactions, and counted in the emission (`minted`). (Testnet schedule. The mainnet plan removes the tail, `TAIL = 0`, which caps the issuance at 42,048,000 VRL: docs/04 section 4.) For useful-work classes (Job, Op, …) a *rebate* bounded by the amount burned at job creation is added (§8). Tested invariant: `Σ balances + escrows + bonds + burned = genesis + minted`.

### 10.5 Difficulty (target), cumulative work, chain selection

After each applied block, with `E = BLOCK_TIME` and `dt = block timestamp − parent timestamp`:

- EMA (fallback): `ema = target × (15E + clamp(dt, E/2, 2E)) / (16E)`, bounded to `[2¹⁶, 2⁶⁴−1]` (integer division).
- **LWMA-1** (production rule): if the number of blocks `< N+1 = 91`, the next target is `ema`. Otherwise, over the window `w` of the last `N+1` blocks (`w[0]` = the oldest): `st_i = clamp(ts(w[i]) − ts(w[i−1]), 1, 6E)`, `num = Σ_{i=1..N} i·st_i`, `sum_t = Σ_{i=1..N} target(w[i])`, `den = E·N(N+1)/2`, `next = (sum_t / N) · num / den` (integers, integer division **before** the multiplication), then bounded to `[target(w[N])/2, 2·target(w[N])]`, then to `[2¹⁶, 2⁶⁴−1]`. (ASERT, `docs/13` §5, has since become the default rule: the genesis `retarget` parameter selects it.)
- Work of a block: `⌊2⁶⁴ / (target + 1)⌋`; the chain kept is the one with maximal **cumulative work** (a candidate only replaces the current chain if its cumulative work is **strictly** greater; on a tie the current chain is kept — see `docs/08` §5).

### 10.6 Verification by a second implementation (measured)

- `vearl-chain-vectors` produces a chain of 330 blocks (transfers, extreme timestamp gaps of 1 s and 130 s, two epochs of native matrices, LWMA active from block 91), frozen in `tests/vectors/chain-native-v1.json`.
- `tools/chain_ref.py`, written from this §10, **fully re-validates** it (recomputed tile, Ed25519 signatures, nonces, balances, LWMA, emission, monetary supply, cumulative work) and recovers the exact final state; it **refuses for the right reason** 13 altered blocks (target, prev, timestamp, amount, miner, extra_nonce, tile, version, null signature, signature of another account, replayed nonce, fee or amount above the balance).
- **Reverse direction**: `chain_ref.py` **mines** 6 blocks (with transfers signed in Python); the Rust validator accepts them (`vearl-chain-vectors verify`, test `chain_vectors_are_frozen_and_python_mined_blocks_are_accepted`).
- Limit: same author for the code, the specification and the second implementer; a real third party remains to be found `[AUDIT ACTION]`. The comparison covers neither the P2P network, nor reorganizations, nor the useful-work classes.

## 11. Compute market: jobs, auctions, bonds, data, fraud (v1)

Scope: **simple matrix-product** jobs (`A·B`, `n×n`). Inference jobs, tiled jobs and tensor programs (`docs/05`) reuse the same skeleton (escrow, bond, challenge window) with their own proofs, **not yet specified** here.

### 11.1 v1 constants and genesis parameters

`BURN_BPS = 1000` (10 % of the escrow burned at creation), `BOND_BPS = 1000` (bond = 10 % of the retained price), `MIN_ESCROW = 10⁸`, `CHALLENGE_WINDOW = 10` blocks, `AUCTION_GRACE_SECS = 60`, `MAX_AUCTION_SECS = 3600`, `FETCH_FEE = 5·10⁶`, `FETCH_EXTEND_SECS = 60`, `MAX_PENDING_FETCHES = 8`. **Genesis** parameters (they change consensus and are part of the genesis hash): `challenge_secs` (default 300) and `serve_secs` (default 120). `bps(x, b) = ⌊x·b / 10000⌋`.

### 11.2 Encoding of market transactions

`Tx = from[32] || nonce u64 || fee u64 || TxKind`; `u32` tag of `TxKind`: `Transfer` 0, `SubmitJob` 1, `ClaimJob` 2, `PostResult` 3, `Accept` 4, `Fraud` 5, `Settle` 6, `SubmitJobAuction` 7, `Bid` 8, `Fetch` 9, `Serve` 10, `SlashUnserved` 11 (12–14: inference, blocks, programs). Fields, in order: `SubmitJob{a_root, b_root, escrow u64, deadline u64}`; `SubmitJobAuction` = same `+ auction_secs u64`; `ClaimJob{job_id}`; `PostResult{job_id, c_root}`; `Accept{job_id}`; `Settle{job_id}`; `SlashUnserved{job_id}`; `Bid{job_id, price u64}`; `Fetch{job_id, i u32, j u32}`; `Serve{job_id, i u32, j u32, row Vec<i32>, path}`; `Fraud{job_id, i u32, j u32, a_row, a_path, b_col, b_path, c_row Vec<i32>, c_path}`. A `path` is a vector of hashes `length u64 || 32·length bytes`; `a_row`, `b_col` are vectors of `i8` (`length u64 || bytes`); `row`, `c_row` are vectors of `i32` (`length u64 || 4·length bytes` LE).

`job_id = H("vearl/job/v1", client, nonce u64, a_root, b_root)` (the nonce is that of the creation transaction). Commitments: `a_root` = root **by rows** of `A`, `b_root` = root **by columns** of `B` (§3); `c_root` = the **by-rows** root of 32-bit integers (§3: leaf `i` = `leaf_hash(i, row i as i32 LE)`, 4n bytes; depth `⌈log2 n⌉`, 7 for `n = 128`).

### 11.3 State machine of a job

States: `Open`, `Claimed{worker, bond}`, `Posted{worker, bond, c_root, at, at_ts}`, `Settled`, `Slashed`, `Expired`. `challenge_open(height, ts, at, at_ts) = ¬(height > at + CHALLENGE_WINDOW ∧ ts ≥ at_ts + challenge_secs)`: **the window stays open as long as one of the two durations (blocks, timestamp seconds) has not elapsed** — a burst of fast blocks therefore does not suffice to be paid before the challenge. Each transaction is evaluated with the height and timestamp **of the block that contains it**.

- **SubmitJob / SubmitJobAuction**: `escrow ≥ MIN_ESCROW`; `deadline > height`; `auction_secs ≤ MAX_AUCTION_SECS` (and `> 0` for the auction variant); job does not exist; debit of `escrow`; `burn = bps(escrow, BURN_BPS)` added to `burned`; `escrow_net = escrow − burn`; `rebate_left = burn`; `auction_end = ts + auction_secs` (0 without an auction).
- **Bid**: `Open` job with an auction, `ts < auction_end`; `0 < price ≤ escrow_net`; **strictly lower** than the current best bid; records `(from, price)`.
- **ClaimJob**: `Open` job, `height ≤ deadline`. Without an auction: price = `escrow_net`. With an auction: refused if `ts < auction_end`; the **winner** claims at the price of its bid; any other worker can only claim, **at the full price**, after `auction_end + AUCTION_GRACE_SECS` or if there was no bid. `bond = bps(price, BOND_BPS)` debited; the client recovers `escrow_net − price`; `escrow_net := price`; state `Claimed`.
- **PostResult**: by the worker, `Claimed` job, `height ≤ deadline`; state `Posted{at = height, at_ts = ts}`.
- **Accept**: by the client, `Posted` job; the worker receives `escrow_net + bond`; reputation `done += 1`; the retained data-request fees return to the client; `Settled`.
- **Settle** (anyone): `Posted` job with pending requests → refused; `Posted` and window **closed** → payment as with `Accept`; `Open` and `height > deadline` → the client recovers `escrow_net`, `Expired`; `Claimed` and `height > deadline` → `expired += 1` for the worker, **bond burned**, client refunded, `Expired`; otherwise refused.
- **Fetch** (client only, `Posted` job, window open): entry `(i, j)` with `i, j < n`, no duplicate, at most `MAX_PENDING_FETCHES` pending; debit of `FETCH_FEE` (retained); deadline `ts + serve_secs`.
- **Serve** (worker only): pending request; Merkle opening of `C[i][j]` against `c_root`; the retained fee goes to the worker; the window is **extended**: `at_ts := max(at_ts, max(ts + FETCH_EXTEND_SECS − challenge_secs, 0))`, `at := max(at, max(height + 2 − CHALLENGE_WINDOW, 0))`.
- **SlashUnserved** (anyone): requires a request with `ts > deadline`; sanction (below).
- **Fraud** (anyone, `Posted` job, window open): `a_row`/`b_col` of length `n`, values `|x| ≤ 64`; valid Merkle openings of `A[i]` (against `a_root`, index `i`), of `B[:, j]` (against `b_root`, index `j`) and of the row `C[i]` (`c_row`, `|c_row| = n`, against `c_root`, leaf `i`); **fails if** `Σ a_row·b_col = c_row[j]` (correct entry). Sanction.
- **Sanction** (Fraud and SlashUnserved): `reward = ⌊bond/2⌋` to the **reporter**; `bond − reward` **burned**; the client receives `escrow_net + retained fees`; `slashed += 1`; `Slashed`.

### 11.4 Useful-work blocks

Class `Job(id)`: the block's miner must be **the worker** of a `Claimed` or `Posted` job (state *before* the block's transactions); `A`, `B` are the matrices **whose roots are the job's `a_root`/`b_root`**; the tile is verified as in §6 (with the corresponding Merkle openings). **Reward**: `base_reward(h)` (§10.4) **plus** a *rebate* `min(⌊base/2⌋, rebate_left)` deducted from `rebate_left`: usefulness never earns more than what was burned at job creation (anti wash trading).

### 11.5 Monetary-supply invariant

`Σ balances + Σ_{Open} escrow_net + Σ_{Claimed,Posted}(escrow_net + bond) + Σ retained fees + burned = genesis + minted`, checked after each block.

### 11.6 Verification by a second implementation (measured)

- `vearl-market-vectors` produces a chain of **52 blocks** covering 12 transaction types: an auction with two bids, claim by the winner, 2 useful blocks (rebate then exhausted), Fetch/Serve, Accept; a wrong result **proven** (Fraud); a requested piece of data that is never served (**SlashUnserved**); a job never claimed (**expiry**); a settlement through the **challenge window** (12 blocks and 240 s); a transfer. Frozen in `tests/vectors/market-v1.json`. (The current vector chains are longer: see `tools/audit_all.py`.)
- `tools/market_ref.py`, written from this §11, **fully re-validates** it (byte-by-byte encoding of each transaction, signatures, state machine, tiles of the useful blocks recomputed with the matrices attached to the committed roots, rewards, burned amount, reputation, invariant) and recovers the exact final state; it **refuses for the right reason** 17 altered cases (non-lower bid, claim during the auction or by a non-winner, acceptance or data request by a third party, premature settlement, fraud on a correct entry or with a falsified opening of `A` or `C`, useful block by a non-worker, serving a wrong value, duplicate request, escrow under the floor, deadline passed, insufficient balance, SlashUnserved without an expired request) and **accepts** the corresponding valid cases.
- **Reverse direction**: `market_ref.py` **mines** 10 blocks (2 jobs: an honest one paid, a wrong result for which it builds the **fraud proof** itself); the Rust validator accepts them (`vearl-market-vectors verify`, test `market_vectors_are_frozen_and_python_mined_market_blocks_are_accepted`).
- **Differential fuzz** (`tools/diff_fuzz.py`, `vearl-market-vectors diff`): random transaction blocks — valid by construction in ~88 % of cases, otherwise faulty (wrong signer, price, moment, indices, altered Merkle openings or values, shifted nonces, extreme amounts and fees) — are presented to Python and Rust, which must **decide the same way**; the final state (emission, burned, accounts, jobs, reputations) must also be identical. Measured: **9 campaigns, 3,000 to 4,500 blocks each, 27,000 identical decisions, 0 divergence**; ~20 % of blocks accepted (per campaign: ≈ 5-12 Fraud, 2-10 Serve, 3-11 Bid, 25-50 Settle), with Settle/SlashUnserved/Expired at every stage. Limit: a Bid is rarely accepted (short auction window), and only Native-class work is exercised.
- Limits: same author for the code, this text and the second implementer `[AUDIT ACTION]`; not covered: inference / tiled jobs, reorganizations, P2P network (see `docs/08`); the vector set does not exercise the `MAX_PENDING_FETCHES` cap nor the extension of the window by several `Serve`s.

## 12. Verifiable tensor programs (v1)

Scope: programs of at most **240 operations** on `n×n` tensors of 8-bit integers (`|x| ≤ 64`), with at most **64 weight matrices** (a large network is split into `n×n` blocks: block products + `Add`, `docs/11`): convolutional networks, residual blocks, transformer encoder blocks (`docs/05`). Tiled products (`TiledTx`) are specified in §13; the historical inference jobs (`MlpTx`) are obsolete (replaced by programs).

### 12.1 Tensors, commitments, indexing

`T_0` is the client's input; operation `k` (0-indexed) produces `T_{k+1}`. All the inputs of an operation are **already computed** tensors (index `≤ k`). A tensor `T` is committed by the Merkle root **by columns** (leaf `j` = `leaf_hash(j, column j)`, §3); a weight `W` by the root **by rows**; the integer result `Z` of a GEMM by the **by-rows** root of 32-bit integers (§3: leaf `i` = `leaf_hash(i, row i as i32 LE)`, 4n bytes; depth `⌈log2 n⌉`, 7 for `n = 128`). Unused areas of a tensor are 0.

### 12.2 Integer semantics of the operations (exact arithmetic, no floating point)

Notation: `floor` = integer division rounding down (also for negatives); `clamp(v, lo, hi)`.

- **Gemm{w, x, shift, relu}**: `Z = W_w · T_x` (i32); output = `clamp(Z >> shift, relu ? 0 : −64, 64)` (arithmetic shift). **GemmAT{a, x, shift, relu}**: `Z = T_aᵀ · T_x`, same activation.
- **Add{a, b, relu}**: `clamp(T_a + T_b, relu ? 0 : −64, 64)` cell by cell.
- **Map{x, Transpose}**: `out[i][j] = T_x[j][i]`.
- **Map{x, Im2col{c_in, h, w, batch, kh, kw, pad}}**: the input tensor stores the channel as the row and `(y, x, batch)` as the column (`column = (y·w + x)·batch + b`). Output: row `r = c·kh·kw + dy·kw + dx`, column `q = (py·w + px)·batch + b`; the value is `T_x[c][(sy·w + sx)·batch + b]` with `sy = py + dy − pad`, `sx = px + dx − pad`, and **0 outside the image**; rows `≥ c_in·kh·kw` and columns `≥ h·w·batch` are 0. (A convolution = Im2col then Gemm.)
- **Map{x, Conv{c_in, h, w, batch, kh, kw, pad, stride}}**: GENERAL convolution (any stride and padding). Input `(channel, (y, x, batch))` on the `h×w` grid; output grid `oh = ⌊(h + 2·pad − kh)/stride⌋ + 1`, `ow` likewise; output: row `r = c·kh·kw + dy·kw + dx`, column `q = (oy·ow + ox)·batch + b`, value `T_x[c][(sy·w + sx)·batch + b]` with `sy = oy·stride + dy − pad`, `sx = ox·stride + dx − pad`, **0 outside the image**; rows `≥ c_in·kh·kw` and columns `≥ oh·ow·batch` are 0. With `stride = 1` and `pad = kh/2` it is `Im2col`.
- **Map{x, HeadSpread{dh, heads, l}}** (multi-head attention): `out[i][h·l + j] = T_x[i][j]` for `h = ⌊i / dh⌋`, `i < heads·dh`, `j < l`; 0 elsewhere. With `GemmAT{a: K, x: out}`, the scores of all the heads are computed in **one** product (column block `h` = `K_hᵀ Q_h`).
- **Map{x, HeadGather{dh, heads, l}}**: `out[i][j] = T_x[i][⌊i / dh⌋·l + j]` for `i < heads·dh`, `j < l`; 0 elsewhere (each row takes back the block of its head, after `V·A`).
- **Map{x, Shift{dr, dc}}**: `out[i][j] = T_x[i + dr][j + dc]` if `i + dr < n` and `j + dc < n`, otherwise 0. Used to extract a block from a **wide** input tensor (more than `n` features: the feature blocks are stored side by side in the columns of `T_0`, `docs/12`).
- **Map{x, Flatten{c, h, w, batch}}**: `out[i][j] = T_x[c'][(py·w + px)·batch + j]` with `i = c'·h·w + py·w + px`, for `i < c·h·w` and `j < batch`; 0 elsewhere.
- **Pool{x, Max{c, h, w, batch, k}}**: maximum of the `k²` cells of the window (same layout and bounds as `Avg`; output 0 outside the grid).
- **Pool{x, Avg{c, h, w, batch, k}}**: rounded mean over disjoint `k×k` windows: `⌊(Σ + k²/2) / k²⌋` (division rounding down, including for negative sums); output `(c, h/k, w/k, batch)` in the same layout.
- **Softmax{x, len}** (per column, on the first `len` rows; the rest 0): `m = max` of the column; `e_j = EXP2[clamp(m − v_j, 0, 128)]` with `EXP2[d] = ⌊65536 · 2^(−d/8)⌋` (a 129-entry table computed in exact decimal arithmetic — at exact powers of 2 the exact value is an integer: `EXP2[8] = 32768`); `S = Σ e_j`; output `⌊(64·e_j + ⌊S/2⌋) / S⌋`.
- **SoftmaxLo{x, len}** (the **fine** part of the same softmax, same `m`, `e_j`, `S`): `hi_j = ⌊(64·e_j + ⌊S/2⌋) / S⌋`, `fin_j = ⌊(4096·e_j + ⌊S/2⌋) / S⌋`, output `clamp(fin_j − 64·hi_j, −64, 64)` (in practice within `[−32, 32]`); rows `≥ len`: 0. With `Softmax`, `64·hi + lo` gives the probability to within 1/4096: a model computes `V·A_high ≫ s` + `V·A_fine ≫ (s + 6)`. Over hundreds of tokens (197 for a 224 × 224 image), `Softmax` alone rounds every probability < 1/128 to 0.
- **CausalSoftmax{x, len, l, fine}** (**causal** attention of a language model: a token only sees the preceding tokens): column `j` is query `q = j mod l` (columns `(head, query)` after `HeadSpread`, `l` = width of one head's block); with `len_j = min(len, q + 1)`, the output column is `Softmax{len_j}` of the input column if `fine = 0`, `SoftmaxLo{len_j}` if `fine = 1` (same formulas, rows `≥ len_j`: 0). The column function therefore depends on `j`: a judge recomputes it with the index `j` of the proof.
- **Bias{x, w, row, relu}**: `out[i][j] = clamp(T_x[i][j] + W_w[row][i], relu ? 0 : −64, 64)` for all `i, j < n`: a **bias vector is a row of a weight block** (a single block carries up to `n` bias vectors; bias rows are 0 beyond the useful dimension). This is what allows importing a model with biases without changing the semantics of the other operations: use `Gemm{relu: false}` then `Bias{relu}`.
- **Lut{x, w, row}**: `out[i][j] = W_w[row][clamp(T_x[i][j], −64, 63) + 64]`: a **128-entry table** (a row of a weight block) applied to each cell; the input `+64` is treated as `+63`. Serves any quantized activation (GELU, SiLU, tanh, sigmoid…) with no floating point in the consensus: the table is computed off-chain by the client and committed with the weights.
- **Mul{x, w, row, shift}** (gamma of a LayerNorm): `out[i][j] = clamp((T_x[i][j] · g[i]) >> shift, −64, 64)` with `g = W_w[row]` (any 8-bit integers), **arithmetic** shift (rounding toward −∞). Followed by a `Bias` (beta), it implements the affine LayerNorm of BERT/MiniLM.
- **LayerNorm{x, len}** (per column, first `len` rows): `μ = ⌊(Σv + ⌊len/2⌋) / len⌋`; `var = ⌊Σ(v−μ)² / len⌋`; `σ = max(1, ⌊(⌊√(4·(var+1))⌋ + 1) / 2⌋)` (exact integer square root); output `clamp(⌊((v−μ)·16 + ⌊σ/2⌋) / σ⌋, −64, 64)`.

**Documented limit**: on a column of very low variance (standard deviation ≲ 7), the integer normalisation departs from the real normalisation (8-bit quantisation); this is a property of the format, not an error.

### 12.3 Validation of a program (`Submit`)

`1 ≤ #ops ≤ 240`, `#weights ≤ 64` (single-byte indices). For operation `k`, a tensor index `t` is valid if `t ≤ k`. Gemm: `w <` #weights, `shift ≤ 24`. Softmax/SoftmaxLo/LayerNorm: `x ≥ 1`, `1 ≤ len ≤ n`. CausalSoftmax: same and `1 ≤ l ≤ n`. Flatten: `c, h, w, batch > 0`, `c·h·w ≤ n`, `h·w·batch ≤ n`, `c ≤ n`. Shift: `dr, dc < n`; Conv: same bounds as Im2col with `stride ≥ 1`, `h + 2·pad ≥ kh`, `w + 2·pad ≥ kw` and `oh·ow·batch ≤ n`; Im2col: `c_in, h, w, batch, kh, kw > 0`, `c_in·kh·kw ≤ n`, `h·w·batch ≤ n`, `c_in ≤ n`, `pad < kh + kw`. Pool: `k ≥ 1`, `k | h`, `k | w`, `c ≤ n`, `h·w·batch ≤ n`. Bias, Lut: `x ≤ k`, `w <` #weights, `row < n`. Mul: same and `shift ≤ 24`. HeadSpread, HeadGather: `dh, heads, l > 0`, `heads·dh ≤ n`, `heads·l ≤ n`. At least one Gemm/GemmAT is required (the mined work), `escrow ≥ MIN_ESCROW × #Gemm`, `deadline > height`.

### 12.4 Encoding of program transactions

`TxKind::Prog` = `u32` tag **14**, then the `u32` tag of `ProgTx`: `Submit` 0 `{x0_root, w_roots Vec<[32]>, ops Vec<Op>, escrow u64, deadline u64}`; `Claim` 1 `{job_id}`; `PostOp` 2 `{job_id, op u8, z_root, x_root}`; `Accept` 3; `Fraud` 4 `{job_id, op u8, kind}`; `Settle` 5; `PostBatch` 6 `{job_id, op u8, posts Vec<([32], [32])>}` (pairs `(z_root, x_root)`); `Fetch` 7 `{job_id, tensor u8, col u32}`; `Serve` 8 `{job_id, tensor u8, col u32, vals Vec<i8>, path Vec<[32]>}`; `SlashUnserved` 9 `{job_id}`.

`Op` (`u32` tag): `Gemm` 0 `{w u8, x u8, shift u8, relu u8}`; `Add` 1 `{a, b, relu}`; `Map` 2 `{x u8, map}` with `map` = `Transpose` 0 | `Im2col` 1 `{c_in u16, h u16, w u16, batch u16, kh u8, kw u8, pad u8}` | `Flatten` 2 `{c u16, h u16, w u16, batch u16}` | `Conv` 3 `{c_in u16, h u16, w u16, batch u16, kh u8, kw u8, pad u8, stride u8}` | `Shift` 4 `{dr u16, dc u16}` | `HeadSpread` 5 `{dh u16, heads u16, l u16}` | `HeadGather` 6 `{dh u16, heads u16, l u16}`; `Pool` 3 `{x u8, pool}` with `Avg` 0 or `Max` 1, `{c u16, h u16, w u16, batch u16, k u8}`; `GemmAT` 4 `{a, x, shift, relu}`; `Softmax` 5 `{x u8, len u16}`; `LayerNorm` 6 `{x u8, len u16}`; `Bias` 7 `{x u8, w u8, row u16, relu u8}`; `Lut` 8 `{x u8, w u8, row u16}`; `Mul` 9 `{x u8, w u8, row u16, shift u8}`; `SoftmaxLo` 10 `{x u8, len u16}`; `CausalSoftmax` 11 `{x u8, len u16, l u16, fine u8}`.

`ProgFraud` (`u32` tag): `Gemm` 0 `{i u32, j u32, w_row Vec<i8>, w_path, x_col Vec<i8>, x_path, z_row Vec<i32>, z_path}`; `Act` 1 `{i, j, z_row Vec<i32>, z_path, out_col, out_path}`; `Elem` 2 `{i, j, ins Vec<{tensor u8, col u32, vals Vec<i8>, path}>, out_col, out_path}`; `Col` 3 `{j u32, in_col, in_path, out_col, out_path}`; `Bias` 4 `{j u32, in_col Vec<i8>, in_path, b_row Vec<i8>, b_path, out_col Vec<i8>, out_path}`; `Lut` 5 `{j u32, in_col Vec<i8>, in_path, t_row Vec<i8>, t_path, out_col Vec<i8>, out_path}`.

`job_id = H("vearl/prog/v1", client, nonce u64, x0_root, bincode(ops), w_root_0, …, w_root_m)`.

### 12.5 State machine

States: `Open`, `Working{worker, bond}`, `Complete{worker, bond, at, at_ts}`, `Settled`, `Slashed`, `Expired`. `Submit`: debit of the escrow, `burn = bps(escrow, BURN_BPS)`, `escrow_net = escrow − burn`, `rebate_left = burn`. `Claim` (`Open` job, `height ≤ deadline`): bond `bps(escrow_net, BOND_BPS)`. **`PostBatch`** (worker only, same conditions): publishes `posts.len() ≥ 1` consecutive operations starting at `op` (= number already published), with the same checks for each as `PostOp` (`z_root` null except for a Gemm); **exactly equivalent** to the corresponding sequence of `PostOp`s and reduces the load on the chain by a factor `#ops` (the reference worker uses it: one transaction per job). **`PostOp`** (worker only, `height ≤ deadline`): `op` = number of operations already published (**imposed order**), `z_root` null except for a Gemm; at the last one: `Complete{at = height, at_ts = ts}`. `Accept` (client, `Complete`): the worker receives `escrow_net + bond`. `Settle`: `Complete` and window closed (`docs/07` §11.3) → payment as with `Accept`; `Open` and `height > deadline` → `Expired`, client refunded; `Working` and `height > deadline` → `Expired`, **bond burned**, client refunded.

**Fraud** (anyone; `Working` job — **as soon as a wrong operation is published, without waiting for the end** — or `Complete` with the window open): operation `op` must be published; the proof must match the operation's type:
- **Gemm** (on Gemm/GemmAT): opening of the row of `A` (root of `W_w` by rows, or the **by-columns** root of `T_a` for GemmAT, whose transpose has this root by rows), of the column of `T_x`, and of the row `Z[i]` (`z_row`) against `z_root` (leaf `i`); **fails if** `Σ w_row·x_col = z_row[j]`.
- **Act** (on Gemm/GemmAT): opening of the row `Z[i]` against `z_root` and of column `j` of the output against `x_root`; **fails if** `out_col[i] = act(z_row[j])`.
- **Elem** (on Add/Map/Pool): the openings of the source columns, **in the order of the operation's source cells** (12.2: `Add` → 2 cells `(a,i,j),(b,i,j)`; `Transpose` → `(x,j,i)`; `Im2col`, `Flatten`, `Pool` → cells defined above, no cell if the output is structurally 0) and of the output column; **fails if** the output cell equals the function of the source cells.
- **Col** (on Softmax/SoftmaxLo/LayerNorm/CausalSoftmax): opening of column `j` of the input and of the output; **fails if** the column function (12.2) gives back the output column.
- **Bias** (on Bias): opening of column `j` of the input (root of `T_x`), of the **bias row** `row` of weight `w` (root of the weight by rows, leaf `row`) and of column `j` of the output; **fails if** `clamp(in_col + b_row) = out_col` (cell by cell).
- **Bias on Mul** (the proof of a `Mul` has the **form** of a `Bias` proof): input column `j`, row `row` of weight `w` (any 8-bit values, no ±64 bound), output column; **fails if** `clamp((in_col[i] · b_row[i]) >> shift, −64, 64) = out_col[i]` for all `i`.
- **Lut** (on Lut): same openings (input column, **table row** `row` of weight `w`, output column); **fails if** `table[clamp(in_col[i], −64, 63) + 64] = out_col[i]` for all `i`.
Sanction: `reward = ⌊bond/2⌋` to the reporter, `bond − reward` burned, the client receives `escrow_net` (plus its pending data-request fees), `slashed += 1`, `Slashed`.

**Data availability** (same rules and same constants as simple-product jobs, §11.3): without it, a worker that withholds its results would be paid when the window expires, the client being unable to verify anything. `Fetch` (client, `Complete` job, window open, `1 ≤ tensor ≤ #ops`, `col < n`, at most `MAX_PENDING_FETCHES` distinct pending requests): the client pays `FETCH_FEE`, set aside; the request expires `serve_secs` after the block's timestamp. `Serve` (worker): `vals` (n values in `[−64, 64]`) opened against the **published** root of tensor `tensor` (leaf `col`); the fee is paid to the worker, the request disappears, the challenge window is extended (`at_ts ← max(at_ts, ts + FETCH_EXTEND_SECS − challenge_secs)`, `at ← max(at, height + 2 − CHALLENGE_WINDOW)`). `SlashUnserved` (anyone, `Complete` job): a request has exceeded its deadline (`ts > deadline`) → sanction like a proven fraud (half of the bond to the reporter, half burned, escrow and fees returned to the client). `Settle` is refused while a request is pending; `Accept` returns the fees set aside. Nodes monitor and trigger `SlashUnserved` by themselves.

### 12.6 `Op` work blocks

Class `Op{job, op}` (`u32` tag 4, `job` 32 B, `op` `u8`): the miner must be **the worker** of a `Working` or `Complete` program, and `op` a **Gemm whose inputs are published** (`T_x` already published, and `T_a` for GemmAT). The work matrices are `A = W_w` (or `T_aᵀ`) and `B = T_x`, whose roots (by rows / by columns) are those committed on-chain. Reward: `base_reward` plus *rebate* `min(⌊base/2⌋, rebate_left)` (§11.4). A worker can mine **any** of its published Gemms, in rotation, as long as the job is in progress or in the challenge window.

### 12.7 Verification by a second implementation (measured)

- `vearl-prog-vectors` produces **229 blocks**: a CNN with a residual block accepted, a transformer encoder block (14 operations, softmax, 2 LayerNorms, GemmAT) settled by the window, an MLP with biases and an activation table (Gemm, Bias, tanh Lut, Gemm, Bias) accepted, the same published **in a single `PostBatch` transaction** then accepted, a column demanded then served (accepted) and a column demanded and not served (worker sanctioned by `SlashUnserved`), and **all six fraud types** proven (wrong GEMM, wrong activation during `Working`, wrong residual addition, wrong softmax column, wrong bias, wrong table). Frozen in `tests/vectors/prog-v1.json`. (The current chain is longer, 483 blocks once the attention operations were added; see `tools/audit_all.py`.)
- `tools/prog_ref.py`, written from this §12 (semantics of the operations: the independent engine `vearl_ref.py`, validated on ~2,000 random programs; encoding of transactions and proofs; state machine; `Op` blocks with matrices attached to the roots), **fully re-validates** the chain: 229 blocks, 22 programs more than before (including the MLP with biases and a table), exact final state. It **refuses for the right reason** 15 cases (fraud on a correct entry, falsified opening, unsuitable proof type, acceptance by a third party or of an incomplete program, premature settlement, useful block by a non-worker, out-of-order publication or by a third party, double claim, escrow under the floor, deadline passed, program without a Gemm, forward reference, shift out of bounds, non-dividing pooling) and accepts 5 valid cases (submission; the 4 fraud types replayed by another challenger).
- **Reverse direction**: `prog_ref.py` **defines** a program itself (2 Gemms, softmax, LayerNorm), **computes** it with its engine, submits it, **mines** its `Op` blocks, publishes the operations — once honestly (paid), once with a wrong LayerNorm column for which it **builds the fraud proof** (slashed) — then an **MLP with biases and a table** (honest, paid; with one cell of the `Bias` output wrong, for which it builds the `Bias` proof; with one cell of the table output wrong, for which it builds the `Lut` proof); plus a variant published in one batch (`PostBatch`) then proven wrong — plus a column demanded then served (the client is refunded its fees) and a column not served then `SlashUnserved` — i.e. 78 blocks that the Rust validator accepts (435 in total, with the 4 multi-head attention cases and the 2 fine-attention `SoftmaxLo` cases of the reference chain) (`vearl-prog-vectors verify`, test `prog_vectors_are_frozen_and_python_authored_programs_are_accepted`).
- **Differential fuzz** (`tools/diff_fuzz_prog.py`, `vearl-prog-vectors diff`): random program-transaction blocks — valid or invalid submissions (floor, deadline, invalid programs), claims, publication of operations in order or not with true, falsified or arbitrary roots, acceptances and settlements at any time, fraud proofs **of the four types** with valid or altered openings, on honest or truly falsified operations, shifted nonces, extreme fees — are presented to Python and Rust, which must decide the same way; the final state (accounts, programs, reputations, emission, burned) must be identical. Measured: **9 campaigns of 1,500 to 3,000 blocks, 22,500 identical decisions, 0 divergence** (before `Bias` was added) then, with the bias MLP in the data (3 campaigns) then with biases and a table (3 campaigns) of 2,000 blocks, **12,000 identical decisions, 0 divergence** (then **8,000 more decisions** with batch publications and the three new availability transactions, including accepted `Fetch`, `Serve` and `SlashUnserved`, 0 divergence), including 1 to 4 `Bias` proofs and 1 to 3 `Lut` proofs accepted per campaign; the independent operation engine is also compared tensor by tensor on 3,000 random programs (including ~1,800 `Bias` and ~1,800 `Lut`), 0 divergence; per campaign of 3,000 blocks ≈ 500 accepted `PostOp`s, 30-40 settlements/acceptances, and accepted fraud proofs of each type (≈ 3-8 Act, 6-7 Col, 1-3 Elem, 1-5 Gemm). Limit: only Native-class work is exercised (not the `Op` blocks), and the data comes from the 6 programs of the vectors.
- Limits: same author for the code, this text and the second implementer `[AUDIT ACTION]`; the historical inference jobs (`MlpTx`, obsolete) are not covered; the vector does not exercise `Expired`/`Settle` of a program never claimed, nor falsified GemmATs.


## 13. Tiled matrix products (`tiled`, v1) — large-scale linear algebra

Scope: `C = A·B` with `A` of `I·n × K·n` and `B` of `K·n × J·n` 8-bit integers (`|x| ≤ 64`), `1 ≤ I, K, J ≤ 8` (up to `8n × 8n × 8n`, i.e. 8192³ for `n = 1024`). The computation is exact (32-bit integers: `|P| ≤ 64²·n`, sum of `K ≤ 8` products without overflow). No floating-point semantics is involved; any quantisation is done off-chain by the client.

### 13.1 Splitting and commitments

`A_ik` (`n×n`, row-major) is the block `(i, k)` of `A`; `B_kj` the block `(k, j)` of `B`; `P_ijk = A_ik·B_kj` (i32); `C_ij = Σ_k P_ijk`. The blocks `A_ik` are committed by the Merkle root **by rows**, order `(i, k)` (`a_roots[i·K + k]`); the blocks `B_kj` by the root **by columns**, order `(k, j)` (`b_roots[k·J + j]`); a product or a sum by the **by-rows** root of 32-bit integers (§3: leaf `i` = `leaf_hash(i, row i as i32 LE)`, 4n bytes; depth `⌈log2 n⌉`, 7 for `n = 128`). The worker publishes `product_roots` in order `(i, j, k)` (`index = (i·J + j)·K + k`) and `c_roots` in order `(i, j)` (`i·J + j`).

### 13.2 Encoding

`TxKind::Tiled` = `u32` tag **13**, then the `u32` tag of `TiledTx`: `Submit` 0 `{dims (u8, u8, u8) = (I, K, J), a_roots Vec<[32]>, b_roots Vec<[32]>, escrow u64, deadline u64}`; `Claim` 1 `{job_id}`; `Post` 2 `{job_id, product_roots Vec<[32]>, c_roots Vec<[32]>}`; `Accept` 3; `Fraud` 4 `{job_id, kind}`; `Settle` 5; `Fetch` 6 `{job_id, which u8, i u8, j u8, k u8, idx u32}`; `Serve` 7 `{job_id, which u8, i u8, j u8, k u8, idx u32, row Vec<i32>, path Vec<[32]>}`; `SlashUnserved` 8 `{job_id}`. `TiledFraudKind`: `Product` 0 `{i u8, j u8, k u8, r u32, c u32, a_row Vec<i8>, a_path, b_col Vec<i8>, b_path, z_row Vec<i32>, z_path}`; `Sum` 1 `{i u8, j u8, r u32, c u32, z_rows Vec<Vec<i32>>, z_paths Vec<Vec<[32]>>, c_row Vec<i32>, c_path}`.

`job_id = H("vearl/tiled/v1", client, nonce u64, [I, K, J] (3 bytes), a_root_0, …, b_root_0, …)`.

### 13.3 State machine

States: `Open`, `Claimed{worker, bond}`, `Posted{worker, bond, at, at_ts}`, `Settled`, `Slashed`, `Expired`. `Submit`: `1 ≤ I, K, J ≤ 8`, `#a_roots = I·K`, `#b_roots = K·J`, `escrow ≥ MIN_ESCROW × (I·K·J)` (floor **per product**), `deadline > height`, new identifier; debit, `burn = bps(escrow, BURN_BPS)`, `escrow_net = escrow − burn`, `rebate_left = burn`. `Claim` (`Open`, `height ≤ deadline`): bond `bps(escrow_net, BOND_BPS)`. `Post` (worker, `Claimed`, `height ≤ deadline`): exactly `I·K·J` product roots and `I·J` sum roots → `Posted{at = height, at_ts = ts}`. `Accept` (client, `Posted`) and `Settle` (`Posted`, window closed) pay `escrow_net + bond` to the worker; `Settle` of an `Open` or `Claimed` job after the deadline → `Expired` (client refunded; bond burned if `Claimed`). `Fraud` (`Posted`, window open): `reward = ⌊bond/2⌋`, rest burned, client refunded, `Slashed`.

**Data availability** (same rules, same constants and same window extension as §12.5 / §11.3): `Fetch` (client, `Posted` job, window open) demands on-chain the entry `idx = r·n + c` of a **product** (`which = 0`, block `(i, j, k)`) or of a **sum** (`which = 1`, block `(i, j)`, `k = 0`); `Serve` (worker) opens the **row** `r = idx / n` against `product_roots[(i·J + j)·K + k]` or `c_roots[i·J + j]` (leaf `r`, `|row| = n`); `SlashUnserved` (anyone) sanctions a request left unanswered after `serve_secs` (half of the bond to the reporter, half burned, escrow and fees returned to the client). `Settle` is refused while a request is pending; `Accept` and `Fraud` return the fees.

### 13.4 Fraud proofs (element-level, verifiable on-chain)

- **Product**: block indices within bounds, `r, c < n`, `|a_row| = |b_col| = n`, values in `[−64, 64]`; `a_row` opened against `a_roots[i·K + k]` (leaf `r`), `b_col` against `b_roots[k·J + j]` (leaf `c`), the row `z_row` against `product_roots[(i·J + j)·K + k]` (leaf `r`); **fails if** `Σ a_row·b_col = z_row[c]`. Cost O(n).
- **Sum**: `i < I`, `j < J`, `r, c < n`, `|z_rows| = |z_paths| = K`; each row `z_rows[k]` opened against `product_roots[(i·J + j)·K + k]` (leaf `r`), `c_row` against `c_roots[i·J + j]`; **fails if** `Σ_k z_rows[k][c] = c_row[c]` (64-bit sum).

### 13.5 `Product` work blocks

Class `Product{job, i, j, k}` (`u32` tag 3, `job` 32 B, `i, j, k` `u8`): the miner must be the worker of a `Claimed` or `Posted` job; `A = A_ik`, `B = B_kj` (committed roots). Reward: `base_reward` plus *rebate* `min(⌊base/2⌋, rebate_left)` (§11.4). A worker can mine any of the `I·K·J` products.

### 13.6 What the client does (and why the result does not need `C_ij`)

The client downloads each `P_ijk`, checks its root against `product_roots` then the product with the Freivalds test (24 rounds, error ≤ 2⁻²⁴ per product; localisation of the wrong entry to build the proof), and **sums itself** (exactly) the `P_ijk`: the published `C_ij` only serve the `Sum` proof. Transfers: `I·K·J` products of 4 MiB (n = 1024): this is the cost of the challenge scheme (the gap with the size of `C` is a factor `K`).

### 13.7 Verification by a second implementation (measured)

- `vearl-tiled-vectors` produces **56 blocks**: two honest jobs (accepted; settled by the window), a **wrong product** and a **wrong sum** proven, product and sum entries demanded then served (accepted), an entry not served (`SlashUnserved`), with blocks of class `Product`. Frozen in `tests/vectors/tiled-v1.json` (test `tiled_vectors_are_frozen_and_python_authored_jobs_are_accepted`).
- `tools/tiled_ref.py`, written from this §13, **fully re-validates** the chain (56 blocks, header hash of each block, final state, monetary supply).
- **Reverse direction**: `tiled_ref.py --extend` **defines** two jobs itself (matrices chosen by it, products computed by its exact engine), **mines** `Product` blocks, publishes the roots, settles the first (Accept) and **proves the wrong sum** of the second with a proof it builds; then a request for sum-input data served by the worker (built in Python): 17 more blocks (73 in total) accepted by the Rust validator (`vearl-tiled-vectors verify`).
- **Differential fuzz** (`tools/diff_fuzz_tiled.py`, `vearl-tiled-vectors diff`): random transaction blocks — valid or invalid submissions (dimensions 0/9/200, too many or too few roots, escrow a hair under the floor, deadline passed), claims, publications with true, false or wrongly-numbered roots, acceptances/settlements at any time, `Product` and `Sum` proofs with valid or altered openings, on the right or wrong cell, over 24 data sets (6 shapes × {honest, falsified product, falsified product with honest sums, falsified sum}), shifted nonces, extreme fees. Measured: **6 campaigns (300 then 5 × 1,500 blocks), 7,800 identical decisions, 0 divergence** (then **5 campaigns of 1,500 blocks with the data requests, 7,500 identical decisions, 0 divergence**, including accepted `Fetch`, `Serve` and `SlashUnserved`); final state (accounts, jobs, reputations, emission, burned) identical; per campaign of 1,500 blocks ≈ 50-80 `Post`s, 25-40 settlements, 5-16 `Product` proofs and 6-17 `Sum` proofs accepted.
- Limits: same author for the code, this text and the second implementer `[AUDIT ACTION]`; the fuzz only exercises Native-class work (the `Product` blocks are exercised by the vectors); the availability guarantee is per requested entry (the ultimate recourse), not a normal path.
