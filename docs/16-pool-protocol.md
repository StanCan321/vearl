# 16 - Mining pool protocol (`vpool/1`)

Status: reference implementation (`vearl-core/src/pool.rs`, `src/poolminer.rs`, binaries `vearl-pool` and `vearl-poolminer`), tested on a CPU devnet
(`tests/pool.rs`). **Unaudited prototype, testnet only, tokens have no value.** This document is the specification: a third party can write a
compatible miner or pool from it.

## 1. Why a pool, and what it changes

A Vearl block is found by one `sigma` evaluation (one noised GEMM) whose tile jackpot is below the network target. A small GPU may need hours per
block. A pool lets many miners search together and share the rewards in proportion to their work.

The protocol uses one property of the consensus rules: **a block header carries the reward recipient (`miner`) as a plain public key, and the block
carries no signature of that key.** `sigma = H("vearl/sigma/v1", prev, miner, class, extra_nonce, tx_root, timestamp)` only needs the *public* key.
So:

- the pool is the `miner` of every block it finds; the pool's key receives the whole reward on-chain;
- workers never need the pool's private key, and the pool never needs a worker's key;
- a worker proves its work with **shares**: tile jackpots below an easier *share target*; a share that also meets the network target is a block.

Pools mine the **Native Work** (class 0). Paid jobs (level 1 / level 2) stay reserved to solo workers (`node --worker`): the block rules require
the job's worker to be the miner, which a pool cannot delegate. A pool operator can still run a worker next to the pool.

## 2. Roles

| Role | Program | Holds |
|---|---|---|
| Pool server | `vearl-pool` (a full node without its own mining loop + the pool server) | the pool key (hot wallet), the ledger |
| Worker | `vearl-poolminer` (CPU or CUDA) | a payout address only |
| Explorer / dashboards | any HTTP client | nothing |

## 3. Transport

TCP, UTF-8 JSON, **one message per line** (`\n`). Maximum line: 4096 bytes (the server closes longer lines). No TLS in the protocol: for a public pool
put a TLS terminator in front (stunnel or nginx `stream`) and let workers connect through it. 64-bit integers (targets, `extra_nonce`) are sent as
**16 lowercase hexadecimal digits** (JSON numbers do not hold a `u64` in every language); hashes and keys are 64 hex digits; heights, tile indexes and
counts are ordinary JSON numbers.

Requests: `{"id":N,"method":"...","params":{...}}`. Responses: `{"id":N,"result":{...}}` or `{"id":N,"error":{"code":C,"message":"..."}}`.
Notifications (server to worker, no `id`): `job`, `set_target`.

### 3.1 `login` (first request, once)

```json
{"id":1,"method":"login","params":{"proto":"vpool/1","address":"vrl1...","worker":"rig1","agent":"vearl-poolminer/0.1","sigma_rate":150.0}}
```

- `proto` must be `vpool/1`. `address` is the payout address (Bech32m `vrl1...`; the pool's own address is refused).
- `worker`: 1-32 characters among letters, digits, `.`, `_`, `-`. Several workers may share one address (their shares add up).
- `sigma_rate` (optional): estimated `sigma` evaluations per second; only used to choose the first share target.

Result: `{"proto","slot","share_target","pool_address","fee_bps","min_payout","payout_interval","confirmations","pplns_n","nonce_bits":40,"share_secs"}`.
The server then sends the current `job` immediately.

### 3.2 `job` notification

```json
{"method":"job","params":{
  "job_id":"a1b2c3d4e5f6","clean":true,"height":1234,"prev":"<hash>","miner":"<pool public key>","class":"native",
  "timestamp":1760000000,"tx_root":"<hash>","epoch":12,"target":"<u64 hex>","share_target":"<u64 hex>",
  "slot":17,"nonce_start":"<u64 hex>","nonce_end":"<u64 hex>",
  "params":{"n":1024,"tile":16,"rank":63,"fold_k":64}}}
```

- The worker computes, for `extra_nonce` in `[nonce_start, nonce_end)`:
  `sigma = H("vearl/sigma/v1", prev, miner, bincode(WorkClass::Native), extra_nonce (8 bytes LE), tx_root, timestamp (8 bytes LE))`
  (`H` = BLAKE3 `derive_key` with that context; `bincode(Native)` = four zero bytes), then the noised GEMM of the native matrices of `epoch`
  (`native_matrices(epoch, params)`, see docs/07) which yields one **jackpot per tile**. Tile `(ti, tj)` is index `ti * tiles + tj`, `tiles = n / tile`.
- Every tile whose jackpot is below `share_target` (compare the first 8 bytes of the jackpot, big-endian, as a `u64`) is a share.
- `clean = true`: new parent block, abandon the previous work. `clean = false`: same parent, new transactions or timestamp; switching is advised
  but older jobs of the same parent remain valid (the pool keeps the last 8).
- **Slots.** `extra_nonce = slot << 40 | counter`, so two workers never search the same space. A share with an `extra_nonce` outside the worker's
  slot is rejected. A slot is released when the connection closes.
- The worker needs no chain state: only these fields and the (deterministic) native matrices.

### 3.3 `submit`

```json
{"id":7,"method":"submit","params":{"job_id":"a1b2c3d4e5f6","extra_nonce":"<u64 hex>","ti":3,"tj":12}}
```

Result `{"accepted":true,"block":false,"weight":"..."}`; `block: true` means this share was also a block and the pool submitted it to the network.

| Code | Meaning | Counts toward ban |
|---|---|---|
| 20 | stale or unknown job (a job of the previous parent is still accepted for `stale_grace_secs`, default 4 s) | no |
| 21 | duplicate share (`job`, `extra_nonce`, `ti`, `tj`) | 2 |
| 22 | jackpot not below the share target | 5 |
| 23 | `extra_nonce` outside the worker's slot, or tile out of range | 5 |
| 24 | (reserved) unknown job | |
| 25 | not logged in | |
| 26 | rate limit (50 shares/s sustained, burst 100); not a strike, but more than 5000 violations in a minute closes the connection | no |
| 30 | malformed request / unknown method / bad login | |
| 31 | banned | |
| 32 | too many connections | |

25 strikes within 60 s ban the IP for 10 minutes (loopback is exempt from bans).

### 3.4 `set_target` notification (vardiff)

`{"method":"set_target","params":{"share_target":"<u64 hex>"}}`. Applies to the shares found from now on; the pool also accepts the previous target for
30 s to cover shares in flight.

### 3.5 `hint` (optional, measured rate)

`{"id":N,"method":"hint","params":{"sigma_rate":8000.0}}` tells the pool the worker's measured rate in `sigma` evaluations per second. The pool jumps
at once to the share target that matches (and sends `set_target`) instead of converging by steps of 4: a fast GPU that starts with the default
small-CPU target would otherwise flood the pool for ~15 s. At most 3 hints per session (error 26 beyond); a lying hint only hurts the liar (an easy
target is rate limited, a hard one only lowers its own share rate). The reference worker sends one after 4 s of mining, and never sends more than ~40 shares/s (burst 80) whatever the target.

### 3.6 `ping` and `stats`

`ping` returns `{"pong":<unix time>}`. A worker must send *something* at least every 10 minutes (the reference worker pings every 30 s when idle) or
the connection is closed. `stats` returns `{"accepted","rejected","share_target","balance","paid"}` for the session's address.

## 4. How the pool verifies a share

The pool **recomputes** the jackpot of tile `(ti, tj)` from its own copy of the matrices (`work::tile_jackpot`: `O(tile * n * rank)`, no full GEMM) and
compares it with the share target. A worker can therefore not claim a share it did not compute, and it does not have to be trusted with the
matrices. A forged `extra_nonce`, tile or job only costs the pool one tile computation and earns the sender strikes.

A share that is below the network `target` becomes a block: the pool builds `Header{version 2, height, prev, timestamp, miner = pool key, class Native,
extra_nonce, ti, tj, jackpot, tx_root, target, a_root, b_root, accounts_root}`, the tile proof (`work::build_proof`) and the job's transactions, and
submits it with the normal block validation (`net::submit_block`). `accounts_root` is computed when the job is created (job transactions applied to a copy
of the state, then the block reward). The block is relayed like any other.

## 5. Share target and vardiff

A share of target `T` represents `2^64 / T` tile attempts; its **weight** is `2^96 / T` (fixed point), so weights add up to work whatever
the target of each worker was. The target of every worker is adjusted to about one share every `share_secs` (default 15 s):

- first adjustment after 10 s without a share, otherwise every `retarget_secs` (default 30 s), or earlier after 30 shares;
- factor bounded to 4 per step, no change inside the 0.8-1.25 band;
- never harder than the network target (a block must also be a share);
- the initial target comes from `sigma_rate` (or a small default) and converges in a few steps.

## 6. Accounting: PPLNS

- Window = the last shares whose total weight reaches `pplns_n` (default 2.0) times the weight of one share at the network target. The window
  spans block boundaries (classic PPLNS), which removes the incentive to hop between pools.
- When the pool finds a block, the **net reward** (`emission + fees of the block's transactions`, minus the pool fee `fee_bps`, default 1 %) is split
  across the window in proportion to weight. The split is frozen at that moment and is exact (the remainder of the integer division goes to the largest
  holder; nothing is created or lost).
- The block is `pending` until it is `confirmations` blocks deep (default 10) and still on the best chain; then each credit is added to the miners'
  balances. A block displaced by a reorganisation becomes `orphaned` and credits nothing.
- The finder gets no bonus: the reward is by work only.

## 7. Payouts

Every `payout_interval` seconds (default 600) each miner whose balance is at least `min_payout` (default 1 VRL) receives one `Transfer` signed by the
pool key (fee 0), at most `max_payouts_per_round` (50) per round. The balance is debited when the transaction is created and tracked as *in flight*:

- included (the pool's nonce moved past it) and then final after `confirmations` more blocks, when it moves to the payment history;
- lost from the mempool: re-submitted every maintenance tick (2 s);
- undone by a reorganisation: detected (nonce back below) and re-submitted.

The pool never pays more than its on-chain balance minus what is already in flight. `GET /api/stats` exposes `money.solvent` (pool wallet balance covers
what is owed plus the pending blocks) and the pool prints a warning when it does not.

## 8. Persistence

`<data-dir>/pool/pool-state.bin` (bincode, atomic write via a temporary file, previous copy kept as `.bak`) holds the share window, balances, blocks, payments
and in-flight transfers; saved every 30 s when changed and at shutdown. At start it is **refused** if it belongs to another genesis or another pool key
(it is never overwritten in that case). The node's chain lives in `<data-dir>/node`. Without `--data-dir` nothing is persisted (tests only).

## 9. HTTP statistics (read-only, `--http ADDR`)

`/` (small dashboard), `/health`, `/api/stats`, `/api/blocks`, `/api/payments`, `/api/miner/<vrl1...>` (balance, pending, in flight, paid, share of the
window, workers with estimated rate in sigma/s, last payments). Bind it to localhost and publish it through a reverse proxy.

## 10. Running it

Pool operator (dedicated wallet; testnet example):

```
vearl-wallet new pool.json
vearl-pool --wallet pool.json --password-file pool.pw --genesis genesis.json --peer SEED:9000@KEY --listen 0.0.0.0:9000 \
           --pool-listen 127.0.0.1:3333 --http 127.0.0.1:8650 --data-dir /var/lib/vearl-pool --fee 1 --min-payout 1 --confirmations 10
```

Expose `--pool-listen` through a TLS proxy for the public (see section 3). Fund nothing: the pool key only receives rewards and sends payouts.

Worker:

```
vearl-poolminer --pool pool.example.org:3333 --address vrl1yourpayoutaddress --worker rig1 --gpu       # CUDA build
vearl-poolminer --pool pool.example.org:3333 --address vrl1yourpayoutaddress --threads 8               # CPU
```

The worker reconnects by itself (1 s doubled up to 30 s). It prints its rate, accepted/rejected shares and blocks every 10 s.

## 11. Security notes and limits (read before running a public pool)

- **Hot key.** Rewards land on the pool key and payouts are signed with it: use a dedicated wallet, keep the minimum there, run the pool as an
  unprivileged user. Anyone who gets the key steals the balance.
- **Block withholding.** A worker can submit shares but silently drop a share that would be a block (the pool then loses its reward share). This is inherent
  to pools with untrusted workers; the reference pool has no detection. Mitigations are operational (known miners, monitoring `rejected` vs expected
  blocks per work).
- **No TLS, no authentication.** The payout address is the only identity: anyone can mine for any address (that is harmless: they only give work). Do not
  put secrets in the protocol; terminate TLS in front for public use.
- **Denial of service.** Per-line size, idle and login timeouts, per-IP and total connection caps, token bucket on submissions, strikes and temporary
  bans bound the cost of hostile clients; behind a proxy all clients share the proxy's IP, so apply the per-IP limits in the proxy instead.
- **Single point of failure.** One pool process, one node. If the pool crashes, workers wait and reconnect; the state file keeps the books. No failover.
- **Timestamps.** A job's timestamp is fixed when created (refreshed every `refresh_secs`, 30 s). Blocks found on an older job of the same parent carry
  that older (still valid) timestamp.
- **Only Native Work.** Pool shares do not carry useful jobs; the useful-work market is for solo workers.
- Rounding: balances and payouts are integers (atoms, `1 VRL = 10^8`). The pool fee absorbs rounding remainders of the block split only through the exact
  split rule above (they stay with the largest holder).

## 12. What the tests check (`tests/pool.rs`, `pool::tests`)

- `tile_jackpot` equals the jackpot of the full GEMM and the one `verify_tile_proof` recomputes.
- Two real CPU workers on a real node: blocks found by the pool are accepted by the chain; every block's brute reward is `emission + fees` of the real
  block; the split of each block sums exactly to its reward; funds are conserved (`confirmed rewards = owed + in flight + paid + pool fee`); the pool's
  on-chain balance equals mined rewards minus transfers included; both miners receive real on-chain payments; statistics agree.
- Hostile client: submit before login, bad protocol, bad address, bad worker name, double login, duplicate share, foreign slot, unknown job, out-of-range tile,
  share above the target, oversized line, repeated garbage, per-IP cap; none of them credits anything.
- Vardiff moves the target; state survives a restart (chain and pool), a foreign key or genesis is refused; PPLNS window arithmetic, bounded window,
  one-time confirmation, orphans credit nothing.

Not tested yet: a deep reorganisation that orphans a pending block on a live pool, a pool with thousands of workers, the CUDA worker against the pool for
a long run, TLS in front.
