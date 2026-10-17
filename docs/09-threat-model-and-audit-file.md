# 09 — Threat model and audit file

**Status: testnet, no external audit.** This document is an entry point for a reviewer or auditor: what is to be protected, against whom, what is already covered (and by what reproducible evidence), what is not, and where to look first. It makes no claim of the type "impossible to attack", "100 % anonymous" or "ASIC resistant".

## 1. Scope and trust assumptions

| element | what is assumed |
|---|---|
| Primitives (BLAKE3, Ed25519, Noise XX/X25519/ChaCha20-Poly1305/BLAKE2s, Argon2id, Bech32m) | standard libraries (`blake3`, `ed25519-dalek`, `snow`, `argon2`, `chacha20poly1305`, `bech32`), **no home-made primitive**. Not reviewed by us. |
| The transcript folding (XOR-rotation of the partial sums) | **`[HYPOTHESIS]`**: has no formal analysis; only empirical resistance is studied (`docs/03`). It is the most original and the least proven component. |
| Clocks | tolerance of 60 s (`MAX_FUTURE`); the challenge windows use **two** durations (blocks AND timestamp seconds). |
| Honesty of the majority of work | **no protection against a majority of work**; at launch the security budget is tiny (`docs/04`). |
| User's workstation | out of scope: malware, memory capture, phishing. |

**Assets**: (1) the integrity of the ledger (balances, escrows, bonds, monetary supply); (2) the availability of nodes; (3) the keys (wallets, the node's network key); (4) the fairness of the compute market (an honest worker is paid, a cheater is sanctioned, an honest client does not lose its escrow).

**Adversaries considered**: a hostile network peer (forged messages); a miner trying to obtain blocks with less work than an honest one; a worker that cheats on a result or withholds data; a client that accuses wrongly or burns its workers; a Sybil/eclipse attacker; an attacker with a lot of computing power.

## 2. Catalogue of attacks and status

Legend: **COVERED** (measurement + reproducible evidence), **PARTIAL** (mitigated, limits named), **OPEN**.

| # | threat | measure | status | reproducible evidence |
|---|---|---|---|---|
| A1 | **Structured low-rank inputs**: an attempt becomes `fold_k/rank` times cheaper (0.06× measured at the start) | genesis parameter rule: `fold_k ≤ 1.05·rank`, `rank ≤ 63`, `n` a multiple of 128 | COVERED for the admitted parameters; **`[HYPOTHESIS]`** for other forms of structured inputs | `attackreport`, `tests/attacks.rs`, `docs/03` |
| A2 | Reusing work from one block/job/miner in another | `sigma` binds parent, miner, class, `extra_nonce`, transaction root, timestamp | COVERED (design) | `tests/protocol.rs`, vectors `consensus-work-v1.json` |
| A3 | Forging a tile proof (false rows/columns) | Merkle openings of the 16 rows of `A` and 16 columns of `B` against the committed roots; tile **recomputed** | COVERED | `tests/protocol.rs`, block fuzz (§3) |
| A4 | Wrong job result | Freivalds test on the client side (error ≤ 2⁻²⁴ over 24 rounds) + **single-entry fraud proof** (openings A, B, C); 10 % bond: half to the challenger, half burned | COVERED | `tests/protocol.rs`, `market_ref.py` |
| A5 | Being paid before the end of the challenge through a burst of fast blocks | window = blocks **AND** timestamp seconds | COVERED | `tests/protocol.rs` (block burst refused), market vector E |
| A6 | Withholding the result's data | on-chain `Fetch`/`Serve`, `SlashUnserved` after `serve_secs` | COVERED | `tests/da.rs`, market vector C |
| A7 | Wash trading to inflate the reward | *rebate* ≤ amount burned at job creation | COVERED (design) | `tests/protocol.rs`, `docs/04` |
| A8 | Answer too costly at small `n` (GPU yield) | batched mining, noise generated on the GPU | performance, not security | `gpubench` |
| A9 | Manipulation of the difficulty (timestamps, intermittent miner) | **ASERT** (aserti3-2d, half-life 900 s) by default: absolute schedule, no exploitable window; future tolerance 60 s (next block ≤ 4.7 % easier) | COVERED by simulation: periodic miner +27.8 % (LWMA) → −0.1 % (ASERT); **no proof** | `docs/13` §4–5, `consensus_sim.py daa`, `asert_ref.py` (official vectors) |
| A10 | Forged transaction/block/message that makes a node panic | bounded decoding (16 MiB), validation that returns `Err`, never an `unwrap` on external data | COVERED by fuzz, **not** by proof | `tests/fuzz_state.rs` (1.1 M mutants, 0 panics, conservation checked), `tests/fuzz_net.rs` (462,000 inputs, 0 panics, no modified block accepted); **a running node fuzzed over the network** (`vearl-netfuzz`, `src/netfuzz.rs`): 9,000 attempts / ≈ 18,000 messages (raw bytes, truncated handshakes, random encrypted messages, every client and peer request mutated) against a `node --worker` process: 0 panics in its log, always reachable, mining continuous |
| A11 | Breaking the conservation of the monetary supply | invariant checked after each block; fuzz of mutated signed transactions | COVERED (empirical) | `check_supply_invariant`, `fuzz_state.rs` |
| A12 | Rust node diverging from the specification | byte-by-byte specification + independent Python implementations (work, chain, market, network light client) | COVERED for the listed scope; **same author** | `docs/07`, `docs/08`, `tools/*_ref.py`, `tools/light_client.py`, `tests/vectors_gate.rs` |
| A13 | Eclipse / Sybil | limits per IP and /16, outbound peers in distinct /16s, pinnable keys | **PARTIAL**: basic mitigation, **no Sybil defence** | `tests/net.rs` |
| A14 | Denial of service by volume | token bucket 100 msg/s, connection caps, bounded mempool (5,000, 64 per sender), misbehaviour score + IP ban; blocks ≤ 256 transactions and ≤ 4 MiB; synchronisation answers ≤ 8 MiB | PARTIAL: no large-scale load test | `tests/net.rs`, `tests/consensus.rs` |
| A15 | 51 % attack / selfish mining | MESS braking of deep reorganizations (local rule) | **PARTIAL**: late deep reorganization forbidden as long as `q/(1−q) < f(t)` (simulation); short attack and **selfish mining not addressed** (threshold ≈ 1/3 simulated) | `docs/13` §3–4, `tests/consensus.rs` |
| A16 | Key theft (wallet, network key) | Argon2id + ChaCha20-Poly1305, password never as an argument, keys never sent to the node | PARTIAL: depends on the user's workstation | `tests/wallet.rs` |
| A17 | Confidentiality of transactions and job data | — | **OPEN**: nothing is confidential (public chain, matrices deposited at the worker) | — |
| A18 | Specification errors shared between code, text and second implementer | — | **OPEN**: same author; a third-party implementer is needed | — |
| A19 | Deep reorganizations, transactions lost in an abandoned branch | search for the common ancestor, **atomic** reorganization from state snapshots (restored if a block is invalid), replacement only if greater work (× f(t) with MESS); eviction of nonce gaps | COVERED for a merge of two chains of ~155 blocks (jobs and transfers on each side): convergence in a few seconds, monetary-supply invariant true, merged network functional, and the transfers confirmed in the abandoned branch are **put back into the mempool then included** in the winning chain | `tools/fork_test.py`, `tests/net.rs` |
| A20 | A network test blocked for 60 s under heavy load (seen once) | **probable cause found by the endurance test (A21)**: a worker transaction that had become inapplicable blocked all the following ones | PARTIAL: fix in place, not proven as the single cause | `tools/soak.py` |
| A21 | **Transaction stuck at the head of a sender's nonce sequence** (e.g. a bid that arrives after the end of the auction): all the following wait behind it and `has_pending` prevents the worker from resending them → the worker stops working | eviction of stuck transactions (inapplicable head or missing nonce for more than 30 s); the worker rebuilds its transactions at the current state | COVERED by the endurance test (before: jobs `Open` for minutes; after: 22 honest jobs paid, 13 cheaters sanctioned in 5 min, 4 hard stops); no unit test | `tools/soak.py`, `VEARL_DEBUG_MEMPOOL=1` |
| A22 | A worker that restarts loses the data deposited by clients (matrices, weights) and can no longer run its jobs or serve the requested data → bond lost | deposits persisted on disk (`putdata.bin`, 512 MiB max) and reloaded at start-up; a published job whose result is lost is recomputed | COVERED by the endurance test (17 to 22 deposits reloaded per restart); **the result of programs/inferences in progress is not resumed** (the worker recomputes them from the start) | `tools/soak.py` |
| A23 | **Level 2** (Monte Carlo, rendering): undetected inexact result | **sampling** audit (the client recomputes m chunks) + one-chunk fraud proof, judged by on-chain recomputation (bounded cost) | **PARTIAL by design**: the guarantee is probabilistic (`1−(1−φ)^m`); an isolated falsified chunk is only seen with probability m/N; profitability of cheating not demonstrated | `tests/l2.rs`, `tests/net.rs`, `docs/10` §7 |
| A24 | Level-2 kernels diverging between implementations (a node could not judge another's proof) | integers only, specification `docs/10` §3–4, second implementer | COVERED for 18 chunk outputs (byte for byte); **same author** | `tools/l2_ref.py`, `tests/vectors/l2-kernels-v1.json` |

## 3. How to reproduce the evidence (from `vearl-core`, on a machine with Rust, Python 3, and a GPU for the `gpu` feature)

**In one command**: `cargo build --release --bins && python3 tools/audit_all.py [--quick] [--cargo]` replays the 17 independent checks (six Python validators on the frozen vectors, blocks written by Python replayed by Rust, three Rust/Python differential fuzzers, the independent operation engine, an SLH-DSA verifier written from FIPS 205, ASERT against Bitcoin Cash's official vectors) and prints an OK / FAILED table (`--cargo` adds the whole Rust test suite). Measured: `--quick` (300 blocks per fuzz) **17/17 in 50 s** on the VPS. The detail, check by check:

```bash
cargo test --release --features gpu --no-fail-fast  # 171 tests (protocol, consensus, market, network, light client, GPU, fuzz)
./target/release/vearl-netfuzz 127.0.0.1:9000 3000 1   # fuzz of a running node over the network (exit code 1 if it dies)
FUZZ_ITERS=1500 cargo test --release --test fuzz_state -- --nocapture   # ~1.1 M transaction mutants
FUZZ_ITERS=6000 cargo test --release --test fuzz_net   -- --nocapture   # ~460,000 mutated blocks/messages
python3 tools/vearl_ref.py fuzz 2000 target/release/vearl-model         # 2000 tensor programs, independent engine
python3 tools/soak.py 300 11                                              # endurance: 4 real nodes, chaos (SIGKILL), jobs of every type, cheater
python3 tools/fork_test.py 90                                             # partition of two groups of nodes then merge (deep reorganization)
python3 tools/spec_ref.py tests/vectors/consensus-work-v1.json          # 40 checks of the work layer
python3 tools/chain_ref.py tests/vectors/chain-native-v1.json           # chain of 330 blocks + 13 refusals + 6 blocks mined in Python
python3 tools/market_ref.py tests/vectors/market-v1.json                # market: 52 blocks, 17 refusals, 10 blocks mined in Python
python3 tools/diff_fuzz.py 3000 11                                       # Rust/Python differential fuzz of the market: identical decisions and final state
python3 tools/diff_fuzz_prog.py 3000 11                                  # same for tensor programs (4 fraud types)
python3 tools/l2_ref.py tests/vectors/l2-kernels-v1.json                # level-2 kernels (Monte Carlo, rendering): 18 outputs identical to Rust
python3 tools/prog_ref.py tests/vectors/prog-v1.json                    # tensor programs: 229 blocks, 6 fraud types + data availability, 15 refusals, 78 Python-authored blocks (including an MLP with biases and a table)
python3 tools/tiled_ref.py tests/vectors/tiled-v1.json                  # tiled products: 56 blocks, 2 fraud types + availability; --extend: 17 Python-authored blocks
python3 tools/diff_fuzz_tiled.py 1500 1                                  # Rust/Python differential fuzz of tiled products (7,800 identical decisions measured)
bash tools/interop_p2p.sh                                                # real node + Python light client (Noise XX)
python3 tools/client_demo.py                                             # Python SDK: honest worker paid, cheater sanctioned
```
(Python prerequisites: `pip install blake3 numpy cryptography noiseprotocol`)

## 4. Where an auditor should look first (in order of risk)

1. **`work.rs`, the folding and the `security_check` rule**: the only non-standard construction; look for inputs or strategies that reduce the cost of an attempt below that of an honest computation (`docs/03`).
2. **`state.rs` (`apply_tx`, `verify_fraud`, `challenge_open`) and `prog.rs` / `mlp.rs` / `tiled.rs`**: financial state machines; money rules (escrow, bond, burning, rebate) and fraud proofs with Merkle openings.
3. **`chain.rs` (`validate_inner`, difficulty)**: order of the checks, integer arithmetic, bounds.
4. **`net.rs` / `secure.rs`**: synchronisation by replay, management of memory and rate, key pinning.
5. **`gpu/vearl_gpu.cu`**: bit-for-bit identity with the CPU (tested: work, exact GEMM `gpu_exact_gemm_equals_cpu`, Merkle roots `gpu_merkle_root_equals_cpu`); risks of divergence on other GPU architectures (tested on two cards: A10 and, on 2026-10-06, RTX 5060 Ti / Blackwell sm_120; all 7 GPU tests pass on both).
6. **`prog.rs` integer semantics** (softmax/LayerNorm): documented tolerances; compared with a second engine on ~2,000 programs.

## 5. What this file does not cover

Economic model (`docs/04` is a simulation, not an audit), legal compliance, operational security of a public deployment (TLS, firewall, backups: `docs/06`), formal analysis of the folding, behaviour on other GPUs, confidentiality, and any statement about the value of tokens (**testnet: no value**).
