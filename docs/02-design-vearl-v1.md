# VEARL v1 — Fee-Bound Useful Work (FBUW)

Status: **testnet prototype, not audited, no property formally proven.** Labels: `[IMPL]` implemented and tested in `vearl-core` · `[HYP]` hypothesis · `[RESEARCH NEEDED]` unresolved · `[PLAN]` not done yet.

## 1. The idea in one sentence

A protocol cannot prove that a computation is useful to a third party; it can prove that **someone paid, and burned a share, for that precise computation to be done**. VEARL therefore ties the consensus work to matrices *committed by a client who has put money in escrow*, and makes this economic constraint the only "certificate of usefulness" it claims.

## 2. What VEARL changes compared with Pearl

| Topic | Pearl (according to the sources, §01) | VEARL v1 |
|---|---|---|
| Origin of A and B | chosen by the miner, not checked | **committed by a job** (Merkle roots over the rows of A and the columns of B, in the `SubmitJob` transaction) `[IMPL]` |
| "Usefulness" | an economic property that is not enforced; a random miner is accepted | a `Job`-class miner must be the **declared worker** of an escrowed job; otherwise the block is classed `Native`, with no usefulness claim `[IMPL]` |
| Compute market | outside the protocol / future work | **inside the protocol**: escrow, worker bond, challenge delay, settlement, expiry `[IMPL]` |
| Result check | — | Freivalds on the client side (`O(n²)`, error ≤ 2⁻ᵏ) + **element-level fraud proof** (~1.2 KB, verified in `O(n)`) that slashes the bond `[IMPL]` |
| Incentive for usefulness | none on the protocol side | **rebate bounded by the burn**: `rebate ≤ 10 % of the burned escrow`; a fake job = net loss or zero `[IMPL, tested]` |
| Security with an empty market | yes | yes: `base_reward` is identical for every valid block `[IMPL]` |
| Anti-replay | seed = block hash | seed = parent + miner + class + **tx_root** + timestamp + extra_nonce: changing the beneficiary or the transactions invalidates the work `[IMPL, tested]` |
| Privacy | transparent (Bitcoin fork) | `[PLAN]` phase 2 |
| GPU hardware | int8, vLLM miner | int8 tensor-core CUDA kernel, bit-exact against the CPU `[IMPL, measured: ~80-90 TOPS on A10]` |

What this table does **not** say: that VEARL is "better" in practice. It is a design difference; superiority remains to be shown by measurements (GPU throughput, overhead, real market).

## 3. Protocol

### Work (`work.rs`, spec **v2**) `[IMPL]`
`C' = (A+E)(B+F)`, `E=EL·ER`, `F=FL·FR` of rank `r`, derived from `σ` (blake3-XOF) with noise entries in `{-1,0,1}`: `A,B ∈ [-64,64]` and `r ≤ 63` guarantee that `A+E`, `B+F` fit in int8 (so they can run on int8 tensor cores).
Each `16×16` tile of `C'` is one attempt. The transcript is **folded** every `fold_k` steps of `k` (XOR + a step-dependent rotation, on the current accumulator, down to 64 words), then hashed by a single blake3 chunk (`"vearl/jackpot/v2" ‖ σ ‖ ti ‖ tj ‖ fold`). Winner if `hash[..8] < target`. Decoding `C = C' − AF − EB − EF` in `O(n²r)`.
Why fold: hashing every partial sum costs ~`4/r` bytes per MAC, which would far exceed the tensor-core throughput; folding makes the hash negligible while still forcing the computation of every partial sum. **Security of the folding has not been analysed** `[RESEARCH NEEDED]`.
Verification of a block: Merkle openings of `T` rows of A and `T` columns of B against the job's roots, recomputation of the tile (`O(T²n)`). Testnet: `n=128, T=16, r=63, fold_k=64`. **These parameters come from the attack analysis (docs/03)**: the gain of an attacker who chooses low-rank inputs is ~`fold_k/rank`, so `WorkParams::security_check` (called by `Chain::genesis`) enforces `fold_k ≤ 1.05·rank`, `rank ≤ 63` (so that `A+E` always fits in int8), etc.

### GPU kernel (`gpu/vearl_gpu.cu`, feature `gpu`) `[IMPL, measured]`
int8 WMMA on tensor cores, 128×128 CTA (8 warps, 4×2 sub-tiles per warp, smem staging), folding in registers, **blake3 re-implemented on the GPU**. The `gpu_matches_cpu` test checks **bit-for-bit** equality of `C'` and of all the jackpots with the CPU reference (several `n`, `r`, `fold_k`): a node without a GPU can therefore verify blocks.
Measurements on A10 (`gpubench`, rank 16, attempts = 16×16 tiles). **Warning: these parameters (rank 16, `fold_k` ≥ 256) are UNSAFE** (an attempt costs ~0.06× the honest computation there, see docs/03); they measure the kernel, not a consensus setting. With the safe parameters (rank 63, `fold_k`=64, n=8192) and the current kernel (PTX `mma.m16n8k32`, folding in registers, noise on tensor cores): 12.7 ms of kernel + 0.9 ms of noise, i.e. ~9 % overhead compared with the bare GEMM **of that same kernel**; but this kernel (~87 TOPS) is still ~2.3× slower than cuBLAS (~199 TOPS measured), see docs/03 §3. The tables below date from the original kernel and unsafe parameters: they are kept for the record only.

| n | fold_k | int8 TOPS (kernel) | noise time / kernel time | attempts/s |
|---|---|---|---|---|
| 4096 | 256 | 78.7 | 0.17 / 1.75 ms ≈ 10 % | 2.5·10⁷ |
| 4096 | 1024 | 87.3 | ≈ 10 % | 2.7·10⁷ |
| 8192 | 16 | 23.2 | 0.81 / 47.5 ms ≈ 2 % | 5.1·10⁶ |
| 8192 | 512 | 76.9 | 0.78 / 14.3 ms ≈ 5 % | 1.05·10⁷ |
| 8192 | 4096 (≈ no folding) | 79.0 | 0.76 / 13.9 ms ≈ 5.5 % | 1.06·10⁷ |

Reading: at `n=8192`, `fold_k ≥ 512` costs ≲ 3 % of the kernel throughput; at `fold_k=16` it costs ~3.4×. (Single-measurement figures, average of 5 runs, no error bar.) The `fold_k` parameter therefore directly trades "transcript security" against "efficiency". The kernel reaches ~80-90 TOPS; the dense int8 peak of the A10 was **not** measured here (the cuBLAS FP16 GEMM measured in P1 gives 73 TFLOPS), and a factor of ~2 can probably still be gained (`cp.async` double buffering, `ldmatrix`, larger tiles). Pearl's overhead figures (~4-5 %) are the project's own, on other hardware and models: **no direct comparison is possible**.

### Two work classes `[IMPL]`
- `Native`: matrices derived from the protocol (per epoch). **No usefulness is claimed.** This is the base security.
- `Job(id)`: the job's matrices. The miner must be the `worker` (job `Claimed`/`Posted`).

### Rewards `[IMPL]`
`block = base_reward(h)` (identical for all classes) `+ rebate` if `Job`, with `rebate = min(base/2, rebate_left)` and `rebate_left` initialised to the job's **burned amount**. Hence:
- empty market → security intact;
- usefulness subsidised **at most** up to what a client has burned;
- **wash trading** (being the client and the worker of one's own job): cost = burn (10 %), maximum gain = rebate ≤ burn ⇒ never better than mining `Native` (test `wash_trading_a_fake_job_is_not_profitable`).

### Life cycle of a job `[IMPL]`
`SubmitJob` (escrow, 10 % burned) → `ClaimJob` (10 % bond) → mining in `Job` class (useful result = decoded `C`) → `PostResult(c_root)` → 10-block window → `Accept` or `Settle` (worker paid) or `Fraud` (bond: ½ to the challenger, ½ burned; client refunded). Expiry: client refunded, bond burned if the worker delivered nothing.
Invariant checked in tests: `balances + escrows + bonds + burn = genesis + emission`.

### Difficulty `[IMPL]`
**LWMA-1** (30-block window, solve times bounded to [1, 6T], per-block change bounded to [×0.5; ×2], integers only); genesis parameter `Retarget::{Lwma, Ema, Fixed}`. Choice and comparison with the former EMA in docs/04 §5. (ASERT was added later and is now the default: docs/13 §5.)

### End-to-end GPU mining `[IMPL, measured]`
`gpumine` (n=4096, `fold_k`=256, target ≈ 1 σ/block): every block mined by the GPU is **re-validated by the CPU validator** (Merkle proof + recomputation of **a single tile**). One σ (65,536 attempts) costs ~2.5-3 ms on the A10; verifying a block costs ~5.7 ms on CPU. **These figures do not yet show the intended asymmetry**: with a target of one σ per block, mining and verifying cost the same order of magnitude. At the real difficulty (thousands of σ per block) verification stays ~6 ms while mining is multiplied accordingly — that is the property that matters, and it has not been measured at high difficulty.

### Network (`net.rs`, bins `node` and `vearl-cli`) `[IMPL, prototype]`
TCP + length-prefixed bincode messages (16 MiB max), `Hello` (node id, genesis hash, cumulative work), block and transaction gossip, peer-address exchange, mempool (5,000 tx, 16 per sender, sequential nonces), **chain choice by cumulative work** with reorg by replay from genesis, rejection of timestamps > +60 s, per-peer rate limiting (100 msg/s), banning of peers that send an invalid block.
Tested: 3 nodes (propagation of blocks and transactions, discovery C→B→A); two partitioned miners that converge after reconnecting; a peer sending an invalid block is **disconnected** and the chain is unchanged. Also run as 3 separate processes on the VPS (2 competing miners + 1 passive node, transfer submitted through the CLI to the passive node): same tips on all 3, balance credited everywhere, supply invariant true.

### End-to-end compute market `[IMPL, tested]`
A node started with `--worker` automatically claims (`ClaimJob`) the jobs whose matrices it knows (deposited off-chain by the client through `PutJob`, recognised by their Merkle roots), switches to **Job class** to mine with the client's matrices, publishes the Merkle root of the result (`PostResult`), then settles (`Settle`) after the challenge window. The client (`vearl-cli job-finish`) fetches `C`, verifies it with Freivalds (24 rounds) and sends `Accept`, or locates the wrong entry and publishes the fraud proof.
Verified: (1) honest worker → job `Settled`, result = true `A·B`, ≥ 1 Job-class block mined for that job; (2) `--cheat` worker (one wrong entry) → detected by Freivalds, fraud proof accepted, job `Slashed`, supply invariant intact; (3) also run with real processes on the VPS: submission → `Claimed` → `Posted` → `Accept` → `Settled` in a few seconds.
**Persistence**: append-only block file, replay and re-validation at start-up, corrupted trailing fragment truncated (tested), atomic rewrite after a reorganization. The state (accounts, jobs) is rebuilt by replay, not saved separately.
**What is NOT solved here**: data availability (the worker serves `C` if it wants to; if it withholds it, the client has no recourse other than expiry — see limit 3); a single worker per job, first come first served; no payment for storage/transfer fees.

**Network limits**: no complete Sybil defence (an attacker who controls all of a node's connections controls its view); synchronisation by full replay is `O(chain length)`; no pruning; devnet keys come from **public** seeds.

### Network security (P4c) `[IMPL, tested]`
- **Encrypted and authenticated channel**: Noise XX (`Noise_XX_25519_ChaChaPoly_BLAKE2s`) through the `snow` library, no home-made primitive. A peer's identity = its static X25519 key; the node key is persisted (`node.key`, 0600 on Unix). A plaintext peer (old protocol) is rejected; a tampered encrypted byte kills the session.
- **Pinned keys**: `--peer host:port@KEY_HEX` refuses the connection if the presented key differs (anti-impersonation of a known peer).
- **Basic anti-eclipse**: global inbound connection cap (24), per IP (2) and per /16 (4); never two outbound peers in the same /16; known peers persisted (anchors retried at restart); alert if there is no outbound peer for 30 s; cap on simultaneous handshakes (anti slow-loris).
- **Anti-spam**: per-peer rate (100 msg/s), misbehaviour score (invalid transactions), ban at the first serious fault (invalid block), de-duplication of blocks already seen, anti-DoS filter on the header (announced jackpot under the target *before* the costly verification).
- **Design flaw found and fixed**: the challenge window was measured in blocks only; on a fast network (or with a miner chaining blocks), 10 blocks pass in under a second and a cheating worker was **paid before the client could verify**. It now also requires `CHALLENGE_SECS = 300` s of timestamp (bounded by the validators' clock, so it cannot be consumed by a burst of blocks). Test: 40 blocks at +1 s do not close the window, the fraud proof remains admissible.
- **Remaining limits**: Noise does not protect against an attacker who multiplies identities (**Sybil**: the per-IP/§16 caps make it more expensive, not impossible with many addresses); peers' IPs remain visible (no network anonymity); no external audit of the use of `snow`; the 300 s window is an uncalibrated parameter; no encryption of files on disk.

### Quantized inference jobs (`mlp.rs`) `[IMPL, tested]`
The first useful compute type beyond an isolated GEMM: a network of `L ≤ 8` layers, `Z_l = W_l · X_{l-1}` (exact int32, **this GEMM is what is mined with noise**), `X_l = act_l(Z_l)` (arithmetic shift + saturation; ReLU and `[0,64]` for hidden layers, `[-64,64]` for the last). `X_0` is committed by columns, each `W_l` by rows; the worker publishes layer by layer the root of `Z_l` (one leaf per entry) and of `X_l` (by columns, since `X_l` is operand B of the next layer).
- **Mining**: block class `Layer{job, layer}`; A = `W_l`, B = `X_{l-1}` (roots committed by the client or published by the worker); layer `l` is only unlocked once `X_{l-1}` is published. Same reward rule as simple jobs (base + rebate bounded by the burn).
- **Client-side verification**: Freivalds on each GEMM (`O(n²)`), element-wise check of the activations, consistency of the served roots with those published on-chain; total cost `O(L·n²)` instead of `O(L·n³)`.
- **Two element-level fraud proofs**, verified on-chain in `O(n)`: *GEMM* (`Z_l[i][j] ≠ ⟨W_l[i,:], X_{l-1}[:,j]⟩`, Merkle openings of the row, the column and the Z entry) and *activation* (`X_l[i][j] ≠ act(Z_l[i][j])`). An error in an intermediate layer is provable on its own: the following layers computed from the published value are "correct" relative to it.
- **Tests**: honest cycle and payment; GEMM fraud in layer 1; activation fraud in layer 0; false accusations rejected; access control and layer order; per-layer price floor; bounded rebate; then end to end on the network (3 test nodes: honest worker paid, wrong GEMM slashed, wrong activation slashed) and with real processes on the VPS (4-layer network, 8 `LAYER` blocks, `Settled`).
- **Limits**: int8 integers only (values in `[-64,64]`), `n×n` matrices (the protocol's n); non-linear activations other than ReLU/saturation, normalisations and softmax are not covered; a single worker per job; **availability of the `Z_l`/`X_l` not guaranteed** (the client depends on the worker to retrieve them, as with simple jobs); weights and inputs are visible to the worker (no confidentiality).

### Reverse auctions and reputation `[IMPL, tested]`
- **`SubmitJobAuction`**: like `SubmitJob`, with an auction duration in timestamp seconds. During the auction, workers send `Bid{price}`; a bid must be **strictly lower** than the best (and ≤ net escrow, > 0). Nobody can claim before the end.
- **After the end**: the winner claims **at its price**; the client immediately recovers the difference between the net escrow and that price, and the worker's bond is computed on the price. The winner has `AUCTION_GRACE_SECS = 60` s of exclusivity; after that, **any worker can claim at the full price** (the bid of a defaulting winner is lost). With no bid, claiming at the full price is open as soon as the auction ends.
- **Reputation** (informative, not enforced by the protocol): `done / slashed / expired` counters per worker, updated at settlement, at proven fraud and at expiry (simple and inference jobs); queryable through `vearl-cli reputation`.
- **Worker node**: `--ask VRL` — bids its price if it beats the best bid, claims if it wins. Workers without `--ask` only claim fixed-price jobs (or at the full price after an auction with no taker).
- **Tested**: lowest price wins, equal or higher bid refused, zero price / outside escrow refused, premature claim or by a loser refused, refund of the gap to the client, payment at the bid price, grace delay, fixed-price jobs unchanged, absurd durations refused, reputation counters; then **two competing workers on the network** (bids 20 and 35 VRL): the cheaper one wins and runs the job, settled at 20 VRL, reputation +1 for it alone.
- **Limits**: bids are **public and sequential** (not sealed): a worker can wait until the last second to undercut ("sniping"); the lowest price wins regardless of reputation; a worker can submit bids below its real cost and then walk away (it loses its place but nothing else as long as it does not claim); the auction duration is in timestamp seconds, so it is bounded by the allowed clock drift (±60 s).

### Data availability (`Fetch` / `Serve` / `SlashUnserved`) `[IMPL, tested]`
**Problem.** A worker can publish `c_root` without having the data, or withhold it, and be paid automatically at the end of the challenge window.
**Mechanism (simple jobs).**
- During the window, the client can demand **on-chain** an entry `(i,j)` of the result (`Fetch`, ≤ 8 simultaneous requests, `FETCH_FEE` = 0.05 VRL put in escrow).
- The worker answers with `Serve` (value + Merkle path against `c_root`, verified in `O(log n²)`) within `serve_secs` (120 s in production, a genesis parameter): it then collects the fee, and the challenge window is **reopened** for at least 60 s / 3 blocks so that the client can verify the served value in `O(n)` against its own A and B.
- **No answer** within the delay: anyone sends `SlashUnserved`; the reporter collects half of the bond, the other half is burned, the client recovers escrow + fee. Any node that mines does this automatically.
- Automatic settlement is **blocked** while a request is pending; an `Accept` refunds pending requests.
**What this guarantees**: (1) one cannot be paid for a root without data; (2) a result that was *made up* but served consistently is detected from the first sampled entry (served value ≠ dot product), and the served value directly serves as a fraud proof (tested: "made-up" job slashed after **one** request); (3) a worker that withholds the data is slashed.
**What it does NOT guarantee**: the client does not obtain the complete result through this channel (one entry per request, each request costs fees and block space) — complete delivery stays **off-chain**. Corruption of **a single** entry in a result that the worker refuses to deliver is not detectable by sampling; only Freivalds on the complete result finds it. The mechanism therefore prevents "paid for nothing" and "made-up data", not "refused delivery of a correct result" (fair exchange: a known problem, with no solution without a third party or on-chain data).
- **Durations**: `challenge_secs` and `serve_secs` are genesis parameters (`Timing`, included in the genesis hash); tests shorten them.
- **Tested**: request served → worker paid and window reopened; settlement blocked while a request is pending; root without data → slash after the deadline (and not before) with refund of the client and gain for the reporter; unauthorised or invalid request/server rejected (index, duplicate, cap, third party, wrong path); made-up result detected through one entry; fees refunded at acceptance; then on the network: honest audit, a worker that withholds slashed automatically by a third-party miner, made-up result detected through a single entry.
- **Not covered**: inference jobs (their `Z_l`/`X_l` remain under the worker's sole responsibility); one request per entry is very costly for a wide audit (a full row = n requests): moving to per-row leaves would reduce this cost, not done.

### Tiled products (`tiled.rs`) `[IMPL, tested]`
A job `C = A·B` with `A` (M×K), `B` (K×N), `M, K, N` multiples of `n` (up to 8×8×8 blocks). The client commits each block `A_ik` (by rows) and `B_kj` (by columns); the worker publishes the root of each product `P_ijk = A_ik·B_kj` (one leaf per entry) and of each sum `C_ij = Σ_k P_ijk`.
- **Mining**: block class `Product{job,i,j,k}` = the noisy GEMM of ONE `n×n` block of the client. **No product is required to advance the chain**: the useful computation feeds the lottery continuously and any winning attempt can become a block (otherwise a job of 512 products would need 512 blocks, i.e. hours).
- **Two element-level fraud proofs**: *Product* (`P_ijk[r][c] ≠ ⟨A_ik[r,:], B_kj[:,c]⟩`, `O(n)`) and *Sum* (`C_ij[r][c] ≠ Σ_k P_ijk[r][c]`, `K/n` openings + one opening of `C`).
- **Client-side verification**: Freivalds on each product + exact sum (`O(#products · n²)`), consistency of the served data with the published roots.
- **Tests**: the splitting gives back exactly the naive rectangular product (1×3×2 blocks); honest cycle and payment; mining any product of the job (and refusal of other miners, indices outside the job, data of another block); fraud on a product; fraud on a sum; false accusations rejected; price floor **per product**; end to end on the network (honest rectangular product, wrong product slashed, wrong sum slashed); with real processes (256×384 by 384×256, 12 products).
- **Limits**: bounded size (8 blocks per dimension); the worker's reference computation is on the CPU and under the node lock (to be moved to the GPU and outside the lock for large sizes); data availability not guaranteed (like inference jobs); neither auctions nor data requests for this job type; dimensions that are not multiples of `n` require zero padding on the client side.

### Verifiable tensor programs (`prog.rs`) `[IMPL, tested]`
A sequence (≤ 16) of operations on int8 `n×n` tensors committed **by columns**; `T_0` is the client's input, `T_{k+1}` the output of operation `k`.
| operation | definition (integer, frozen) | fraud proof |
|---|---|---|
| `Gemm{w,x,shift,relu}` | `T = act((W_w·T_x) >> shift)`; **the GEMM is the mined work** (block class `Op{job,op}`) | *Gemm* (row of W, column of the input, entry of Z) and *Act* (entry of Z, output column) |
| `Add{a,b,relu}` | `T = saturation(T_a + T_b)` — residual connection | *Elem*: opens the source columns and the output column |
| `Map{x, Transpose \| Im2col \| Flatten}` | structured copy; `Im2col` (stride 1, padding, channels × window as rows, position × batch as columns) turns a **convolution into a GEMM** | *Elem* |
| `Pool{x, Avg k×k}` | integer mean rounded by `div_euclid` over disjoint windows | *Elem* (k² sources) |
`act`: arithmetic shift then saturation to `[-64,64]` (ReLU: lower bound 0). All the semantics are **integer and unambiguous**: two honest workers obtain bit-identical tensors, a condition for a fraud proof to be indisputable.
- **Verified**: `Im2col` + GEMM gives back **exactly** a naive 3×3 convolution with padding (4→8 channels, 8×8 images, batch 2); rounding, saturation and shifts on negatives; honest cycle and payment of the CNN (im2col → conv → pooling → im2col → conv → conv → **residual addition** → classifier) with one block mined for each Gemm whose input is published; the **six fault types** (middle GEMM, activation, im2col, pooling, residual addition, classifier) are each proven by a single proof and sanctioned; false accusations, unsuitable proof types and refused accesses; validation of the program (forward references, non-existent weights, program with no Gemm, im2col outside the tensor, non-dividing pooling, shift out of bounds, price floor per Gemm); then end to end on the network (honest paid, wrong GEMM/activation/im2col sanctioned) and with real processes (8 operations verified, `OP` blocks mined).
- **Column operations (P10c)**: `Softmax{x,len}` and `LayerNorm{x,len}` apply to each **column** (= one token, in the features × tokens layout), hence to leaves that are already committed: the *Col* fraud proof opens **two columns** (input and output) and recomputes the column in `O(n)`. `Softmax`: `e_j = EXP2_TABLE[max − v_j]` (exact table of `floor(65536·2^(−d/8))`, computed offline in decimal arithmetic, no floating-point function in the consensus), output `(64·e_j + S/2)/S ∈ [0,64]`. `LayerNorm`: mean and standard deviation **rounded to nearest**, `exact integer square root`, output `clamp(floor(((v−μ)·16 + σ/2)/σ), −64, 64)`. `GemmAT{a,x}` = `T_aᵀ·T_x` (attention scores `KᵀQ`, output `V·A` through a transposition): the "by rows" root of `T_aᵀ` **is** the "by columns" root of `T_a`, so no extra commitment is needed and the same *Gemm* fraud proof applies.
- **Accuracy of the integer semantics** (measured by the tests): the integer softmax stays within ±1 unit (out of 64) of the real softmax on 300 random columns; the normalisation stays within (8 + |output|/2)/σ + 2 units of the real normalisation for columns with a standard deviation ≥ ~7. **Below that (standard deviation of the order of 1-2 int8 units) quantisation dominates**: this is a limit of int8, not a bug, and must be measured on real models (P10d).
- **Demonstration transformer**: Q, K, V, `KᵀQ` scores, softmax, `V·A` output, projection, residual, normalisation, feed-forward (ReLU), residual, normalisation; the program is identical to a naive independent attention computation, the honest cycle is paid, six fault types (GemmAT ×2, Gemm, softmax, two normalisations) are proven and sanctioned, and the whole thing passes end to end on the network (honest, wrong GEMM, wrong softmax column).
- **Design choice**: mining is **opportunistic** (it remains possible during the challenge window); publishing all the operations at once therefore does not deprive the worker of its blocks.
- **Limits**: `n×n` tensors with zero padding (a real network must be split: see tiled products, not yet integrated into programs); **biases** fold into the weights through a row of 1s; no max-pooling; a single worker per job; availability of intermediate tensors not guaranteed; no confidentiality of weights/inputs. (Later additions — max-pooling, general convolutions, bias and table operations, model registry — are in docs/05 and docs/12.)

## 4. Known limits (read before believing anything)

1. **"Useful" = "paid and burned"**, not "proven useful". A client can pay for useless computation (at the price of the burn). The same compromise exists in every PoUW with a chosen input; we assume it explicitly. `[HYP]`
2. **Security of the low-rank noise and of the transcript hash: not analysed** for this variant (parameters chosen for the testnet). Any exploitation of correlations between tiles would break the fairness of the lottery. `[RESEARCH NEEDED]` — re-read the proof in the primary source (arXiv 2504.09971) and adapt it formally.
3. **Data availability**: partially handled (see "Data availability"): one can no longer be paid for a root without data nor for a made-up result, but complete delivery of a correct result stays off-chain. A client that withholds `A,B` after a `ClaimJob` still makes the worker lose its bond. Mitigations `[PLAN]`: refunded `Abort`, erasure-coded blobs, per-row leaves.
4. **Freivalds** proves "`C = A·B` with high probability", not "this GPU computed it". That is enough to pay for a result; it is not a proof of hardware work (that comes from the lottery).
5. **Worker selection**: first come first served in the basic mode; auctions and reputation exist (see above) but there is no SLA. `[PLAN]`
6. **Compute types**: initially a single job type (integer GEMM). AI inference = decomposition into GEMMs + non-linearities outside the GEMM; these are now covered by tensor programs (see above and docs/05, docs/12). `[IMPL]`
7. **Network**: basic P2P done (see above); Sybil and eclipse remain open (`docs/09`).
8. **Confidentiality** absent (phase 2: stealth addresses + commitments + proven ZK proofs, e.g. Halo2; the *inputs* of private jobs are a different problem, `[RESEARCH NEEDED]`).
9. **ASIC/specialisation**: an int8 GEMM kernel is in the zone where accelerators already exist. No "ASIC-proof" promise. The main defence is **real demand** (the same GPUs serve AI), not an algorithm.
10. Economic parameters (`R0`, 10 % burn, 10 % bond…) = **provisional**, not calibrated.

## 5. Tests `[IMPL]`
(`cargo test --release`; `--features gpu` adds the GPU tests, including Native and Job blocks mined on the GPU and accepted by the CPU validator.) Network: data availability, concurrent auctions, end-to-end inference, propagation, partition/reorg, invalid block, honest market, market with a cheater, disk restart, plaintext peer rejected, tampered session, pinned key, transaction spam, admission caps, half-open connections. Protocol: exact decoding; valid/altered tile proof/wrong commitment; Freivalds + localisation; beneficiary and tx_root anti-replay; job cycle + supply invariant; only the worker mines in `Job` class; fraud slashed + false accusation rejected; wash trading not profitable; direction of the difficulty adjustment; a burst of blocks against the challenge window. The current suite is ~170 tests (see docs/13 and docs/15), plus the independent cross-checks run by `tools/audit_all.py` (docs/09).

## 6. Roadmap

| Step | Content | Status |
|---|---|---|
| P0 | economic simulation | done |
| P1 | GPU benchmark (A10) | done |
| P2 | protocol core: work, market, fraud, tests, devnet | **done (CPU)** |
| P3 | int8 tensor-core CUDA kernel + folding + GPU blake3, bit-exact vs CPU, A10 measurements | **done** |
| P3b | GPU miner wired to the chain (`mine_gpu`, job resident in VRAM, decoding only for jobs); GPU blocks validated by the CPU validator | **done** |
| P3c | kernel optimisation (cp.async, ldmatrix, larger tiles), GPU decoding | to do |
| P4 | TCP P2P network: gossip, sync, reorg by cumulative work, mempool, basic anti-abuse, CLI; 3 nodes on the VPS | **done (prototype)** |
| P4b | **worker inside the node** (claims, mines in Job class, publishes the result root, settles), disk persistence, market CLI (`job-submit/status/finish`) | **done** |
| P4c | Noise XX encrypted/authenticated channel, pinnable peer keys, admission by IP and /16, fault scoring, challenge window in seconds, attack tests | **done (prototype)** |
| P5 | attack analysis of the work: **structured-input flaw found, attack implemented (identical jackpots), parameters fixed and enforced by the code**; remaining: other structured families, proof of the folding (docs/03) | **partly done** |
| P7 | economics: duplex simulation (equilibrium, security, inflation, shocks), **LWMA-1 difficulty replacing the EMA** (bias and slow recovery found by simulation), parameter recommendations (docs/04) | **done** |
| P10 | extended compute types (docs/05): P10a tiled/rectangular GEMM · P10b verifiable operations (residuals, unfolding, pooling, integer softmax) · P10c softmax/LayerNorm/transformer · P10d int8 model import · P10f level 2 (sandbox, Monte Carlo, tiled rendering) · P10e deterministic compute with checkpoints | see the lines below |
| P8 | **wallet** (24 words, Argon2id + ChaCha20-Poly1305 keystore, Bech32m addresses), **JSON genesis**, **HTTP API + explorer**, **faucet**, systemd units + nginx + install script (docs/06) | **done** |
| P8b | **wallets linked to mining, the worker and the market CLI** (`--miner-wallet`, `wallet:FILE`, `vrl1…` addresses; public seeds refused on a file genesis), per-address history, long wait, HEAD | **done** |
| P10a | **tiled / rectangular matrix products** (`tiled.rs`: mined n×n blocks, product and sum fraud proofs, worker, CLI, API) | **done** |
| P10b | **verifiable tensor programs** (`prog.rs`: mined Gemm + activation, residual addition, im2col/transposition, average pooling; Gemm/Act/Elem fraud proofs; demonstration CNN with a residual block) | **done** |
| P10c | **integer softmax and layer normalisation**, product between two computed tensors (`GemmAT`), verifiable **transformer encoder block** (14 operations, 8 mined GEMMs) | **done** |
| P10d | **int8 model import** (`model.rs`, `vearl-model`, `Flatten` operation), bit-for-bit conformance with an independent engine (1,997 random programs, 0 divergence), real *digits* CNN: 96.97 % in float as in int8 (see docs/05 §6) | **done** |
| P5b | kernel rewritten (PTX mma, noise on tensor cores, folding in registers): consensus overhead ~9 % vs the bare GEMM of the same kernel; 10× faster CPU validation of a block | **done** |
| P5c | bring the kernel closer to cuBLAS (~87 → ~199 TOPS), generate the noise on the GPU, overlap host preparation and compute | partly done (noise on GPU; see the annex) |
| P6a | **multi-layer quantized inference jobs** (noisy mined GEMMs + verified activations, element-level fraud proofs), worker and CLI, network tests | **done** |
| P6b | reverse auctions + reputation; **data availability through on-chain requests (Fetch/Serve/SlashUnserved)**; remaining: GEMM batches, minimum required reputation, SLA, availability for inference jobs | **partly done** |
| P5 | fuzzing, adversarial tests, network simulation; formal analysis of the noise | partly done (docs/09) |
| P6 | confidentiality (private testnet), multi-type jobs, auctions | to do |
| P9 | external audits, public testnet; mainnet **only** if the properties are demonstrated | to do |

Commands: `cd vearl-core && cargo test --release && cargo run --release --bin devnet`; GPU: `--features gpu` (`gpubench`, `gpumine`); network: `node --listen 127.0.0.1:9001 --mine-seed 1`, `node --listen 127.0.0.1:9002 --peer 127.0.0.1:9001`, `vearl-cli status|balance|transfer`.

## Annex — Batched GPU mining (measured)

At small `n`, a sigma only represents `(n/16)²` attempts (64 at n=128): a single CUDA block, hence 1 SM out of 80 busy, and the time is dominated by launches and copies. The miner now processes a **batch of sigmas** per kernel chain (`z` grid = sigma index; noise derived on several CPU threads). The result is **bit-for-bit identical** to the sigma-by-sigma computation and to the CPU reference (`tests/gpu.rs::gpu_batch_equals_sigma_by_sigma`); the mining semantics (first winning `extra_nonce`) is unchanged.

Measured throughput (A10, rank 63, fold_k 64, millions of attempts/s):

| n | before batching (1 sigma, CPU noise) | batch, noise derived on CPU | batch, noise on GPU (v1) | **after optimisation of the auxiliary kernels** |
|---|---|---|---|---|
| 128 | 0.74 | 6.6 | 31.4 | **68** (×92) |
| 256 | 2.3 | 12.2 | 45.5 | **78** |
| 512 | 6.1 | 16.4 | 55.5 | **79** |
| 1024 | — | — | — | **71** |

Optimisations of v1 (`ncu` profiling, `gpubench N 63 BATCH`): noise generation took 980 µs against 146 µs for the tiles at n=128, batch of 512 — cause: the blake3 message permutation read from an array in constant memory (dynamic indexing → local memory) and one-byte writes. Permutation written out explicitly + 16-byte writes: `noise_gen` 980 → 68 µs, and the jackpot hashing of the tile kernel gains 20 % along the way; noise packed into a single 16-byte write per thread. Results still bit-for-bit identical to the CPU reference (6 GPU tests).

The noise (blake3-XOF stream in `derive_key` mode, one thread per 64-byte block) is now generated on the GPU from the 48-byte headers alone: the CPU now sends only ~2.5 KB per batch of 512 sigmas instead of ~16 MB. **Bit-for-bit identical** to the CPU's `derive_noise` (test `gpu_noise_generation_matches_blake3_xof_for_many_shapes`: n ∈ {128, 256}, ranks 1, 16, 63). At large `n` (4096) the kernel reaches ≈ 122 TOPS int8 against ≈ 199 TOPS for cuBLAS `[MEASURED]`, so there is still room on the tile kernel itself.

**Test note.** `two_competing_workers_…` (network, auction) was unstable 1 time in ~15: with three miners without waiting, the chain often reorganises and the state read just after "Posted" could be undone a second later; the chain clock also advances faster than the wall clock. The test now checks the result once the job is settled. This was not an observed protocol flaw, but a 60 s hang was seen once under heavy parallel load and was not reproduced in 350 isolated runs `[UNEXPLAINED]`.

**GPU measurement on RTX 5060 Ti (Blackwell, sm_120), 2026-10-06** `[MEASURED]`: all 7 GPU tests pass bit for bit (blocks, noise, exact GEMM, batches); `gpubench 1024 63` gives ≈ 49 million attempts/s at a batch of 8 (72 TOPS int8 on the tile kernel, up to 80 TOPS at larger batches), against ≈ 71 million quoted above for the A10. Development measurement on a rented card, not a published figure.
