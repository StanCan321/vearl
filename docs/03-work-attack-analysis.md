# 03 — Attack analysis of the consensus work (P5)

Labels: `[MEASURED]` executed and reproducible (`attackreport`, `tests/attacks.rs`) · `[MODEL]` analytical derivation validated by counting · `[HYP]` hypothesis · `[RESEARCH NEEDED]`.
This document does **not** prove that the work is safe. It closes **one** family of attacks that was found, and honestly lists those that remain.

## 1. The flaw found: structured inputs make attempts cheaper `[MEASURED]`

**Scenario.** An accomplice client (or the attacker himself, as client *and* worker) commits in a job **low-rank** matrices A, B, extreme case `A = B = 0` (valid for the protocol: values in [-64, 64]). The worker then mines `Job`-class blocks, which collect the base reward like any block.

**Why this is a security problem and not only a usefulness one.** Every attempt (tile) is the same lottery ticket, whatever the real cost of obtaining it. If degenerate inputs make attempts *k* times cheaper, the attacker obtains *k* times more hashrate per euro: the cost of a 51 % attack is divided by *k*. This is the same class of flaw as the one noted at Pearl (inputs chosen by the miner), but here measured on VEARL.

**Attack.** With `A' = A + EL·ER = [UA | EL]·[VA ; ER]` (rank `a = a0 + r`) and `B' = B + FL·FR = [WB | FL]·[ZB ; FR]` (rank `b = b0 + r`), the accumulator of a tile after step `s` is
`acc_s = UA'_rows · G_s · ZB'_columns`, where `G_s = Σ_{k<K_s} VA'[:,k]·WB'[k,:]` (an `a×b` matrix, **common to all tiles**).
Cost per tile and per folding step: `≈ T²·b` MACs, against `T²·fold_k` for the honest computation. **Attacker gain ≈ fold_k / b, with b ≥ rank.**

**Results.**
- `tests/attacks.rs`: the jackpots obtained by the attack are **bit-for-bit identical** to those of the honest dense computation, for several ranks (0/0, 4/4, 0/8, 16/16) over the 64 tiles: the attack is a real mining shortcut. The cost model coincides exactly with the number of operations actually executed.
- Old parameters (`rank=16`, `fold_k=256`, n=4096): an attempt costs **0.06 × the honest computation** (~16× cheaper). CPU measurement on a full σ (n=512, naive loops on both sides): attack **~11× faster** (time ratio 0.09; model 0.07).
- The parameters that had been used for the previous GPU benchmark were therefore **unsafe**; the corresponding throughput figures remain valid as kernel measurements, not as consensus parameters.

## 2. Fix: the noise rank must cover the folding interval `[MODEL, MEASURED]`

The noise imposes `b ≥ rank`: no input can have a lower effective rank. It is therefore enough that `fold_k ≤ ~rank` for the gain to fall to ~1.

Analytical table (n=4096, attacker cost / honest cost per attempt; `ok` = accepted by `WorkParams::security_check`) for `A = B = 0`:

| rank \ fold_k | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|
| 16 | 1.00 ok | 0.50 | 0.25 | 0.13 | 0.06 | 0.03 |
| 32 | 2.02 ok | 1.01 ok | 0.50 | 0.25 | 0.13 | 0.06 |
| 48 | 3.04 ok | 1.52 ok | 0.76 | 0.38 | 0.19 | 0.09 |
| 63 | 4.00 ok | 2.00 ok | **1.00 ok** | 0.50 | 0.25 | 0.13 |

Measurement on a full σ, n=512, `rank=63, fold_k=64`: the attack is **1.7× slower** than the honest computation (model 1.12).

**Decision.** `WorkParams::TESTNET = { n:128, tile:16, rank:63, fold_k:64 }`. `Chain::genesis` **refuses** any parameter set that does not pass `security_check` (tile 16, `n` a multiple of 128, `rank ≤ 63` so that `A+E` always fits in int8 — |A| ≤ 64 and |E| ≤ rank —, `fold_k` a multiple of 16 dividing `n`, and **`fold_k ≤ 1.05·rank`**).

**Why `rank ≤ 63`**: a value that overflowed int8 on the GPU but not on the CPU reference (i32) would break consensus. The upper bound is the worst case, not the typical case.

## 3. The price of this fix, and the kernel optimisation (P5b) `[MEASURED, A10]`

First costing (original WMMA kernel, n=8192, `rank=63, fold_k=64`): 19.7 ms of kernel + 4.7 ms of noise = 24.4 ms, i.e. **+75 %** compared with the bare GEMM of that same kernel (13.9 ms). The kernel was then rewritten:

| step | kernel (ms) | noise (ms) | comment |
|---|---|---|---|
| original WMMA | 19.7 | 4.7 | folding through shared memory, noise in scalar loops |
| noise on tensor cores | 19.4 | 0.97 | the noise is a GEMM of depth `rank`: ×5 faster |
| **PTX `mma.m16n8k32` + `ldmatrix` + `cp.async` + folding in registers** | **12.7** | **0.93** | register layout documented (PTX ISA): folding is done without going through shared memory |

**Current state (n=8192, `rank=63, fold_k=64`)**: 12.7 ms + 0.9 ms. Compared with the bare GEMM of the **same** kernel (12.55 ms): folding ≈ +1 %, noise ≈ +7 %, i.e. **~9 % consensus overhead**. The kernel stays bit-exact with the CPU reference (3 GPU tests, including Native and Job blocks mined on the GPU and validated by the CPU).

**What these 9 % do NOT say**: the kernel itself only reaches ~87 int8 TOPS, whereas **cuBLAS measures ~199 TOPS** on this card (`proto1_gpu/int8_ceiling.cu`, n=8192: 5.53 ms). A useful GEMM done with cuBLAS would therefore be ~2.3× faster than the VEARL kernel; the **real** overhead for a miner who would otherwise have used cuBLAS is ~2.5× in time. Closing this gap requires software pipelining of the fragments (register double buffering), larger CTA tiles, possibly a modified CUTLASS GEMM. Not done; not to be confused with the 4-5 % announced by Pearl (measured by the project on other hardware and models, against their own kernel).

**Next bottleneck: the host.** The "wall" throughput (~11·10⁶ attempts/s at n=8192) is limited by the CPU work per σ (blake3 noise generation, conversion, ~10 ms of transfers) and not by the GPU (~19·10⁶ attempts/s for the kernel alone). To do: generate the noise on the GPU and overlap preparation and compute. (Done since then: see docs/02, annex.)

**Validator-side verification**: recomputing a tile at `rank=63` cost ~25 ms (n=4096) with badly ordered loops; after reordering the loops (identical results) it costs **~2.6 ms**.

Constraint added by the kernel: `fold_k` a multiple of **32** (`mma` has k=32), checked by `security_check`.

## 4. What remains open `[RESEARCH NEEDED]`

1. **Other families of structured inputs.** I closed the **low rank**. *Toeplitz / circulant / low displacement-rank* matrices, or *block-sparse* ones, admit fast products (FFT). `A + E` stays "structured + rank r"; an algorithm exploiting the displacement rank could reduce the cost of the prefix partial sums. Not analysed.
2. **Security of the folding.** XOR + rotation of the current accumulators, then blake3. I have no proof that an attacker cannot predict the folded digest without computing all the partial sums. The formal argument of the primary source (arXiv 2504.09971) concerns its own construction, not mine.
3. **Noise with entries in {-1,0,1}.** Chosen to fit in int8. Its distribution is narrower than that of uniform noise over a large interval; the effect on the indistinguishability of tiles is not analysed.
4. **Defence in depth not implemented**: rejecting jobs whose rows/columns opened in a tile proof are degenerate (zeros, repetitions). Useful but circumventable (an adaptive attacker chooses "clean" low-rank inputs); it does not replace the bound given by the noise.
5. **Economics**: even with a cost per attempt ≥ honest, a fraudulent job stays almost free (escrow recovered, only the 10 % burn is lost); the real bound is `rebate ≤ burn`, not the price of the computation (see 02, wash trading).

## 5. Reproduce
```bash
cd vearl-core
cargo test --release --test attacks      # valid attack + cost model + refusal of parameters
cargo run --release --bin attackreport   # tables and timing measurements
cargo run --release --features gpu --bin gpubench 8192 63   # GPU price of the safe parameters
```
