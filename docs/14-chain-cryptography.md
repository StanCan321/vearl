# 14 — Chain cryptography

Labels: `[IMPL]` implemented and tested · `[LIT]` based on a published standard · `[HYP]` not demonstrated. **No home-made primitive**: standard building blocks are composed. Testnet, no external audit.

## 0. Block header v2 and accounts root `[IMPL]`

The block header (`docs/07` §2) now carries, in addition to the v1 fields: `a_root`, `b_root` (roots of the block's work) and `accounts_root` (Merkle root of the accounts **after** the block, `docs/07` §4 quater).

**What this enables**: a client that does not hold the complete state can prove a balance through a Merkle path against `accounts_root` (`State::accounts_proof` / `verify_account_proof`), and verify a tile proof against `a_root`/`b_root` without knowing the job's data. Without these two roots, a light client already had to download and replay the whole chain (which is what `tools/light_client.py` does) to obtain any guarantee at all.

**Header-based light client `[IMPL]`** (equivalent to an SPV client). Network messages `GetHeaders{from}` → `Headers(Vec<(header, tile proof)>)` and `GetAccountProof(key)` → `AccountProof{height, tip, balance, nonce, index, count, path}`. The client (`chain.rs::LightClient`, and independently `tools/light_client.py --headers ADDRESS`) verifies each header **without the state**: chaining, increasing timestamp, target recomputed by the genesis rule from the headers alone (ASERT, LWMA-1, EMA or fixed), native work roots recomputed for the Native class, **tile proof recomputed** from its 16 rows of `A` and 16 columns of `B` alone (`spec_ref.tile_jackpot`, checked equal to the full jackpot on the frozen vectors), jackpot under the target; then it verifies a **balance** through a Merkle path against the tip's `accounts_root`.
Measured: on a real node (`tools/interop_p2p.sh`), 85 headers verified in 0.2 s and the recipient's balance (4.5 VRL) proven, at fixed difficulty and in ASERT; in Rust, same target and same cumulative work as the full chain for the four rules, and 8 header falsifications refused (jackpot, target, native root, parent, timestamp, miner, proof row, version) (`tests/light.rs`).
**Assumption and limits (SPV)**: the validity of the transactions, the authorisation of a useful block's miner (being the job's worker) and the value of `accounts_root` itself are **not** verified — they rest on the honest majority of work, as for any SPV client; no proof of the **absence** of an account (an `Ack` answer is refused); a single source of headers (no choice between several peers nor MESS braking); `accounts_root` covers accounts, not jobs in progress. This is not finality (`docs/13` §6): a root can disappear in a reorganization. A tile proof weighs ≈ 6 KiB at `n = 128` and ≈ 43 KiB at `n = 1024` per header (against 80 bytes for a Bitcoin header): this is the price of a useful proof of work verifiable without the matrices.

## 1. Post-quantum signatures `[IMPL]` `[LIT]`

An account can be **classical (Ed25519)** or **post-quantum (SLH-DSA-SHAKE-128s, FIPS 205)**. The protocol accepts both; nothing forces a migration.

| | Ed25519 | SLH-DSA-SHAKE-128s |
|---|---|---|
| Public key = address | 32 B | 32 B |
| Signature | 64 B | **7,856 B** |
| Signing (compute) | µs | **≈ 1.6 s** (measured, pure-Rust crate) |
| Verification | µs | **≈ 1.6 ms** |
| Assumption | discrete logarithm (broken by Shor) | hash functions (SHAKE256) |
| Bech32m address | `vrl1…` | `vrlq1…` |

**Design choices**
- The signature length distinguishes the scheme (`crypto::verify_sig`): 64 → Ed25519, 7,856 → SLH-DSA, any other length refused. The 32-byte public key remains the address in both cases, so no transaction format changed.
- **Domain separation**: the signed message is `bincode(tx)` with the **FIPS 205 context** `vearl/tx/pq/v1`; a signature made in another context is refused (test).
- **"Hedged" signature** (system randomness); verification does not depend on it.
- **Deterministic derivation** from the wallet phrase: `blake3` (domain `vearl/wallet/pq-account/v1`) → 32-byte seed → `blake3` XOF → the three FIPS 205 seeds (`SK.seed`, `SK.prf`, `PK.seed`). The same phrase restores the same PQ account (test). The address prefix `vrlq` avoids confusing the two schemes.
- Tools: `vearl-wallet pq-address --file F`, `vearl-wallet send --pq …`.

**Independent verification**: `tools/pq_ref.py` is an SLH-DSA-SHAKE-128s verifier **written from the standard** (Algorithms 4, 8, 11, 13, 17, 20). On signatures produced by the Rust crate `fips205`, the two implementations give **16/16 identical verdicts** (2 valid signatures accepted by both; alterations in R, FORS, WOTS+ chain, authentication path, last layer, message, key: refused by both). Replayed by `tools/audit_all.py` (the 16th check).

**Tests** (`tests/pq.rs`): deterministic derivation, every alteration refused, a different context refused, PQ transfer **inside a validated block** (balance, nonce, monetary-supply invariant), replay refused, altered signature refused by the state validation.

### Limits (to be read)
- The `fips205` 0.4.1 crate is **pure Rust, with no `unsafe`, but unaudited**; our Python verifier does not replace an audit.
- **The P2P channel is not post-quantum** (Noise XX / X25519): an adversary who records traffic today could decrypt it later. Since chain data is public the impact is limited, but session keys are not protected against Shor.
- An Ed25519 public key that has **already been published** remains exposed to a future quantum computer; only migrating the funds to a `vrlq` account fixes this.
- Cost in bytes: 7.8 KiB per transaction; **no per-byte fee schedule** exists yet (the cap of 256 transactions per block bounds the worst case at ≈ 2 MiB).
- The 1.6 s signature is slow for a service that signs a lot (the `s` parameter = small signatures, slow signing; the `f` parameter would sign ≈ 10× faster at the price of 17 KiB signatures).
- The market CLI (`vearl-cli`) still signs with Ed25519; only wallet transfers have the `--pq` option. The **protocol** (state, validation, mempool) accepts PQ transactions of any type.
- No aggregated signature, no multisig.
- The PoW (blake3, hashing) is not concerned by Shor; the effect of Grover on 256-bit hashes is a quadratic reduction of effort (of the order of 2^128), not a break `[LIT]`.
