# State of the art: Pearl (PRL) and matrix Proof-of-Useful-Work

Web research of 2026-09-25. Labels: `[SOURCE]` stated by the cited source · `[NOTE]` remark by VEARL. Market figures (hashrate, price) come from third parties and have not been re-checked.

## 1. What Pearl does

- A Bitcoin-fork L1 (transparent transactions), mainnet launched on 27 April 2026, developed by Pearl Research Labs. `[SOURCE: hashrateindex, alearesearch]`
- "cuPOW / NoisyGEMM" proof of work: the miner computes `C' = (A+E)(B+F)` where `E`, `F` are **low-rank** matrices derived from a public seed (the block hash). It hashes the **transcript** of the partial sums per tile; if the hash is under the target, it is a "jackpot". The noise is removed afterwards (`O(n²r)`) to recover the true `A·B`. Announced overhead `1+o(1)`; Pearl announces +5.1 % (Llama 70B, 4×H200) and +3.9 % (DeepSeek V3.2, 8×H200), measurements **by the project**, not verified. `[SOURCE: pearlresearch.ai/research/blog/proof-of-useful-work]`
- Verification: recompute **one tile** + Merkle proof of the opened rows/columns; the verifier checks values in [−64, 64], Merkle consistency, recomputed tile = jackpot, jackpot < target. `[SOURCE: arXiv 2606.04819]`
- Theoretical basis: Komargodski, Schen, Weinstein, *Proofs of Useful Work from Arbitrary Matrix Multiplication* (arXiv 2504.09971 / ePrint 2025/685). `[SOURCE]`
- Code: repository `pearl-research-labs/pearl` (`pearld` node in Go, Oyster wallet, vLLM miner, post-quantum XMSS, Plonky2, ISC licence). `[SOURCE: GitHub]`
- `[NOTE]` Divergence between sources: several articles speak of a ~60 KB Plonky2 SNARK; the empirical study concludes that the current protocol **is not ZK** (a plain Merkle proof) and the official blog rules out ZK for large models. The repository contains Plonky2/STARK code. Real status: **to be verified in the code**.

## 2. Its central flaw, measured by a third party

Empirical study *The Usefulness Gap in Proof-of-Useful-Work* (arXiv 2606.04819, June 2026) `[SOURCE]`:

- The verifier **does not check where A and B come from**. The miner chooses its matrices; "validity" is GEMM correctness, **not** usefulness. The protocol treats usefulness as an "economic property, not a cryptographic guarantee".
- The authors build a miner ("turbo4") that runs the whole NoisyGEMM pipeline with **random matrices**, without any AI inference, and is accepted.
- Of 8,012 observed workers (~21 % of the hashrate), all have inference-capable hardware, but the dominant software **contains no inference code**. The study's conclusion: the network (24 EH/s, ~320,000 GPU equivalents, ~112 MW estimated) "produces zero useful AI compute".
- Statistical defence (kurtosis test to tell network weights from randomness): can be bypassed by sampling a truncated Gaussian (kurtosis 2.97 against 3.33).
- Market effect: GPU rental price +38 %, utilization 57 % → 94 %, displacing legitimate research workloads.
- Leads the authors list: on-chain compute contracts, TEE attestation, cryptographic provenance commitments, differentiated **economic incentives**. `[SOURCE]`
- `[NOTE]` These figures come from a single study; to be cross-checked before any public citation.

## 3. Useful theoretical framework

*The Economics of Proof-of-Useful-Work* (R. Pass, arXiv 2606.06700) `[SOURCE]`: at competitive equilibrium, the cost of a majority attack stays tied to the **block reward**, even if the work is useful; three regimes (Bitconia = classic PoW, Fortessia = "duplex" replaces mining, Duplexia = the reward subsidizes inference). `[NOTE]` this validates the VEARL separation: security = block reward, usefulness = client fees.

## 4. Lessons for VEARL

To take over: tile proof + Merkle (`O(T²n)` verification), low-rank noise, hashed transcript, `O(n²r)` decoding.
To avoid: letting the miner choose A and B with no link to a client; claiming "useful" without a mechanism that enforces it; relying on a statistical distribution test.
To add: cryptographic job → matrices binding, an on-chain market, a usefulness rebate bounded by burn, element-level fraud proofs, privacy (Pearl is transparent).

## Sources
- [Pearl – Proof of Useful Work (official blog)](https://pearlresearch.ai/research/blog/proof-of-useful-work)
- [GitHub repository pearl-research-labs/pearl](https://github.com/pearl-research-labs/pearl)
- [The Usefulness Gap in Proof-of-Useful-Work (arXiv 2606.04819)](https://arxiv.org/abs/2606.04819)
- [Proofs of Useful Work from Arbitrary Matrix Multiplication (arXiv 2504.09971)](https://arxiv.org/abs/2504.09971) · [ePrint 2025/685](https://eprint.iacr.org/2025/685)
- [The Economics of Proof-of-Useful-Work (arXiv 2606.06700)](https://arxiv.org/abs/2606.06700)
- [Pearl (PRL) explained – Hashrate Index](https://hashrateindex.com/blog/pearl-prl-ai-compute-cryptocurrency/)
- [Pearl – Alea Research](https://alearesearch.substack.com/p/pearl-proof-of-useful-work-for-ai)
