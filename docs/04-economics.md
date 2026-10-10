# 04 — Economics: simulation, calibration, difficulty

Labels: `[HYP]` model hypothesis · `[MEASURED]` measured on the hardware · `[SIM]` reproducible simulation result (`proto0_econ/`) · `[IMPL]` implemented and tested.
**Nothing here predicts a price, a demand or a profitability.** These are orders of magnitude conditional on explicit assumptions, meant to spot bad settings before a public testnet.

## 1. The model changes: "duplex", not "time-sharing" `[HYP]`

The old simulation (`sim.py`) assumed that a GPU does *either* mining *or* a client job. In VEARL a GPU that runs a job does **the client's noisy GEMM**: it produces mining attempts at the same rate as a native GPU (measured overhead ~9 %, `docs/03`). Clients **add to** mining instead of replacing it; this is the "Duplexia" regime of Pass's framework (arXiv 2606.06700). `sim2.py` models this.

Inputs: cost of a GPU-hour `c = 0.15 kW × $0.10/kWh + $0.15 = $0.165` (power **measured** at ~150 W under load; electricity and amortisation assumed), rental for an attacker $0.50/GPU-h `[HYP]`, duplex overhead 8 % `[HYP]`, emission `R(t)·3600/T·price`.

## 2. Competitive equilibrium (year 0) `[SIM]`

All GPUs produce attempts at the same rate. At the marginal equilibrium: if the demand `D ≤ E/c` (E = emission in $/h), **clients are served at the duplex overhead alone** (mining subsidises the computation); beyond that, GPUs must be attracted, and the price rises toward `c(1+δ)`.

| token price | demand (GPU-h/h) | GPUs | job price ($/GPU-h) | $/GEMM 8192³ | gross attack cost ($/h) |
|---|---|---|---|---|---|
| $0.01 | 0 | 145 | 0.013 | 5.5·10⁻⁸ | 73 |
| $0.01 | 500 | 500 | 0.130 | 5.5·10⁻⁷ | 250 |
| $0.10 | 0 | 1,455 | 0.013 | 5.5·10⁻⁸ | 727 |
| $0.10 | 5,000 | 5,000 | 0.130 | 5.5·10⁻⁷ | 2,500 |
| $1.00 | 500 | 14,545 | 0.013 | 5.5·10⁻⁸ | 7,273 |

**Readings and warnings.**
1. As long as demand is small compared with `E/c`, the computation is **almost free** for the client: it is a *subsidy by inflation*, not a real market. Workers live on emission; the token must therefore have a value independent of usage.
2. Client fees are then negligible (< $1/h of burn in most cases): the **10 % burn is neither a revenue mechanism nor a deflation mechanism** before strong demand. Its real role is to **bound the rebate** (anti wash trading, `docs/02`).
3. **Security grows with demand** since job GPUs also mine (e.g. $0.01: $73/h with no client, $2,500/h with 5,000 GPU-h/h of demand). This is an advantage of duplex over time-sharing **if** the assumption "attempts at the same rate" holds in practice (to be checked on a real network).
4. The price per 8192³ GEMM is of the order of 10⁻⁷ to 10⁻⁶ $: any **fixed price floor in VRL is arbitrary** (see §7).

## 3. Security: how many confirmations? `[SIM]`

Cost for an attacker to rent as much power as the honest miners (≈ 50 %) for `k` blocks of 30 s, to be compared with the value `V` at risk:

| scenario | GPUs | gross cost $/h | V = $1k | V = $10k | V = $100k |
|---|---|---|---|---|---|
| $0.01, D=0 | 145 | 73 | 13.7 h | 137 h | 1,375 h |
| $0.10, D=500 | 1,455 | 727 | 1.4 h | 13.7 h | 137 h |
| $1.00, D=5,000 | 14,545 | 7,273 | 0.1 h | 1.4 h | 13.7 h |

**Consequence**: at launch, the security budget (emission × price) is **very low**: $10,000 would only be safe after more than 13 h of confirmations at $0.10. This is the classic problem of young PoW chains, not specific to VEARL. A testnet has no value to protect; **a mainnet should only accept amounts below the attack cost over the confirmation time**, and no parameter choice here creates security without an economic value behind the token.

## 4. Emission and inflation `[SIM]` (provisional parameters)

`R0 = 20` VRL/block, annual halving, floor 0.5 VRL/block: 21.0 M issued in the 1st year (1,051,200 blocks × 20), 10.5 M in the 2nd, supply ≈ 41.4 M after 6 years (figures recomputed from the constants, 30 s blocks); inflation (year's emission / supply at the start of the year) 50 % (year 2) → 3.3 % (year 5) → ≈ 1.3 %/year with the floor (year 7, uncapped supply). A choice to revisit: a gentler curve reduces the concentration of supply among the first miners; no data here allows a decision.

## 5. Difficulty: the flaw found and the fix `[SIM] [IMPL]`

With the original rule (EMA 1/16, block time bounded to [T/2, 2T]):
- **bias at steady state**: average block time 32.7 s instead of 30 s (truncating the solve time skews the mean);
- **slow recovery**: after a ×100 power drop, **124 blocks** and **up to ~2 h between two blocks** (a slow *and* cheap-to-attack chain during that time).

Comparison (200-300 trials, Poisson process, rule applied identically):

| algorithm | steady state: mean / std dev | /2: blocks (worst interval) | /10 | /100 | ×10 |
|---|---|---|---|---|---|
| A — current EMA (dt ≤ 2T) | 32.7 s / 33.3 s | 38 (200 s) | 82 (761 s) | 124 (7,003 s) | 87 (91 s) |
| B — EMA, dt ≤ 10T | 27.1 s / 27.8 s | 23 (152 s) | 34 (433 s) | 40 (3,056 s) | 92 (87 s) |
| **C — LWMA-1 (N=30)** | **30.3 s / 31.3 s** | 26 (166 s) | 40 (513 s) | 50 (4,328 s) | **50** (92 s) |

B is fast on the way down but **biased** (27 s) and slow on the way up; **C has no bias and reacts quickly in both directions**. **Decision: LWMA-1** (Zawy) `[LIT]`, implemented in integers (`lwma_next_target`, solve times bounded to [1, 6T], per-block change bounded to [×0.5; ×2]), genesis parameter `Retarget::{Lwma, Ema, Fixed}` (`Fixed` for tests and fast devnets). Tested: target unchanged when blocks are on time, reaction in the right direction and bounded, no overflow at extreme values, aberrant timestamp tolerated, and **return to ~30 s after a 95 % power drop** with the exact protocol rule. Observed on a real node: the pace falls from "instant blocks" to the 30 s scale within a few dozen blocks.
**Limits**: LWMA is known for timestamp-manipulation attacks (bounded here by `MAX_FUTURE = 60 s` and solve time ≥ 1); the parameters (N=30, bounds) are not optimised; no hop-mining attack simulation. (ASERT replaced LWMA-1 as the default afterwards: see `docs/13` §5.)

## 6. Price shock `[SIM]`

A drop of 80 % in the price over 10 days, with GPUs leaving at 20 %/day of the gap: the fleet falls to ~34 % of the initial level in a month; the attack cost follows **the token price**, not only the number of GPUs. The difficulty absorbs the gradual departure (§5); a sudden departure (×100) remains the most dangerous situation.

## 7. Parameter recommendations (testnet → to be revalidated before any mainnet)

| parameter | current value | recommendation |
|---|---|---|
| retargeting | LWMA-1, N=30 | **adopted** (replaces the EMA); ASERT later (`docs/13`) |
| burn at job creation | 10 % | keep as long as it serves to **bound the rebate**; do not present it as revenue |
| worker bond | 10 % of net escrow | keep; capital cost negligible (locked ≥ 5 min) |
| escrow floor `MIN_ESCROW` (fixed 1 VRL) | fixed in VRL | **badly specified**: a fixed token price makes no sense against a GEMM at 10⁻⁷ $; it only serves to discourage job spam. To be replaced by an **adaptive** floor (fee market, EIP-1559 style, or a multiple of the median of recent escrows) `[PLAN]` |
| emission | 20 VRL, annual halving, floor 0.5 | acceptable for a testnet; **to be decided with market data** |
| challenge window / answer delay | 300 s / 120 s | consistent with the 60 s clock drift; never below |

## 8. What is not modelled
Strategic behaviours (selfish mining, block withholding, eclipse attacks), endogenous price volatility, bandwidth and storage costs, real demand, competition from other chains for the same GPUs, regulation. The equilibrium model assumes a fluid entry/exit of homogeneous GPUs (A10).

## 9. Reproduce
```bash
python3 proto0_econ/sim2.py                 # equilibrium, security, emission, difficulty shock, price shock
python3 proto0_econ/difficulty_compare.py   # EMA vs LWMA-1
cd vearl-core && cargo test --release --test protocol lwma
```
