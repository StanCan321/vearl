#!/usr/bin/env python3
"""VEARL - economic simulation v2 (stdlib only). Replaces the "time-sharing" model of sim.py.

Why a new model: in VEARL a GPU that runs a client job does the client's noised GEMM, so it produces
mining attempts at the SAME rate as a native GPU ("duplex", see Pass, arXiv 2606.06700 for the
theoretical framework). Clients therefore do not displace mining: they add to it. sim.py assumed the opposite.

ALL the economic parameters are HYPOTHESES (nothing here predicts a price or a profitability). The only figures
measured come from the hardware (proto1_gpu, docs/03): ~150 W under load on A10, ~9 % consensus overhead.

Sections: 1) empty market -> saturated equilibrium  2) security (attack cost, confirmations)  3) emission and inflation
           4) difficulty dynamics after a hashrate shock  5) price shock with delayed response of the miners
"""
import math
import random
import statistics as st
from dataclasses import dataclass


@dataclass
class Econ:
    # hardware (A10): MEASURED power; capex and rental = hypotheses
    gpu_power_kw: float = 0.15
    elec_usd_kwh: float = 0.10
    capex_usd_h: float = 0.15
    rental_usd_h: float = 0.50          # what an attacker who RENTS a GPU pays (hypothesis)
    duplex_overhead: float = 0.08       # overhead of a job GPU-hour vs native mining (transfers, bond, decoding)
    # protocol (values from docs/02, provisional)
    block_time_s: float = 30.0
    reward0: float = 20.0
    halvings_per_year: float = 1.0      # 1 halving ~ 1 051 200 blocks
    tail_frac: float = 0.025            # floor = 0.5 VRL = 2.5 % of reward0
    burn_frac: float = 0.10
    token_usd: float = 0.10

    @property
    def cost_gpu_h(self):
        return self.gpu_power_kw * self.elec_usd_kwh + self.capex_usd_h

    def reward(self, years):
        return max(self.reward0 * 0.5 ** (years * self.halvings_per_year), self.reward0 * self.tail_frac)

    def emission_usd_h(self, years=0.0):
        return self.reward(years) * (3600 / self.block_time_s) * self.token_usd


def equilibrium(e: Econ, demand_gpu_h: float, years=0.0):
    """Competitive equilibrium (zero marginal profit).
    N GPUs in total ALL produce attempts at the same rate (native AND jobs): each receives r = E/N per hour.
      - D <= E/c: clients are served at the marginal price of the duplex overhead (mining subsidises the computation)
      - D  > E/c: N = D GPUs must be attracted; price p = c(1+delta) - E/N  (>0)
    """
    E, c, d = e.emission_usd_h(years), e.cost_gpu_h, e.duplex_overhead
    n_native = E / c
    if demand_gpu_h <= n_native:
        n, price = n_native, d * c
    else:
        n = demand_gpu_h
        price = max(c * (1 + d) - E / n, d * c)
    r = E / n
    # client revenue -> workers; 10 % of the escrow is burned, so the client pays price/(1-burn) in all
    client_pays = price / (1 - e.burn_frac)
    burn_usd_h = demand_gpu_h * client_pays * e.burn_frac
    attack_gross = n * e.rental_usd_h            # rent as much hashrate as the honest ones (50 %)
    attack_net = max(attack_gross - E, 0.0)      # the attacker also mines during the attack and collects the attempts
    return dict(n=n, job_price=price, client_pays=client_pays, r=r, E=E, burn_usd_h=burn_usd_h,
                attack_gross=attack_gross, attack_net=attack_net, native=n - min(demand_gpu_h, n))


def section1(e):
    print("## 1. Equilibrium (year 0): empty market -> saturated\n")
    print(f"cost of a GPU-hour c = {e.cost_gpu_h:.3f} $; emission E = R*3600/T*price. GEMM 8192^3 ~ 13.6 ms => "
          f"{13.6e-3 / 3600 * 1e6:.1f} micro-GPU-hours.\n")
    print("| token price | demand (GPU-h/h) | total GPUs | job price ($/GPU-h) | client pays ($/GPU-h) | $/GEMM 8192^3 | tickets $/GPU-h | burn $/h | gross attack cost $/h | net $/h |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for price in (0.01, 0.10, 1.00):
        for d in (0, 50, 500, 5000):
            x = equilibrium(Econ(token_usd=price), d)
            gemm = x["client_pays"] * 13.6e-3 / 3600
            print(f"| ${price:.2f} | {d} | {x['n']:.0f} | {x['job_price']:.3f} | {x['client_pays']:.3f} | {gemm:.2e} | {x['r']:.3f} | {x['burn_usd_h']:.1f} | {x['attack_gross']:.0f} | {x['attack_net']:.0f} |")
    print("\nReading: as long as the demand is below E/c, the computation is almost FREE for the client (mining")
    print("subsidises it); beyond that, the price rises toward c(1+delta). Security (attack cost) grows with the demand")
    print("since the job GPUs also mine. This result depends on the duplex hypothesis (attempts at the same rate).\n")


def section2(e):
    print("## 2. Security: confirmations needed depending on the value at risk\n")
    print("50 % attacker (double spend). Cost to hold the majority for k blocks =")
    print("n * rental * (k * T / 3600), net of the attempts it mines itself. k such that this cost >= value at risk V.\n")
    print("| scenario (price, demand) | GPUs | gross cost $/h | V=$1 k | V=$10 k | V=$100 k | V=$1 M |")
    print("|---|---|---|---|---|---|---|")
    for price, d in ((0.01, 0), (0.10, 0), (0.10, 500), (1.00, 500), (1.00, 5000)):
        x = equilibrium(Econ(token_usd=price), d)
        row = []
        for v in (1e3, 1e4, 1e5, 1e6):
            hours = v / x["attack_gross"]
            k = hours * 3600 / e.block_time_s
            row.append(f"{k:,.0f} blocks ({hours:,.1f} h)")
        print(f"| ${price:.2f}, D={d} | {x['n']:.0f} | {x['attack_gross']:.0f} | " + " | ".join(row) + " |")
    print("\nReading: at launch (little value, few GPUs) the cost of an attack is very low. A testnet has no value")
    print("to protect; a mainnet should not accept amounts > attack cost over the confirmation time.\n")


def section3(e):
    print("## 3. Emission and inflation (30 s/block, yearly halving, floor 0.5 VRL/block)\n")
    blocks_year = 365.25 * 86400 / e.block_time_s
    supply, rows = 0.0, []
    for y in range(0, 10):
        r_start, r_end = e.reward(y), e.reward(y + 1)
        # integral of the exponentially decaying emission over the year, floor included (approximated by 52 steps)
        emitted = sum(e.reward(y + (k + 0.5) / 52) * blocks_year / 52 for k in range(52))
        inflation = emitted / supply * 100 if supply else float("inf")
        supply += emitted
        rows.append((y, r_start, emitted, supply, inflation))
    print("| year | starting reward | emitted in the year (M VRL) | cumulative supply (M VRL) | annual inflation |")
    print("|---|---|---|---|---|")
    for y, r0, em, sp, inf in rows:
        print(f"| {y} | {r0:.2f} | {em/1e6:.2f} | {sp/1e6:.2f} | {'-' if inf == float('inf') else f'{inf:.1f} %'} |")
    print("\nWith the floor (tail emission) the supply does not cap: inflation tends toward floor*blocks/supply (~ 2 %/year "
          "at 40 M). The 10 % burn of the escrows offsets it in part if the demand is strong.\n")


def next_target(target, dt, block_time=30):
    e = block_time
    adj = min(max(dt, e // 2), e * 2)
    return max(int(target * (15 * e + adj) // (16 * e)), 1 << 16)


def simulate_blocks(hash_before, hash_after_fn, n_blocks, seed, block_time=30):
    """Poisson process of the blocks; difficulty = EXACT protocol RULE (next_target, integers, dt truncated to the second)."""
    rnd = random.Random(seed)
    two64 = 2 ** 64
    target = int(two64 / (hash_before * block_time))     # initial equilibrium: hashrate * probability = 1/T
    t = 0.0
    last_ts = 0
    out = []
    for i in range(n_blocks):
        h = hash_after_fn(i)
        rate = h * target / two64
        dt_real = rnd.expovariate(rate)
        t += dt_real
        ts = max(int(t), last_ts + 1)
        out.append((dt_real, target))
        target = min(next_target(target, ts - last_ts, block_time), two64 - 1)
        last_ts = ts
    return out


def section4(e):
    print("## 4. Difficulty: reaction to a hashrate shock (exact protocol rule, 300 trials)\n")
    print("| shock | blocks to return to 30 s +/- 25 % (median) | worst interval between 2 blocks (median) | total duration of disturbance (median) |")
    print("|---|---|---|---|")
    for label, factor in (("hashrate /2", 0.5), ("hashrate /10", 0.1), ("hashrate /100", 0.01), ("hashrate x10", 10.0)):
        need, worst, dur = [], [], []
        for seed in range(300):
            blocks = simulate_blocks(1000.0, lambda i, f=factor: 1000.0 * (f if i >= 20 else 1.0), 1200, seed)
            post = blocks[20:]
            # first index where the 20-block moving average is within +/- 25 % of 30 s
            ok = None
            for j in range(0, len(post) - 20):
                m = sum(b[0] for b in post[j:j + 20]) / 20
                if 22.5 <= m <= 37.5:
                    ok = j + 20
                    break
            need.append(ok if ok is not None else 1180)
            worst.append(max(b[0] for b in post[:max(need[-1], 1)]))
            dur.append(sum(b[0] for b in post[:max(need[-1], 1)]))
        print(f"| {label} | {st.median(need):.0f} | {st.median(worst):.0f} s | {st.median(dur)/3600:.2f} h |")
    print("\nReading: the difficulty varies by at most ~6 %/block; after 90 % of the miners leave, it takes several")
    print("dozens of blocks (hence hours) to get back to 30 s. During that time the chain is slow AND cheap to attack.\n")


def section5(e):
    print("## 5. Price shock with progressive departure of the miners (day by day)\n")
    print("The token price falls by 80 % in 10 days; the GPUs leave the network at 20 %/day of the gap with the equilibrium.\n")
    base = Econ(token_usd=0.10)
    n0 = equilibrium(base, 500)["n"]
    n, rows = n0, []
    price_path = [0.10] * 5 + [0.10 * (1 - 0.8 * min(k, 10) / 10) for k in range(0, 40)]
    minratio = 1.0
    for day, p in enumerate(price_path):
        target_n = equilibrium(Econ(token_usd=p), 500)["n"]
        n = n + 0.2 * (target_n - n)
        ratio = n / n0
        minratio = min(minratio, ratio)
        if day in (0, 5, 8, 10, 15, 20, 30, 44):
            rows.append((day, p, target_n, n, equilibrium(Econ(token_usd=p), 500)["attack_gross"] * n / target_n if target_n else 0))
    print("| day | price | equilibrium GPUs | GPUs present | gross attack cost $/h (with the GPUs present) |")
    print("|---|---|---|---|---|")
    for day, p, tn, nn, ac in rows:
        print(f"| {day} | ${p:.3f} | {tn:.0f} | {nn:.0f} | {ac:.0f} |")
    print(f"\nLowest GPU level: {minratio*100:.0f} % of the initial level. A progressive departure is absorbed by the difficulty "
          "(section 4), but the attack cost follows the price, not only the number of GPUs.\n")


def main():
    e = Econ()
    print("# VEARL economic simulation v2 (hypotheses, no promise of profitability)\n")
    section1(e)
    section2(e)
    section3(e)
    section4(e)
    section5(e)


if __name__ == "__main__":
    main()
