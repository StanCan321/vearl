#!/usr/bin/env python3
"""Consensus simulations (docs/13): what the VEARL rules change against three families of attacks, with reproducible figures.

    python3 tools/consensus_sim.py [reorg|lwma|timewarp|selfish|daa|all] [--runs N]

A. `reorg`    : attacker with share `q` of the hashrate who privately mines a branch from a common ancestor; it publishes when its branch is
                accepted by the local rule (simple work, or MESS). Probability of replacing >= d honest blocks within a given horizon.
B. `lwma`     : "on/off" miner who brings power when the difficulty is low (LWMA-1 EXACTLY as chain.rs: window 900 s / T blocks,
                solve times bounded to [1, 6T], variation per block bounded to [x0.5; x2]).
C. `timewarp` : attacker who timestamps its blocks as far in the future as possible (bound MAX_FUTURE = 60 s; the honest clock stays below).
D. `selfish`  : selfish mining (Eyal-Sirer): share of the revenue for q and gamma (the MESS rule does not change it; checks the known threshold ~ 1/3).

These are MODELS (exponential solve times, zero latency, blocks of equal work in A): they bound orders of magnitude, they do not
prove the security of the protocol. Fixed seed.
"""
import sys
import numpy as np

T = 10                      # BLOCK_TIME (state.rs); every duration below is expressed in SECONDS and converted to blocks with T
K = 30 // T * 1               # blocks per 'old block' (a block was 30 s before the change to 10 s): depths and windows keep their duration
MESS_T_MAX = 25_132          # ECIP-1100 (chain.rs)
MESS_MAX = 31


def mess_f(t):
    """f(t) like chain.rs::mess_factor_milli (floating point here)."""
    s = np.minimum(t, MESS_T_MAX) / MESS_T_MAX
    return 1 + (MESS_MAX - 1) * s * s * (3 - 2 * s)


# ---------------------------------------------------------------- A. private reorganisation
def reorg(runs, rng):
    print("A. Private branch attack: probability of replacing >= d honest blocks (4 h horizon, blocks of equal work)")
    print("   q = attacker's share of the hashrate; rule = work comparison (`simple`) or MESS (`mess`)")
    horizon = 4 * 3600
    depths = [6 * K, 30 * K, 100 * K]      # the same durations as with 30 s blocks (3 min, 15 min, 50 min)
    print(f"   {'q':>5} | " + " | ".join(f"d>={d:<3} simple  mess " for d in depths))
    for q in (0.10, 0.30, 0.40, 0.45, 0.55, 0.70, 0.90):
        row = []
        for d in depths:
            ok_simple = ok_mess = 0
            n_ev = int(horizon / T * 3) + 50
            for _ in range(runs):
                dt = rng.exponential(T / 1.0, n_ev)          # time between events of the whole network (rate 1/T)
                ts = np.cumsum(dt)
                ts = ts[ts <= horizon]
                who = rng.random(len(ts)) < q                # True = attacker's block
                A = np.cumsum(who)
                H = np.cumsum(~who)
                # time elapsed on the local chain = date of the last honest block
                last_h = np.maximum.accumulate(np.where(~who, ts, 0.0))
                cond_h = H >= d
                s_ok = np.any(cond_h & (A > H))
                m_ok = np.any(cond_h & (A > H * mess_f(last_h)))
                ok_simple += s_ok
                ok_mess += m_ok
            row.append(f"{ok_simple / runs:6.3f} {ok_mess / runs:6.3f}      ")
        print(f"   {q:5.2f} | " + " | ".join(row))
    print("   Reading: MESS does not remove the short attack (small t => f ~ 1), it makes it harder; it forbids deep reorganisation as long as\n"
          "   q/(1-q) < f(t) (f rises up to 31 after 7 h, ECIP-1100 parameters). WARNING: the cells d>=100 of q = 0.90 are a horizon artefact (fewer than 100 honest blocks in 4 h).\n"
          "   Counterpart: two halves of comparable work separated for more than a few tens of minutes no longer merge by themselves: MESS is OPTIONAL (docs/13 section 3).")


# ---------------------------------------------------------------- B. LWMA and on/off miner
def lwma_next(bt, bs, fallback):
    """Target of the next block from the targets `bt` and timestamps `bs` of the blocks already mined (chain.rs::lwma_next_target)."""
    N = 900 // T                      # LWMA window: 900 s, the half-life of ASERT
    if len(bt) < N + 1:
        return fallback
    wt, ws = bt[-N - 1:], bs[-N - 1:]
    num = st_sum = 0
    for i in range(1, N + 1):
        num += i * min(max(ws[i] - ws[i - 1], 1), 6 * T)
        st_sum += wt[i]
    den = T * (N * (N + 1) // 2)
    nxt = (st_sum // N) * num // den
    prev = wt[N]
    return max(min(max(nxt, prev // 2), prev * 2), 1 << 16)


def lwma(runs, rng):
    print("B. LWMA-1 against an on/off miner (base = 1 unit of power; the attacker brings X units when the target is easy)")
    T0 = 1 << 60                        # reference target: 1 unit of power -> T seconds per block
    print(f"   {'X':>3} {'on/off rule':>28} | mean interval | gap to the target time | share of the blocks it wins while present: X/(1+X)")
    for X in (2, 5, 20):
        res = []
        for policy in ("on when target > 1.3 x ref", f"on {900 // T} blocks / off {1800 // T} blocks"):
            tot_t = tot_b = att_b = 0
            on = False
            for r in range(max(1, runs // 50)):
                bt, bs = [], []
                tg = T0
                t = 0.0
                for blk in range(3000 * K):
                    if policy.startswith("on when"):
                        on = tg > 1.3 * T0 if not on else tg > 0.9 * T0
                    else:
                        on = (blk % (2700 // T)) < (900 // T)
                    power = 1 + (X if on else 0)
                    dt = rng.exponential(T * (T0 / tg) / power)
                    t += dt
                    bt.append(tg)
                    bs.append(max(int(t), (bs[-1] + 1) if bs else 1))
                    tg = lwma_next(bt, bs, tg)
                    if blk >= 300 * K:
                        tot_t += dt
                        tot_b += 1
                        att_b += (X / (1 + X)) if on else 0
            avg = tot_t / tot_b
            res.append((policy, avg, att_b / tot_b))
        for policy, avg, share in res:
            print(f"   {X:3d} {policy:>28} | {avg:14.1f} s | {100 * (avg / T - 1):+18.1f} % | average share of the blocks that go to it {100 * share:5.1f} %")
    print("   Reading: the reactive rule (on if the target is easy) hardly disturbs the block time (+1 %). A periodic miner (30 active blocks, 60 inactive)\n"
          "   slows the network by up to +27 % (X = 20): LWMA-30 reacts with a delay, known property [LIT]. Not fixed here; leads: shorter window, cap on the variation.")


# ---------------------------------------------------------------- C. extreme timestamps
def timewarp(runs, rng):
    print("C. Attacker who timestamps its blocks as far in the future as possible (+60 s above the clock, MAX_FUTURE): drift of the target")
    T0 = 1 << 60
    print(f"   {'q':>5} | final target / reference target | mean interval (real time) ")
    for q in (0.10, 0.30, 0.45, 0.55):
        drift = []
        avg_real = []
        for r in range(max(1, runs // 100)):
            bt, bs = [], []
            tg = T0
            t = 0.0
            for blk in range(4000):
                dt = rng.exponential(T * (T0 / tg))
                t += dt
                mine = rng.random() < q
                stamp = int(t) + (60 if mine else 0)
                stamp = max(stamp, (bs[-1] + 1) if bs else 1)      # the timestamp must grow strictly
                bt.append(tg)
                bs.append(stamp)
                tg = lwma_next(bt, bs, tg)
            drift.append(tg / T0)
            avg_real.append(t / 4000)
        print(f"   {q:5.2f} | {np.mean(drift):20.3f} (standard deviation {np.std(drift):.3f}) | {np.mean(avg_real):8.1f} s")
    print("   Reading: the cap (+60 s) and the floor (strict growth + solve time bounded to [1, 6T]) limit the drift; a sustained\n"
          "   drift of the target remains possible for q > 0.5 (the attacker then controls the majority anyway).")


# ---------------------------------------------------------------- D. selfish mining
def selfish(runs, rng):
    print("D. Selfish mining (Eyal-Sirer), share of the attacker's revenue; gamma = share of the honest network that follows the attacker in a tie")
    print(f"   {'q':>5} | gamma=0: revenue | gamma=0.5 | honest = q")
    for q in (0.20, 0.25, 0.30, 0.33, 0.36, 0.40):
        out = []
        for gamma in (0.0, 0.5):
            mine = other = 0
            lead = 0
            state0p = False
            n = 400_000
            r = rng.random(n)
            r2 = rng.random(n)
            for i in range(n):
                if lead == 0 and not state0p:
                    if r[i] < q:
                        lead = 1
                    else:
                        other += 1
                elif state0p:           # 1-1 tie after publication
                    if r[i] < q:
                        mine += 2
                    elif r2[i] < gamma:
                        mine += 1
                        other += 1
                    else:
                        other += 2
                    lead = 0
                    state0p = False
                elif lead == 1:
                    if r[i] < q:
                        lead = 2
                    else:
                        state0p = True
                        lead = 0
                elif lead == 2:
                    if r[i] < q:
                        lead = 3
                    else:
                        mine += 2
                        lead = 0
                else:
                    if r[i] < q:
                        lead += 1
                    else:
                        lead -= 1
                        mine += 1
            out.append(mine / max(1, mine + other))
        print(f"   {q:5.2f} | {out[0]:16.3f} | {out[1]:9.3f} | {q:11.2f}")
    print("   Reading: above about 1/3 (gamma = 0) or 1/4 (gamma = 0.5) selfish mining earns more than its share; VEARL does not prevent it\n"
          "   (chain choice by work): it is a known property of Nakamoto chains [LIT], MESS does not change it.")


# ---------------------------------------------------------------- E. LWMA-1 against ASERT (aserti3-2d)
def daa(runs, rng):
    """Same model as B, with several rules: LWMA-1 (chain.rs) and ASERT with several half-lives (tools/asert_ref.py, verified on the
    official Bitcoin Cash vectors). Scenarios: constant power, periodic miner (30 active blocks / 60 inactive, X times the base),
    reactive miner, power jump x10 then /10."""
    import asert_ref as A
    T0 = 1 << 60
    rules = [(f"LWMA-1 (N={900 // T})", None)] + [(f"ASERT half-life {h} s", h) for h in (600, 900, 1800, 3600)]

    def simulate(rule_h, power_of, n_blocks, seed):
        r = np.random.default_rng(seed)
        bt, bs = [], []
        tg, t = T0, 0.0
        dts, share = [], []
        for blk in range(n_blocks):
            power, att = power_of(blk, tg)
            dt = r.exponential(T * (T0 / tg) / power)
            t += dt
            bt.append(tg)
            bs.append(max(int(t), (bs[-1] + 1) if bs else 1))
            dts.append(dt)
            share.append(att / power)
            if rule_h is None:
                tg = lwma_next(bt, bs, tg)
            else:
                tg = A.vearl_asert(T0, bs[0], bs[-1], len(bs) - 1, T, rule_h)
        return np.array(dts), np.array(share)

    reps = max(2, runs // 100)
    print("E. LWMA-1 against ASERT: mean interval (target " + str(T) + " s) and gap to the target time; " + str(3000 * K) + " blocks (= 25 h) per draw, " + str(reps) + " draws")
    scen = [
        ("constant power", lambda b, tg: (1.0, 0.0)),
        ("periodic X=5 (15 min on / 30 min off)", lambda b, tg: (6.0, 5.0) if b % (2700 // T) < (900 // T) else (1.0, 0.0)),
        ("periodic X=20 (15 min on / 30 min off)", lambda b, tg: (21.0, 20.0) if b % (2700 // T) < (900 // T) else (1.0, 0.0)),
    ]
    print(f"   {'rule':22s} | " + " | ".join(f"{name:34s}" for name, _ in scen))
    for rname, h in rules:
        cells = []
        for sname, pw in scen:
            m = []
            sd = []
            for rep in range(reps):
                dts, _ = simulate(h, pw, 3000 * K, 1000 + rep)
                m.append(dts[300 * K:].mean())
                sd.append(dts[300 * K:].std())
            mean = float(np.mean(m))
            cells.append(f"{mean:6.1f} s ({100 * (mean / T - 1):+5.1f} %), standard deviation {np.mean(sd):5.1f} s")
        print(f"   {rname:22s} | " + " | ".join(f"{c:34s}" for c in cells))
    print(f"   Power jump x10 at t = {1000 * K} blocks then /10 at {2000 * K}: mean interval of the {100 * K} blocks (50 min) following each jump (ideal {T} s)")
    for rname, h in rules:
        up, down = [], []
        for rep in range(reps):
            dts, _ = simulate(h, lambda b, tg: (10.0 if 1000 * K <= b < 2000 * K else 1.0, 0.0), 3000 * K, 2000 + rep)
            up.append(dts[1000 * K:1100 * K].mean())
            down.append(dts[2000 * K:2100 * K].mean())
        print(f"   {rname:22s} | after x10: {np.mean(up):6.1f} s | after /10: {np.mean(down):6.1f} s")
    print("   Reading: see docs/13 section 5 (choice of the rule and of the half-life).")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    what = args[0] if args else "all"
    runs = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 2000
    rng = np.random.default_rng(12345)
    for name, fn in (("reorg", reorg), ("lwma", lwma), ("timewarp", timewarp), ("selfish", selfish), ("daa", daa)):
        if what in (name, "all"):
            fn(runs, rng)
            print()
