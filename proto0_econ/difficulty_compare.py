#!/usr/bin/env python3
"""Comparison of difficulty adjustment algorithms after a hashrate shock (stdlib).
A = current rule (EMA 1/16, dt bounded to [T/2, 2T]); B = same rule with dt bounded to [T/2, 10T];
C = LWMA-1 (Zawy, window N=30, solvetime bounded to [1, 6T]) [LIT: Zawy12 "LWMA difficulty algorithm"].
"""
import random, statistics as st

T = 30
TWO64 = 2 ** 64


def alg_A(hist, cap=2):
    tgt, ts = hist[-1]
    dt = ts - hist[-2][1] if len(hist) > 1 else T
    adj = min(max(dt, T // 2), T * cap)
    return max(int(tgt * (15 * T + adj) // (16 * T)), 1 << 16)


def alg_B(hist):
    return alg_A(hist, cap=10)


def alg_C(hist, N=30):
    if len(hist) < N + 1:
        return alg_A(hist)
    w = hist[-(N + 1):]
    num, den, tsum = 0, N * (N + 1) // 2, 0
    for i in range(1, N + 1):
        st_i = min(max(w[i][1] - w[i - 1][1], 1), 6 * T)
        num += i * st_i
        tsum += w[i][0]
    avg_t = tsum // N
    return max(int(avg_t * num // (T * den)), 1 << 16)


def run(alg, hash_fn, n_blocks, seed):
    rnd = random.Random(seed)
    target = int(TWO64 / (1000.0 * T))
    hist = [(target, 0)]            # genesis pseudo-block
    t, last_ts, out = 0.0, 0, []
    for i in range(n_blocks):
        rate = hash_fn(i) * target / TWO64
        dt_real = rnd.expovariate(rate)
        t += dt_real
        ts = max(int(t), last_ts + 1)
        out.append(dt_real)
        hist.append((target, ts))   # (target used for THIS block, its timestamp)
        target = min(alg(hist), TWO64 - 1)
        last_ts = ts
    return out


def recovery(alg, factor, trials=200):
    need, worst = [], []
    for seed in range(trials):
        d = run(alg, lambda i: 1000.0 * (factor if i >= 60 else 1.0), 1500, seed)
        post = d[60:]
        ok = next((j + 20 for j in range(len(post) - 20) if 22.5 <= sum(post[j:j + 20]) / 20 <= 37.5), 1400)
        need.append(ok)
        worst.append(max(post[:max(ok, 1)]))
    return st.median(need), st.median(worst)


def steady(alg, trials=100):
    devs = []
    for seed in range(trials):
        d = run(alg, lambda i: 1000.0, 1500, seed)[200:]
        devs.append((sum(d) / len(d), st.pstdev(d)))
    return st.mean(x[0] for x in devs), st.mean(x[1] for x in devs)


if __name__ == "__main__":
    algs = [("A current (EMA, dt<=2T)", alg_A), ("B EMA, dt<=10T", alg_B), ("C LWMA-1 (N=30)", alg_C)]
    print("| algorithm | steady state: mean / standard deviation of the block time | /2: blocks (worst interval) | /10 | /100 | x10 |")
    print("|---|---|---|---|---|---|")
    for name, alg in algs:
        m, s = steady(alg)
        cells = []
        for f in (0.5, 0.1, 0.01, 10.0):
            n, w = recovery(alg, f)
            cells.append(f"{n:.0f} ({w:.0f} s)")
        print(f"| {name} | {m:.1f} s / {s:.1f} s | " + " | ".join(cells) + " |")
