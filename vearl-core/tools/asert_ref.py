#!/usr/bin/env python3
"""ASERT (aserti3-2d, Bitcoin Cash, Nov. 2020): INDEPENDENT Python implementation written from the published specification
(https://upgradespecs.bitcoincashnode.org/2020-11-15-asert/), checked on OFFICIAL vectors (BCHN qa-assets repository,
test_vectors/aserti3-2d, runs 1, 3, 5, 8: first lines copied below), then compared with the Rust node (docs/13).

        python3 tools/asert_ref.py                                   # official vectors
        python3 tools/asert_ref.py --rust ../target/release/vearl-chain-vectors [N]   # + N random cases against the Rust (VEARL u64 domain)
"""
import random, subprocess, sys

RADIX = 65536


def trunc_div(a, b):
    """Integer division truncated toward zero (the spec requires it for the exponent; Python // rounds toward -inf)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def asert_core(anchor_target, time_delta, height_delta, ideal, halflife):
    """Spec formula, without bounds: anchor_target * 2^((time_delta - ideal*(height_delta+1)) / halflife), cubic polynomial in 16.16."""
    exponent = trunc_div((time_delta - ideal * (height_delta + 1)) * RADIX, halflife)
    num_shifts = exponent >> 16                         # arithmetic shift (floor)
    exponent -= num_shifts * RADIX                      # fractional part in [0, 65536)
    factor = ((195766423245049 * exponent + 971821376 * exponent ** 2 + 5127 * exponent ** 3 + 2 ** 47) >> 48) + RADIX
    t = anchor_target * factor
    t = t >> -num_shifts if num_shifts < 0 else t << num_shifts
    return t >> 16


# ---------------------------------------------------------------- compact targets (nBits) of Bitcoin, for the official vectors
def bits_to_target(bits):
    size, mant = bits >> 24, bits & 0x007FFFFF
    return mant >> 8 * (3 - size) if size <= 3 else mant << 8 * (size - 3)


def target_to_bits(t):
    size = (t.bit_length() + 7) // 8
    compact = t << 8 * (3 - size) if size <= 3 else t >> 8 * (size - 3)
    if compact & 0x00800000:
        compact >>= 8
        size += 1
    return compact | (size << 24)


POW_LIMIT = bits_to_target(0x1D00FFFF)


def bch_next_bits(anchor_bits, anchor_parent_time, anchor_height, height, time):
    t = asert_core(bits_to_target(anchor_bits), time - anchor_parent_time, height - anchor_height, 600, 172800)
    if t == 0:
        return target_to_bits(1)
    if t > POW_LIMIT:
        return 0x1D00FFFF
    return target_to_bits(t)


# (description, anchor nBits, anchor parent time, anchor height, [(height, time, expected nBits)]) - copied from the official files
VECTORS = [
    ("run1 - 600 s blocks at the limit target", 0x1D00FFFF, 0, 1, [(2, 1200, 0x1D00FFFF), (3, 1800, 0x1D00FFFF), (11, 6600, 0x1D00FFFF)]),
    ("run3 - 600 s blocks at the minimum target 0x01010000", 0x01010000, 0, 1, [(2, 1200, 0x01010000), (6, 3600, 0x01010000), (11, 6600, 0x01010000)]),
    ("run5 - jumps of one half-life in height without time advance", 0x1D00FFFF, 0, 1,
     [(2, 0, 0x1D00FEC5), (290, 0, 0x1C7F62C0), (578, 0, 0x1C3FB160), (866, 0, 0x1C1FD8B0), (1154, 0, 0x1C0FEC58), (1442, 0, 0x1C07F62C),
      (1730, 0, 0x1C03FB16), (2018, 0, 0x1C01FD8B), (2306, 0, 0x1C00FEC5), (2594, 0, 0x1B7F62C0), (2882, 0, 0x1B3FB160), (3170, 0, 0x1B1FD8B0)]),
    ("run8 - pseudo-random solve times, decreasing hashrate", 0x1802AEE8, 0, 1,
     [(2, 1200, 0x1802AEE8), (3, 1310, 0x1802AD91), (4, 1327, 0x1802ABF8), (5, 1739, 0x1802AB73), (6, 2219, 0x1802AB20), (7, 2604, 0x1802AA89),
      (8, 2746, 0x1802A94B), (9, 7099, 0x1802B39C), (10, 7099, 0x1802B1F2), (11, 7657, 0x1802B1D4), (12, 9595, 0x1802B58D)]),
]

# ---------------------------------------------------------------- VEARL domain (docs/13 section 5): u64 target, bounds [2^16, 2^64 - 1]
T_MIN, T_MAX = 1 << 16, (1 << 64) - 1


def vearl_asert(anchor_target, first_ts, tip_ts, tip_height, ideal, halflife):
    """Anchor = the FIRST block (initial target, timestamp assumed on schedule): target of the block following the tip (height tip_height)."""
    t = asert_core(anchor_target, tip_ts - (first_ts - ideal), tip_height, ideal, halflife)
    return min(max(t, T_MIN), T_MAX)


def main():
    bad = n = 0
    for desc, ab, apt, ah, rows in VECTORS:
        for (h, t, want) in rows:
            got = bch_next_bits(ab, apt, ah, h, t)
            n += 1
            if got != want:
                bad += 1
                print(f"FAIL {desc}: height {h}, time {t}: got {got:#010x}, expected {want:#010x}")
    print(f"{n - bad}/{n} targets identical to the official aserti3-2d vectors (BCHN qa-assets)")
    if "--rust" in sys.argv:
        exe = sys.argv[sys.argv.index("--rust") + 1]
        count = int(sys.argv[sys.argv.index("--rust") + 2]) if len(sys.argv) > sys.argv.index("--rust") + 2 else 20000
        rng = random.Random(7)
        cases = []
        for _ in range(count):
            anchor = rng.choice([rng.randrange(T_MIN, T_MAX), T_MAX, T_MIN, (1 << 64) // 1000, rng.randrange(1 << 40)]) or T_MIN
            anchor = max(anchor, T_MIN)
            first = rng.randrange(0, 2 ** 40)
            height = rng.choice([0, 1, 2, rng.randrange(0, 10 ** 6)])
            drift = rng.choice([0, rng.randrange(-86400 * 30, 86400 * 30), rng.randrange(-100, 100)])
            tip = max(first + 30 * height + drift, first)
            halflife = rng.choice([1800, 3600, 8640, 172800])
            cases.append((anchor, first, tip, height, 30, halflife))
        inp = "\n".join(" ".join(map(str, c)) for c in cases) + "\n"
        out = subprocess.run([exe, "asert"], input=inp, capture_output=True, text=True, check=True).stdout.split()
        dif = sum(int(o) != vearl_asert(*c) for o, c in zip(out, cases))
        bad += dif + (len(out) != len(cases))
        print(f"{len(cases) - dif}/{len(cases)} targets identical between the Rust node and this implementation (VEARL u64 domain)")
    print("RESULT: ASERT MATCHES" if bad == 0 else f"RESULT: {bad} DIVERGENCE(S)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
