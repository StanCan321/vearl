#!/usr/bin/env python3
"""DIFFERENTIAL FUZZ of the market state machine: the independent Python validator (market_ref.py) and the Rust validator
receive EXACTLY the same blocks (random transactions, valid or faulty: bids, claims, true or false results,
data requests/services, fraud proofs with valid or altered openings, settlements at arbitrary moments) and
must TAKE THE SAME DECISION (accept / refuse) for each one. Any divergence is a defect of one of the two or of the
specification (docs/07 §11).

    python3 tools/diff_fuzz.py [NB_BLOCS=600] [SEED=1] [--rust ../target/release/vearl-market-vectors]
"""
import base64, copy, json, os, random, struct, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import spec_ref as S
import chain_ref as C
import market_ref as M

N = 128
COIN = 10**8
COUNT = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else 600
SEED = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 1
RUST = sys.argv[sys.argv.index("--rust") + 1] if "--rust" in sys.argv else "target/release/vearl-market-vectors"
rng = random.Random(SEED)
nrng = np.random.default_rng(SEED)

# ---------------- genesis and reference state ----------------
alloc = [{"key": C.pub_hex(s), "amount": 10_000 * COIN} for s in range(1, 5)]
G = {"params": {"n": N, "tile": 16, "rank": 63, "fold_k": 64}, "alloc": alloc, "timestamp": 1_700_000_000, "target": (1 << 64) // 16 - 1, "retarget": "Fixed",
     "timing": {"challenge_secs": 60, "serve_secs": 30},
     "constants": {"block_time": 30, "epoch_len": 100, "r0": 20 * COIN, "halving": 1_051_200, "tail": COIN // 2, "lwma_n": 30, "max_future": 60,
                   "burn_bps": 1000, "bond_bps": 1000, "min_escrow": COIN, "challenge_window": 10, "auction_grace_secs": 60, "max_auction_secs": 3600,
                   "fetch_fee": COIN // 20, "fetch_extend_secs": 60, "max_pending_fetches": 8}}

# ---------------- pool of jobs (known matrices) ----------------


class Pool:
    def __init__(self):
        self.a = nrng.integers(-64, 65, (N, N)).astype(np.int64)
        self.b = nrng.integers(-64, 65, (N, N)).astype(np.int64)
        self.ar = S.merkle_root([S.leaf_hash(i, S.i8bytes(self.a[i])) for i in range(N)])
        self.br = S.merkle_root([S.leaf_hash(j, S.i8bytes(self.b[:, j])) for j in range(N)])
        self.a_lv = S.merkle_levels([S.leaf_hash(i, S.i8bytes(self.a[i])) for i in range(N)])
        self.b_lv = S.merkle_levels([S.leaf_hash(j, S.i8bytes(self.b[:, j])) for j in range(N)])
        c = self.a @ self.b
        cbad = c.copy()
        cbad[rng.randrange(N), rng.randrange(N)] += rng.choice([1, -1, 1000])
        self.c = {"ok": c, "bad": cbad}
        self.c_lv = {k: S.c_levels(m) for k, m in self.c.items()}

    @staticmethod
    def proof(levels, idx):
        out = []
        for lv in levels[:-1]:
            out.append(lv[idx ^ 1].hex())
            idx >>= 1
        return out

    def c_root(self, which):
        return self.c_lv[which][-1][0].hex()


POOL = [Pool() for _ in range(5)]


def jobs_data():
    return [{"a": base64.b64encode(S.i8bytes(p.a)).decode(), "b": base64.b64encode(S.i8bytes(p.b)).decode()} for p in POOL]


# ---------------- action generation ----------------
def rand_key():
    return rng.randint(1, 4)


def pick_job(st, statuses=None):
    ids = [i for i, j in st.jobs.items() if statuses is None or j["status"] in statuses]
    return rng.choice(ids) if ids else "00" * 32


def pool_of(st, jid):
    j = st.jobs.get(jid)
    if not j:
        return rng.choice(POOL)
    for p in POOL:
        if p.ar.hex() == j["a_root"] and p.br.hex() == j["b_root"]:
            return p
    return rng.choice(POOL)


def garbage_hex():
    return bytes(rng.randrange(256) for _ in range(32)).hex()


def gen_kind(st, ts, height):
    """A plausible or faulty transaction, depending on the current state. Returns (signer_seed, kind)."""
    r = rng.random()
    guided = rng.random() < 0.88
    if r < 0.10:
        return rand_key(), {"type": "Transfer", "to": C.pub_hex(rand_key()) if rng.random() < .8 else garbage_hex(), "amount": rng.choice([COIN, 5 * COIN, 10**14, 0, 12345678])}
    if r < 0.25:
        p = rng.choice(POOL)
        kind = {"type": "SubmitJob", "a_root": p.ar.hex(), "b_root": p.br.hex(), "escrow": rng.choice([COIN // 2, 20 * COIN, 50 * COIN, 20 * COIN, 50 * COIN, 10**15]), "deadline": max(0, height + rng.choice([-1, 8, 60, 500, 500, 60]))}
        if rng.random() < 0.6:
            kind = dict(kind, type="SubmitJobAuction", auction_secs=rng.choice([20, 60, 200, 60, 4000, 0]))
        return rand_key(), kind
    if r < 0.35:
        live = [i for i, jb in st.jobs.items() if jb["status"] == "Open" and jb["end"] > ts]   # auctions still open
        jid = rng.choice(live) if (guided and live) else pick_job(st, ("Open",) if guided else None)
        j = st.jobs.get(jid)
        base = j["best"][1] if j and j["best"] else (j["escrow_net"] if j else COIN)
        return rand_key(), {"type": "Bid", "job_id": jid, "price": max(0, base + rng.choice([-COIN, -1, 0, 1, COIN, -base]))}
    if r < 0.47:
        jid = pick_job(st, ("Open",) if guided else None)
        j = st.jobs.get(jid)
        who = rand_key()
        if guided and j and j["best"] and rng.random() < .7:
            who = next((s for s in range(1, 5) if C.pub_hex(s) == j["best"][0]), who)
        return who, {"type": "ClaimJob", "job_id": jid}
    if r < 0.57:
        jid = pick_job(st, ("Claimed",) if guided else None)
        j = st.jobs.get(jid)
        p = pool_of(st, jid)
        who = next((s for s in range(1, 5) if j and C.pub_hex(s) == j.get("worker")), rand_key()) if guided else rand_key()
        return who, {"type": "PostResult", "job_id": jid, "c_root": p.c_root(rng.choice(["ok", "ok", "bad"])) if rng.random() < .9 else garbage_hex()}
    if r < 0.64:
        jid = pick_job(st, ("Posted",) if guided else None)
        j = st.jobs.get(jid)
        who = next((s for s in range(1, 5) if j and C.pub_hex(s) == j["client"]), rand_key()) if guided else rand_key()
        return who, {"type": rng.choice(["Accept", "Accept", "Settle"]), "job_id": jid}
    if r < 0.70:
        jid = pick_job(st)
        return rand_key(), {"type": "Settle", "job_id": jid}
    if r < 0.78:
        jid = pick_job(st, ("Posted",) if guided else None)
        j = st.jobs.get(jid)
        who = next((s for s in range(1, 5) if j and C.pub_hex(s) == j["client"]), rand_key()) if guided else rand_key()
        return who, {"type": "Fetch", "job_id": jid, "i": rng.choice([0, 3, 5, 127, 128, 10**6]), "j": rng.choice([0, 7, 127, 128])}
    if r < 0.86:
        jid = pick_job(st, ("Posted",) if guided else None)
        j = st.jobs.get(jid)
        p = pool_of(st, jid)
        which = "ok" if not j or j.get("c_root") == p.c_root("ok") else "bad"
        f = rng.choice(j["fetches"]) if j and j["fetches"] and rng.random() < .8 else (rng.choice([0, 3, 5]), rng.choice([0, 7, 127]), 0)
        i, jj = f[0], f[1]
        if i >= N or jj >= N:
            i, jj = 0, 0
        row = [int(v) for v in p.c[which][i]]
        row[jj] += rng.choice([0, 0, 0, 1])
        who = next((s for s in range(1, 5) if j and C.pub_hex(s) == j.get("worker")), rand_key()) if guided else rand_key()
        return who, {"type": "Serve", "job_id": jid, "i": i, "j": jj, "row": row, "path": Pool.proof(p.c_lv[which], i)}
    if r < 0.93:
        jid = pick_job(st, ("Posted",) if guided else None)
        return rand_key(), {"type": "SlashUnserved", "job_id": jid}
    # fraud proof (with valid, possibly altered openings)
    jid = pick_job(st, ("Posted",) if guided else None)
    p = pool_of(st, jid)
    j = st.jobs.get(jid)
    which = "bad" if (j and j.get("c_root") == p.c_root("bad")) or rng.random() < .3 else "ok"
    if which == "bad" and rng.random() < .8:
        cb = p.c["bad"]
        diff = np.argwhere(cb != p.c["ok"])[0]
        i, jj = int(diff[0]), int(diff[1])
    else:
        i, jj = rng.randrange(N), rng.randrange(N)
    k = {"type": "Fraud", "job_id": jid, "i": i, "j": jj, "a_row": [int(v) for v in p.a[i]], "a_path": Pool.proof(p.a_lv, i), "b_col": [int(v) for v in p.b[:, jj]],
         "b_path": Pool.proof(p.b_lv, jj), "c_row": [int(v) for v in p.c[which][i]], "c_path": Pool.proof(p.c_lv[which], i)}
    m = rng.random()
    if m < 0.12:
        k["a_row"][rng.randrange(N)] += 1
    elif m < 0.22:
        k["c_row"][jj] += 1
    elif m < 0.28:
        k["b_path"][rng.randrange(len(k["b_path"]))] = garbage_hex()
    return rand_key(), k


# ---------------- loop: Python decides, we record ----------------
V0 = {"genesis": G, "jobs_data": jobs_data()}
st = M.Market(V0)
attempts = []
stats = {"accepted": 0, "rejected": 0, "kinds_ok": {}, "kinds_ko": {}}
ts = G["timestamp"]
for it in range(COUNT):
    ts += rng.choice([1, 3, 5, 10, 20, 20, 40, 70, 130])
    height = len(st.headers)
    txs = []
    for _ in range(rng.choice([1, 1, 1, 1, 1, 2])):
        seed, kind = gen_kind(st, ts, height)
        # nonce: that of the state, +1 per transaction of the same signer already in the block (otherwise a deliberate fault from time to time)
        who = C.pub_hex(seed)
        nonce = st.nonce.get(who, 0) + sum(1 for t in txs if t["from"] == who)
        if rng.random() < 0.02:
            nonce += rng.choice([-1, 1, 5])
        t = {"from": who, "nonce": max(nonce, 0), "fee": rng.choice([0, 0, 0, 1000, 10**14]), "kind": kind}
        t["sig"] = C.priv(seed).sign(M.tx_bytes(t)).hex()
        t["id"] = M.tx_id(t).hex()
        txs.append(t)
    miner = C.pub_hex(rng.randint(1, 4))
    blk = M.mine(st, miner, ("Native",), txs, ts)
    trial = copy.deepcopy(st)
    try:
        trial.apply(blk)
        ok, why = True, ""
    except M.Invalid as e:
        ok, why = False, str(e)
    names = "+".join(t["kind"]["type"] for t in txs)
    if ok:
        st = trial
        stats["accepted"] += 1
        for t in txs:
            stats["kinds_ok"][t["kind"]["type"]] = stats["kinds_ok"].get(t["kind"]["type"], 0) + 1
    else:
        stats["rejected"] += 1
        for t in txs:
            stats["kinds_ko"][t["kind"]["type"]] = stats["kinds_ko"].get(t["kind"]["type"], 0) + 1
    attempts.append({"block": blk, "py_ok": ok, "why": why, "names": names})

path = f"/tmp/diff_fuzz_{SEED}.json" if os.name != "nt" else f"diff_fuzz_{SEED}.json"
final = {"minted": st.minted, "burned": st.burned, "height": len(st.headers),
         "accounts": [{"key": a["key"], "balance": st.bal.get(a["key"], 0), "nonce": st.nonce.get(a["key"], 0)} for a in alloc],
         "jobs": {i: {"status": j["status"], "escrow_net": j["escrow_net"], "rebate_left": j["rebate_left"]} for i, j in st.jobs.items()},
         "reputation": {a["key"]: st.rep.get(a["key"], {"done": 0, "slashed": 0, "expired": 0}) for a in alloc}}
json.dump({"genesis": G, "jobs_data": V0["jobs_data"], "attempts": attempts, "final": final}, open(path, "w"))
print(f"[diff] {COUNT} blocks tried: {stats['accepted']} accepted, {stats['rejected']} refused by Python; money supply {'OK' if st.invariant() else 'BROKEN'}")
print("[diff] accepted types:", dict(sorted(stats["kinds_ok"].items())))
print("[diff] refused types:", dict(sorted(stats["kinds_ko"].items())))
r = subprocess.run([RUST, "diff", path], capture_output=True, text=True)
print(r.stdout.strip()[-1800:], r.stderr.strip()[-400:])
sys.exit(r.returncode)
