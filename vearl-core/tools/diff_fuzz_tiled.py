#!/usr/bin/env python3
"""DIFFERENTIAL FUZZ of the BLOCK-WISE JOBS (`tiled`): the independent Python validator (tiled_ref.py) and the Rust validator receive exactly
the same blocks (random transactions: valid or invalid submissions - dimensions, roots, escrow, deadline -, claims, publications of
true or false roots, acceptances, settlements, and PRODUCT and SUM fraud proofs with valid or altered openings, on
honest or falsified data, at the right or wrong cell) and must TAKE THE SAME DECISION; the final state must be identical.

    python3 tools/diff_fuzz_tiled.py [NB_BLOCS=400] [SEED=1] [--rust target/release/vearl-tiled-vectors]
"""
import base64, copy, json, os, random, struct, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import spec_ref as S
import chain_ref as C
import prog_ref as P
import tiled_ref as TR

N = 128
COIN = 10**8
args = [a for a in sys.argv[1:] if not a.startswith("--")]
COUNT = int(args[0]) if args else 400
SEED = int(args[1]) if len(args) > 1 else 1
RUST = sys.argv[sys.argv.index("--rust") + 1] if "--rust" in sys.argv else "target/release/vearl-tiled-vectors"
VEC = os.environ.get("TILED_VECTORS", "tests/vectors/tiled-v1.json")
rng = random.Random(SEED)
nrng = np.random.default_rng(SEED)
G = json.load(open(VEC))["genesis"]
hl = lambda b: list(b)
mm = lambda x, y: np.rint(np.asarray(x, dtype=np.float64) @ np.asarray(y, dtype=np.float64)).astype(np.int64)


def levels(z):
    return S.c_levels(np.asarray(z).reshape(N, N))


def _proof(lv, idx):
    out = []
    for l in lv[:-1]:
        out.append(hl(l[idx ^ 1]))
        idx >>= 1
    return out


def rows_levels(m):
    return S.merkle_levels([S.leaf_hash(i, S.i8bytes(m[i])) for i in range(N)])


def cols_levels(m):
    return S.merkle_levels([S.leaf_hash(j, S.i8bytes(m[:, j])) for j in range(N)])


class Pool:
    """A data set: A, B, products and sums PUBLISHED by the worker (possibly falsified)."""

    def __init__(self, dims, tamper):
        bi, bk, bj = self.dims = dims
        self.tamper = tamper
        self.a = nrng.integers(-64, 65, (bi * N, bk * N)).astype(np.int64)
        self.b = nrng.integers(-64, 65, (bk * N, bj * N)).astype(np.int64)
        self.ar = [TR.rows_root(TR.block(self.a, i, k)) for i in range(bi) for k in range(bk)]
        self.br = [TR.cols_root(TR.block(self.b, k, j)) for k in range(bk) for j in range(bj)]
        self.honest = [mm(TR.block(self.a, i, k), TR.block(self.b, k, j)) for i in range(bi) for j in range(bj) for k in range(bk)]
        self.prods = [x.copy() for x in self.honest]
        if tamper in ("prod", "prodonly"):
            self.prods[rng.randrange(len(self.prods))][rng.randrange(N), rng.randrange(N)] += rng.choice([1, -3, 1000])
        self.sums = [sum(self.prods[ij * bk + k] for k in range(bk)) for ij in range(bi * bj)]
        if tamper == "prodonly":
            self.sums = [sum(self.honest[ij * bk + k] for k in range(bk)) for ij in range(bi * bj)]
        if tamper == "sum":
            self.sums[rng.randrange(len(self.sums))][rng.randrange(N), rng.randrange(N)] += rng.choice([1, -1, 77])
        self.pr = [TR.z_root(x) for x in self.prods]
        self.cr = [TR.z_root(x) for x in self.sums]
        self._lv, self._rl, self._cl = {}, {}, {}

    def plv(self, k):
        if ("p", k) not in self._lv:
            self._lv[("p", k)] = levels(self.prods[k])
        return self._lv[("p", k)]

    def slv(self, ij):
        if ("s", ij) not in self._lv:
            self._lv[("s", ij)] = levels(self.sums[ij])
        return self._lv[("s", ij)]

    def a_lv(self, i, k):
        if ("a", i, k) not in self._rl:
            self._rl[("a", i, k)] = rows_levels(TR.block(self.a, i, k))
        return self._rl[("a", i, k)]

    def b_lv(self, k, j):
        if ("b", k, j) not in self._cl:
            self._cl[("b", k, j)] = cols_levels(TR.block(self.b, k, j))
        return self._cl[("b", k, j)]

    def bad_products(self):
        return [k for k in range(len(self.prods)) if (self.prods[k] != self.honest[k]).any()]

    def bad_sums(self):
        bi, bk, bj = self.dims
        return [ij for ij in range(bi * bj) if (self.sums[ij] != sum(self.prods[ij * bk + k] for k in range(bk))).any()]

    def product_proof(self, k, r, c):
        bi, bk, bj = self.dims
        i, j, kk = k // (bj * bk), (k % (bj * bk)) // bk, k % bk
        ab, bb = TR.block(self.a, i, kk), TR.block(self.b, kk, j)
        return {"Product": dict(i=i, j=j, k=kk, r=r, c=c, a_row=[int(v) for v in ab[r]], a_path=_proof(self.a_lv(i, kk), r), b_col=[int(v) for v in bb[:, c]],
                                b_path=_proof(self.b_lv(kk, j), c), z_row=[int(v) for v in self.prods[k][r]], z_path=_proof(self.plv(k), r))}

    def sum_proof(self, ij, r, c):
        bi, bk, bj = self.dims
        i, j = ij // bj, ij % bj
        return {"Sum": dict(i=i, j=j, r=r, c=c, z_rows=[[int(v) for v in self.prods[ij * bk + k][r]] for k in range(bk)],
                            z_paths=[_proof(self.plv(ij * bk + k), r) for k in range(bk)], c_row=[int(v) for v in self.sums[ij][r]], c_path=_proof(self.slv(ij), r))}


DIMS = [(1, 1, 1), (1, 2, 1), (2, 1, 2), (2, 2, 1), (1, 2, 2), (2, 2, 2)]
POOL = [Pool(d, t) for d in DIMS for t in ("ok", "prod", "prodonly", "sum")]

G2 = copy.deepcopy(G)
G2["alloc"] = [{"key": C.pub_hex(s), "amount": 10_000 * COIN} for s in range(1, 5)]
st = TR.TiledChain({"genesis": G2, "tiled_data": []})
pool_of = {}
garbage = lambda: hl(bytes(rng.randrange(256) for _ in range(32)))


def pick_job(statuses):
    ids = [i for i, j in st.jobs.items() if j["status"] in statuses]
    return rng.choice(ids) if ids else None


def who_for(jid, field):
    j = st.jobs.get(jid)
    target = j.get(field) if j else None
    return next((s for s in range(1, 5) if C.pub_hex(s) == target), rng.randint(1, 4))


def jb(jid):
    return hl(bytes.fromhex(jid))


def gen_da(guided):
    """Data availability: Fetch (client), Serve (worker), SlashUnserved (anyone)."""
    m = rng.random()
    if m < 0.4:
        jid = pick_job(("Posted",) if guided else st.jobs.keys()) or "00" * 32
        j = st.jobs.get(jid)
        bi, bk, bj = j["dims"] if j else (1, 1, 1)
        w = rng.choice([0, 1, 1, 0, 2])
        body = dict(job_id=jb(jid), which=w, i=rng.randrange(bi + (1 if rng.random() < 0.05 else 0)), j=rng.randrange(bj), k=rng.randrange(bk) if w == 0 else (0 if rng.random() < 0.9 else 1),
                    idx=rng.randrange(N * N) if rng.random() < 0.97 else N * N)
        return (who_for(jid, "client") if guided else rng.randint(1, 4)), {"Tiled": {"Fetch": body}}, "Fetch"
    if m < 0.85:
        cands = [i for i, j in st.jobs.items() if j["status"] == "Posted" and j["fetches"]]
        if not cands:
            return rng.randint(1, 4), {"Tiled": {"SlashUnserved": dict(job_id=garbage())}}, "SlashUnserved"
        jid = rng.choice(cands)
        j = st.jobs[jid]
        w, i, jj, k, idx, _ = rng.choice(j["fetches"])
        p = pool_of.get(jid, rng.choice(POOL))
        bi, bk, bj = p.dims
        try:
            if w == 0:
                pk = (i * bj + jj) * bk + k
                row, path = [int(v) for v in np.asarray(p.prods[pk]).reshape(N, N)[idx // N]], _proof(p.plv(pk), idx // N)
            else:
                row, path = [int(v) for v in np.asarray(p.sums[i * bj + jj]).reshape(N, N)[idx // N]], _proof(p.slv(i * bj + jj), idx // N)
        except Exception:
            row, path = [0] * N, [garbage() for _ in range(7)]
        mm = rng.random()
        if mm < 0.08:
            row[idx % N] += 1
        elif mm < 0.14 and path:
            path[rng.randrange(len(path))] = garbage()
        elif mm < 0.18:
            idx = (idx + 1) % (N * N)
        body = dict(job_id=jb(jid), which=w, i=i, j=jj, k=k, idx=idx, row=row, path=path)
        return (who_for(jid, "worker") if guided else rng.randint(1, 4)), {"Tiled": {"Serve": body}}, "Serve"
    jid = pick_job(("Posted",) if guided else st.jobs.keys()) or "00" * 32
    return rng.randint(1, 4), {"Tiled": {"SlashUnserved": dict(job_id=jb(jid))}}, "SlashUnserved"


def gen(height):
    r = rng.random()
    guided = rng.random() < 0.88
    if rng.random() < 0.12:
        return gen_da(guided)
    if r < 0.05:
        return rng.randint(1, 4), {"Transfer": {"to": hl(bytes.fromhex(C.pub_hex(rng.randint(1, 4)))), "amount": rng.choice([COIN, 10**14, 0])}}, "Transfer"
    if r < 0.24:
        p = rng.choice(POOL)
        bi, bk, bj = p.dims
        products = bi * bk * bj
        esc = rng.choice([30 * COIN * products] * 5 + [COIN * products, COIN * products - 1, 10**15])
        dims, ar, br = list(p.dims), [hl(x) for x in p.ar], [hl(x) for x in p.br]
        m = rng.random()
        if m < 0.05:
            dims[rng.randrange(3)] = rng.choice([0, 9, 200])
        elif m < 0.09:
            ar = ar[:-1]
        elif m < 0.12:
            br = br + [garbage()]
        sub = dict(dims=dims, a_roots=ar, b_roots=br, escrow=esc, deadline=max(0, height + rng.choice([-1, 8, 500, 500, 60])))
        return rng.randint(1, 4), {"Tiled": {"Submit": sub}}, "Submit", p
    if r < 0.36:
        jid = pick_job(("Open",) if guided else st.jobs.keys()) or "00" * 32
        return rng.randint(1, 4), {"Tiled": {"Claim": {"job_id": jb(jid)}}}, "Claim"
    if r < 0.52:
        jid = pick_job(("Claimed",) if guided else st.jobs.keys())
        if jid is None:
            return rng.randint(1, 4), {"Tiled": {"Claim": {"job_id": garbage()}}}, "Claim"
        p = pool_of.get(jid, rng.choice(POOL))
        pr, cr = [hl(x) for x in p.pr], [hl(x) for x in p.cr]
        m = rng.random()
        if m < 0.06:
            pr = pr[:-1]
        elif m < 0.10:
            cr = cr + [garbage()]
        elif m < 0.16:
            pr[rng.randrange(len(pr))] = garbage()
        return who_for(jid, "worker") if guided else rng.randint(1, 4), {"Tiled": {"Post": dict(job_id=jb(jid), product_roots=pr, c_roots=cr)}}, "Post"
    if r < 0.62:
        jid = pick_job(("Posted",) if guided else st.jobs.keys()) or "00" * 32
        return who_for(jid, "client") if guided else rng.randint(1, 4), {"Tiled": {rng.choice(["Accept", "Accept", "Settle"]): {"job_id": jb(jid)}}}, "Accept/Settle"
    if r < 0.68:
        jid = pick_job(st.jobs.keys()) or "00" * 32
        return rng.randint(1, 4), {"Tiled": {"Settle": {"job_id": jb(jid)}}}, "Settle"
    # fraud proof
    jid = pick_job(("Posted", "Claimed", "Open") if not guided else ("Posted",))
    if jid is None:
        return rng.randint(1, 4), {"Tiled": {"Settle": {"job_id": garbage()}}}, "Settle"
    p = pool_of.get(jid, rng.choice(POOL))
    nprod, nsum = len(p.prods), len(p.sums)
    if rng.random() < 0.5:
        bad = p.bad_products()
        k = rng.choice(bad) if bad and rng.random() < 0.85 else rng.randrange(nprod)
        if (p.prods[k] != p.honest[k]).any() and rng.random() < 0.9:
            cells = np.argwhere(p.prods[k] != p.honest[k])
            r_, c_ = map(int, cells[rng.randrange(len(cells))])
        else:
            r_, c_ = rng.randrange(N), rng.randrange(N)
        fr = p.product_proof(k, r_, c_)
    else:
        bad = p.bad_sums()
        ij = rng.choice(bad) if bad and rng.random() < 0.85 else rng.randrange(nsum)
        bk = p.dims[1]
        if bad and ij in bad and rng.random() < 0.9:
            cells = np.argwhere(p.sums[ij] != sum(p.prods[ij * bk + k] for k in range(bk)))
            r_, c_ = map(int, cells[rng.randrange(len(cells))])
        else:
            r_, c_ = rng.randrange(N), rng.randrange(N)
        fr = p.sum_proof(ij, r_, c_)
    body = next(iter(fr.values()))
    m = rng.random()
    if m < 0.07 and "a_row" in body:
        body["a_row"][rng.randrange(N)] += 1
    elif m < 0.12 and "b_col" in body:
        body["b_col"][rng.randrange(N)] ^= 1
    elif m < 0.17 and "z_row" in body:
        body["z_row"][body["c"]] += 1
    elif m < 0.22 and "c_row" in body:
        body["c_row"][body["c"]] += 1
    elif m < 0.27:
        pth = next((body[x] for x in ("z_path", "a_path", "b_path", "c_path") if x in body), None)
        if pth:
            pth[rng.randrange(len(pth))] = garbage()
    elif m < 0.31 and "i" in body:
        body["i"] = rng.choice([0, 5, 9])
    elif m < 0.34 and "z_rows" in body:
        body["z_rows"] = body["z_rows"][:-1]
    return rng.randint(1, 4), {"Tiled": {"Fraud": dict(job_id=jb(jid), kind=fr)}}, "Fraud(" + next(iter(fr)) + ")"


attempts, stats = [], {"accepted": 0, "rejected": 0, "ok": {}, "ko": {}}
ts = G["timestamp"]
for it in range(COUNT):
    ts += rng.choice([1, 3, 10, 20, 20, 40, 70, 130])
    height = len(st.headers)
    txs, names, new_pools = [], [], []
    for g in [gen(height) for _ in range(rng.choice([1, 1, 1, 1, 2]))]:
        seed, kind, name = g[0], g[1], g[2]
        who = C.pub_hex(seed)
        nonce = st.nonce.get(who, 0) + sum(1 for t in txs if t["from"] == who)
        if rng.random() < 0.02:
            nonce += rng.choice([-1, 1])
        t = {"from": who, "nonce": max(nonce, 0), "fee": rng.choice([0, 0, 0, 1000, 10**14]), "kind": kind}
        t["sig"] = C.priv(seed).sign(TR.tx_bytes(t)).hex()
        t["id"] = TR.tx_id(t).hex()
        txs.append(t)
        names.append(name)
        if name == "Submit":
            sub = kind["Tiled"]["Submit"]
            try:
                jid = TR.tiled_id(who, t["nonce"], tuple(sub["dims"]), [bytes(x) for x in sub["a_roots"]], [bytes(x) for x in sub["b_roots"]]).hex()
                new_pools.append((jid, g[3]))
            except Exception:
                pass
    blk = TR.mine(st, C.pub_hex(rng.randint(1, 4)), ("Native",), txs, ts)
    trial = copy.deepcopy(st)
    try:
        trial.apply(blk)
        ok, why = True, ""
    except P.Invalid as e:
        ok, why = False, str(e)
    if ok:
        st = trial
        for jid, pl in new_pools:
            pool_of.setdefault(jid, pl)
    stats["accepted" if ok else "rejected"] += 1
    for nm in names:
        d = stats["ok" if ok else "ko"]
        d[nm] = d.get(nm, 0) + 1
    attempts.append({"block": blk, "py_ok": ok, "why": why, "names": "+".join(names)})

final = {"minted": st.minted, "burned": st.burned, "height": len(st.headers),
         "accounts": [{"key": a["key"], "balance": st.bal.get(a["key"], 0), "nonce": st.nonce.get(a["key"], 0)} for a in G2["alloc"]],
         "jobs": {i: {"status": j["status"], "escrow_net": j["escrow_net"], "rebate_left": j["rebate_left"], "posted": bool(j["product_roots"])} for i, j in st.jobs.items()},
         "reputation": {a["key"]: st.rep.get(a["key"], {"done": 0, "slashed": 0, "expired": 0}) for a in G2["alloc"]}}
path = f"/tmp/diff_tiled_{SEED}.json" if os.name != "nt" else f"diff_tiled_{SEED}.json"
json.dump({"genesis": G2, "attempts": attempts, "final": final}, open(path, "w"))
print(f"[diff] {COUNT} blocks tried: {stats['accepted']} accepted, {stats['rejected']} refused by Python; money supply {'OK' if st.invariant() else 'BROKEN'}; job statuses: "
      + str({s: sum(1 for j in st.jobs.values() if j['status'] == s) for s in sorted({j['status'] for j in st.jobs.values()})}))
print("[diff] accepted:", dict(sorted(stats["ok"].items())))
print("[diff] refused:", dict(sorted(stats["ko"].items())))
r = subprocess.run([RUST, "diff", path], capture_output=True, text=True)
print(r.stdout.strip()[-1800:], r.stderr.strip()[-400:])
sys.exit(r.returncode)
