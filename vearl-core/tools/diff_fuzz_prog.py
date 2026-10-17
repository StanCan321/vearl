#!/usr/bin/env python3
"""DIFFERENTIAL FUZZ of the TENSOR PROGRAMS: the independent Python validator (prog_ref.py) and the Rust validator receive
exactly the same blocks (random program transactions: valid or invalid submissions, claims, publications
of operations in order or not, with true, falsified or arbitrary roots, acceptances, settlements, and fraud
proofs of the FOUR kinds with valid or altered openings, on honest or falsified operations) and must TAKE THE SAME
DECISION; the final state must be identical. The data come from the programs of tests/vectors/prog-v1.json (CNN, transformer,
and their falsified variants).

    python3 tools/diff_fuzz_prog.py [NB_BLOCKS=400] [SEED=1] [--rust ../target/release/vearl-prog-vectors]
"""
import base64, copy, json, os, random, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import spec_ref as S
import chain_ref as C
import vearl_ref as R
import market_ref as M
import prog_ref as P

N = 128
COIN = 10**8
args = [a for a in sys.argv[1:] if not a.startswith("--")]
COUNT = int(args[0]) if args else 400
SEED = int(args[1]) if len(args) > 1 else 1
RUST = sys.argv[sys.argv.index("--rust") + 1] if "--rust" in sys.argv else "target/release/vearl-prog-vectors"
VEC = os.environ.get("PROG_VECTORS", "tests/vectors/prog-v1.json")
rng = random.Random(SEED)

V = json.load(open(VEC))
G = V["genesis"]


# ---------------- pool of programs (known data, including falsified variants) ----------------
def hl(b):
    return list(b)


class Pool:
    def __init__(self, sub, data):
        dec = lambda s: np.frombuffer(base64.b64decode(s), dtype=np.int8).astype(np.int64).reshape(N, N)
        self.sub = sub
        self.ops = sub["ops"]
        self.x0, self.ws = dec(data["x0"]), [dec(w) for w in data["ws"]]
        self.tens = [dec(t) for t in data["tensors"]]     # published by the worker (possibly wrong): T_0 .. T_posted
        self.zs = [None if z is None else np.frombuffer(base64.b64decode(z), dtype="<i4").astype(np.int64).reshape(N, N) for z in data["zs"]]
        self.x0_root = P.ProgChain.croot(self.x0)
        self.w_roots = [P.ProgChain.rroot(w) for w in self.ws]
        self.gemms = sum(1 for o in self.ops if next(iter(o)) in ("Gemm", "GemmAT"))
        self._c, self._r, self._z, self._bad = {}, {}, {}, {}

    def cols(self, t):
        if t not in self._c:
            self._c[t] = S.merkle_levels([S.leaf_hash(x, S.i8bytes(self.tens[t][:, x])) for x in range(N)])
        return self._c[t]

    def rows(self, key, mat):
        if key not in self._r:
            self._r[key] = S.merkle_levels([S.leaf_hash(x, S.i8bytes(mat[x])) for x in range(N)])
        return self._r[key]

    def zlv(self, k):
        if k not in self._z:
            self._z[k] = S.c_levels(self.zs[k])
        return self._z[k]

    def x_root(self, k):
        return self.cols(k + 1)[-1][0]

    def z_root(self, k):
        return self.zlv(k)[-1][0] if k < len(self.zs) and self.zs[k] is not None else b"\x00" * 32

    def mismatch(self, k):
        """Cells (i, j) where operation k of the program, applied to the PUBLISHED tensors, does not give back the published output:
        (possible_proof_kind -> list of cells)."""
        if k in self._bad:
            return self._bad[k]
        op = self.ops[k]
        ((kind, p),) = op.items()
        out = {}
        T = self.tens
        if kind in ("Gemm", "GemmAT"):
            a = self.ws[p["w"]] if kind == "Gemm" else T[p["a"]].T
            zt = R._mm(a, T[p["x"]])
            g = np.argwhere(zt != self.zs[k])
            out["Gemm"] = [tuple(map(int, c)) for c in g[:3]]
            act = np.clip(self.zs[k] >> p["shift"], 0 if p["relu"] else -64, 64)
            ga = np.argwhere(act != T[k + 1])
            out["Act"] = [tuple(map(int, c)) for c in ga[:3]]
        elif kind in ("Softmax", "SoftmaxLo", "LayerNorm", "CausalSoftmax", "CausalSoftmaxS", "RmsNorm"):
            _, ref = R.eval_op(op, T, self.ws)
            cols = np.nonzero((ref != T[k + 1]).any(axis=0))[0]
            out["Col"] = [(0, int(c)) for c in cols[:3]]
        elif kind == "MulW":
            _, ref = R.eval_op(op, T, self.ws)
            g = np.argwhere(ref != T[k + 1])
            out["MulW"] = [tuple(map(int, c)) for c in g[:3]]
        elif kind in ("Bias", "Lut", "Mul"):
            _, ref = R.eval_op(op, T, self.ws)
            cols = np.nonzero((ref != T[k + 1]).any(axis=0))[0]
            out["Bias" if kind == "Mul" else kind] = [(0, int(c)) for c in cols[:3]]     # Mul: proof of the Bias shape
        else:
            _, ref = R.eval_op(op, T, self.ws)
            g = np.argwhere(ref != T[k + 1])
            out["Elem"] = [tuple(map(int, c)) for c in g[:3]]
        self._bad[k] = out
        return out

    def proof(self, k, kind, i, j):
        """Fraud proof of kind `kind` on operation k, cell (i, j), with VALID openings."""
        op = self.ops[k]
        ((okind, p),) = op.items()
        T = self.tens
        pr = lambda lv, idx: [hl(h) for h in _proof(lv, idx)]
        if kind == "Gemm":
            a = self.ws[p["w"]] if okind == "Gemm" else T[p["a"]].T
            key = ("w", p["w"]) if okind == "Gemm" else ("at", p["a"])
            b = T[p["x"]]
            return {"Gemm": dict(i=i, j=j, w_row=[int(v) for v in a[i]], w_path=pr(self.rows(key, a), i), x_col=[int(v) for v in b[:, j]],
                                 x_path=pr(self.cols(p["x"]), j), z_row=[int(v) for v in self.zs[k][i]], z_path=pr(self.zlv(k), i))}
        if kind == "Act":
            return {"Act": dict(i=i, j=j, z_row=[int(v) for v in self.zs[k][i]], z_path=pr(self.zlv(k), i), out_col=[int(v) for v in T[k + 1][:, j]],
                                out_path=pr(self.cols(k + 1), j))}
        if kind == "Lut":
            bt = self.ws[p["w"]]
            return {"Lut": dict(j=j, in_col=[int(v) for v in T[p["x"]][:, j]], in_path=pr(self.cols(p["x"]), j), t_row=[int(v) for v in bt[p["row"]]],
                                t_path=pr(self.rows(("w", p["w"]), bt), p["row"]), out_col=[int(v) for v in T[k + 1][:, j]], out_path=pr(self.cols(k + 1), j))}
        if kind == "Bias":
            bt = self.ws[p["w"]]
            return {"Bias": dict(j=j, in_col=[int(v) for v in T[p["x"]][:, j]], in_path=pr(self.cols(p["x"]), j), b_row=[int(v) for v in bt[p["row"]]],
                                 b_path=pr(self.rows(("w", p["w"]), bt), p["row"]), out_col=[int(v) for v in T[k + 1][:, j]], out_path=pr(self.cols(k + 1), j))}
        if kind == "Col":
            return {"Col": dict(j=j, in_col=[int(v) for v in T[p["x"]][:, j]], in_path=pr(self.cols(p["x"]), j), out_col=[int(v) for v in T[k + 1][:, j]],
                                out_path=pr(self.cols(k + 1), j))}
        if kind == "MulW":
            bt = self.ws[p["w"]]
            return {"MulW": dict(i=i, j=j, in_col=[int(v) for v in T[p["x"]][:, j]], in_path=pr(self.cols(p["x"]), j), w_row=[int(v) for v in bt[i]],
                                 w_path=pr(self.rows(("w", p["w"]), bt), i), out_col=[int(v) for v in T[k + 1][:, j]], out_path=pr(self.cols(k + 1), j))}
        ins = []
        for (t, si, sj) in P.op_inputs(op, i, j):
            ins.append({"tensor": t, "col": sj, "vals": [int(v) for v in T[t][:, sj]], "path": pr(self.cols(t), sj)})
        return {"Elem": dict(i=i, j=j, ins=ins, out_col=[int(v) for v in T[k + 1][:, j]], out_path=pr(self.cols(k + 1), j))}


def struct_i32(v):
    import struct
    return struct.pack("<i", v)


def _proof(levels, idx):
    out = []
    for lv in levels[:-1]:
        out.append(lv[idx ^ 1])
        idx >>= 1
    return out


submits = [t["kind"]["Prog"]["Submit"] for b in V["blocks"] for t in b["txs"] if "Prog" in t["kind"] and "Submit" in t["kind"]["Prog"]]
POOL = [Pool(sub, d) for sub, d in zip(submits, V["prog_data"])]

# ---------------- reference state ----------------
G2 = copy.deepcopy(G)
G2["alloc"] = [{"key": C.pub_hex(s), "amount": 10_000 * COIN} for s in range(1, 5)]
st = P.ProgChain({"genesis": G2, "prog_data": []})
st.bind = False
pool_of = {}      # id du job -> Pool
KINDS_P = ["Gemm", "Act", "Elem", "Col", "Bias", "Lut", "MulW"]


def garbage():
    return hl(bytes(rng.randrange(256) for _ in range(32)))


def pick_job(statuses):
    ids = [i for i, j in st.progs.items() if j["status"] in statuses]
    return rng.choice(ids) if ids else None


def who_for(jid, field):
    j = st.progs.get(jid)
    target = j.get(field) if j else None
    return next((s for s in range(1, 5) if C.pub_hex(s) == target), rng.randint(1, 4))


def gen_da(guided):
    """Data availability: Fetch (client), Serve (worker), SlashUnserved (anyone)."""
    m = rng.random()
    if m < 0.4:
        jid = pick_job(("Complete",) if guided else st.progs.keys()) or "00" * 32
        j = st.progs.get(jid)
        nops = len(j["ops"]) if j else 4
        tensor = rng.randint(1, nops) if rng.random() < 0.9 else rng.randint(0, nops + 2)
        col = rng.randrange(N) if rng.random() < 0.95 else N + rng.randint(0, 3)
        return (who_for(jid, "client") if guided else rng.randint(1, 4)), {"Prog": {"Fetch": dict(job_id=hl(bytes.fromhex(jid)), tensor=tensor, col=col)}}, "Fetch"
    if m < 0.85:
        cands = [i for i, j in st.progs.items() if j["status"] == "Complete" and j["fetches"]]
        if not cands:
            return rng.randint(1, 4), {"Prog": {"SlashUnserved": dict(job_id=garbage())}}, "SlashUnserved"
        jid = rng.choice(cands)
        j = st.progs[jid]
        tensor, col, _ = rng.choice(j["fetches"])
        p = pool_of.get(jid, rng.choice(POOL))
        tt = min(tensor, len(p.tens) - 1)
        vals = [int(v) for v in p.tens[tt][:, col]]
        path = [hl(h) for h in _proof(p.cols(tt), col)]
        mm = rng.random()
        if mm < 0.08:
            vals[rng.randrange(N)] += 1
        elif mm < 0.14:
            path[rng.randrange(len(path))] = garbage()
        elif mm < 0.18:
            vals = vals[:-1]
        elif mm < 0.22:
            col = (col + 1) % N
        return (who_for(jid, "worker") if guided else rng.randint(1, 4)), {"Prog": {"Serve": dict(job_id=hl(bytes.fromhex(jid)), tensor=tensor, col=col, vals=vals, path=path)}}, "Serve"
    jid = pick_job(("Complete",) if guided else st.progs.keys()) or "00" * 32
    return rng.randint(1, 4), {"Prog": {"SlashUnserved": dict(job_id=hl(bytes.fromhex(jid)))}}, "SlashUnserved"


def gen(height):
    r = rng.random()
    guided = rng.random() < 0.88
    if rng.random() < 0.12:
        return gen_da(guided)
    if r < 0.06:
        return rng.randint(1, 4), {"Transfer": {"to": hl(bytes.fromhex(C.pub_hex(rng.randint(1, 4)))), "amount": rng.choice([COIN, 10**14, 0])}}, "Transfer"
    if r < 0.24:
        p = rng.choice(POOL)
        esc = rng.choice([30 * COIN * p.gemms] * 5 + [COIN, 10 ** 15])
        ops = p.ops
        if rng.random() < 0.06:
            ops = rng.choice([[{"Softmax": {"x": 1, "len": 4}}], [{"Gemm": {"w": 0, "x": 5, "shift": 3, "relu": True}}], ops[:1] + [{"Gemm": {"w": 99, "x": 0, "shift": 3, "relu": True}}]])
        sub = dict(x0_root=hl(p.x0_root), w_roots=[hl(w) for w in p.w_roots], ops=ops, escrow=esc, deadline=max(0, height + rng.choice([-1, 8, 500, 500, 60])))
        return rng.randint(1, 4), {"Prog": {"Submit": sub}}, "Submit", p
    if r < 0.36:
        jid = pick_job(("Open",) if guided else st.progs.keys()) or "00" * 32
        return rng.randint(1, 4), {"Prog": {"Claim": {"job_id": hl(bytes.fromhex(jid))}}}, "Claim"
    if r < 0.60:
        jid = pick_job(("Working",) if guided else st.progs.keys())
        if jid is None:
            return rng.randint(1, 4), {"Prog": {"Claim": {"job_id": garbage()}}}, "Claim"
        j, p = st.progs[jid], pool_of.get(jid, rng.choice(POOL))
        k = len(j["posted"]) if (guided or rng.random() < .5) else rng.randrange(len(j["ops"]))
        kk = min(k, len(p.ops) - 1)
        zr, xr = hl(p.z_root(kk)), hl(p.x_root(kk)) if kk + 1 < len(p.tens) else garbage()
        if rng.random() < 0.06:
            xr = garbage()
        if rng.random() < 0.05:
            zr = garbage()
        if kk + 1 >= len(p.tens):
            xr = garbage()
        if rng.random() < 0.3:
            # BATCH publication: `cnt` consecutive operations in one transaction
            cnt = rng.randint(1, max(1, len(j["ops"]) - k)) if guided else rng.randint(1, len(j["ops"]) + 1)
            posts = []
            for kq in range(k, k + cnt):
                k2 = min(kq, len(p.tens) - 2)
                z2, x2 = hl(p.z_root(k2)), (hl(p.x_root(k2)) if k2 + 1 < len(p.tens) else garbage())
                if rng.random() < 0.05:
                    x2 = garbage()
                posts.append([z2, x2])
            return who_for(jid, "worker") if guided else rng.randint(1, 4), {"Prog": {"PostBatch": dict(job_id=hl(bytes.fromhex(jid)), op=k, posts=posts)}}, "PostBatch"
        return who_for(jid, "worker") if guided else rng.randint(1, 4), {"Prog": {"PostOp": dict(job_id=hl(bytes.fromhex(jid)), op=k, z_root=zr, x_root=xr)}}, "PostOp"
    if r < 0.68:
        jid = pick_job(("Complete",) if guided else st.progs.keys()) or "00" * 32
        return who_for(jid, "client") if guided else rng.randint(1, 4), {"Prog": {rng.choice(["Accept", "Accept", "Settle"]): {"job_id": hl(bytes.fromhex(jid))}}}, "Accept/Settle"
    if r < 0.73:
        jid = pick_job(st.progs.keys()) or "00" * 32
        return rng.randint(1, 4), {"Prog": {"Settle": {"job_id": hl(bytes.fromhex(jid))}}}, "Settle"
    # fraud proof
    jid = pick_job(("Working", "Complete") if guided else st.progs.keys())
    if jid is None:
        return rng.randint(1, 4), {"Prog": {"Settle": {"job_id": garbage()}}}, "Settle"
    j, p = st.progs[jid], pool_of.get(jid, rng.choice(POOL))
    npost = max(1, len(j["posted"]))
    k = rng.randrange(min(npost, len(p.ops)))
    # preferably an operation that is really wrong in the published data
    bad_ops = [x for x in range(min(len(j["posted"]), len(p.ops), len(p.tens) - 1)) if any(p.mismatch(x).values())]
    if bad_ops and rng.random() < 0.8:
        k = rng.choice(bad_ops)
    k = min(k, len(p.tens) - 2)
    ops_kind = next(iter(p.ops[k]))
    kinds_ok = {"Gemm": ["Gemm", "Act"], "GemmAT": ["Gemm", "Act"], "Softmax": ["Col"], "SoftmaxLo": ["Col"], "CausalSoftmax": ["Col"], "CausalSoftmaxS": ["Col"], "LayerNorm": ["Col"], "RmsNorm": ["Col"], "Bias": ["Bias"], "Lut": ["Lut"], "Mul": ["Bias"], "MulW": ["MulW"]}.get(ops_kind, ["Elem"])
    kind = rng.choice(kinds_ok if rng.random() < 0.9 else KINDS_P)
    mm = p.mismatch(k).get(kind, [])
    if mm and rng.random() < 0.85:
        i, jj = rng.choice(mm)
    else:
        i, jj = rng.randrange(N), rng.randrange(N)
    try:
        fr = p.proof(k, kind, i, jj)
    except Exception:
        fr = p.proof(k, kinds_ok[0], i, jj)
    body = next(iter(fr.values()))
    m = rng.random()
    if m < 0.08 and "w_row" in body:
        body["w_row"][rng.randrange(N)] += 1
    elif m < 0.11 and ("b_row" in body or "t_row" in body):
        body["b_row" if "b_row" in body else "t_row"][rng.randrange(N)] += 1
    elif m < 0.14 and "out_col" in body:
        body["out_col"][rng.randrange(N)] ^= 1
    elif m < 0.18 and "z_row" in body:
        body["z_row"][body["j"]] += 1
    elif m < 0.22:
        pth = next((body[x] for x in ("z_path", "out_path", "w_path", "in_path", "b_path", "t_path") if x in body), None)
        if pth:
            pth[rng.randrange(len(pth))] = garbage()
    return rng.randint(1, 4), {"Prog": {"Fraud": dict(job_id=hl(bytes.fromhex(jid)), op=k, kind=fr)}}, "Fraud(" + next(iter(fr)) + ")"


# ---------------- loop: Python decides, we record ----------------
attempts, stats = [], {"accepted": 0, "rejected": 0, "ok": {}, "ko": {}}
ts = G["timestamp"]
for it in range(COUNT):
    ts += rng.choice([1, 3, 10, 20, 20, 40, 70, 130])
    height = len(st.headers)
    txs, names, new_pools = [], [], []
    burst = None
    if rng.random() < 0.25:
        wj = [i for i, j in st.progs.items() if j["status"] == "Working"]
        if wj:
            jid = rng.choice(wj)
            j, p = st.progs[jid], pool_of.get(jid)
            if p is not None:
                left = len(j["ops"]) - len(j["posted"])
                burst = (jid, p, len(j["posted"]), rng.randint(1, max(1, left)))
    gens = []
    if burst:
        jid, p, k0, cnt = burst
        for k in range(k0, k0 + cnt):
            kk = min(k, len(p.tens) - 2)
            gens.append((who_for(jid, "worker"), {"Prog": {"PostOp": dict(job_id=hl(bytes.fromhex(jid)), op=k, z_root=hl(p.z_root(kk)), x_root=hl(p.x_root(kk)))}}, "PostOp"))
    else:
        gens = [gen(height) for _ in range(rng.choice([1, 1, 1, 1, 2]))]
    for g in gens:
        seed, kind, name = g[0], g[1], g[2]
        who = C.pub_hex(seed)
        nonce = st.nonce.get(who, 0) + sum(1 for t in txs if t["from"] == who)
        if rng.random() < 0.02:
            nonce += rng.choice([-1, 1])
        t = {"from": who, "nonce": max(nonce, 0), "fee": rng.choice([0, 0, 0, 1000, 10**14]), "kind": kind}
        t["sig"] = C.priv(seed).sign(P.tx_bytes(t)).hex()
        t["id"] = P.tx_id(t).hex()
        txs.append(t)
        names.append(name)
        if name == "Submit":
            sub = kind["Prog"]["Submit"]
            new_pools.append((P.prog_id(who, t["nonce"], bytes(sub["x0_root"]), [bytes(w) for w in sub["w_roots"]], sub["ops"]).hex(), g[3]))
    blk = P._mine(st, C.pub_hex(rng.randint(1, 4)), ("Native",), txs, ts)
    trial = copy.deepcopy(st)
    try:
        trial.apply(blk)
        ok, why = True, ""
    except P.Invalid as e:
        ok, why = False, str(e)
    if ok:
        st = trial
        for jid, pl in new_pools:
            pool_of[jid] = pl
    stats["accepted" if ok else "rejected"] += 1
    for nm in names:
        d = stats["ok" if ok else "ko"]
        d[nm] = d.get(nm, 0) + 1
    attempts.append({"block": blk, "py_ok": ok, "why": why, "names": "+".join(names)})

final = {"minted": st.minted, "burned": st.burned, "height": len(st.headers),
         "accounts": [{"key": a["key"], "balance": st.bal.get(a["key"], 0), "nonce": st.nonce.get(a["key"], 0)} for a in G2["alloc"]],
         "jobs": {i: {"status": j["status"], "escrow_net": j["escrow_net"], "rebate_left": j["rebate_left"], "posted": len(j["posted"])} for i, j in st.progs.items()},
         "reputation": {a["key"]: st.rep.get(a["key"], {"done": 0, "slashed": 0, "expired": 0}) for a in G2["alloc"]}}
path = f"/tmp/diff_prog_{SEED}.json" if os.name != "nt" else f"diff_prog_{SEED}.json"
json.dump({"genesis": G2, "attempts": attempts, "final": final}, open(path, "w"))
print(f"[diff] {COUNT} blocks tried: {stats['accepted']} accepted, {stats['rejected']} refused by Python; money supply {'OK' if st.invariant() else 'BROKEN'}; program statuses: "
      + str({s: sum(1 for j in st.progs.values() if j['status'] == s) for s in sorted({j['status'] for j in st.progs.values()})}))
print("[diff] accepted:", dict(sorted(stats["ok"].items())))
print("[diff] refused:", dict(sorted(stats["ko"].items())))
r = subprocess.run([RUST, "diff", path], capture_output=True, text=True)
print(r.stdout.strip()[-1800:], r.stderr.strip()[-400:])
sys.exit(r.returncode)
