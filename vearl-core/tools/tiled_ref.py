#!/usr/bin/env python3
"""INDEPENDENT validator of BLOCK-WISE matrix product jobs (`tiled`): state machine (Submit, Claim, Post, Accept, Fraud, Settle),
`Product` work blocks (A = block A_ik, B = block B_kj) and the TWO fraud proofs (Product, Sum). Written from
docs/07-work-specification.md section 13. Re-validates a chain produced by `vearl-tiled-vectors`.

    ../target/release/vearl-tiled-vectors > tiled.json
    python3 tiled_ref.py tiled.json        # pip install blake3 numpy cryptography
"""
import base64, copy, json, struct, sys
import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
import spec_ref as S
import chain_ref as C
import market_ref as M
import prog_ref as P
from prog_ref import Invalid, u8, u32, u64, hb, i8b, vec, depth_for, vp

N = 128
MAX_BLOCKS = 8


# ---------------- bincode encoding of the transactions ----------------
def enc_fraud(f):
    ((k, p),) = f.items()
    path = lambda x: vec(x, lambda h: hb(h))
    if k == "Product":
        return (u32(0) + u8(p["i"]) + u8(p["j"]) + u8(p["k"]) + u32(p["r"]) + u32(p["c"]) + vec(p["a_row"], lambda v: i8b([v])) + path(p["a_path"])
                + vec(p["b_col"], lambda v: i8b([v])) + path(p["b_path"]) + vec(p["z_row"], lambda v: struct.pack("<i", v)) + path(p["z_path"]))
    if k == "Sum":
        return (u32(1) + u8(p["i"]) + u8(p["j"]) + u32(p["r"]) + u32(p["c"]) + vec(p["z_rows"], lambda zr: vec(zr, lambda v: struct.pack("<i", v)))
                + vec(p["z_paths"], lambda pp: path(pp)) + vec(p["c_row"], lambda v: struct.pack("<i", v)) + path(p["c_path"]))
    raise ValueError(k)


def enc_kind(kind):
    ((k, p),) = kind.items()
    if k != "Tiled":
        return P.enc_kind(kind)
    ((tk, q),) = p.items()
    j = lambda: hb(q["job_id"])
    if tk == "Submit":
        d = q["dims"]
        return (u32(13) + u32(0) + u8(d[0]) + u8(d[1]) + u8(d[2]) + vec(q["a_roots"], lambda h: hb(h)) + vec(q["b_roots"], lambda h: hb(h)) + u64(q["escrow"]) + u64(q["deadline"]))
    if tk == "Claim":
        return u32(13) + u32(1) + j()
    if tk == "Post":
        return u32(13) + u32(2) + j() + vec(q["product_roots"], lambda h: hb(h)) + vec(q["c_roots"], lambda h: hb(h))
    if tk == "Accept":
        return u32(13) + u32(3) + j()
    if tk == "Fraud":
        return u32(13) + u32(4) + j() + enc_fraud(q["kind"])
    if tk == "Settle":
        return u32(13) + u32(5) + j()
    if tk == "Fetch":
        return u32(13) + u32(6) + j() + u8(q["which"]) + u8(q["i"]) + u8(q["j"]) + u8(q["k"]) + u32(q["idx"])
    if tk == "Serve":
        return u32(13) + u32(7) + j() + u8(q["which"]) + u8(q["i"]) + u8(q["j"]) + u8(q["k"]) + u32(q["idx"]) + vec(q["row"], lambda v: struct.pack("<i", v)) + vec(q["path"], lambda h: hb(h))
    if tk == "SlashUnserved":
        return u32(13) + u32(8) + j()
    raise ValueError(tk)


def tx_bytes(t):
    return bytes.fromhex(t["from"]) + u64(t["nonce"]) + u64(t["fee"]) + enc_kind(t["kind"])


def tx_id(t):
    sig = bytes.fromhex(t["sig"])
    return S.H("vearl/tx/v1", tx_bytes(t) + u64(len(sig)) + sig)


def tiled_id(client_hex, nonce, dims, a_roots, b_roots):
    return S.H("vearl/tiled/v1", bytes.fromhex(client_hex), u64(nonce), bytes(dims), *a_roots, *b_roots)


# ---------------- engagements ----------------
def rows_root(m):
    return S.merkle_root([S.leaf_hash(i, S.i8bytes(m[i])) for i in range(N)])


def cols_root(m):
    return S.merkle_root([S.leaf_hash(j, S.i8bytes(m[:, j])) for j in range(N)])


def z_root(z):
    return S.c_root(np.asarray(z).reshape(N, N))


def block(m, bi, bj):
    return m[bi * N:(bi + 1) * N, bj * N:(bj + 1) * N]


class TiledChain:
    def __init__(self, V):
        g = V["genesis"]
        self.g, self.c, self.p, self.timing = g, g["constants"], g["params"], g["timing"]
        self.bal = {a["key"]: a["amount"] for a in g["alloc"]}
        self.genesis_supply = sum(self.bal.values())
        self.nonce, self.rep, self.jobs = {}, {}, {}
        self.minted = self.burned = 0
        self.headers, self.native = [], {}
        self.native_roots = {}
        self.tip_ts, self.tip_hash, self.target = g["timestamp"], b"\x00" * 32, g["target"]
        # data of the jobs, attached to their COMMITMENTS: (dims, roots of A, roots of B) -> (A, B)
        self.data = {}
        for d in V.get("tiled_data", []):
            dims = tuple(d["dims"])
            a = np.frombuffer(base64.b64decode(d["a"]), dtype=np.int8).astype(np.int64).reshape(dims[0] * N, dims[1] * N)
            b = np.frombuffer(base64.b64decode(d["b"]), dtype=np.int8).astype(np.int64).reshape(dims[1] * N, dims[2] * N)
            ar = tuple(rows_root(block(a, i, k)) for i in range(dims[0]) for k in range(dims[1]))
            br = tuple(cols_root(block(b, k, j)) for k in range(dims[1]) for j in range(dims[2]))
            self.data[(dims, ar, br)] = (a, b)

    def credit(self, k, x):
        self.bal[k] = self.bal.get(k, 0) + x

    def debit(self, k, x):
        if self.bal.get(k, 0) < x:
            raise Invalid("insufficient balance")
        self.bal[k] -= x

    def challenge_open(self, height, ts, at, at_ts):
        return not (height > at + self.c["challenge_window"] and ts >= at_ts + self.timing["challenge_secs"])

    def repd(self, k):
        return self.rep.setdefault(k, {"done": 0, "slashed": 0, "expired": 0})

    @staticmethod
    def products(j):
        return j["dims"][0] * j["dims"][1] * j["dims"][2]

    @staticmethod
    def pidx(j, i, jj, k):
        return (i * j["dims"][2] + jj) * j["dims"][1] + k

    @staticmethod
    def block_roots(j, i, jj, k):
        bi, bk, bj = j["dims"]
        if i >= bi or jj >= bj or k >= bk:
            return None
        return j["a_roots"][i * bk + k], j["b_roots"][k * bj + jj]

    def held(self):
        h = 0
        for j in self.jobs.values():
            if j["status"] == "Open":
                h += j["escrow_net"]
            elif j["status"] in ("Claimed", "Posted"):
                h += j["escrow_net"] + j["bond"] + j["fetch_held"]
        return h

    # ---------------- transactions ----------------
    def apply_tx(self, t, height, ts, miner):
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(t["from"])).verify(bytes.fromhex(t["sig"]), tx_bytes(t))
        except (InvalidSignature, ValueError):
            raise Invalid("signature")
        if self.nonce.get(t["from"], 0) != t["nonce"]:
            raise Invalid("nonce")
        self.nonce[t["from"]] = t["nonce"] + 1
        self.debit(t["from"], t["fee"])
        self.credit(miner, t["fee"])
        me, c = t["from"], self.c
        ((k, p),) = t["kind"].items()
        if k == "Transfer":
            self.debit(me, p["amount"])
            self.credit(hb(p["to"]).hex(), p["amount"])
            return
        assert k == "Tiled", k
        ((tk, q),) = p.items()
        if tk == "Submit":
            bi, bk, bj = q["dims"]
            ar, br = [hb(x) for x in q["a_roots"]], [hb(x) for x in q["b_roots"]]
            if any(d == 0 or d > MAX_BLOCKS for d in (bi, bk, bj)) or len(ar) != bi * bk or len(br) != bk * bj:
                raise Invalid("invalid specification")
            products = bi * bk * bj
            if q["escrow"] < c["min_escrow"] * products:
                raise Invalid("escrow below the per-product floor")
            if q["deadline"] <= height:
                raise Invalid("deadline passed")
            jid = tiled_id(me, t["nonce"], (bi, bk, bj), ar, br).hex()
            if jid in self.jobs:
                raise Invalid("duplicate job")
            self.debit(me, q["escrow"])
            burn = M.bps(q["escrow"], c["burn_bps"])
            self.burned += burn
            self.jobs[jid] = dict(client=me, dims=(bi, bk, bj), a_roots=ar, b_roots=br, escrow_net=q["escrow"] - burn, deadline=q["deadline"], status="Open",
                                  rebate_left=burn, product_roots=[], c_roots=[], fetches=[], fetch_held=0)
            return
        j = self.jobs.get(hb(q["job_id"]).hex())
        if j is None:
            raise Invalid("unknown job")
        if tk == "Claim":
            if j["status"] != "Open" or height > j["deadline"]:
                raise Invalid("job not claimable")
            bond = M.bps(j["escrow_net"], c["bond_bps"])
            self.debit(me, bond)
            j["status"], j["worker"], j["bond"] = "Claimed", me, bond
        elif tk == "Post":
            if j["status"] != "Claimed":
                raise Invalid("the job is not in progress")
            if j["worker"] != me or height > j["deadline"]:
                raise Invalid("cannot publish")
            if len(q["product_roots"]) != self.products(j) or len(q["c_roots"]) != j["dims"][0] * j["dims"][2]:
                raise Invalid("number of roots")
            j["product_roots"] = [hb(x) for x in q["product_roots"]]
            j["c_roots"] = [hb(x) for x in q["c_roots"]]
            j["status"], j["at"], j["at_ts"] = "Posted", height, ts
        elif tk == "Accept":
            if not (j["status"] == "Posted" and j["client"] == me):
                raise Invalid("cannot accept")
            self.credit(j["worker"], j["escrow_net"] + j["bond"])
            self.repd(j["worker"])["done"] += 1
            held, j["fetch_held"], j["fetches"], j["status"] = j["fetch_held"], 0, [], "Settled"
            self.credit(j["client"], held)
        elif tk == "Fetch":
            if j["status"] != "Posted":
                raise Invalid("no result published")
            if j["client"] != me:
                raise Invalid("only the client can request (client)")
            bi, bk, bj = j["dims"]
            w, i, jj, k, idx = q["which"], q["i"], q["j"], q["k"], q["idx"]
            ok = idx < N * N and i < bi and jj < bj and ((k < bk) if w == 0 else (k == 0 if w == 1 else False))
            if not ok:
                raise Invalid("index out of bounds")
            if not self.challenge_open(height, ts, j["at"], j["at_ts"]):
                raise Invalid("window closed")
            if len(j["fetches"]) >= c["max_pending_fetches"] or any(f[:5] == (w, i, jj, k, idx) for f in j["fetches"]):
                raise Invalid("duplicate request or too many")
            self.debit(me, c["fetch_fee"])
            j["fetch_held"] += c["fetch_fee"]
            j["fetches"].append((w, i, jj, k, idx, ts + self.timing["serve_secs"]))
        elif tk == "Serve":
            if j["status"] != "Posted":
                raise Invalid("no result published")
            if j["worker"] != me:
                raise Invalid("only the worker can serve")
            w, i, jj, k, idx = q["which"], q["i"], q["j"], q["k"], q["idx"]
            pos = next((x for x, f in enumerate(j["fetches"]) if f[:5] == (w, i, jj, k, idx)), None)
            if pos is None:
                raise Invalid("no pending request")
            root = j["product_roots"][self.pidx(j, i, jj, k)] if w == 0 else j["c_roots"][i * j["dims"][2] + jj]
            if len(q["row"]) != N or not vp(root, S.leaf_hash(idx // N, S.i32bytes(q["row"])), idx // N, q["path"], depth_for(N)):
                raise Invalid("invalid Merkle proof (Merkle)")
            del j["fetches"][pos]
            j["fetch_held"] -= c["fetch_fee"]
            j["at_ts"] = max(max(ts + c["fetch_extend_secs"] - self.timing["challenge_secs"], 0), j["at_ts"])
            j["at"] = max(max(height + 2 - c["challenge_window"], 0), j["at"])
            self.credit(me, c["fetch_fee"])
        elif tk == "SlashUnserved":
            if j["status"] != "Posted":
                raise Invalid("no result published")
            if not any(ts > f[5] for f in j["fetches"]):
                raise Invalid("no expired request (no expired)")
            reward = j["bond"] // 2
            self.credit(me, reward)
            self.burned += j["bond"] - reward
            self.credit(j["client"], j["escrow_net"] + j["fetch_held"])
            self.repd(j["worker"])["slashed"] += 1
            j["fetch_held"], j["fetches"], j["status"] = 0, [], "Slashed"
        elif tk == "Fraud":
            if j["status"] != "Posted":
                raise Invalid("no result published")
            if not self.challenge_open(height, ts, j["at"], j["at_ts"]):
                raise Invalid("dispute window closed")
            self.verify_fraud(j, q["kind"])
            reward = j["bond"] // 2
            self.credit(me, reward)
            self.burned += j["bond"] - reward
            self.credit(j["client"], j["escrow_net"] + j["fetch_held"])
            self.repd(j["worker"])["slashed"] += 1
            j["fetch_held"], j["fetches"], j["status"] = 0, [], "Slashed"
        elif tk == "Settle":
            if j["status"] == "Posted" and j["fetches"]:
                raise Invalid("pending requests")
            if j["status"] == "Posted" and not self.challenge_open(height, ts, j["at"], j["at_ts"]):
                self.credit(j["worker"], j["escrow_net"] + j["bond"])
                self.repd(j["worker"])["done"] += 1
                j["status"] = "Settled"
            elif j["status"] == "Open" and height > j["deadline"]:
                self.credit(j["client"], j["escrow_net"])
                j["status"] = "Expired"
            elif j["status"] == "Claimed" and height > j["deadline"]:
                self.repd(j["worker"])["expired"] += 1
                self.burned += j["bond"]
                self.credit(j["client"], j["escrow_net"])
                j["status"] = "Expired"
            else:
                raise Invalid("settlement impossible for now")

    # ---------------- fraud proofs ----------------
    def verify_fraud(self, j, f):
        n, d, cd = N, depth_for(N), depth_for(N)
        ((kind, p),) = f.items()
        zrow_ok = lambda root, r, row, path: len(row) == n and vp(root, S.leaf_hash(r, S.i32bytes(row)), r, path, cd)
        if kind == "Product":
            i, jj, k, r, c = p["i"], p["j"], p["k"], p["r"], p["c"]
            roots = self.block_roots(j, i, jj, k)
            if roots is None:
                raise Invalid("block index out of bounds")
            if r >= n or c >= n or len(p["a_row"]) != n or len(p["b_col"]) != n:
                raise Invalid("malformed proof")
            if any(abs(v) > 64 for v in p["a_row"] + p["b_col"]):
                raise Invalid("values out of bounds")
            if not vp(roots[0], S.leaf_hash(r, i8b(p["a_row"])), r, p["a_path"], d):
                raise Invalid("invalid opening of A")
            if not vp(roots[1], S.leaf_hash(c, i8b(p["b_col"])), c, p["b_path"], d):
                raise Invalid("invalid opening of B")
            if not zrow_ok(j["product_roots"][self.pidx(j, i, jj, k)], r, p["z_row"], p["z_path"]):
                raise Invalid("invalid opening of the product")
            if sum(a * b for a, b in zip(p["a_row"], p["b_col"])) == p["z_row"][c]:
                raise Invalid("correct entry: not a fraud")
        elif kind == "Sum":
            i, jj, r, c = p["i"], p["j"], p["r"], p["c"]
            bk = j["dims"][1]
            if i >= j["dims"][0] or jj >= j["dims"][2] or r >= n or c >= n or len(p["z_rows"]) != bk or len(p["z_paths"]) != bk:
                raise Invalid("malformed proof")
            for k in range(bk):
                if not zrow_ok(j["product_roots"][self.pidx(j, i, jj, k)], r, p["z_rows"][k], p["z_paths"][k]):
                    raise Invalid("invalid opening of the product")
            if not zrow_ok(j["c_roots"][i * j["dims"][2] + jj], r, p["c_row"], p["c_path"]):
                raise Invalid("invalid opening of the sum")
            if sum(z[c] for z in p["z_rows"]) == p["c_row"][c]:
                raise Invalid("correct sum: not a fraud")
        else:
            raise Invalid("unknown proof kind")

    # ---------------- blocks ----------------
    def matrices(self, epoch):
        if epoch not in self.native:
            self.native[epoch] = S.native_matrices(epoch, N)
        return self.native[epoch]

    def product_matrices(self, j, i, jj, k):
        a, b = self.data[(j["dims"], tuple(j["a_roots"]), tuple(j["b_roots"]))]
        ab, bb = block(a, i, k), block(b, k, jj)
        if (rows_root(ab), cols_root(bb)) != self.block_roots(j, i, jj, k):
            raise Invalid("the data does not match the committed roots")
        return ab, bb

    def apply(self, b):
        hd, p, c = b["header"], self.p, self.c
        if hd["version"] != 2:
            raise Invalid("version")
        if hd["height"] != len(self.headers):
            raise Invalid("height")
        if bytes.fromhex(hd["prev"]) != self.tip_hash:
            raise Invalid("prev")
        if hd["timestamp"] <= self.tip_ts:
            raise Invalid("timestamp not increasing")
        if hd["target"] != self.target:
            raise Invalid("target")
        jackpot = bytes.fromhex(hd["jackpot"])
        if not S.meets_target(jackpot, hd["target"]):
            raise Invalid("jackpot above the target")
        ids = [tx_id(t) for t in b["txs"]]
        if len(b["txs"]) > 256:                                   # MAX_BLOCK_TXS (consensus rule)
            raise Invalid("too many transactions")
        if sum(len(tx_bytes(t)) + 8 + len(bytes.fromhex(t["sig"])) for t in b["txs"]) > 4 << 20:   # MAX_BLOCK_TX_BYTES (bincode(SignedTx))
            raise Invalid("block transactions too heavy")
        root = C.merkle_tx_root(ids)
        if bytes.fromhex(hd["tx_root"]) != root:
            raise Invalid("tx_root")
        cls = hd["class"]
        if cls["type"] == "Native":
            a, bm = self.matrices(hd["height"] // c["epoch_len"])
            ct = ("Native",)
        else:
            assert cls["type"] == "Product"
            j = self.jobs.get(cls["job"])
            if j is None:
                raise Invalid("unknown job")
            if not (j["status"] in ("Claimed", "Posted") and j["worker"] == hd["miner"]):
                raise Invalid("the miner is not the active worker of this job")
            if self.block_roots(j, cls["i"], cls["j"], cls["k"]) is None:
                raise Invalid("block index out of bounds")
            a, bm = self.product_matrices(j, cls["i"], cls["j"], cls["k"])
            ct = ("Product", bytes.fromhex(cls["job"]), cls["i"], cls["j"], cls["k"])
        if (bytes.fromhex(hd["a_root"]), bytes.fromhex(hd["b_root"])) != self.work_roots(cls, hd["height"]):
            raise Invalid("work roots")
        sg = S.sigma_of(bytes.fromhex(hd["prev"]), bytes.fromhex(hd["miner"]), ct, hd["extra_nonce"], root, hd["timestamp"])
        T = p["n"] // p["tile"]
        if not (hd["ti"] < T and hd["tj"] < T):
            raise Invalid("tile out of bounds")
        jp, _, _ = S.work(a, bm, sg, p["n"], p["tile"], p["rank"], p["fold_k"])
        if jp[hd["ti"] * T + hd["tj"]] != jackpot:
            raise Invalid("jackpot different from the recomputation")
        snap = copy.deepcopy((self.bal, self.nonce, self.rep, self.jobs, self.burned))
        try:
            for t in b["txs"]:
                self.apply_tx(t, hd["height"], hd["timestamp"], hd["miner"])
            reward = C.base_reward(c, hd["height"])
            bonus = 0
            if cls["type"] == "Product" and cls["job"] in self.jobs:
                jb = self.jobs[cls["job"]]
                bonus = min(reward // 2, jb["rebate_left"])
                jb["rebate_left"] -= bonus
            self.credit(hd["miner"], reward + bonus)
            if bytes.fromhex(hd["accounts_root"]) != S.accounts_root(self.bal, self.nonce):
                raise Invalid("accounts_root")
        except Invalid:
            self.bal, self.nonce, self.rep, self.jobs, self.burned = snap
            raise
        self.minted += reward + bonus
        self.headers.append(hd)
        self.tip_ts = hd["timestamp"]
        self.tip_hash = S.H("vearl/header/v1", header_bytes(hd))

    def work_roots(self, cls, height):
        """Roots (A, B) of the work of a block (v2 header)."""
        if cls["type"] == "Native":
            ep = height // self.c["epoch_len"]
            if ep not in self.native_roots:
                a, bm = self.matrices(ep)
                self.native_roots[ep] = (S.rows_root(a), S.cols_root(bm))
            return self.native_roots[ep]
        return self.block_roots(self.jobs[cls["job"]], cls["i"], cls["j"], cls["k"])

    def post_root(self, miner, cls, txs, height, ts):
        """Account root after applying the block (reference miner); null root if the transactions are invalid."""
        snap = copy.deepcopy((self.bal, self.nonce, self.rep, self.jobs, self.burned))
        try:
            for t in txs:
                self.apply_tx(t, height, ts, miner)
            reward = C.base_reward(self.c, height)
            bonus = 0
            if cls["type"] == "Product" and cls["job"] in self.jobs:
                bonus = min(reward // 2, self.jobs[cls["job"]]["rebate_left"])
            self.credit(miner, reward + bonus)
            return S.accounts_root(self.bal, self.nonce)
        except Invalid:
            return b"\x00" * 32
        finally:
            self.bal, self.nonce, self.rep, self.jobs, self.burned = snap

    def invariant(self):
        return sum(self.bal.values()) + self.held() + self.burned == self.genesis_supply + self.minted


def class_tuple(c):
    return ("Native",) if c["type"] == "Native" else ("Product", bytes.fromhex(c["job"]), c["i"], c["j"], c["k"])


def header_bytes(hd):
    return S.header_bytes(hd["version"], hd["height"], bytes.fromhex(hd["prev"]), hd["timestamp"], bytes.fromhex(hd["miner"]), class_tuple(hd["class"]),
                          hd["extra_nonce"], hd["ti"], hd["tj"], bytes.fromhex(hd["jackpot"]), bytes.fromhex(hd["tx_root"]), hd["target"],
                          bytes.fromhex(hd["a_root"]), bytes.fromhex(hd["b_root"]), bytes.fromhex(hd["accounts_root"]))


def signed(st, seed, kind):
    who = C.pub_hex(seed)
    t = {"from": who, "nonce": st.nonce.get(who, 0), "fee": 0, "kind": kind}
    t["sig"] = C.priv(seed).sign(tx_bytes(t)).hex()
    t["id"] = tx_id(t).hex()
    return t


def mine(st, miner, cls, txs, ts):
    height = len(st.headers)
    if cls[0] == "Native":
        a, bm = st.matrices(height // st.c["epoch_len"])
    else:
        j = st.jobs[cls[1].hex()]
        a, bm = st.product_matrices(j, cls[2], cls[3], cls[4])
    root = C.merkle_tx_root([tx_id(t) for t in txs])
    T = st.p["n"] // st.p["tile"]
    for en in range(1 << 20):
        sg = S.sigma_of(st.tip_hash, bytes.fromhex(miner), cls, en, root, ts)
        jp, _, _ = S.work(a, bm, sg, st.p["n"], st.p["tile"], st.p["rank"], st.p["fold_k"])
        for idx, jk in enumerate(jp):
            if S.meets_target(jk, st.target):
                cj = {"type": "Native"} if cls[0] == "Native" else {"type": "Product", "job": cls[1].hex(), "i": cls[2], "j": cls[3], "k": cls[4]}
                ra, rb = st.work_roots(cj, height)
                hd = {"version": 2, "height": height, "prev": st.tip_hash.hex(), "timestamp": ts, "miner": miner, "class": cj, "extra_nonce": en,
                      "ti": idx // T, "tj": idx % T, "jackpot": jk.hex(), "tx_root": root.hex(), "target": st.target,
                      "a_root": ra.hex(), "b_root": rb.hex(), "accounts_root": st.post_root(miner, cj, txs, height, ts).hex()}
                return {"header": hd, "txs": txs}
    raise RuntimeError("no solution")


def _proof(levels, idx):
    out = []
    for lv in levels[:-1]:
        out.append(list(lv[idx ^ 1]))
        idx >>= 1
    return out


def build_extension(V, path):
    """The Python validator AUTHOR: it defines two block-wise jobs (A, B chosen by it), COMPUTES the products with its exact engine, mines
    `Product` work blocks, publishes the roots, settles the first honestly (Accept) and proves the wrong SUM of the second with a
    proof that it builds itself. The file is replayed by the RUST validator (vearl-tiled-vectors verify)."""
    import numpy as np
    m = TiledChain(V)
    for b in V["blocks"]:
        m.apply(b)
    ext, data = list(V["blocks"]), list(V["tiled_data"])
    rng = np.random.default_rng(99)
    ts = m.tip_ts
    client, worker, chall = C.pub_hex(1), C.pub_hex(2), C.pub_hex(3)
    T = lambda **kw: {"Tiled": kw}
    hl = lambda h: list(h)

    def push(miner, cls, txs, dt=20):
        nonlocal ts
        ts += dt
        b = mine(m, miner, cls, txs, ts)
        m.apply(b)
        b["hash"] = m.tip_hash.hex()
        ext.append(b)

    def sub_case(dims, honest):
        bi, bk, bj = dims
        a = rng.integers(-64, 65, (bi * N, bk * N)).astype(np.int64)
        b = rng.integers(-64, 65, (bk * N, bj * N)).astype(np.int64)
        ar = [rows_root(block(a, i, k)) for i in range(bi) for k in range(bk)]
        br = [cols_root(block(b, k, j)) for k in range(bk) for j in range(bj)]
        t = signed(m, 1, T(Submit=dict(dims=list(dims), a_roots=[hl(x) for x in ar], b_roots=[hl(x) for x in br], escrow=30 * 10**8 * bi * bk * bj, deadline=len(m.headers) + 300)))
        jid = tiled_id(client, t["nonce"], dims, ar, br)
        push(C.pub_hex(4), ("Native",), [t])
        push(C.pub_hex(4), ("Native",), [signed(m, 2, T(Claim=dict(job_id=hl(jid))))])
        prods = [[R_mm(block(a, i, k), block(b, k, j)) for k in range(bk)] for i in range(bi) for j in range(bj)]
        sums = [sum(p) for p in prods]
        if not honest:
            sums[0] = sums[0].copy()
            sums[0][5, 7] += 3
        data.append({"dims": list(dims), "a": base64.b64encode(a.astype(np.int8).tobytes()).decode(), "b": base64.b64encode(b.astype(np.int8).tobytes()).decode(),
                     "products": [base64.b64encode(np.asarray(x, dtype="<i4").tobytes()).decode() for ps in prods for x in ps],
                     "sums": [base64.b64encode(np.asarray(x, dtype="<i4").tobytes()).decode() for x in sums]})
        m.data[(dims, tuple(ar), tuple(br))] = (a, b)
        push(worker, ("Product", jid, bi - 1, bj - 1, bk - 1), [])
        pr = [z_root(x) for ps in prods for x in ps]
        cr = [z_root(x) for x in sums]
        push(C.pub_hex(4), ("Native",), [signed(m, 2, T(Post=dict(job_id=hl(jid), product_roots=[hl(x) for x in pr], c_roots=[hl(x) for x in cr])))])
        return jid, prods, sums

    R_mm = lambda x, y: np.rint(np.asarray(x, dtype=np.float64) @ np.asarray(y, dtype=np.float64)).astype(np.int64)
    zl = lambda z: S.c_levels(np.asarray(z).reshape(N, N))
    # 1) honest, accepted by the client
    jid1, _, _ = sub_case((1, 2, 2), True)
    push(C.pub_hex(4), ("Native",), [signed(m, 1, T(Accept=dict(job_id=hl(jid1))))])
    # 2) wrong sum (correct products), proven
    jid2, prods, sums = sub_case((2, 1, 2), False)
    fr = {"Sum": dict(i=0, j=0, r=5, c=7, z_rows=[[int(v) for v in prods[0][k][5]] for k in range(len(prods[0]))], z_paths=[_proof(zl(prods[0][k]), 5) for k in range(len(prods[0]))],
                      c_row=[int(v) for v in sums[0][5]], c_path=_proof(zl(sums[0]), 5))}
    push(C.pub_hex(4), ("Native",), [signed(m, 3, T(Fraud=dict(job_id=hl(jid2), kind=fr)))])
    # 3) availability: the client demands a SUM entry on-chain, the worker serves it (proof built here), acceptance
    jid3, prods3, sums3 = sub_case((1, 1, 2), True)
    idx3 = 9
    push(C.pub_hex(4), ("Native",), [signed(m, 1, T(Fetch=dict(job_id=hl(jid3), which=1, i=0, j=1, k=0, idx=idx3)))])
    push(C.pub_hex(4), ("Native",), [signed(m, 2, T(Serve=dict(job_id=hl(jid3), which=1, i=0, j=1, k=0, idx=idx3, row=[int(v) for v in np.asarray(sums3[1]).reshape(N, N)[idx3 // N]], path=_proof(zl(sums3[1]), idx3 // N))))])
    push(C.pub_hex(4), ("Native",), [signed(m, 1, T(Accept=dict(job_id=hl(jid3))))])
    assert m.jobs[jid3.hex()]["status"] == "Settled" and m.jobs[jid3.hex()]["fetch_held"] == 0
    assert m.jobs[jid1.hex()]["status"] == "Settled" and m.jobs[jid2.hex()]["status"] == "Slashed" and m.invariant()
    final = {"height": len(m.headers), "minted": m.minted, "burned": m.burned,
             "accounts": [{"key": a["key"], "balance": m.bal.get(a["key"], 0), "nonce": m.nonce.get(a["key"], 0)} for a in V["genesis"]["alloc"]],
             "jobs": [{"id": i, "status": j["status"], "rebate_left": j["rebate_left"], "escrow_net": j["escrow_net"], "posted": bool(j["product_roots"])} for i, j in m.jobs.items()],
             "reputation": []}
    json.dump({"genesis": V["genesis"], "tiled_data": data, "blocks": ext, "final": final}, open(path, "w"))
    print(f"Python extension written: {path} ({len(ext)} blocks including {len(ext) - len(V['blocks'])} mined and signed by the Python validator)")


def main(path):
    V = json.load(open(path))
    if "--extend" in sys.argv:
        return build_extension(V, sys.argv[sys.argv.index("--extend") + 1])
    m = TiledChain(V)
    for k, b in enumerate(V["blocks"]):
        try:
            m.apply(b)
        except Invalid as e:
            print(f"REFUSED at block {k}: {e}")
            sys.exit(1)
        assert m.invariant(), f"money supply broken at block {k}"
        assert bytes.fromhex(b["hash"]) == m.tip_hash, f"different header hash at block {k}"
    f = V["final"]
    bad = 0
    for name, got, want in [("height",len(m.headers), f["height"]), ("emission", m.minted, f["minted"]), ("burned",m.burned, f["burned"])]:
        ok = got == want
        bad += not ok
        print(("ok   " if ok else "FAIL"), name, got if ok else f"{got} != {want}")
    for a in f["accounts"]:
        ok = (m.bal.get(a["key"], 0), m.nonce.get(a["key"], 0)) == (a["balance"], a["nonce"])
        bad += not ok
        print(("ok   " if ok else "FAIL"), "account", a["key"][:8], m.bal.get(a["key"], 0))
    for jj in f["jobs"]:
        pj = m.jobs[jj["id"]]
        ok = (pj["status"], pj["escrow_net"], pj["rebate_left"], bool(pj["product_roots"])) == (jj["status"], jj["escrow_net"], jj["rebate_left"], jj["posted"])
        bad += not ok
        print(("ok   " if ok else "FAIL"), "job", jj["id"][:8], pj["status"])
    print(f"{len(V['blocks'])} blocks replayed by the independent Python validator; money supply {'OK' if m.invariant() else 'BROKEN'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1])
