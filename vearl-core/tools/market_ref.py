#!/usr/bin/env python3
"""INDEPENDENT validator of the compute market (jobs, auctions, bonds, useful-work blocks, data, fraud),
written from docs/07-work-specification.md section 11. It re-validates a chain produced by the reference node
(`vearl-market-vectors`), finds the final state and refuses, for the right reason, altered transactions and blocks.

    ../target/release/vearl-market-vectors > market.json
    python3 market_ref.py market.json      # pip install blake3 numpy cryptography
"""
import base64, copy, json, struct, sys
import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
import spec_ref as S
import chain_ref as C


class Invalid(Exception):
    pass


def bps(x, b):
    return x * b // 10_000


def u64(x):
    return struct.pack("<Q", x)


def u32(x):
    return struct.pack("<I", x)


def vec_hash(path):
    return u64(len(path)) + b"".join(bytes.fromhex(p) for p in path)


def kind_bytes(k):
    t = k["type"]
    hb = lambda name: bytes.fromhex(k[name])
    if t == "Transfer":
        return u32(0) + hb("to") + u64(k["amount"])
    if t == "SubmitJob":
        return u32(1) + hb("a_root") + hb("b_root") + u64(k["escrow"]) + u64(k["deadline"])
    if t == "ClaimJob":
        return u32(2) + hb("job_id")
    if t == "PostResult":
        return u32(3) + hb("job_id") + hb("c_root")
    if t == "Accept":
        return u32(4) + hb("job_id")
    if t == "Fraud":
        return (u32(5) + hb("job_id") + u32(k["i"]) + u32(k["j"]) + u64(len(k["a_row"])) + bytes(v & 0xFF for v in k["a_row"]) + vec_hash(k["a_path"]) +
                u64(len(k["b_col"])) + bytes(v & 0xFF for v in k["b_col"]) + vec_hash(k["b_path"]) + u64(len(k["c_row"])) + S.i32bytes(k["c_row"]) + vec_hash(k["c_path"]))
    if t == "Settle":
        return u32(6) + hb("job_id")
    if t == "SubmitJobAuction":
        return u32(7) + hb("a_root") + hb("b_root") + u64(k["escrow"]) + u64(k["deadline"]) + u64(k["auction_secs"])
    if t == "Bid":
        return u32(8) + hb("job_id") + u64(k["price"])
    if t == "Fetch":
        return u32(9) + hb("job_id") + u32(k["i"]) + u32(k["j"])
    if t == "Serve":
        return u32(10) + hb("job_id") + u32(k["i"]) + u32(k["j"]) + u64(len(k["row"])) + S.i32bytes(k["row"]) + vec_hash(k["path"])
    if t == "SlashUnserved":
        return u32(11) + hb("job_id")
    raise ValueError(t)


def tx_bytes(t):
    return bytes.fromhex(t["from"]) + u64(t["nonce"]) + u64(t["fee"]) + kind_bytes(t["kind"])


def tx_id(t):
    sig = bytes.fromhex(t["sig"])
    return S.H("vearl/tx/v1", tx_bytes(t) + u64(len(sig)) + sig)


def job_id_of(client, nonce, a_root, b_root):
    return S.H("vearl/job/v1", bytes.fromhex(client), u64(nonce), a_root, b_root)


def verify_path(root, leaf, index, path, depth):
    if len(path) != depth or index >= (1 << depth):
        return False
    cur = leaf
    for sib in path:
        cur = S.H("vearl/merkle/node/v1", cur, sib) if index & 1 == 0 else S.H("vearl/merkle/node/v1", sib, cur)
        index >>= 1
    return cur == root


def depth_for(count):
    d = 0
    while (1 << d) < count:
        d += 1
    return d


def header_hash(hd):
    return S.H("vearl/header/v1", header_bytes(hd))


def class_tuple(c):
    return ("Native",) if c["type"] == "Native" else ("Job", bytes.fromhex(c["job"]))


def header_bytes(hd):
    return S.header_bytes(hd["version"], hd["height"], bytes.fromhex(hd["prev"]), hd["timestamp"], bytes.fromhex(hd["miner"]), class_tuple(hd["class"]),
                          hd["extra_nonce"], hd["ti"], hd["tj"], bytes.fromhex(hd["jackpot"]), bytes.fromhex(hd["tx_root"]), hd["target"],
                          bytes.fromhex(hd["a_root"]), bytes.fromhex(hd["b_root"]), bytes.fromhex(hd["accounts_root"]))


class Market:
    def __init__(self, V):
        g = V["genesis"]
        self.g, self.c, self.p = g, g["constants"], g["params"]
        self.timing = g["timing"]
        self.bal = {a["key"]: a["amount"] for a in g["alloc"]}
        self.genesis_supply = sum(a["amount"] for a in g["alloc"])
        self.nonce, self.rep, self.jobs = {}, {}, {}
        self.minted = self.burned = 0
        self.headers = []
        self.tip_ts, self.tip_hash = g["timestamp"], b"\x00" * 32
        self.target = g["target"]
        self.native = {}
        self.native_roots = {}
        n = self.p["n"]
        # data of the jobs, attached to their COMMITMENTS (roots): a job is only mineable if the roots match
        self.data = {}
        for d in V["jobs_data"]:
            a = np.frombuffer(base64.b64decode(d["a"]), dtype=np.int8).astype(np.int64).reshape(n, n)
            b = np.frombuffer(base64.b64decode(d["b"]), dtype=np.int8).astype(np.int64).reshape(n, n)
            ar = S.merkle_root([S.leaf_hash(i, S.i8bytes(a[i])) for i in range(n)])
            br = S.merkle_root([S.leaf_hash(j, S.i8bytes(b[:, j])) for j in range(n)])
            self.data[(ar, br)] = (a, b)

    # ---------- state utilities ----------
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

    def held(self):
        h = 0
        for j in self.jobs.values():
            h += j["fetch_held"]
            if j["status"] == "Open":
                h += j["escrow_net"]
            elif j["status"] in ("Claimed", "Posted"):
                h += j["escrow_net"] + j["bond"]
        return h

    # ---------- transactions ----------
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
        k, me, c = t["kind"], t["from"], self.c
        typ = k["type"]
        if typ == "Transfer":
            self.debit(me, k["amount"])
            self.credit(k["to"], k["amount"])
        elif typ in ("SubmitJob", "SubmitJobAuction"):
            secs = k["auction_secs"] if typ == "SubmitJobAuction" else 0
            if typ == "SubmitJobAuction" and secs == 0:
                raise Invalid("auction_secs is zero")
            if k["escrow"] < c["min_escrow"]:
                raise Invalid("escrow below the floor")
            if k["deadline"] <= height:
                raise Invalid("deadline passed")
            if secs > c["max_auction_secs"]:
                raise Invalid("auction too long")
            jid = job_id_of(me, t["nonce"], bytes.fromhex(k["a_root"]), bytes.fromhex(k["b_root"])).hex()
            if jid in self.jobs:
                raise Invalid("duplicate job")
            self.debit(me, k["escrow"])
            burn = bps(k["escrow"], c["burn_bps"])
            self.burned += burn
            self.jobs[jid] = {"client": me, "a_root": k["a_root"], "b_root": k["b_root"], "escrow_net": k["escrow"] - burn, "deadline": k["deadline"],
                              "status": "Open", "rebate_left": burn, "end": ts + secs if secs > 0 else 0, "best": None, "fetches": [], "fetch_held": 0}
        elif typ == "Bid":
            j = self.jobs.get(k["job_id"])
            if j is None:
                raise Invalid("unknown job")
            if j["status"] != "Open" or j["end"] == 0 or ts >= j["end"]:
                raise Invalid("no open auction")
            if k["price"] == 0 or k["price"] > j["escrow_net"]:
                raise Invalid("invalid price")
            if j["best"] and k["price"] >= j["best"][1]:
                raise Invalid("bid not lower than the best (bid)")
            j["best"] = (me, k["price"])
        elif typ == "ClaimJob":
            j = self.jobs.get(k["job_id"])
            if j is None:
                raise Invalid("unknown job")
            if j["status"] != "Open" or height > j["deadline"]:
                raise Invalid("job not claimable")
            price = j["escrow_net"]
            if j["end"] > 0:
                if ts < j["end"]:
                    raise Invalid("auction still open (auction)")
                if j["best"] and j["best"][0] == me:
                    price = j["best"][1]
                elif ts >= j["end"] + c["auction_grace_secs"] or j["best"] is None:
                    pass
                else:
                    raise Invalid("only the winner can claim (winning) during the grace period")
            bond = bps(price, c["bond_bps"])
            self.debit(me, bond)
            if price < j["escrow_net"]:
                self.credit(j["client"], j["escrow_net"] - price)
            j["escrow_net"] = price
            j["status"], j["worker"], j["bond"] = "Claimed", me, bond
        elif typ == "PostResult":
            j = self.jobs.get(k["job_id"])
            if j is None:
                raise Invalid("unknown job")
            if not (j["status"] == "Claimed" and j["worker"] == me and height <= j["deadline"]):
                raise Invalid("cannot publish (post)")
            j["status"], j["c_root"], j["at"], j["at_ts"] = "Posted", k["c_root"], height, ts
        elif typ == "Accept":
            j = self.jobs.get(k["job_id"])
            if j is None:
                raise Invalid("unknown job")
            if not (j["status"] == "Posted" and j["client"] == me):
                raise Invalid("cannot accept (accept)")
            self.credit(j["worker"], j["escrow_net"] + j["bond"])
            self.repd(j["worker"])["done"] += 1
            held, j["fetch_held"], j["fetches"], j["status"] = j["fetch_held"], 0, [], "Settled"
            self.credit(j["client"], held)
        elif typ == "Fraud":
            j = self.jobs.get(k["job_id"])
            if j is None:
                raise Invalid("unknown job")
            if j["status"] != "Posted":
                raise Invalid("no result published")
            if not self.challenge_open(height, ts, j["at"], j["at_ts"]):
                raise Invalid("dispute window closed")
            self.verify_fraud(j, k)
            self.sanction(j, me)
        elif typ == "Fetch":
            j = self.jobs.get(k["job_id"])
            n = self.p["n"]
            if k["i"] >= n or k["j"] >= n:
                raise Invalid("index out of bounds")
            if j is None or j["status"] != "Posted":
                raise Invalid("no result published")
            if j["client"] != me:
                raise Invalid("only the client can request (client)")
            if not self.challenge_open(height, ts, j["at"], j["at_ts"]):
                raise Invalid("window closed")
            if len(j["fetches"]) >= c["max_pending_fetches"] or any((f[0], f[1]) == (k["i"], k["j"]) for f in j["fetches"]):
                raise Invalid("duplicate request or too many")
            self.debit(me, c["fetch_fee"])
            j["fetch_held"] += c["fetch_fee"]
            j["fetches"].append((k["i"], k["j"], ts + self.timing["serve_secs"]))
        elif typ == "Serve":
            j = self.jobs.get(k["job_id"])
            if j is None or j["status"] != "Posted":
                raise Invalid("no result published")
            if j["worker"] != me:
                raise Invalid("only the worker can serve")
            pos = next((x for x, f in enumerate(j["fetches"]) if (f[0], f[1]) == (k["i"], k["j"])), None)
            if pos is None:
                raise Invalid("no pending request")
            n = self.p["n"]
            # ROW i is served (row-wise commitment)
            if len(k["row"]) != n or not verify_path(bytes.fromhex(j["c_root"]), S.leaf_hash(k["i"], S.i32bytes(k["row"])), k["i"], [bytes.fromhex(x) for x in k["path"]], depth_for(n)):
                raise Invalid("invalid Merkle proof (Merkle)")
            del j["fetches"][pos]
            j["fetch_held"] -= c["fetch_fee"]
            j["at_ts"] = max(max(ts + c["fetch_extend_secs"] - self.timing["challenge_secs"], 0), j["at_ts"])
            j["at"] = max(max(height + 2 - c["challenge_window"], 0), j["at"])
            self.credit(me, c["fetch_fee"])
        elif typ == "SlashUnserved":
            j = self.jobs.get(k["job_id"])
            if j is None or j["status"] != "Posted":
                raise Invalid("no result published")
            if not any(ts > f[2] for f in j["fetches"]):
                raise Invalid("no expired request (no expired)")
            self.sanction(j, me)
        elif typ == "Settle":
            j = self.jobs.get(k["job_id"])
            if j is None:
                raise Invalid("unknown job")
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
                raise Invalid("settlement impossible for now (cannot settle)")

    def sanction(self, j, beneficiary):
        """Bond: half to the reporter, half burned; the client is refunded (escrow + retained request fees)."""
        reward = j["bond"] // 2
        self.credit(beneficiary, reward)
        self.burned += j["bond"] - reward
        self.credit(j["client"], j["escrow_net"] + j["fetch_held"])
        self.repd(j["worker"])["slashed"] += 1
        j["fetch_held"], j["fetches"], j["status"] = 0, [], "Slashed"

    def verify_fraud(self, j, k):
        n = self.p["n"]
        i, jj = k["i"], k["j"]
        if i >= n or jj >= n or len(k["a_row"]) != n or len(k["b_col"]) != n:
            raise Invalid("malformed proof")
        if any(abs(v) > 64 for v in k["a_row"] + k["b_col"]):
            raise Invalid("values out of bounds")
        d = depth_for(n)
        P = lambda name: [bytes.fromhex(x) for x in k[name]]
        if not verify_path(bytes.fromhex(j["a_root"]), S.leaf_hash(i, S.i8bytes(k["a_row"])), i, P("a_path"), d):
            raise Invalid("invalid opening of A")
        if not verify_path(bytes.fromhex(j["b_root"]), S.leaf_hash(jj, S.i8bytes(k["b_col"])), jj, P("b_path"), d):
            raise Invalid("invalid opening of B")
        if len(k["c_row"]) != n or not verify_path(bytes.fromhex(j["c_root"]), S.leaf_hash(i, S.i32bytes(k["c_row"])), i, P("c_path"), d):
            raise Invalid("invalid opening of C")
        dot = sum(a * b for a, b in zip(k["a_row"], k["b_col"]))
        if dot == k["c_row"][jj]:
            raise Invalid("correct entry: not a fraud")

    # ---------- blocks ----------
    def matrices(self, epoch):
        if epoch not in self.native:
            self.native[epoch] = S.native_matrices(epoch, self.p["n"])
        return self.native[epoch]

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
        else:
            j = self.jobs.get(cls["job"])
            if j is None:
                raise Invalid("unknown job")
            if not (j["status"] in ("Claimed", "Posted") and j["worker"] == hd["miner"]):
                raise Invalid("the miner is not the active worker of this job (miner)")
            key = (bytes.fromhex(j["a_root"]), bytes.fromhex(j["b_root"]))
            if key not in self.data:
                raise Invalid("job data missing")
            a, bm = self.data[key]
        if (bytes.fromhex(hd["a_root"]), bytes.fromhex(hd["b_root"])) != self.work_roots(cls, hd["height"]):
            raise Invalid("work roots")
        sg = S.sigma_of(bytes.fromhex(hd["prev"]), bytes.fromhex(hd["miner"]), class_tuple(cls), hd["extra_nonce"], root, hd["timestamp"])
        T = p["n"] // p["tile"]
        if not (hd["ti"] < T and hd["tj"] < T):
            raise Invalid("tile out of bounds")
        jp, _, _ = S.work(a, bm, sg, p["n"], p["tile"], p["rank"], p["fold_k"])
        if jp[hd["ti"] * T + hd["tj"]] != jackpot:
            raise Invalid("jackpot different from the recomputation")
        # transactions on a copy of the state (atomic)
        snap = copy.deepcopy((self.bal, self.nonce, self.rep, self.jobs, self.burned))
        try:
            for t in b["txs"]:
                self.apply_tx(t, hd["height"], hd["timestamp"], hd["miner"])
            reward = C.base_reward(c, hd["height"])
            bonus = 0
            if cls["type"] == "Job" and cls["job"] in self.jobs:
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
        self.tip_ts, self.tip_hash = hd["timestamp"], header_hash(hd)

    def work_roots(self, cls, height):
        """Roots (A, B) of the work of a block (v2 header): native matrices of the epoch, or roots committed by the job."""
        if cls["type"] == "Native":
            ep = height // self.c["epoch_len"]
            if ep not in self.native_roots:
                a, bm = self.matrices(ep)
                self.native_roots[ep] = (S.rows_root(a), S.cols_root(bm))
            return self.native_roots[ep]
        j = self.jobs[cls["job"]]
        return bytes.fromhex(j["a_root"]), bytes.fromhex(j["b_root"])

    def post_root(self, miner, cls, txs, height, ts):
        """Account root after applying the block (for the reference miner); null root if the transactions are invalid."""
        snap = copy.deepcopy((self.bal, self.nonce, self.rep, self.jobs, self.burned))
        try:
            for t in txs:
                self.apply_tx(t, height, ts, miner)
            reward = C.base_reward(self.c, height)
            bonus = 0
            if cls["type"] == "Job" and cls["job"] in self.jobs:
                bonus = min(reward // 2, self.jobs[cls["job"]]["rebate_left"])
            self.credit(miner, reward + bonus)
            return S.accounts_root(self.bal, self.nonce)
        except Invalid:
            return b"\x00" * 32
        finally:
            self.bal, self.nonce, self.rep, self.jobs, self.burned = snap

    def invariant(self):
        return sum(self.bal.values()) + self.held() + self.burned == self.genesis_supply + self.minted


# ---------------- Python mining of blocks (for the negative tests) ----------------
def mine(m, miner, cls, txs, ts):
    p, c = m.p, m.c
    height = len(m.headers)
    if cls[0] == "Native":
        a, bm = m.matrices(height // c["epoch_len"])
    else:
        j = m.jobs[cls[1].hex()]
        a, bm = m.data[(bytes.fromhex(j["a_root"]), bytes.fromhex(j["b_root"]))]
    root = C.merkle_tx_root([tx_id(t) for t in txs])
    T = p["n"] // p["tile"]
    for en in range(1 << 20):
        sg = S.sigma_of(m.tip_hash, bytes.fromhex(miner), cls, en, root, ts)
        jp, _, _ = S.work(a, bm, sg, p["n"], p["tile"], p["rank"], p["fold_k"])
        for idx, jk in enumerate(jp):
            if S.meets_target(jk, m.target):
                cj = {"type": "Native"} if cls[0] == "Native" else {"type": "Job", "job": cls[1].hex()}
                ra, rb = m.work_roots(cj, height)
                hd = {"version": 2, "height": height, "prev": m.tip_hash.hex(), "timestamp": ts, "miner": miner, "class": cj, "extra_nonce": en,
                      "ti": idx // T, "tj": idx % T, "jackpot": jk.hex(), "tx_root": root.hex(), "target": m.target,
                      "a_root": ra.hex(), "b_root": rb.hex(), "accounts_root": m.post_root(miner, cj, txs, height, ts).hex()}
                return {"header": hd, "hash": header_hash(hd).hex(), "txs": txs}
    raise RuntimeError("no solution")


def signed(m, seed, kind, nonce=None, fee=0):
    who = C.pub_hex(seed)
    t = {"from": who, "nonce": m.nonce.get(who, 0) if nonce is None else nonce, "fee": fee, "kind": kind}
    t["sig"] = C.priv(seed).sign(tx_bytes(t)).hex()
    t["id"] = tx_id(t).hex()
    return t


def main(path):
    V = json.load(open(path))
    bad = 0

    def chk(name, got, want):
        nonlocal bad
        ok = got == want
        bad += not ok
        print(("ok   " if ok else "FAIL"), name, "" if ok else f"\n   got {got}\n   expected {want}")

    m = Market(V)
    snaps = []
    kinds = {}
    for b in V["blocks"]:
        for t in b["txs"]:
            chk_enc = tx_bytes(t).hex() == t["tx_bincode"] and tx_id(t).hex() == t["id"]
            if not chk_enc:
                bad += 1
                print("FAIL transaction encoding", t["kind"]["type"])
            kinds[t["kind"]["type"]] = kinds.get(t["kind"]["type"], 0) + 1
        m.apply(b)
        assert header_hash(b["header"]).hex() == b["hash"], "header hash"
        snaps.append(copy.deepcopy(m))
    chk("%d valid blocks; transaction types covered: %s" % (len(V["blocks"]), dict(sorted(kinds.items()))), len(m.headers), V["final"]["height"])
    F = V["final"]
    chk("emission", m.minted, F["minted"])
    chk("burned", m.burned, F["burned"])
    for a in F["accounts"]:
        chk("account " + a["key"][:8], (m.bal.get(a["key"], 0), m.nonce.get(a["key"], 0)), (a["balance"], a["nonce"]))
    for jb in F["jobs"]:
        j = m.jobs[jb["id"]]
        chk("job " + jb["id"][:8] + " " + jb["status"], (j["status"], j["rebate_left"], j["escrow_net"]), (jb["status"], jb["rebate_left"], jb["escrow_net"]))
    for r in F["reputation"]:
        chk("reputation " + r["key"][:8], m.rep.get(r["key"], {"done": 0, "slashed": 0, "expired": 0}), {"done": r["done"], "slashed": r["slashed"], "expired": r["expired"]})
    chk("invariant: balances + escrows + bonds + burned = genesis + emission", m.invariant(), True)

    # ---------- negative tests: each alteration is refused, for the right reason ----------
    def at(pred):
        for k, b in enumerate(V["blocks"]):
            if pred(b):
                return k
        raise KeyError

    def has(b, typ):
        return any(t["kind"]["type"] == typ for t in b["txs"])

    def state_after(k):
        return copy.deepcopy(snaps[k])

    def attempt(name, st, seed_txs, why, miner=None, cls=("Native",), dt=20):
        nonlocal bad
        miner = miner or C.pub_hex(4)
        blk = mine(st, miner, cls, seed_txs, st.tip_ts + dt)
        try:
            copy.deepcopy(st).apply(blk)
            got = "ACCEPTED"
        except Invalid as e:
            got = str(e)
        ok = why in got
        bad += not ok
        print(("ok   " if ok else "FAIL"), "refusal:", name, "->", got)

    def K(typ, **kw):
        return dict(type=typ, **kw)

    ka = at(lambda b: has(b, "SubmitJobAuction"))
    ida = next(t for t in V["blocks"][ka]["txs"] if t["kind"]["type"] == "SubmitJobAuction")
    ja = next(iter(state_after(ka).jobs))
    st = state_after(ka + 1)  # after the 1st bid (35)
    attempt("bid not lower than the best", st, [signed(st, 3, K("Bid", job_id=ja, price=40 * 10**8))], "bid")
    st = state_after(ka + 2)  # after the bid of 20
    attempt("bid equal to the best", st, [signed(st, 3, K("Bid", job_id=ja, price=20 * 10**8))], "bid", dt=5)
    attempt("claim during the auction", st, [signed(st, 2, K("ClaimJob", job_id=ja))], "auction", dt=5)
    kc = at(lambda b: has(b, "ClaimJob") and b["txs"][0]["kind"]["job_id"] == ja)
    st = state_after(kc - 1)
    attempt("claim by a non-winner during the grace period", st, [signed(st, 3, K("ClaimJob", job_id=ja))], "winning")
    kp = at(lambda b: has(b, "PostResult") and b["txs"][0]["kind"]["job_id"] == ja)
    st = state_after(kp)
    attempt("acceptance by a third party", st, [signed(st, 3, K("Accept", job_id=ja))], "accept")
    attempt("fetch by a third party", st, [signed(st, 3, K("Fetch", job_id=ja, i=0, j=0))], "client")
    attempt("settlement before the end of the window", st, [signed(st, 4, K("Settle", job_id=ja))], "cannot settle")
    attempt("fraud on a correct entry", st, [signed(st, 1, K("Fraud", **fraud_for(st, ja, 3, 5, correct=True)))], "not a fraud")
    attempt("useful block mined by a non-worker", st, [], "miner", miner=C.pub_hex(3), cls=("Job", bytes.fromhex(ja)))
    attempt("SlashUnserved without an expired request", st, [signed(st, 4, K("SlashUnserved", job_id=ja))], "no expired")
    kf = at(lambda b: has(b, "Fetch") and b["txs"][0]["kind"]["job_id"] == ja)
    st = state_after(kf)
    attempt("service of a wrong value", st, [signed(st, 2, K("Serve", job_id=ja, i=3, j=5, row=[v + (123456 if x == 5 else 0) for x, v in enumerate(V["blocks"][kf + 1]["txs"][0]["kind"]["row"])], path=V["blocks"][kf + 1]["txs"][0]["kind"]["path"]))], "Merkle")
    attempt("duplicate Fetch", st, [signed(st, 1, K("Fetch", job_id=ja, i=3, j=5))], "duplicate")
    k0 = at(lambda b: has(b, "SubmitJob"))
    st = state_after(k0 - 1)
    sj = next(t for t in V["blocks"][k0]["txs"] if t["kind"]["type"] == "SubmitJob")
    attempt("escrow below the floor", st, [signed(st, 1, K("SubmitJob", a_root=sj["kind"]["a_root"], b_root=sj["kind"]["b_root"], escrow=10**7, deadline=10_000))], "floor")
    attempt("deadline in the past", st, [signed(st, 1, K("SubmitJob", a_root=sj["kind"]["a_root"], b_root=sj["kind"]["b_root"], escrow=10**9, deadline=1))], "deadline")
    attempt("insufficient balance for the escrow", st, [signed(st, 1, K("SubmitJob", a_root=sj["kind"]["a_root"], b_root=sj["kind"]["b_root"], escrow=10**15, deadline=10_000))], "balance")
    ke = at(lambda b: has(b, "Settle") and b["txs"][0]["from"] == C.pub_hex(4) and m.jobs[b["txs"][0]["kind"]["job_id"]]["status"] == "Settled")
    st = state_after(ke - 1)
    attempt("settlement once the window is closed: accepted", st, [signed(st, 4, K("Settle", job_id=V["blocks"][ke]["txs"][0]["kind"]["job_id"]))], "ACCEPTED")
    kfr = at(lambda b: has(b, "Fraud"))
    st = state_after(kfr - 1)
    fr = V["blocks"][kfr]["txs"][0]
    attempt("valid fraud: accepted", st, [signed(st, 1, fr["kind"])], "ACCEPTED")
    bad_fr = copy.deepcopy(fr["kind"])
    bad_fr["c_row"][bad_fr["j"]] += 1
    attempt("fraud with a value of C that is not opened", st, [signed(st, 1, bad_fr)], "opening of C")
    bad_fr = copy.deepcopy(fr["kind"])
    bad_fr["a_row"][0] = (bad_fr["a_row"][0] + 1) if bad_fr["a_row"][0] < 64 else 0
    attempt("fraud with a falsified row of A", st, [signed(st, 1, bad_fr)], "opening of A")
    build_extension(V, path)
    print("\nRESULT:", "MARKET VALIDATED BY THE INDEPENDENT IMPLEMENTATION" if bad == 0 else f"{bad} FAILURE(S)")
    return bad == 0


def build_extension(V, path):
    """The Python validator MINES the rest of the chain itself (2 new jobs: an honest paid one, a false one proven and
    slashed; matrices, signatures, useful-work blocks): the produced file is replayed by the RUST validator."""
    m = Market(V)
    for b in V["blocks"]:
        m.apply(b)
    n = m.p["n"]
    ext, jobs_data = list(V["blocks"]), list(V["jobs_data"])
    rng = np.random.default_rng(2024)
    ts = m.tip_ts

    def nxt(miner, cls, txs, dt=20):
        nonlocal ts
        ts += dt
        blk = mine(m, miner, cls, txs, ts)
        m.apply(blk)
        ext.append(blk)

    def newjob():
        a = rng.integers(-64, 65, (n, n)).astype(np.int64)
        b = rng.integers(-64, 65, (n, n)).astype(np.int64)
        ar = S.merkle_root([S.leaf_hash(i, S.i8bytes(a[i])) for i in range(n)])
        br = S.merkle_root([S.leaf_hash(j, S.i8bytes(b[:, j])) for j in range(n)])
        m.data[(ar, br)] = (a, b)
        enc = lambda x: base64.b64encode(S.i8bytes(x)).decode()
        jobs_data.append({"a": enc(a), "b": enc(b)})
        return a, b, ar, br

    def croot(c):
        return S.c_root(c)

    client, w1, w2 = C.pub_hex(1), C.pub_hex(2), C.pub_hex(3)
    native = ("Native",)
    # F: honest
    a, b, ar, br = newjob()
    jid = job_id_of(client, m.nonce.get(client, 0), ar, br).hex()
    nxt(C.pub_hex(4), native, [signed(m, 1, dict(type="SubmitJob", a_root=ar.hex(), b_root=br.hex(), escrow=25 * 10**8, deadline=len(m.headers) + 500))])
    nxt(C.pub_hex(4), native, [signed(m, 2, dict(type="ClaimJob", job_id=jid))])
    nxt(w1, ("Job", bytes.fromhex(jid)), [])
    nxt(C.pub_hex(4), native, [signed(m, 2, dict(type="PostResult", job_id=jid, c_root=croot(a @ b).hex()))])
    nxt(C.pub_hex(4), native, [signed(m, 1, dict(type="Accept", job_id=jid))])
    # G: wrong result -> fraud proof built by the Python validator
    a, b, ar, br = newjob()
    jid2 = job_id_of(client, m.nonce.get(client, 0), ar, br).hex()
    nxt(C.pub_hex(4), native, [signed(m, 1, dict(type="SubmitJob", a_root=ar.hex(), b_root=br.hex(), escrow=40 * 10**8, deadline=len(m.headers) + 500))])
    nxt(C.pub_hex(4), native, [signed(m, 3, dict(type="ClaimJob", job_id=jid2))])
    nxt(w2, ("Job", bytes.fromhex(jid2)), [])
    cbad = a @ b
    cbad[2, 9] += 1
    nxt(C.pub_hex(4), native, [signed(m, 3, dict(type="PostResult", job_id=jid2, c_root=croot(cbad).hex()))])
    fp = fraud_for(m, jid2, 2, 9, correct=False, c_override=cbad)
    nxt(C.pub_hex(4), native, [signed(m, 1, dict(type="Fraud", **fp))])
    assert m.jobs[jid]["status"] == "Settled" and m.jobs[jid2]["status"] == "Slashed" and m.invariant()
    out = path.replace(".json", "") + ".ext.json"
    json.dump({"genesis": V["genesis"], "jobs_data": jobs_data, "blocks": ext, "final": V["final"]}, open(out, "w"))
    print("extension mined in Python (%d blocks, 2 jobs) written to %s -> Rust validator: vearl-market-vectors verify %s" % (len(ext) - len(V["blocks"]), out, out))


def fraud_for(st, jid, i, j, correct, c_override=None):
    """Fraud proof (with valid openings) on the entry (i, j); correct=True: the opened value is the true value."""
    n = st.p["n"]
    jb = st.jobs[jid]
    a, b = st.data[(bytes.fromhex(jb["a_root"]), bytes.fromhex(jb["b_root"]))]
    # trees built by the verifier
    def proof(leaves, idx):
        return [x.hex() for x in S.merkle_proof(leaves, idx)]
    ar = [S.leaf_hash(x, S.i8bytes(a[x])) for x in range(n)]
    br = [S.leaf_hash(x, S.i8bytes(b[:, x])) for x in range(n)]
    c = (a @ b) if c_override is None else c_override
    cl = S.row_leaves(c)
    return dict(job_id=jid, i=i, j=j, a_row=[int(v) for v in a[i]], a_path=proof(ar, i), b_col=[int(v) for v in b[:, j]], b_path=proof(br, j),
                c_row=[int(v) for v in c[i]], c_path=proof(cl, i))


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
