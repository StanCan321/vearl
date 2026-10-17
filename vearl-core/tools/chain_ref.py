#!/usr/bin/env python3
"""INDEPENDENT chain validator (Native Work + transfers), written from docs/07-work-specification.md section 10.
It re-validates each block produced by the reference node (recomputation of the winning tile, LWMA-1 retargeting,
signed transactions, rewards) then compares the final state. It also refuses altered blocks (negative tests).

    ../target/release/vearl-chain-vectors > chain.json
    python3 chain_ref.py chain.json          # requires: pip install blake3 numpy cryptography
"""
import copy, json, struct, sys
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
import spec_ref as S


class Invalid(Exception):
    pass


def base_reward(c, height):
    return max(c["r0"] >> min(height // c["halving"], 63), c["tail"])


def block_work(target):
    return (1 << 64) // (target + 1)


def ema_next(c, target, dt):
    e = c["block_time"]
    adj = min(max(dt, e // 2), e * 2)
    t = target * (15 * e + adj) // (16 * e)
    return max(min(t, (1 << 64) - 1), 1 << 16)


def lwma_next(c, headers, fallback):
    N, T = c["lwma_n"], c["block_time"]
    if len(headers) < N + 1:
        return fallback
    w = headers[-N - 1:]
    num = sum_t = 0
    for i in range(1, N + 1):
        st = min(max(w[i]["timestamp"] - w[i - 1]["timestamp"], 1), 6 * T)
        num += i * st
        sum_t += w[i]["target"]
    den = T * (N * (N + 1) // 2)
    nxt = (sum_t // N) * num // den
    prev = w[N]["target"]
    nxt = min(max(nxt, prev // 2), prev * 2)
    return min(max(nxt, 1 << 16), (1 << 64) - 1)


def tx_bytes(t):
    # Tx = from || nonce u64 || fee u64 || TxKind::Transfer (tag u32 = 0) || to || amount u64
    return (bytes.fromhex(t["from"]) + struct.pack("<QQ", t["nonce"], t["fee"]) + struct.pack("<I", 0) +
            bytes.fromhex(t["to"]) + struct.pack("<Q", t["amount"]))


def tx_id(t):
    sig = bytes.fromhex(t["sig"])
    return S.H("vearl/tx/v1", tx_bytes(t) + struct.pack("<Q", len(sig)) + sig)


def header_hash(hd):
    hb = S.header_bytes(hd["version"], hd["height"], bytes.fromhex(hd["prev"]), hd["timestamp"], bytes.fromhex(hd["miner"]),
                        ("Native",), hd["extra_nonce"], hd["ti"], hd["tj"], bytes.fromhex(hd["jackpot"]), bytes.fromhex(hd["tx_root"]), hd["target"],
                        bytes.fromhex(hd["a_root"]), bytes.fromhex(hd["b_root"]), bytes.fromhex(hd["accounts_root"]))
    return S.H("vearl/header/v1", hb)


class Chain:
    def __init__(self, g):
        self.g = g
        self.c = g["constants"]
        self.p = g["params"]
        self.bal = {a["key"]: a["amount"] for a in g["alloc"]}
        self.nonce = {}
        self.minted = 0
        self.headers = []
        self.target = g["target"]
        self.tip_ts = g["timestamp"]
        self.tip_hash = b"\x00" * 32
        self.native = {}

    def matrices(self, epoch):
        if epoch not in self.native:
            self.native[epoch] = S.native_matrices(epoch, self.p["n"])
        return self.native[epoch]

    def roots(self, epoch):
        a, b = self.matrices(epoch)
        return S.rows_root(a), S.cols_root(b)

    def apply(self, b):
        """Validates and applies the block; raises Invalid. The state is only modified on success."""
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
        root = merkle_tx_root(ids)
        if bytes.fromhex(hd["tx_root"]) != root:
            raise Invalid("tx_root")
        # work: recomputation of the announced tile (Native class: matrices of the epoch height // epoch_len)
        a, bm = self.matrices(hd["height"] // c["epoch_len"])
        if (bytes.fromhex(hd["a_root"]), bytes.fromhex(hd["b_root"])) != self.roots(hd["height"] // c["epoch_len"]):
            raise Invalid("work roots")
        sg = S.sigma_of(bytes.fromhex(hd["prev"]), bytes.fromhex(hd["miner"]), ("Native",), hd["extra_nonce"], root, hd["timestamp"])
        T = p["n"] // p["tile"]
        if not (hd["ti"] < T and hd["tj"] < T):
            raise Invalid("tile out of bounds")
        jp, _, _ = S.work(a, bm, sg, p["n"], p["tile"], p["rank"], p["fold_k"])
        if jp[hd["ti"] * T + hd["tj"]] != jackpot:
            raise Invalid("jackpot different from the recomputation")
        # transactions on a copy of the state
        bal, nonce = dict(self.bal), dict(self.nonce)
        miner = hd["miner"]
        for t in b["txs"]:
            try:
                Ed25519PublicKey.from_public_bytes(bytes.fromhex(t["from"])).verify(bytes.fromhex(t["sig"]), tx_bytes(t))
            except (InvalidSignature, ValueError):
                raise Invalid("signature")
            if nonce.get(t["from"], 0) != t["nonce"]:
                raise Invalid("nonce")
            nonce[t["from"]] = t["nonce"] + 1
            if bal.get(t["from"], 0) < t["fee"]:
                raise Invalid("balance (fee)")
            bal[t["from"]] -= t["fee"]
            bal[miner] = bal.get(miner, 0) + t["fee"]
            if bal.get(t["from"], 0) < t["amount"]:
                raise Invalid("balance (amount)")
            bal[t["from"]] -= t["amount"]
            bal[t["to"]] = bal.get(t["to"], 0) + t["amount"]
        reward = base_reward(c, hd["height"])
        bal[miner] = bal.get(miner, 0) + reward
        if bytes.fromhex(hd["accounts_root"]) != S.accounts_root(bal, nonce):
            raise Invalid("accounts_root")
        # success: validation of the state and retargeting
        self.bal, self.nonce = bal, nonce
        self.minted += reward
        dt = hd["timestamp"] - self.tip_ts
        self.headers.append(hd)
        self.tip_ts = hd["timestamp"]
        self.tip_hash = header_hash(hd)
        ema = ema_next(c, self.target, dt)
        self.target = lwma_next(c, self.headers, ema)

    def post_accounts_root(self, miner, txs, height):
        """Account root after applying (assumed valid) the transactions and the reward: used by the reference miner."""
        bal, nonce = dict(self.bal), dict(self.nonce)
        for t in txs:
            nonce[t["from"]] = t["nonce"] + 1
            bal[t["from"]] = bal.get(t["from"], 0) - t["fee"] - t["amount"]
            bal[miner] = bal.get(miner, 0) + t["fee"]
            bal[t["to"]] = bal.get(t["to"], 0) + t["amount"]
        bal[miner] = bal.get(miner, 0) + base_reward(self.c, height)
        return S.accounts_root({k: max(v, 0) for k, v in bal.items()}, nonce)     # (an invalid test block can have a negative balance)

    def total_work(self):
        return sum(block_work(h["target"]) for h in self.headers)


def merkle_tx_root(ids):
    if not ids:
        return b"\x00" * 32
    return S.merkle_root([S.leaf_hash(i, x) for i, x in enumerate(ids)])


def priv(seed):
    return Ed25519PrivateKey.from_private_bytes(bytes([seed]) * 32)


def pub_hex(seed):
    return priv(seed).public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()


def make_tx(seed, nonce, fee, to, amount):
    t = {"from": pub_hex(seed), "nonce": nonce, "fee": fee, "to": to, "amount": amount}
    t["sig"] = priv(seed).sign(tx_bytes(t)).hex()
    t["id"] = tx_id(t).hex()
    return t


def mine_block(ch, miner, txs, ts):
    """Reference miner in Python: first (extra_nonce, tile) whose jackpot is under the target."""
    p, c = ch.p, ch.c
    height = len(ch.headers)
    a, bm = ch.matrices(height // c["epoch_len"])
    root = merkle_tx_root([tx_id(t) for t in txs])
    T = p["n"] // p["tile"]
    for en in range(1 << 20):
        sg = S.sigma_of(ch.tip_hash, bytes.fromhex(miner), ("Native",), en, root, ts)
        jp, _, _ = S.work(a, bm, sg, p["n"], p["tile"], p["rank"], p["fold_k"])
        for idx, j in enumerate(jp):
            if S.meets_target(j, ch.target):
                ra, rb = ch.roots(height // c["epoch_len"])
                hd = {"version": 2, "height": height, "prev": ch.tip_hash.hex(), "timestamp": ts, "miner": miner, "extra_nonce": en,
                      "ti": idx // T, "tj": idx % T, "jackpot": j.hex(), "tx_root": root.hex(), "target": ch.target,
                      "a_root": ra.hex(), "b_root": rb.hex(), "accounts_root": ch.post_accounts_root(miner, txs, height).hex()}
                return {"header": hd, "hash": header_hash(hd).hex(), "txs": txs}
    raise RuntimeError("no solution")


def main(path):
    V = json.load(open(path))
    ch = Chain(V["genesis"])
    for b in V["blocks"]:
        ch.apply(b)
        assert header_hash(b["header"]).hex() == b["hash"], "header hash"
        for t in b["txs"]:
            assert tx_id(t).hex() == t["id"], "transaction identifier"
    f = V["final"]
    bad = 0

    def chk(name, got, want):
        nonlocal bad
        ok = got == want
        bad += not ok
        print(("ok   " if ok else "FAIL"), name, "" if ok else f"\n   got {got}\n   expected {want}")

    chk(f"{len(V['blocks'])} valid blocks, height", len(ch.headers), f["height"])
    chk("target of the next block (LWMA-1)", ch.target, f["next_target"])
    chk("total emission", ch.minted, f["minted"])
    chk("cumulative work", str(ch.total_work()), f["total_work"])
    for a in f["accounts"]:
        chk("account " + a["key"][:8], (ch.bal.get(a["key"], 0), ch.nonce.get(a["key"], 0)), (a["balance"], a["nonce"]))
    supply = sum(x["amount"] for x in V["genesis"]["alloc"]) + ch.minted
    chk("money supply invariant", sum(ch.bal.values()), supply)

    # negative tests: each alteration must be refused, for the right reason
    def state_at(k):
        c2 = Chain(V["genesis"])
        for b in V["blocks"][:k]:
            c2.apply(b)
        return c2

    k = next(i for i, b in enumerate(V["blocks"]) if b["txs"] and i > V["genesis"]["constants"]["lwma_n"] + 10)  # a block with a transaction after the LWMA window
    base = state_at(k)

    def mutate(fn):
        b = copy.deepcopy(V["blocks"][k])
        fn(b)
        return b

    cases = [
        ("wrong target", lambda b: b["header"].__setitem__("target", b["header"]["target"] - 1), "target"),
        ("wrong prev", lambda b: b["header"].__setitem__("prev", "00" * 32), "prev"),
        ("non-increasing timestamp", lambda b: b["header"].__setitem__("timestamp", base.tip_ts), "timestamp"),
        ("altered amount (tx_root unchanged)", lambda b: b["txs"][0].__setitem__("amount", b["txs"][0]["amount"] + 1), "tx_root"),
        ("altered miner (sigma changes)", lambda b: b["header"].__setitem__("miner", "11" * 32), "jackpot"),
        ("altered extra_nonce", lambda b: b["header"].__setitem__("extra_nonce", b["header"]["extra_nonce"] + 1), "jackpot"),
        ("altered tile", lambda b: b["header"].__setitem__("tj", (b["header"]["tj"] + 1) % 8), "jackpot"),
        ("version 1", lambda b: b["header"].__setitem__("version", 1), "version"),
    ]
    for name, fn, why in cases:
        try:
            copy.deepcopy(base).apply(mutate(fn))
            got = "ACCEPTED (!)"
        except Invalid as e:
            got = str(e)
        ok = why in got
        bad += not ok
        print(("ok   " if ok else "FAIL"), "refusal:", name, "->", got)

    # does the Python validator accept a block MINED in Python? and does it refuse, for the right reason, blocks
    # whose work is valid but whose transactions are wrong?
    full = Chain(V["genesis"])
    for b in V["blocks"]:
        full.apply(b)
    s1, s2 = 1, 2
    n1 = full.nonce.get(pub_hex(s1), 0)
    dest = pub_hex(3)
    ts0 = full.tip_ts

    def try_block(name, txs, why):
        nonlocal bad
        blk = mine_block(full, pub_hex(1), txs, ts0 + 31)
        try:
            copy.deepcopy(full).apply(blk)
            got = "ACCEPTED"
        except Invalid as e:
            got = str(e)
        ok = why in got
        bad += not ok
        print(("ok   " if ok else "FAIL"), name, "->", got)

    good = make_tx(s1, n1, 5, dest, 7 * 10**8)
    try_block("block mined in Python, valid transaction", [good], "ACCEPTED")
    forged = dict(good, sig="00" * 64)
    try_block("null signature (valid work)", [forged], "signature")
    try_block("replayed nonce", [make_tx(s1, n1 + 1, 5, dest, 1)], "nonce")
    try_block("fee > balance", [make_tx(s1, n1, 10**15, dest, 1)], "balance (fee)")
    try_block("amount > balance", [make_tx(s1, n1, 0, dest, 10**15)], "balance (amount)")
    stolen = make_tx(s2, full.nonce.get(pub_hex(s2), 0), 0, dest, 1)
    stolen["from"] = pub_hex(s1)  # signature of s2 under the identity of s1
    try_block("signature of another account", [stolen], "signature")

    # extension: 6 blocks mined in Python (including transfers), written for the RUST validator
    ext = copy.deepcopy(V["blocks"])
    ch2 = Chain(V["genesis"])
    for b in V["blocks"]:
        ch2.apply(b)
    ts = ch2.tip_ts
    for i in range(6):
        ts += 20 + 9 * i
        txs = []
        if i % 2 == 0:
            seed = 1 + i % 3
            txs = [make_tx(seed, ch2.nonce.get(pub_hex(seed), 0), 1000 + i, pub_hex(1 + (seed % 3)), (i + 1) * 10**7)]
        blk = mine_block(ch2, pub_hex(1 + i % 3), txs, ts)
        ch2.apply(blk)
        ext.append(blk)
    out = path.replace(".json", "") + ".ext.json"
    json.dump({"genesis": V["genesis"], "blocks": ext}, open(out, "w"))
    print("extension mined in Python written to", out, "-> to be validated by: vearl-chain-vectors verify", out)

    print("\nRESULT:", "CHAIN VALIDATED BY THE INDEPENDENT IMPLEMENTATION" if bad == 0 else f"{bad} FAILURE(S)")
    return bad == 0


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
