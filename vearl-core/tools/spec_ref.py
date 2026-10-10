#!/usr/bin/env python3
"""INDEPENDENT implementation of the VEARL work layer, written from docs/07-work-specification.md
(without reading the Rust code) and verified against the vectors produced by `vearl-vectors`.

    ../target/release/vearl-vectors > vectors.json
    python3 spec_ref.py vectors.json          # requires: pip install blake3 numpy
"""
import json, struct, sys
import numpy as np
import blake3

JACKPOT_DOMAIN = b"vearl/jackpot/v2"


# ---------------- primitives (spec section 1) ----------------
def H(domain, *parts):
    h = blake3.blake3(derive_key_context=domain)
    for p in parts:
        h.update(p)
    return h.digest()


def xof(domain, data, length):
    h = blake3.blake3(derive_key_context=domain)
    h.update(data)
    return h.digest(length=length)


# ---------------- Merkle (section 3) ----------------
def leaf_hash(index, data):
    return H("vearl/merkle/leaf/v1", struct.pack("<Q", index), data)


def merkle_levels(leaves):
    size = 1
    while size < max(len(leaves), 1):
        size *= 2
    lv = [list(leaves) + [b"\x00" * 32] * (size - len(leaves))]
    while len(lv[-1]) > 1:
        cur = lv[-1]
        lv.append([H("vearl/merkle/node/v1", cur[i], cur[i + 1]) for i in range(0, len(cur), 2)])
    return lv


def merkle_root(leaves):
    return merkle_levels(leaves)[-1][0]


def merkle_proof(leaves, idx):
    lv = merkle_levels(leaves)
    out = []
    for level in lv[:-1]:
        out.append(level[idx ^ 1])
        idx >>= 1
    return out


def i8bytes(a):
    return np.asarray(a, dtype=np.int8).tobytes()


def i32bytes(row):
    """Bytes of a sequence of i32 values (little-endian)."""
    return np.asarray(row, dtype="<i4").tobytes()


def row_leaves(c):
    """v2 commitment of an INTEGER matrix (C, Z, products, sums): leaf i = leaf_hash(i, row i as i32 LE)."""
    c = np.asarray(c)
    assert c.ndim == 2, "n x n matrix expected"
    return [leaf_hash(i, i32bytes(c[i])) for i in range(c.shape[0])]


def c_root(c):
    return merkle_root(row_leaves(c))


def c_levels(c):
    return merkle_levels(row_leaves(c))


# ---------------- native matrices (section 4) ----------------
def native_matrices(epoch, n):
    buf = xof("vearl/native/v1", struct.pack("<Q", epoch), 2 * n * n)
    v = np.frombuffer(buf, dtype=np.uint8).astype(np.int64) % 129 - 64
    return v[: n * n].reshape(n, n), v[n * n:].reshape(n, n)


# ---------------- noise (section 5) ----------------
def noise(sigma, n, r):
    buf = xof("vearl/noise/v1", sigma, 4 * n * r)
    v = np.frombuffer(buf, dtype=np.uint8).astype(np.int64) % 3 - 1
    el = v[: n * r].reshape(n, r)
    er = v[n * r: 2 * n * r].reshape(r, n)
    fl = v[2 * n * r: 3 * n * r].reshape(n, r)
    fr = v[3 * n * r:].reshape(r, n)
    return el, er, fl, fr


# ---------------- work (section 6) ----------------
def rotl32(x, k):
    x = x.astype(np.uint64)
    return (((x << np.uint64(k)) | (x >> np.uint64(32 - k))) & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def work(a, b, sigma, n, tile, r, fold_k):
    el, er, fl, fr = noise(sigma, n, r)
    ap = a + el @ er
    bp = b + fl @ fr
    T = n // tile
    acc = np.zeros((n, n), dtype=np.int64)
    fold = np.zeros((T, T, 64), dtype=np.uint32)
    for s in range(n // fold_k):
        k0 = s * fold_k
        acc = (acc + ap[:, k0:k0 + fold_k] @ bp[k0:k0 + fold_k, :]).astype(np.int32).astype(np.int64)  # i32 arithmetic (wrapping)
        rot = s % 31 + 1
        rotated = rotl32((acc & 0xFFFFFFFF).astype(np.uint32), rot)
        for ti in range(T):
            for tj in range(T):
                blk = rotated[ti * tile:(ti + 1) * tile, tj * tile:(tj + 1) * tile].reshape(-1)  # i = x*tile + y
                fold[ti, tj] ^= np.bitwise_xor.reduce(blk.reshape(-1, 64), axis=0)  # word i % 64
    jackpots = []
    for ti in range(T):
        for tj in range(T):
            m = JACKPOT_DOMAIN + sigma + struct.pack("<II", ti, tj) + fold[ti, tj].astype("<u4").tobytes()
            jackpots.append(blake3.blake3(m).digest())
    return jackpots, acc, (el, er, fl, fr)


def tile_jackpot(a_rows, b_cols, ti, tj, sigma, n, tile, r, fold_k):
    """Jackpot of the SINGLE tile (ti, tj), from its `tile` rows of A and `tile` columns of B (without the rest of the matrices):
    what a light client recomputes. Same arithmetic as `work` restricted to the tile."""
    el, er, fl, fr = noise(sigma, n, r)
    rows = [ti * tile + x for x in range(tile)]
    cols = [tj * tile + x for x in range(tile)]
    ap = np.asarray(a_rows, dtype=np.int64) + el[rows] @ er             # (tile, n): rows of A + EL.ER
    bp = np.asarray(b_cols, dtype=np.int64).T + fl @ fr[:, cols]        # (n, tile): columns of B + FL.FR
    acc = np.zeros((tile, tile), dtype=np.int64)
    fold = np.zeros(64, dtype=np.uint32)
    for s in range(n // fold_k):
        k0 = s * fold_k
        acc = (acc + ap[:, k0:k0 + fold_k] @ bp[k0:k0 + fold_k, :]).astype(np.int32).astype(np.int64)
        rotated = rotl32((acc & 0xFFFFFFFF).astype(np.uint32), s % 31 + 1)
        fold ^= np.bitwise_xor.reduce(rotated.reshape(-1, 64), axis=0)
    m = JACKPOT_DOMAIN + sigma + struct.pack("<II", ti, tj) + fold.astype("<u4").tobytes()
    return blake3.blake3(m).digest()


def decode(cn, a, b, nz):
    el, er, fl, fr = nz
    c = cn - a @ fl @ fr - el @ er @ b - el @ er @ fl @ fr
    return ((c + 2**31) % 2**32 - 2**31)


# ---------------- sigma and header (section 2, 7) ----------------
def class_bytes(c):
    kind = c[0]
    if kind == "Native":
        return struct.pack("<I", 0)
    if kind == "Job":
        return struct.pack("<I", 1) + c[1]
    if kind == "Layer":
        return struct.pack("<I", 2) + c[1] + bytes([c[2]])
    if kind == "Product":
        return struct.pack("<I", 3) + c[1] + bytes(c[2:5])
    if kind == "Op":
        return struct.pack("<I", 4) + c[1] + bytes([c[2]])
    raise ValueError(kind)


def sigma_of(prev, miner, cls, extra_nonce, tx_root, ts):
    return H("vearl/sigma/v1", prev, miner, class_bytes(cls), struct.pack("<Q", extra_nonce), tx_root, struct.pack("<Q", ts))


def header_bytes(version, height, prev, ts, miner, cls, extra_nonce, ti, tj, jackpot, tx_root, target, a_root, b_root, accounts_root):
    return (struct.pack("<IQ", version, height) + prev + struct.pack("<Q", ts) + miner + class_bytes(cls) +
            struct.pack("<QII", extra_nonce, ti, tj) + jackpot + tx_root + struct.pack("<Q", target) + a_root + b_root + accounts_root)


def rows_root(a):
    """Merkle root of the ROWS of A (leaf i = i-th row as int8)."""
    return merkle_root([leaf_hash(i, i8bytes(a[i])) for i in range(len(a))])


def cols_root(b):
    """Merkle root of the COLUMNS of B."""
    b = np.asarray(b)
    return merkle_root([leaf_hash(j, i8bytes(b[:, j])) for j in range(b.shape[1])])


def accounts_root(bal, nonce):
    """Account root (v2 header): leaf 0 = number of live accounts, leaf i+1 = (key, balance, nonce) in key order.
    bal / nonce: dictionaries hexadecimal key -> integer. An account is live if its balance or nonce is non-zero."""
    live = sorted(k for k in set(bal) | set(nonce) if bal.get(k, 0) or nonce.get(k, 0))
    leaves = [leaf_hash(0, b"count" + struct.pack("<Q", len(live)))]
    for i, k in enumerate(live):
        leaves.append(leaf_hash(i + 1, bytes.fromhex(k) + struct.pack("<QQ", bal.get(k, 0), nonce.get(k, 0))))
    return merkle_root(leaves)


def meets_target(jackpot, target):
    return int.from_bytes(jackpot[:8], "big") < target


# ---------------- verification of the vectors ----------------
def main(path):
    V = json.load(open(path))
    bad = 0

    def chk(name, got, want):
        nonlocal bad
        ok = got == want
        bad += not ok
        print(("ok   " if ok else "FAIL"), name)
        if not ok:
            print("   got:", got, "\n   expected:", want)

    m = V["merkle"]
    leaves = [bytes.fromhex(x) for x in m["leaves"]]
    chk("merkle: root of 5 leaves", merkle_root(leaves).hex(), m["root5"])
    chk("merkle: proof of leaf 3", [p.hex() for p in merkle_proof(leaves, 3)], m["proof_idx3"])
    chk("merkle: leaf i = leaf_hash(i, [i]*7)", [leaf_hash(i, bytes([i]) * 7).hex() for i in range(5)], m["leaves"])
    M = np.array(m["matrix8"], dtype=np.int64).reshape(8, 8)
    chk("merkle: row-wise root", merkle_root([leaf_hash(i, i8bytes(M[i])) for i in range(8)]).hex(), m["rows_root"])
    chk("merkle: column-wise root", merkle_root([leaf_hash(j, i8bytes(M[:, j])) for j in range(8)]).hex(), m["cols_root"])

    for w in V["work"]:
        p = w["params"]
        n, t, r, fk = p["n"], p["tile"], p["rank"], p["fold_k"]
        tag = f"work n={n} rank={r} fold_k={fk}"
        a, b = native_matrices(w["epoch"], n)
        chk(tag + ": native A", blake3.blake3(i8bytes(a)).hexdigest(), w["a_blake3"])
        chk(tag + ": native B", blake3.blake3(i8bytes(b)).hexdigest(), w["b_blake3"])
        sg = bytes.fromhex(w["sigma"])
        nz = noise(sg, n, r)
        chk(tag + ": noise", {k: blake3.blake3(i8bytes(x)).hexdigest() for k, x in zip(["el", "er", "fl", "fr"], nz)}, w["noise_blake3"])
        jp, cn, _ = work(a, b, sg, n, t, r, fk)
        chk(tag + ": C' (i32)", blake3.blake3(cn.astype("<i4").tobytes()).hexdigest(), w["c_noised_blake3"])
        chk(tag + ": jackpots", [j.hex() for j in jp], w["jackpots"])
        T_ = n // t
        for (ti, tj) in ((0, 0), (T_ - 1, T_ - 1), (T_ // 2, 0)):          # light client: one tile from its rows/columns only
            tjp = tile_jackpot(a[ti * t:(ti + 1) * t], b[:, tj * t:(tj + 1) * t].T, ti, tj, sg, n, t, r, fk)
            chk(tag + f": tile ({ti},{tj}) alone = complete jackpot", tjp.hex(), w["jackpots"][ti * T_ + tj])
        c = decode(cn, a, b, nz)
        chk(tag + ": C decode = A*B", (c == a @ b).all(), True)
        chk(tag + ": hash of C", blake3.blake3(c.astype("<i4").tobytes()).hexdigest(), w["c_blake3"])

    prev, miner, txr, jid = b"\xaa" * 32, b"\xbb" * 32, b"\xcc" * 32, b"\xdd" * 32
    classes = [("Native",), ("Job", jid), ("Layer", jid, 3), ("Product", jid, 1, 2, 3), ("Op", jid, 5)]
    for i, (c, s) in enumerate(zip(classes, V["sigma"])):
        chk(f"class {c[0]}: encoding", class_bytes(c).hex(), s["class_bincode"])
        chk(f"class {c[0]}: sigma", sigma_of(prev, miner, c, s["extra_nonce"], txr, s["timestamp"]).hex(), s["sigma"])
    hb = header_bytes(2, 42, prev, 1_700_000_042, miner, ("Job", jid), 77, 3, 5, b"\xee" * 32, txr, 0x123456789abcdef0, b"\x11" * 32, b"\x22" * 32, b"\x33" * 32)
    chk("header: encoding", hb.hex(), V["header"]["bincode"])
    chk("header: hash", H("vearl/header/v1", hb).hex(), V["header"]["hash"])
    for tv in V["target"]:
        j = bytes.fromhex(tv["jackpot"]).ljust(32, b"\x00")
        j = j[:7] + bytes([j[7]]) + j[8:]
        chk("target", meets_target(j, tv["target"]), tv["meets"])
    print("\nRESULT:", "ALL THE VECTORS MATCH" if bad == 0 else f"{bad} FAILURE(S)")
    return bad == 0


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
