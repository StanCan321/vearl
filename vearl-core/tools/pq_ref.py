#!/usr/bin/env python3
"""INDEPENDENT SLH-DSA-SHAKE-128s verifier (FIPS 205, Algorithms 4, 8, 11, 13, 17, 20; SHAKE256), written from the standard.
It verifies the signatures produced by the Rust `fips205` crate (vearl-pq-vectors): if both implementations agree on valid
signatures AND on every alteration, VEARL's derivation, context and domain separation (docs/14) do not depend on a bug in only one of them.

        python3 tools/pq_ref.py --rust ../target/release/vearl-pq-vectors     # or: vearl-pq-vectors > pq.json && python3 tools/pq_ref.py pq.json
"""
import hashlib, json, sys

N, H, D, HP, A, K, LG_W = 16, 63, 7, 9, 12, 14, 4
W = 1 << LG_W
LEN1 = 32
LEN2 = 3
LEN = LEN1 + LEN2
M_BYTES = 30
SIG_LEN = N + K * (A + 1) * N + (H + D * LEN) * N   # 7856
CTX = b"vearl/tx/pq/v1"

WOTS_HASH, WOTS_PK, TREE, FORS_TREE, FORS_ROOTS = 0, 1, 2, 3, 4


def shake(data, outlen):
    return hashlib.shake_256(data).digest(outlen)


class Adrs:
    """32 bytes: layer (4) | tree (12) | type (4) | word1 (4) | word2 (4) | word3 (4), big-endian."""
    def __init__(self):
        self.b = bytearray(32)

    def copy(self):
        a = Adrs()
        a.b = bytearray(self.b)
        return a

    def set_layer(self, v): self.b[0:4] = v.to_bytes(4, "big")
    def set_tree(self, v): self.b[4:16] = v.to_bytes(12, "big")

    def set_type(self, t):
        self.b[16:20] = t.to_bytes(4, "big")
        self.b[20:32] = bytes(12)                       # setType clears the three words

    def set_keypair(self, v): self.b[20:24] = v.to_bytes(4, "big")
    def get_keypair(self): return int.from_bytes(self.b[20:24], "big")
    def set_chain(self, v): self.b[24:28] = v.to_bytes(4, "big")
    def set_hash(self, v): self.b[28:32] = v.to_bytes(4, "big")
    def set_tree_height(self, v): self.b[24:28] = v.to_bytes(4, "big")
    def set_tree_index(self, v): self.b[28:32] = v.to_bytes(4, "big")
    def get_tree_index(self): return int.from_bytes(self.b[28:32], "big")


def F(pk_seed, adrs, m): return shake(pk_seed + bytes(adrs.b) + m, N)
def Hh(pk_seed, adrs, m): return shake(pk_seed + bytes(adrs.b) + m, N)
def T(pk_seed, adrs, m): return shake(pk_seed + bytes(adrs.b) + m, N)


def base_2b(x, b, out_len):
    """FIPS 205 Alg. 4: b bits at a time, most significant first."""
    total, bits, out, i = 0, 0, [], 0
    for _ in range(out_len):
        while bits < b:
            total = (total << 8) + x[i]
            i += 1
            bits += 8
        bits -= b
        out.append((total >> bits) & ((1 << b) - 1))
        total &= (1 << bits) - 1
    return out


def chain(x, i, s, pk_seed, adrs):
    tmp = x
    for j in range(i, i + s):
        adrs.set_hash(j)
        tmp = F(pk_seed, adrs, tmp)
    return tmp


def wots_pk_from_sig(sig, m, pk_seed, adrs):
    msg = base_2b(m, LG_W, LEN1)
    csum = sum(W - 1 - v for v in msg)
    csum <<= (8 - ((LEN2 * LG_W) % 8)) % 8
    msg += base_2b(csum.to_bytes((LEN2 * LG_W + 7) // 8, "big"), LG_W, LEN2)
    tmp = b""
    for i in range(LEN):
        adrs.set_chain(i)
        tmp += chain(sig[i * N:(i + 1) * N], msg[i], W - 1 - msg[i], pk_seed, adrs)
    wp = adrs.copy()
    wp.set_type(WOTS_PK)
    wp.set_keypair(adrs.get_keypair())
    return T(pk_seed, wp, tmp)


def xmss_pk_from_sig(idx, sig_xmss, m, pk_seed, adrs):
    adrs.set_type(WOTS_HASH)
    adrs.set_keypair(idx)
    sig, auth = sig_xmss[:LEN * N], sig_xmss[LEN * N:]
    node = wots_pk_from_sig(sig, m, pk_seed, adrs)
    adrs.set_type(TREE)
    adrs.set_tree_index(idx)
    for k in range(HP):
        adrs.set_tree_height(k + 1)
        a = auth[k * N:(k + 1) * N]
        if (idx >> k) % 2 == 0:
            adrs.set_tree_index(adrs.get_tree_index() // 2)
            node = Hh(pk_seed, adrs, node + a)
        else:
            adrs.set_tree_index((adrs.get_tree_index() - 1) // 2)
            node = Hh(pk_seed, adrs, a + node)
    return node


def ht_verify(m, sig_ht, pk_seed, idx_tree, idx_leaf, pk_root):
    adrs = Adrs()
    adrs.set_tree(idx_tree)
    xl = (HP + LEN) * N
    node = xmss_pk_from_sig(idx_leaf, sig_ht[:xl], m, pk_seed, adrs)
    for j in range(1, D):
        idx_leaf = idx_tree % (1 << HP)
        idx_tree >>= HP
        adrs.set_layer(j)
        adrs.set_tree(idx_tree)
        node = xmss_pk_from_sig(idx_leaf, sig_ht[j * xl:(j + 1) * xl], node, pk_seed, adrs)
    return node == pk_root


def fors_pk_from_sig(sig_fors, md, pk_seed, adrs):
    idxs = base_2b(md, A, K)
    roots = b""
    for i in range(K):
        blk = sig_fors[i * (A + 1) * N:(i + 1) * (A + 1) * N]
        sk, auth = blk[:N], blk[N:]
        adrs.set_tree_height(0)
        adrs.set_tree_index(i * (1 << A) + idxs[i])
        node = F(pk_seed, adrs, sk)
        for j in range(A):
            adrs.set_tree_height(j + 1)
            a = auth[j * N:(j + 1) * N]
            if (idxs[i] >> j) % 2 == 0:
                adrs.set_tree_index(adrs.get_tree_index() // 2)
                node = Hh(pk_seed, adrs, node + a)
            else:
                adrs.set_tree_index((adrs.get_tree_index() - 1) // 2)
                node = Hh(pk_seed, adrs, a + node)
        roots += node
    fp = adrs.copy()
    fp.set_type(FORS_ROOTS)
    fp.set_keypair(adrs.get_keypair())
    return T(pk_seed, fp, roots)


def verify(pk, msg, sig, ctx=CTX):
    if len(pk) != 2 * N or len(sig) != SIG_LEN or len(ctx) > 255:
        return False
    pk_seed, pk_root = pk[:N], pk[N:]
    m_prime = bytes([0, len(ctx)]) + ctx + msg
    r = sig[:N]
    sig_fors = sig[N:N + K * (A + 1) * N]
    sig_ht = sig[N + K * (A + 1) * N:]
    digest = shake(r + pk_seed + pk_root + m_prime, M_BYTES)
    md = digest[:(K * A + 7) // 8]                                   # 21 bytes
    tmp_tree = digest[len(md):len(md) + (H - H // D + 7) // 8]       # 7 bytes
    tmp_leaf = digest[len(md) + len(tmp_tree):len(md) + len(tmp_tree) + (H // D + 7) // 8]   # 2 bytes
    idx_tree = int.from_bytes(tmp_tree, "big") % (1 << (H - H // D))
    idx_leaf = int.from_bytes(tmp_leaf, "big") % (1 << (H // D))
    adrs = Adrs()
    adrs.set_tree(idx_tree)
    adrs.set_type(FORS_TREE)
    adrs.set_keypair(idx_leaf)
    pk_fors = fors_pk_from_sig(sig_fors, md, pk_seed, adrs)
    return ht_verify(pk_fors, sig_ht, pk_seed, idx_tree, idx_leaf, pk_root)


def main(path):
    if path == "--rust":
        import subprocess
        V = json.loads(subprocess.run([sys.argv[2]], capture_output=True, text=True, check=True).stdout)
    else:
        V = json.load(open(path))
    bad = 0
    n = 0
    for c in V["cases"]:
        pk, msg, sig = bytes.fromhex(c["pk"]), bytes.fromhex(c["msg"]), bytes.fromhex(c["sig"])
        got = verify(pk, msg, sig)
        n += 1
        ok = got == c["valid"]
        bad += not ok
        print(("ok   " if ok else "FAIL"), c["name"], f"(Rust: {c['valid']}, Python: {got})")
    print(f"{n - bad}/{n} identical verdicts between the Rust `fips205` crate and the independent Python verifier")
    print("RESULT: SLH-DSA MATCHES" if bad == 0 else "RESULT: DIVERGENCE")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1])
