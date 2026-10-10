#!/usr/bin/env python3
"""INDEPENDENT light client: connects to a REAL VEARL node (Noise_XX_25519_ChaChaPoly_BLAKE2s over TCP), does the Hello,
downloads the chain through GetBlocks and RE-VALIDATES it entirely with the independent validator (chain_ref.py): recomputed
work, signatures, LWMA, rewards. Written from docs/08-network-specification.md, without reading the node's code.

    python3 light_client.py GENESIS.json HOST:PORT [--min-height N]                       # full blocks, everything revalidated
    python3 light_client.py GENESIS.json HOST:PORT --headers ADDRESS [--min-height N]      # LIGHT CLIENT: headers only + balance proof
Prerequisites: pip install noiseprotocol blake3 numpy cryptography

--headers mode (docs/14 section 0): GetHeaders -> each v2 header is verified WITHOUT the state (chaining, timestamp, target recomputed by the genesis
rule, native work roots recomputed, tile proof recomputed on its rows/columns alone, jackpot under the target),
then GetAccountProof -> the balance of ADDRESS is verified by a Merkle path against `accounts_root` of the tip. SPV assumption: the
validity of the transactions is NOT verified (majority of the work assumed honest).
"""
import json, socket, struct, sys, time
from noise.connection import NoiseConnection, Keypair
import spec_ref as S
import chain_ref as C
import asert_ref as A

A_HALFLIFE = 900   # ASERT half-life of VEARL (docs/13 section 5)

PATTERN = b"Noise_XX_25519_ChaChaPoly_BLAKE2s"
MAX_FRAME, TAG = 65535, 16
CONSTANTS = {"block_time": 30, "epoch_len": 100, "r0": 20 * 10**8, "halving": 1_051_200, "tail": 5 * 10**7, "lwma_n": 30, "max_future": 60}


# ---------------- Bech32m addresses (BIP-350) ----------------
_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def _polymod(values):
    gen = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for v in values:
        b = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ v
        for i in range(5):
            chk ^= gen[i] if (b >> i) & 1 else 0
    return chk


def bech32m_decode(addr):
    addr = addr.strip().lower()
    pos = addr.rfind("1")
    hrp, data = addr[:pos], [_CHARSET.index(c) for c in addr[pos + 1:]]
    if _polymod([ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp] + data) != 0x2BC830A3:
        raise ValueError("invalid Bech32m checksum")
    five = data[:-6]
    acc = bits = 0
    out = bytearray()
    for v in five:
        acc = (acc << 5) | v
        bits += 5
        while bits >= 8:
            bits -= 8
            out.append((acc >> bits) & 0xFF)
    return hrp, bytes(out)


def parse_amount(s):
    i, _, f = s.partition(".")
    return int(i or 0) * 10**8 + int((f + "00000000")[:8] or 0)


# ---------------- genesis ----------------
def load_genesis(path):
    f = json.load(open(path))
    alloc = []
    for a in f["alloc"]:
        hrp, pk = bech32m_decode(a["address"])
        assert hrp == "vrl" and len(pk) == 32
        alloc.append((pk, parse_amount(a["amount"])))
    p = f["params"]
    target = int(f["target"])
    retarget = f["retarget"]
    g = {"params": p, "alloc": [{"key": k.hex(), "amount": v} for k, v in alloc], "timestamp": f["timestamp"], "target": target,
         "constants": CONSTANTS, "retarget": retarget}
    tag = {"Ema": 0, "Lwma": 1, "Fixed": 2, "Asert": 3}[retarget]
    enc = (struct.pack("<QQQQ", p["n"], p["tile"], p["rank"], p["fold_k"]) + struct.pack("<Q", len(alloc)) +
           b"".join(k + struct.pack("<Q", v) for k, v in alloc) + struct.pack("<QQ", f["timestamp"], target) +
           struct.pack("<QQ", f["timing"]["challenge_secs"], f["timing"]["serve_secs"]) + struct.pack("<I", tag))
    return g, S.H("vearl/genesis/v1", enc)


# ---------------- bincode (reading) ----------------
class Rd:
    def __init__(self, b):
        self.b, self.p = b, 0

    def take(self, n):
        if self.p + n > len(self.b):
            raise ValueError("end of buffer")
        x = self.b[self.p:self.p + n]
        self.p += n
        return x

    def u8(self): return self.take(1)[0]
    def u32(self): return struct.unpack("<I", self.take(4))[0]
    def u64(self): return struct.unpack("<Q", self.take(8))[0]
    def u128(self): return int.from_bytes(self.take(16), "little")
    def h32(self): return self.take(32)
    def vec(self, fn):
        n = self.u64()
        if n > 1 << 24:
            raise ValueError("vector too long")
        return [fn() for _ in range(n)]


class Unsupported(Exception):
    pass


def parse_class(r):
    tag = r.u32()
    if tag == 0:
        return {"type": "Native"}
    if tag == 1:
        return {"type": "Job", "job": r.h32().hex()}
    if tag == 2:
        r.h32(); r.u8()
    elif tag == 3:
        r.h32(); r.take(3)
    elif tag == 4:
        r.h32(); r.u8()
    else:
        raise ValueError("unknown class")
    raise Unsupported("useful-work class (Layer/Product/Op)")


def parse_block(r):
    hd = {"version": r.u32(), "height": r.u64(), "prev": r.h32().hex(), "timestamp": r.u64(), "miner": r.h32().hex()}
    cls = parse_class(r)
    hd.update({"extra_nonce": r.u64(), "ti": r.u32(), "tj": r.u32(), "jackpot": r.h32().hex(), "tx_root": r.h32().hex(), "target": r.u64(),
               "a_root": r.h32().hex(), "b_root": r.h32().hex(), "accounts_root": r.h32().hex()})   # v2 header (docs/14)
    proof = {"ti": r.u32(), "tj": r.u32(), "a_rows": r.vec(lambda: r.vec(lambda: struct.unpack("<b", r.take(1))[0])),
             "a_paths": r.vec(lambda: r.vec(r.h32)), "b_cols": r.vec(lambda: r.vec(lambda: struct.unpack("<b", r.take(1))[0])), "b_paths": r.vec(lambda: r.vec(r.h32))}
    txs = []
    for _ in range(r.u64()):
        frm, nonce, fee, kind = r.h32().hex(), r.u64(), r.u64(), r.u32()
        if kind != 0:
            raise Unsupported(f"market transaction (tag {kind})")
        t = {"from": frm, "nonce": nonce, "fee": fee, "to": r.h32().hex(), "amount": r.u64()}
        t["sig"] = bytes(r.vec(r.u8)).hex()
        txs.append(t)
    return {"header": hd, "txs": txs, "proof": proof, "class": cls}


# ---------------- Noise transport ----------------
class Peer:
    def __init__(self, addr):
        host, port = addr.rsplit(":", 1)
        self.s = socket.create_connection((host, int(port)), timeout=10)
        n = NoiseConnection.from_name(PATTERN)
        n.set_as_initiator()
        n.set_keypair_from_private_bytes(Keypair.STATIC, bytes(range(1, 33)))  # ephemeral test identity
        n.start_handshake()
        self._send_frame(n.write_message())            # -> e
        n.read_message(self._read_frame())            # <- e, ee, s, es
        self.remote = n.noise_protocol.handshake_state.rs.public_bytes  # authenticated static key of the node
        self._send_frame(n.write_message())            # -> s, se
        assert n.handshake_finished
        self.n, self.buf = n, b""

    def _send_frame(self, b):
        self.s.sendall(struct.pack(">H", len(b)) + b)

    def _read_exact(self, k):
        out = b""
        while len(out) < k:
            c = self.s.recv(k - len(out))
            if not c:
                raise EOFError("connection closed")
            out += c
        return out

    def _read_frame(self):
        (ln,) = struct.unpack(">H", self._read_exact(2))
        return self._read_exact(ln)

    def send(self, payload):
        plain = struct.pack("<I", len(payload)) + payload
        for i in range(0, len(plain), MAX_FRAME - TAG):
            self._send_frame(self.n.encrypt(plain[i:i + MAX_FRAME - TAG]))

    def recv(self):
        while True:
            if len(self.buf) >= 4:
                ln = struct.unpack("<I", self.buf[:4])[0]
                if len(self.buf) >= 4 + ln:
                    msg, self.buf = self.buf[4:4 + ln], self.buf[4 + ln:]
                    return msg
            self.buf += self.n.decrypt(self._read_frame())


# ---------------- light client mode (headers only) ----------------
def parse_class_any(r):
    tag = r.u32()
    if tag == 0:
        return ("Native",)
    if tag == 1:
        return ("Job", r.h32())
    if tag == 2:
        return ("Layer", r.h32(), r.u8())
    if tag == 3:
        j = r.h32()
        return ("Product", j, r.u8(), r.u8(), r.u8())
    if tag == 4:
        return ("Op", r.h32(), r.u8())
    raise ValueError("unknown class")


def parse_header_v2(r):
    hd = {"version": r.u32(), "height": r.u64(), "prev": r.h32(), "timestamp": r.u64(), "miner": r.h32(), "class": parse_class_any(r),
          "extra_nonce": r.u64(), "ti": r.u32(), "tj": r.u32(), "jackpot": r.h32(), "tx_root": r.h32(), "target": r.u64(),
          "a_root": r.h32(), "b_root": r.h32(), "accounts_root": r.h32()}
    i8 = lambda: struct.unpack("<b", r.take(1))[0]
    proof = {"ti": r.u32(), "tj": r.u32(), "a_rows": r.vec(lambda: r.vec(i8)), "a_paths": r.vec(lambda: r.vec(r.h32)),
             "b_cols": r.vec(lambda: r.vec(i8)), "b_paths": r.vec(lambda: r.vec(r.h32))}
    return hd, proof


def header_hash_v2(h):
    return S.H("vearl/header/v1", S.header_bytes(h["version"], h["height"], h["prev"], h["timestamp"], h["miner"], h["class"], h["extra_nonce"],
                                                 h["ti"], h["tj"], h["jackpot"], h["tx_root"], h["target"], h["a_root"], h["b_root"], h["accounts_root"]))


def depth_for(count):
    d = 0
    while (1 << d) < count:
        d += 1
    return d


def verify_path(root, leaf, idx, path, depth):
    if len(path) != depth or idx >= (1 << depth):
        return False
    cur = leaf
    for sib in path:
        cur = S.H("vearl/merkle/node/v1", cur, sib) if idx % 2 == 0 else S.H("vearl/merkle/node/v1", sib, cur)
        idx >>= 1
    return cur == root


class HeaderChain:
    """Follows the headers without state (same rules as chain.rs::LightClient, written here from docs/07, docs/13 and docs/14)."""

    def __init__(self, g):
        self.g, self.p, self.c = g, g["params"], g["constants"]
        self.hist, self.tip, self.last, self.target, self.work = [], b"\x00" * 32, None, g["target"], 0
        self.native = {}

    def native_roots(self, epoch):
        if epoch not in self.native:
            a, b = S.native_matrices(epoch, self.p["n"])
            self.native[epoch] = (S.rows_root(a), S.cols_root(b))
        return self.native[epoch]

    def apply(self, h, pr):
        p, n, t = self.p, self.p["n"], self.p["tile"]
        assert h["version"] == 2, "version"
        assert h["height"] == len(self.hist) and h["prev"] == self.tip, "chaining"
        prev_ts = self.hist[-1][0] if self.hist else self.g["timestamp"]
        assert h["timestamp"] > prev_ts, "timestamp not increasing"
        assert h["target"] == self.target, f"target {h['target']} instead of {self.target}"
        assert S.meets_target(h["jackpot"], h["target"]), "jackpot above the target"
        if h["class"][0] == "Native":
            assert (h["a_root"], h["b_root"]) == self.native_roots(h["height"] // self.c["epoch_len"]), "native roots"
        assert (pr["ti"], pr["tj"]) == (h["ti"], h["tj"]) and len(pr["a_rows"]) == t and len(pr["b_cols"]) == t, "tile proof"
        d = depth_for(n)
        for x in range(t):
            i, j = h["ti"] * t + x, h["tj"] * t + x
            assert len(pr["a_rows"][x]) == n and len(pr["b_cols"][x]) == n and max(abs(v) for v in pr["a_rows"][x] + pr["b_cols"][x]) <= 64
            assert verify_path(h["a_root"], S.leaf_hash(i, S.i8bytes(pr["a_rows"][x])), i, pr["a_paths"][x], d), "row of A"
            assert verify_path(h["b_root"], S.leaf_hash(j, S.i8bytes(pr["b_cols"][x])), j, pr["b_paths"][x], d), "column of B"
        sg = S.sigma_of(h["prev"], h["miner"], h["class"], h["extra_nonce"], h["tx_root"], h["timestamp"])
        jp = S.tile_jackpot(pr["a_rows"], pr["b_cols"], h["ti"], h["tj"], sg, n, t, p["rank"], p["fold_k"])
        assert jp == h["jackpot"], "recomputed tile different from the jackpot"
        self.hist.append((h["timestamp"], h["target"]))
        mode = self.g.get("retarget", "Lwma")
        ema = C.ema_next(self.c, self.target, h["timestamp"] - prev_ts)
        if mode == "Ema":
            self.target = ema
        elif mode == "Lwma":
            self.target = C.lwma_next(self.c, [{"timestamp": a, "target": b} for a, b in self.hist[-(self.c["lwma_n"] + 1):]], ema)
        elif mode == "Asert":
            (f_ts, f_tg), (l_ts, _) = self.hist[0], self.hist[-1]
            self.target = A.vearl_asert(f_tg, f_ts, l_ts, len(self.hist) - 1, self.c["block_time"], A_HALFLIFE)
        self.work += C.block_work(h["target"])
        self.tip, self.last = header_hash_v2(h), h


def main_headers(gpath, addr, address, min_height=0):
    g, gh = load_genesis(gpath)
    hrp, pk = bech32m_decode(address)
    assert hrp in ("vrl", "vrlq") and len(pk) == 32
    peer = Peer(addr)
    peer.send(hello(gh))
    hc = HeaderChain(g)
    deadline, proof, t0 = time.time() + 120, None, time.time()
    while time.time() < deadline:
        r = Rd(peer.recv())
        tag = r.u32()
        if tag == 0:                                                   # Hello of the node
            r.u64()
            assert r.h32() == gh, "the node announces another genesis"
            print(f"Hello of the node: height {r.u64()}")
            peer.send(struct.pack("<I", 51) + struct.pack("<Q", 0))     # GetHeaders{from: 0}
        elif tag == 52:                                                # Headers
            batch = r.vec(lambda: parse_header_v2(r))
            for h, pr in batch:
                hc.apply(h, pr)
            if batch:
                peer.send(struct.pack("<I", 51) + struct.pack("<Q", len(hc.hist)))
            else:
                peer.send(struct.pack("<I", 53) + pk)                   # GetAccountProof(key)
        elif tag == 54:                                                # AccountProof
            proof = {"height": r.u64(), "tip": r.h32(), "balance": r.u64(), "nonce": r.u64(), "index": r.u64(), "count": r.u64(), "path": r.vec(r.h32)}
            if proof["height"] > len(hc.hist):                         # the node moved ahead: we catch up then ask again
                peer.send(struct.pack("<I", 51) + struct.pack("<Q", len(hc.hist)))
                continue
            if proof["height"] == len(hc.hist) and proof["tip"] == hc.tip:
                break
            peer.send(struct.pack("<I", 53) + pk)
        elif tag == 11:                                                # Ack (unknown account)
            ok, info = r.u8(), r.take(r.u64()).decode()
            raise SystemExit(f"the node refuses the account proof: {info}")
    assert proof and proof["tip"] == hc.tip, "no proof on the followed tip"
    assert len(hc.hist) >= max(min_height, 1), "too few headers"
    leaf = S.leaf_hash(proof["index"] + 1, pk + struct.pack("<QQ", proof["balance"], proof["nonce"]))
    ok = proof["index"] < proof["count"] and verify_path(hc.last["accounts_root"], leaf, proof["index"] + 1, proof["path"], depth_for(proof["count"] + 1))
    assert ok, "invalid balance proof"
    print(f"{len(hc.hist)} HEADERS VERIFIED WITHOUT THE STATE (tile proofs, targets {g.get('retarget')}, chaining) in {time.time() - t0:.1f} s; work {hc.work}")
    print(f"BALANCE PROVEN against accounts_root of header {len(hc.hist) - 1}: {proof['balance'] / 1e8:.8f} VRL (nonce {proof['nonce']}) for {address}")
    return hc, proof


def hello(genesis_hash):
    return struct.pack("<I", 0) + struct.pack("<Q", 0xC0FFEE) + genesis_hash + struct.pack("<Q", 0) + (0).to_bytes(16, "little") + b"\x00"


def main(gpath, addr, min_height=0):
    g, gh = load_genesis(gpath)
    print("computed genesis hash:", gh.hex())
    peer = Peer(addr)
    print("Noise XX handshake ok; static key of the node:", peer.remote.hex())
    peer.send(hello(gh))
    remote_h, blocks = None, []
    deadline = time.time() + 120
    while time.time() < deadline:
        r = Rd(peer.recv())
        tag = r.u32()
        if tag == 0:  # Hello of the node
            r.u64()
            ghash, remote_h, work = r.h32(), r.u64(), r.u128()
            assert ghash == gh, "the node announces another genesis"
            print(f"Hello of the node: height {remote_h}, work {work}")
            peer.send(struct.pack("<I", 1) + struct.pack("<Q", 0))            # GetBlocks{from: 0}
        elif tag == 2 and remote_h is not None:  # Blocks
            batch = r.vec(lambda: parse_block(r))
            blocks += batch
            if not batch or len(blocks) >= remote_h:
                break
            peer.send(struct.pack("<I", 1) + struct.pack("<Q", len(blocks)))  # suite
        # the other messages (GetPeers, NewBlock, NewTx...) are ignored by this light client
    print(f"{len(blocks)} blocks received")
    assert len(blocks) >= max(min_height, 1), "too few blocks"
    ch = Chain(g)
    for k, b in enumerate(blocks):
        ch.apply({"header": b["header"], "txs": b["txs"]})
        # the tile proof carried must contain the real rows of A and columns of B (native)
        a, bm = ch.matrices(b["header"]["height"] // CONSTANTS["epoch_len"])
        pr = b["proof"]
        assert (pr["ti"], pr["tj"]) == (b["header"]["ti"], b["header"]["tj"])
        for x in range(16):
            assert list(a[pr["ti"] * 16 + x]) == pr["a_rows"][x] and list(bm[:, pr["tj"] * 16 + x]) == pr["b_cols"][x], "inconsistent tile proof"
    print(f"CHAIN OF {len(blocks)} BLOCKS RE-VALIDATED BY THE INDEPENDENT IMPLEMENTATION (work, signatures, difficulty, rewards)")
    print("expected target of the next block:", ch.target, "; emission:", ch.minted, "; cumulative work:", ch.total_work())
    return ch


class Chain(C.Chain):
    """chain_ref.Chain + retargeting rule chosen by the genesis."""

    def apply(self, b):
        mode = self.g.get("retarget", "Lwma")
        old_target, old_ts = self.target, self.tip_ts
        super().apply(b)
        if mode == "Fixed":
            self.target = old_target
        elif mode == "Ema":
            self.target = C.ema_next(self.c, old_target, b["header"]["timestamp"] - old_ts)
        elif mode == "Asert":
            # ASERT (aserti3-2d), independent implementation verified on the official vectors (tools/asert_ref.py); anchor = first block
            h0 = self.headers[0]
            self.target = A.vearl_asert(h0["target"], h0["timestamp"], self.tip_ts, len(self.headers) - 1, self.c["block_time"], A_HALFLIFE)


if __name__ == "__main__":
    mh = int(sys.argv[sys.argv.index("--min-height") + 1]) if "--min-height" in sys.argv else 0
    if "--headers" in sys.argv:
        main_headers(sys.argv[1], sys.argv[2], sys.argv[sys.argv.index("--headers") + 1], mh)
    else:
        main(sys.argv[1], sys.argv[2], mh)
