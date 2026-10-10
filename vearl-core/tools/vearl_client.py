#!/usr/bin/env python3
"""Minimal Python SDK to USE the VEARL network from a script (data science, on-demand computation):
transfers, delegation of a matrix product to a worker, verification of the result (Freivalds test) and payment or
fraud proof. Speaks the REAL protocol (Noise XX + bincode, docs/07-08); no dependency on the Rust code.

    from vearl_client import Client
    c = Client("127.0.0.1:9300", seed=4)              # PUBLIC seed key (devnet); otherwise Client(addr, secret=bytes32)
    job = c.submit_matmul(worker="127.0.0.1:9300", a=A, b=B, escrow_vrl=5)      # A, B: int8 in [-64, 64], n x n
    c.wait_status(job, "Posted")
    c.wait_status(job, "Posted")
C_int32 = c.result_and_settle(job, worker="127.0.0.1:9300", a=A, b=B)      # verifies; Accept or Fraud automatically
Prerequisites: pip install noiseprotocol blake3 numpy cryptography
"""
import os, struct, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import spec_ref as S
import light_client as L
import market_ref as M
import chain_ref as C
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

COIN = 10**8
T_HELLO, T_ACK, T_ACCOUNT, T_JOBINFO, T_JOBRESULT = 0, 11, 10, 14, 20


def _string(r):
    return bytes(r.take(r.u64())).decode()


class Client:
    def __init__(self, addr, seed=None, secret=None):
        assert (seed is None) != (secret is None), "seed OR secret"
        self.addr = addr
        self.sk = Ed25519PrivateKey.from_private_bytes(bytes([seed]) * 32 if seed is not None else secret)
        self.pk = self.sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    # ---------------- transport ----------------
    def _rpc(self, node, payload, want):
        """One request on a fresh Noise connection; returns the first message with tag `want`."""
        p = L.Peer(node)
        p.s.settimeout(30)
        p.send(payload)
        while True:
            r = L.Rd(p.recv())
            tag = r.u32()
            if tag == want:
                out = (tag, r)
                p.s.close()
                return out
            if tag == T_ACK and want != T_ACK:
                ok, info = bool(r.u8()), _string(r)
                p.s.close()
                raise RuntimeError(f"the node answered Ack(ok={ok}): {info}")

    # ---------------- accounts and transactions ----------------
    def account(self):
        _, r = self._rpc(self.addr, struct.pack("<I", 9) + self.pk, T_ACCOUNT)
        balance, nonce, next_nonce = r.u64(), r.u64(), r.u64()
        return dict(balance=balance / COIN, nonce=nonce, next_nonce=next_nonce)

    def send(self, kind, fee=0):
        """Signs and broadcasts a transaction (`kind` = dict in the market_ref format, e.g. {"type": "Transfer", ...})."""
        t = {"from": self.pk.hex(), "nonce": self.account()["next_nonce"], "fee": fee, "kind": kind}
        sig = self.sk.sign(M.tx_bytes(t))
        raw = M.tx_bytes(t) + struct.pack("<Q", len(sig)) + sig
        _, r = self._rpc(self.addr, struct.pack("<I", 4) + raw, T_ACK)
        ok, info = bool(r.u8()), _string(r)
        if not ok:
            raise RuntimeError(f"transaction refused: {info}")
        return M.tx_id({**t, "sig": sig.hex()}).hex()

    def transfer(self, to_pubkey_hex, amount_vrl):
        return self.send({"type": "Transfer", "to": to_pubkey_hex, "amount": int(round(amount_vrl * COIN))})

    # ---------------- matrix product jobs ----------------
    @staticmethod
    def roots(a, b):
        n = a.shape[0]
        ar = S.merkle_root([S.leaf_hash(i, S.i8bytes(a[i])) for i in range(n)])
        br = S.merkle_root([S.leaf_hash(j, S.i8bytes(b[:, j])) for j in range(n)])
        return ar, br

    def submit_matmul(self, worker, a, b, escrow_vrl=5, auction_secs=None, deadline_blocks=400):
        a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
        assert a.shape == b.shape and a.shape[0] == a.shape[1] and abs(a).max() <= 64 and abs(b).max() <= 64, "n x n matrices, integers in [-64, 64]"
        n = a.shape[0]
        ar, br = self.roots(a, b)
        nonce = self.account()["next_nonce"]
        job = M.job_id_of(self.pk.hex(), nonce, ar, br).hex()
        # deposit of the matrices with the worker (off-chain): PutJob{a, b}
        payload = struct.pack("<I", 12) + struct.pack("<Q", n * n) + S.i8bytes(a) + struct.pack("<Q", n * n) + S.i8bytes(b)
        _, r = self._rpc(worker, payload, T_ACK)
        assert r.u8() == 1, "the worker refused the data"
        height = self.status()["height"]
        kind = dict(type="SubmitJob", a_root=ar.hex(), b_root=br.hex(), escrow=int(escrow_vrl * COIN), deadline=height + deadline_blocks)
        if auction_secs:
            kind = dict(kind, type="SubmitJobAuction", auction_secs=auction_secs)
        self.send(kind)
        return job

    def status(self):
        _, r = self._rpc(self.addr, struct.pack("<I", 7), 8)
        return dict(height=r.u64(), tip=r.h32().hex(), work=r.u128())

    def job_status(self, job):
        _, r = self._rpc(self.addr, struct.pack("<I", 13) + bytes.fromhex(job), T_JOBINFO)
        return _string(r)

    def wait_status(self, job, prefix, timeout=120):
        t0, st = time.time(), "unknown"
        while time.time() - t0 < timeout:
            try:
                st = self.job_status(job)
            except RuntimeError as e:
                if "unknown job" not in str(e):  # the creation transaction is not yet included in a block
                    raise
            if st.startswith(prefix):
                return st
            time.sleep(1)
        raise TimeoutError(f"job {job[:8]}: status '{prefix}' not reached (last: {st})")

    def result_and_settle(self, job, worker, a, b, freivalds_rounds=24):
        """Fetches C from the worker, VERIFIES it (Freivalds, error <= 2^-rounds), then Accept; otherwise locates the wrong entry and
        publishes a fraud proof (Merkle openings of A, B and C). Returns C if accepted, None if fraud."""
        a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
        n = a.shape[0]
        _, r = self._rpc(worker, struct.pack("<I", 19) + bytes.fromhex(job), T_JOBRESULT)
        c = np.array(r.vec(lambda: struct.unpack("<i", r.take(4))[0]), dtype=np.int64).reshape(n, n)
        rng = np.random.default_rng()
        bad_rows = set()
        for _ in range(freivalds_rounds):
            x = rng.integers(0, 2, n)
            bad_rows |= set(np.nonzero(a @ (b @ x) != c @ x)[0].tolist())
        if not bad_rows:
            self.send(dict(type="Accept", job_id=job))
            return c
        i = min(bad_rows)
        j = int(np.nonzero(a[i] @ b != c[i])[0][0])
        ar_leaves = [S.leaf_hash(x, S.i8bytes(a[x])) for x in range(n)]
        br_leaves = [S.leaf_hash(x, S.i8bytes(b[:, x])) for x in range(n)]
        cl = S.row_leaves(c)
        P = lambda leaves, idx: [h.hex() for h in S.merkle_proof(leaves, idx)]
        self.send(dict(type="Fraud", job_id=job, i=i, j=j, a_row=[int(v) for v in a[i]], a_path=P(ar_leaves, i), b_col=[int(v) for v in b[:, j]],
                       b_path=P(br_leaves, j), c_row=[int(v) for v in c[i]], c_path=P(cl, i)))
        return None


if __name__ == "__main__":
    # demonstration: python3 vearl_client.py NODE_CLIENT NODE_WORKER [SEED]  (test network with a worker)
    node, worker = sys.argv[1], sys.argv[2]
    c = Client(node, seed=int(sys.argv[3]) if len(sys.argv) > 3 else 4)
    print("account:", c.account())
    rng = np.random.default_rng(0)
    A, B = rng.integers(-64, 65, (128, 128)), rng.integers(-64, 65, (128, 128))
    job = c.submit_matmul(worker, A, B, escrow_vrl=5)
    print("job submitted:", job[:16], "...")
    print("status:", c.wait_status(job, "Posted"))
    res = c.result_and_settle(job, worker, A, B)
    print("VERIFIED and accepted result: C == A@B:", res is not None and bool((res == A @ B).all()) if res is not None else "FRAUD detected -> proof published")
    time.sleep(4)
    print("final status:", c.job_status(job))
