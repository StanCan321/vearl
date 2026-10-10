#!/usr/bin/env python3
"""LINEAR ALGEBRA CLIENT for VEARL: int8 matrix products of ANY SIZE entrusted to a GPU worker, each block verified
before payment (`vearl-cli tiled-run`, docs/07 section 13), with what the protocol does not do by itself:
  - padding with zeros up to a multiple of n (1024) and splitting into jobs of at most 8 x 8 x 8 blocks (`MAX_BLOCKS`),
  - sum of the jobs over the K dimension (exact: 32-bit integers) and cutting of C to the requested dimensions,
  - Gram X^T X, similarity Q D^T.

    from la_client import LA
    la = LA("127.0.0.1:9821", "127.0.0.1:9821", "seed:4", escrow_per_product=5)
    C = la.matmul(A, B)            # A (M x K), B (K x N) int8 in [-64, 64]; C (M x N) int64 EXACT
    G = la.gram(X)                 # X^T X

The jobs are SEQUENTIAL here (a simple client); the protocol accepts as many as one wants in parallel.
"""
import os, subprocess, sys, tempfile, time
import numpy as np

MAX_BLOCKS = 8


class LA:
    def __init__(self, node, worker, key, escrow_per_product=5, n=1024, cli="target/release/vearl-cli", verbose=True):
        self.node, self.worker, self.key, self.esc, self.n, self.cli, self.verbose = node, worker, key, escrow_per_product, n, cli, verbose
        self.jobs = 0
        self.seconds = 0.0

    def _tiled(self, a_pad, b_pad):
        """One job: a_pad (I*n x K*n), b_pad (K*n x J*n) int8, I, K, J <= 8 -> C int64 (I*n x J*n)."""
        n = self.n
        I, K, J = a_pad.shape[0] // n, a_pad.shape[1] // n, b_pad.shape[1] // n
        with tempfile.TemporaryDirectory() as d:
            pa, pb, pc = f"{d}/a.i8", f"{d}/b.i8", f"{d}/c.i32"
            np.ascontiguousarray(a_pad, dtype=np.int8).tofile(pa)
            np.ascontiguousarray(b_pad, dtype=np.int8).tofile(pb)
            t = time.time()
            r = subprocess.run([self.cli, "tiled-run", self.node, self.worker, self.key, str(self.esc), pa, pb, str(I), str(K), str(J), pc], capture_output=True, text=True)
            out = r.stdout + r.stderr
            if r.returncode != 0 or not os.path.exists(pc) or "Accept: Ack { ok: true" not in out:
                raise RuntimeError(f"job refused or fraud detected:\n{out[-1500:]}")
            self.seconds += time.time() - t
            self.jobs += 1
            return np.fromfile(pc, dtype=np.int32).reshape(I * n, J * n).astype(np.int64)

    def matmul(self, A, B):
        A, B = np.asarray(A), np.asarray(B)
        assert A.ndim == 2 and B.ndim == 2 and A.shape[1] == B.shape[0], "incompatible shapes"
        assert abs(A).max() <= 64 and abs(B).max() <= 64, "integers in [-64, 64] (quantise on the client side)"
        n = self.n
        M, K = A.shape
        N = B.shape[1]
        pad = lambda x, r, c: np.pad(x.astype(np.int8), ((0, r * n - x.shape[0]), (0, c * n - x.shape[1])))
        bi, bk, bj = -(-M // n), -(-K // n), -(-N // n)
        Ap, Bp = pad(A, bi, bk), pad(B, bk, bj)
        C = np.zeros((bi * n, bj * n), np.int64)
        step = MAX_BLOCKS
        for i0 in range(0, bi, step):
            for j0 in range(0, bj, step):
                for k0 in range(0, bk, step):
                    i1, j1, k1 = min(i0 + step, bi), min(j0 + step, bj), min(k0 + step, bk)
                    sub = self._tiled(Ap[i0 * n:i1 * n, k0 * n:k1 * n], Bp[k0 * n:k1 * n, j0 * n:j1 * n])
                    C[i0 * n:i1 * n, j0 * n:j1 * n] += sub
                    if self.verbose:
                        print(f"  job {self.jobs}: blocks A[{i0}:{i1}, {k0}:{k1}] x B[{k0}:{k1}, {j0}:{j1}] verified ({self.seconds:.1f} s in total)", flush=True)
        return C[:M, :N]

    def gram(self, X):
        """X^T X (X: samples x dimensions)."""
        X = np.asarray(X)
        return self.matmul(X.T, X)

    def similarity(self, Q, D):
        """Q D^T (dot products between rows)."""
        return self.matmul(np.asarray(Q), np.asarray(D).T)


if __name__ == "__main__":
    print(__doc__)
