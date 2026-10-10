//! BLOCK-WISE matrix product jobs (rectangular shapes): C = A*B with A (M x K), B (K x N), M, K, N multiples of n.
//!
//! Split: A_ik, B_kj, P_ijk = A_ik * B_kj (n x n blocks, exact int32), C_ij = sum_k P_ijk.
//! The client commits each block (A_ik by rows, B_kj by columns) in the transaction; the worker publishes the root of
//! each product P_ijk (one leaf per entry) and of each C_ij. Two element-wise fraud proofs, verifiable on-chain:
//!   - Product: P_ijk[r][c] != <row r of A_ik, column c of B_kj>          (O(n))
//!   - Sum:     C_ij[r][c]  != sum_k P_ijk[r][c]                          (O(K/n * log n^2))
//! The consensus work stays the noised GEMM of ONE n x n block: the class `Product{job,i,j,k}` mines with A_ik and B_kj.
//! A worker does NOT need one block per product: the useful computation produces attempts continuously, and any
//! of the winning attempts can become a block (no product is required to make the chain advance).
//! Limits: bounded size (MAX_BLOCKS^3 products), no guarantee of data availability (see docs/02).
use crate::crypto::*;
use crate::state::*;
use crate::types::*;
use crate::work::*;
use serde::{Deserialize, Serialize};

pub const MAX_BLOCKS: usize = 8;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum TiledFraudKind {
    Product { i: u8, j: u8, k: u8, r: u32, c: u32, a_row: Vec<i8>, a_path: Vec<Hash>, b_col: Vec<i8>, b_path: Vec<Hash>, z_row: Vec<i32>, z_path: Vec<Hash> },
    /// `z_rows[k]` = row `r` of product k, `c_row` = row `r` of the sum (row-wise commitment); the accused entry is column `c`.
    Sum { i: u8, j: u8, r: u32, c: u32, z_rows: Vec<Vec<i32>>, z_paths: Vec<Vec<Hash>>, c_row: Vec<i32>, c_path: Vec<Hash> },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum TiledTx {
    /// dims = (row blocks of A, K blocks, column blocks of B).
    Submit { dims: (u8, u8, u8), a_roots: Vec<Hash>, b_roots: Vec<Hash>, escrow: u64, deadline: u64 },
    Claim { job_id: JobId },
    Post { job_id: JobId, product_roots: Vec<Hash>, c_roots: Vec<Hash> },
    Accept { job_id: JobId },
    Fraud { job_id: JobId, kind: TiledFraudKind },
    Settle { job_id: JobId },
    /// DATA AVAILABILITY (as for programs, docs/07 section 12.5): the client demands, on-chain, an entry of a product (`which` = 0,
    /// block (i, j, k)) or of a sum (`which` = 1, block (i, j), `k` = 0) published, at index `idx` (r * n + c).
    Fetch { job_id: JobId, which: u8, i: u8, j: u8, k: u8, idx: u32 },
    /// The worker serves the ROW `idx / n` containing the entry, with its Merkle path against the published root; the dispute window is extended.
    Serve { job_id: JobId, which: u8, i: u8, j: u8, k: u8, idx: u32, row: Vec<i32>, path: Vec<Hash> },
    /// Anyone: a request was left unanswered after SERVE_SECS -> penalty for the worker.
    SlashUnserved { job_id: JobId },
}

pub fn tiled_id(client: &PubKey, nonce: u64, dims: (u8, u8, u8), a_roots: &[Hash], b_roots: &[Hash]) -> JobId {
    let (nb, db) = (nonce.to_le_bytes(), [dims.0, dims.1, dims.2]);
    let mut parts: Vec<&[u8]> = vec![client, &nb, &db];
    for r in a_roots.iter().chain(b_roots.iter()) {
        parts.push(r);
    }
    h("vearl/tiled/v1", &parts)
}

#[derive(Clone, Debug, PartialEq)]
pub enum TiledStatus {
    Open,
    Claimed { worker: PubKey, bond: u64 },
    Posted { worker: PubKey, bond: u64, at: u64, at_ts: u64 },
    Settled,
    Slashed,
    Expired,
}

#[derive(Clone, Debug)]
pub struct TiledJob {
    pub client: PubKey,
    /// (I, K, J) in blocks.
    pub dims: (usize, usize, usize),
    pub a_roots: Vec<Hash>,
    pub b_roots: Vec<Hash>,
    pub escrow_net: u64,
    pub deadline: u64,
    pub status: TiledStatus,
    pub rebate_left: u64,
    pub product_roots: Vec<Hash>,
    pub c_roots: Vec<Hash>,
    pub fetches: Vec<TFetch>,
    pub fetch_held: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TFetch {
    pub which: u8,
    pub i: u8,
    pub j: u8,
    pub k: u8,
    pub idx: u32,
    pub deadline_ts: u64,
}

impl TiledJob {
    pub fn products(&self) -> usize {
        self.dims.0 * self.dims.1 * self.dims.2
    }
    /// Index of product (i, j, k) in `product_roots`.
    pub fn pidx(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.dims.2 + j) * self.dims.1 + k
    }
    /// (root of A_ik, root of B_kj).
    pub fn block_roots(&self, i: usize, j: usize, k: usize) -> Option<(Hash, Hash)> {
        let (bi, bk, bj) = self.dims;
        if i >= bi || j >= bj || k >= bk {
            return None;
        }
        Some((self.a_roots[i * bk + k], self.b_roots[k * bj + j]))
    }
}

// ---------------- client / worker side utilities ----------------

/// Block (bi, bj) of a `rows x cols` matrix (row-major), n x n blocks.
pub fn get_block(m: &[i8], cols: usize, bi: usize, bj: usize, n: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(n * n);
    for r in 0..n {
        let start = (bi * n + r) * cols + bj * n;
        out.extend_from_slice(&m[start..start + n]);
    }
    out
}

/// Roots of the blocks of A (by rows), order (i, k).
pub fn a_block_roots(a: &[i8], k_total: usize, bi: usize, bk: usize, n: usize) -> Vec<Hash> {
    (0..bi).flat_map(|i| (0..bk).map(move |k| (i, k))).map(|(i, k)| rows_tree(&get_block(a, k_total, i, k, n), n).root()).collect()
}

/// Roots of the blocks of B (by columns), order (k, j).
pub fn b_block_roots(b: &[i8], n_total: usize, bk: usize, bj: usize, n: usize) -> Vec<Hash> {
    (0..bk).flat_map(|k| (0..bj).map(move |j| (k, j))).map(|(k, j)| cols_tree(&get_block(b, n_total, k, j, n), n).root()).collect()
}

/// Complete reference computation (CPU): products P_ijk (order (i,j,k)) and sums C_ij (order (i,j)).
pub fn compute_all(a: &[i8], b: &[i8], dims: (usize, usize, usize), n: usize) -> (Vec<Vec<i32>>, Vec<Vec<i32>>) {
    let (bi, bk, bj) = dims;
    let (k_total, n_total) = (bk * n, bj * n);
    let (mut products, mut sums) = (Vec::new(), Vec::new());
    for i in 0..bi {
        for j in 0..bj {
            let mut c = vec![0i32; n * n];
            for k in 0..bk {
                let p = plain_gemm(&get_block(a, k_total, i, k, n), &get_block(b, n_total, k, j, n), n);
                for (x, v) in c.iter_mut().zip(&p) {
                    *x += *v;
                }
                products.push(p);
            }
            sums.push(c);
        }
    }
    (products, sums)
}

/// Assembles C (M x N) from the blocks C_ij (order (i,j)).
pub fn assemble(sums: &[Vec<i32>], bi: usize, bj: usize, n: usize) -> Vec<i32> {
    let n_total = bj * n;
    let mut c = vec![0i32; bi * n * n_total];
    for i in 0..bi {
        for j in 0..bj {
            for r in 0..n {
                for col in 0..n {
                    c[(i * n + r) * n_total + j * n + col] = sums[i * bj + j][r * n + col];
                }
            }
        }
    }
    c
}

fn bad<T>(m: &str) -> Result<T, String> {
    Err(m.to_string())
}

impl State {
    pub(crate) fn apply_tiled(&mut self, from: &PubKey, nonce: u64, m: &TiledTx, height: u64, ts: u64, p: &WorkParams) -> Result<(), String> {
        match m {
            TiledTx::Submit { dims, a_roots, b_roots, escrow, deadline } => {
                let (bi, bk, bj) = (dims.0 as usize, dims.1 as usize, dims.2 as usize);
                if [bi, bk, bj].iter().any(|d| *d == 0 || *d > MAX_BLOCKS) || a_roots.len() != bi * bk || b_roots.len() != bk * bj {
                    return bad("bad tiled spec");
                }
                let products = (bi * bk * bj) as u64;
                if *escrow < MIN_ESCROW * products {
                    return bad("escrow below floor price (per product)");
                }
                if *deadline <= height {
                    return bad("deadline in the past");
                }
                let id = tiled_id(from, nonce, *dims, a_roots, b_roots);
                if self.tiled.contains_key(&id) {
                    return bad("duplicate job");
                }
                self.debit(from, *escrow)?;
                let burn = bps(*escrow, BURN_BPS);
                self.burned += burn;
                self.tiled.insert(
                    id,
                    TiledJob {
                        client: *from,
                        dims: (bi, bk, bj),
                        a_roots: a_roots.clone(),
                        b_roots: b_roots.clone(),
                        escrow_net: escrow - burn,
                        deadline: *deadline,
                        status: TiledStatus::Open,
                        rebate_left: burn,
                        product_roots: vec![],
                        c_roots: vec![],
                        fetches: vec![],
                        fetch_held: 0,
                    },
                );
            }
            TiledTx::Claim { job_id } => {
                let (net, status, deadline) = {
                    let j = self.tiled.get(job_id).ok_or("unknown job")?;
                    (j.escrow_net, j.status.clone(), j.deadline)
                };
                if status != TiledStatus::Open || height > deadline {
                    return bad("job not claimable");
                }
                let bond = bps(net, BOND_BPS);
                self.debit(from, bond)?;
                self.tiled.get_mut(job_id).unwrap().status = TiledStatus::Claimed { worker: *from, bond };
            }
            TiledTx::Post { job_id, product_roots, c_roots } => {
                let j = self.tiled.get_mut(job_id).ok_or("unknown job")?;
                let TiledStatus::Claimed { worker, bond } = j.status.clone() else { return bad("job is not in progress") };
                if worker != *from || height > j.deadline {
                    return bad("cannot post result");
                }
                if product_roots.len() != j.products() || c_roots.len() != j.dims.0 * j.dims.2 {
                    return bad("wrong number of roots");
                }
                j.product_roots = product_roots.clone();
                j.c_roots = c_roots.clone();
                j.status = TiledStatus::Posted { worker, bond, at: height, at_ts: ts };
            }
            TiledTx::Accept { job_id } => {
                let (client, status, net) = {
                    let j = self.tiled.get(job_id).ok_or("unknown job")?;
                    (j.client, j.status.clone(), j.escrow_net)
                };
                match status {
                    TiledStatus::Posted { worker, bond, .. } if client == *from => {
                        self.credit(&worker, net + bond);
                        self.rep.entry(worker).or_default().done += 1;
                        let jm = self.tiled.get_mut(job_id).unwrap();
                        jm.status = TiledStatus::Settled;
                        let held = std::mem::take(&mut jm.fetch_held);
                        jm.fetches.clear();
                        self.credit(&client, held);
                    }
                    _ => return bad("cannot accept"),
                }
            }
            TiledTx::Fraud { job_id, kind } => {
                let job = self.tiled.get(job_id).ok_or("unknown job")?.clone();
                let TiledStatus::Posted { worker, bond, at, at_ts } = job.status.clone() else { return bad("no posted result") };
                if !self.timing.challenge_open(height, ts, at, at_ts) {
                    return bad("challenge window closed");
                }
                verify_tiled_fraud(&job, kind, p)?;
                let reward = bond / 2;
                self.credit(from, reward);
                self.burned += bond - reward;
                self.credit(&job.client, job.escrow_net + job.fetch_held);
                self.rep.entry(worker).or_default().slashed += 1;
                let jm = self.tiled.get_mut(job_id).unwrap();
                jm.status = TiledStatus::Slashed;
                jm.fetch_held = 0;
                jm.fetches.clear();
            }
            TiledTx::Fetch { job_id, which, i, j: jj, k, idx } => {
                let job = self.tiled.get(job_id).ok_or("unknown job")?;
                let TiledStatus::Posted { at, at_ts, .. } = job.status.clone() else { return bad("no posted result") };
                if job.client != *from {
                    return bad("only the client can request data");
                }
                let (bi, bk, bj) = job.dims;
                let n2 = (p.n * p.n) as u32;
                let in_range = *idx < n2 && (*i as usize) < bi && (*jj as usize) < bj && match *which { 0 => (*k as usize) < bk, 1 => *k == 0, _ => false };
                if !in_range {
                    return bad("index out of range");
                }
                if !self.timing.challenge_open(height, ts, at, at_ts) {
                    return bad("challenge window closed");
                }
                if job.fetches.len() >= MAX_PENDING_FETCHES || job.fetches.iter().any(|f| (f.which, f.i, f.j, f.k, f.idx) == (*which, *i, *jj, *k, *idx)) {
                    return bad("too many or duplicate pending requests");
                }
                self.debit(from, FETCH_FEE)?;
                let jm = self.tiled.get_mut(job_id).unwrap();
                jm.fetch_held += FETCH_FEE;
                jm.fetches.push(TFetch { which: *which, i: *i, j: *jj, k: *k, idx: *idx, deadline_ts: ts + self.timing.serve_secs });
            }
            TiledTx::Serve { job_id, which, i, j: jj, k, idx, row, path } => {
                let job = self.tiled.get(job_id).ok_or("unknown job")?;
                let TiledStatus::Posted { worker, bond, at, at_ts } = job.status.clone() else { return bad("no posted result") };
                if worker != *from {
                    return bad("only the worker can serve");
                }
                let Some(pos) = job.fetches.iter().position(|f| (f.which, f.i, f.j, f.k, f.idx) == (*which, *i, *jj, *k, *idx)) else { return bad("no such pending request") };
                let root = if *which == 0 { job.product_roots[job.pidx(*i as usize, *jj as usize, *k as usize)] } else { job.c_roots[*i as usize * job.dims.2 + *jj as usize] };
                if !verify_row(&root, *idx as usize / p.n, row, path, p) {
                    return bad("bad Merkle proof for the served entry");
                }
                let jm = self.tiled.get_mut(job_id).unwrap();
                jm.fetches.remove(pos);
                jm.fetch_held -= FETCH_FEE;
                let new_at_ts = (ts + FETCH_EXTEND_SECS).saturating_sub(self.timing.challenge_secs).max(at_ts);
                let new_at = (height + 2).saturating_sub(CHALLENGE_WINDOW).max(at);
                jm.status = TiledStatus::Posted { worker, bond, at: new_at, at_ts: new_at_ts };
                self.credit(&worker, FETCH_FEE);
            }
            TiledTx::SlashUnserved { job_id } => {
                let job = self.tiled.get(job_id).ok_or("unknown job")?.clone();
                let TiledStatus::Posted { worker, bond, .. } = job.status else { return bad("no posted result") };
                if !job.fetches.iter().any(|f| ts > f.deadline_ts) {
                    return bad("no expired data request");
                }
                let reward = bond / 2;
                self.credit(from, reward);
                self.burned += bond - reward;
                self.credit(&job.client, job.escrow_net + job.fetch_held);
                self.rep.entry(worker).or_default().slashed += 1;
                let jm = self.tiled.get_mut(job_id).unwrap();
                jm.fetch_held = 0;
                jm.fetches.clear();
                jm.status = TiledStatus::Slashed;
            }
            TiledTx::Settle { job_id } => {
                let (client, status, net, deadline, pending) = {
                    let j = self.tiled.get(job_id).ok_or("unknown job")?;
                    (j.client, j.status.clone(), j.escrow_net, j.deadline, !j.fetches.is_empty())
                };
                match status {
                    TiledStatus::Posted { .. } if pending => return bad("data requests pending: serve or slash first"),
                    TiledStatus::Posted { worker, bond, at, at_ts } if !self.timing.challenge_open(height, ts, at, at_ts) => {
                        self.credit(&worker, net + bond);
                        self.rep.entry(worker).or_default().done += 1;
                        self.tiled.get_mut(job_id).unwrap().status = TiledStatus::Settled;
                    }
                    TiledStatus::Open if height > deadline => {
                        self.credit(&client, net);
                        self.tiled.get_mut(job_id).unwrap().status = TiledStatus::Expired;
                    }
                    TiledStatus::Claimed { worker, bond } if height > deadline => {
                        self.rep.entry(worker).or_default().expired += 1;
                        self.burned += bond;
                        self.credit(&client, net);
                        self.tiled.get_mut(job_id).unwrap().status = TiledStatus::Expired;
                    }
                    _ => return bad("cannot settle yet"),
                }
            }
        }
        Ok(())
    }

    pub(crate) fn tiled_held(&self) -> u64 {
        self.tiled
            .values()
            .map(|j| match &j.status {
                TiledStatus::Open => j.escrow_net,
                TiledStatus::Claimed { bond, .. } => j.escrow_net + bond,
                TiledStatus::Posted { bond, .. } => j.escrow_net + bond + j.fetch_held,
                _ => 0,
            })
            .sum()
    }
}

fn verify_tiled_fraud(job: &TiledJob, kind: &TiledFraudKind, p: &WorkParams) -> Result<(), String> {
    let n = p.n;
    match kind {
        TiledFraudKind::Product { i, j, k, r, c, a_row, a_path, b_col, b_path, z_row, z_path } => {
            let (iu, ju, ku, ru, cu) = (*i as usize, *j as usize, *k as usize, *r as usize, *c as usize);
            let (ra, rb) = job.block_roots(iu, ju, ku).ok_or("block index out of range")?;
            if ru >= n || cu >= n || a_row.len() != n || b_col.len() != n {
                return bad("malformed fraud proof");
            }
            if a_row.iter().chain(b_col.iter()).any(|v| (*v as i32).abs() > VAL_MAX) {
                return bad("values out of range");
            }
            if !verify_proof(&ra, leaf_hash(ru as u64, &i8_bytes(a_row)), ru, a_path, p.depth()) {
                return bad("bad A opening");
            }
            if !verify_proof(&rb, leaf_hash(cu as u64, &i8_bytes(b_col)), cu, b_path, p.depth()) {
                return bad("bad B opening");
            }
            let zr = job.product_roots[job.pidx(iu, ju, ku)];
            if !verify_row(&zr, ru, z_row, z_path, p) {
                return bad("bad product opening");
            }
            let dot: i32 = a_row.iter().zip(b_col).map(|(a, b)| *a as i32 * *b as i32).sum();
            if dot == z_row[cu] {
                return bad("entry is correct: not a fraud");
            }
            Ok(())
        }
        TiledFraudKind::Sum { i, j, r, c, z_rows, z_paths, c_row, c_path } => {
            let (iu, ju, ru, cu) = (*i as usize, *j as usize, *r as usize, *c as usize);
            let bk = job.dims.1;
            if iu >= job.dims.0 || ju >= job.dims.2 || ru >= n || cu >= n || z_rows.len() != bk || z_paths.len() != bk {
                return bad("malformed fraud proof");
            }
            for k in 0..bk {
                let zr = job.product_roots[job.pidx(iu, ju, k)];
                if !verify_row(&zr, ru, &z_rows[k], &z_paths[k], p) {
                    return bad("bad product opening");
                }
            }
            let cr = job.c_roots[iu * job.dims.2 + ju];
            if !verify_row(&cr, ru, c_row, c_path, p) {
                return bad("bad sum opening");
            }
            let s: i64 = z_rows.iter().map(|z| z[cu] as i64).sum();
            if s == c_row[cu] as i64 {
                return bad("sum is correct: not a fraud");
            }
            Ok(())
        }
    }
}

/// Deterministic demonstration matrices (values in [-64,64]): A (I*n x K*n), B (K*n x J*n).
pub fn demo_matrices(n: usize, dims: (usize, usize, usize), seed: u64) -> (Vec<i8>, Vec<i8>) {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);
    let (bi, bk, bj) = dims;
    let a = (0..bi * n * bk * n).map(|_| rng.gen_range(-64..=64) as i8).collect();
    let b = (0..bk * n * bj * n).map(|_| rng.gen_range(-64..=64) as i8).collect();
    (a, b)
}
