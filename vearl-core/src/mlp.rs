//! Quantised INFERENCE jobs (network of L layers): the first "useful" kind of computation beyond an isolated GEMM.
//!
//!   Z_l = W_l * X_{l-1}          (int32, exact; n x n GEMM, THIS is the computation that is mined with noise)
//!   X_l = act_l(Z_l)             (int8; act = arithmetic shift + saturation, ReLU for the hidden layers)
//!
//! The client commits X_0 (columns) and the W_l (rows). The worker publishes, layer by layer, the root of Z_l
//! (one leaf per entry) and the root of X_l (one leaf per column: X_l serves as the B operand of the next layer).
//! Verification: Freivalds on the client side for each GEMM (O(n^2)), elementary test for the activations, and two kinds of
//! element-wise FRAUD PROOFS (verifiable on-chain in O(n)):
//!   - GEMM: Z_l[i][j] != <row i of W_l, column j of X_{l-1}>
//!   - Act:  X_l[i][j] != act_l(Z_l[i][j])
//! Limits (see docs/02): integers only, sizes = n of the protocol, a single worker, no guaranteed availability of the Z/X.
use crate::crypto::*;
use crate::state::*;
use crate::types::*;
use crate::work::*;
use serde::{Deserialize, Serialize};

pub const MAX_LAYERS: usize = 8;
pub const MAX_SHIFT: u8 = 24;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum MlpFraudKind {
    Gemm { i: u32, j: u32, a_row: Vec<i8>, a_path: Vec<Hash>, b_col: Vec<i8>, b_path: Vec<Hash>, z_row: Vec<i32>, z_path: Vec<Hash> },
    Act { i: u32, j: u32, z_row: Vec<i32>, z_path: Vec<Hash>, x_col: Vec<i8>, x_path: Vec<Hash> },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum MlpTx {
    Submit { x0_root: Hash, w_roots: Vec<Hash>, shifts: Vec<u8>, escrow: u64, deadline: u64 },
    Claim { job_id: JobId },
    PostLayer { job_id: JobId, layer: u8, z_root: Hash, x_root: Hash },
    Accept { job_id: JobId },
    Fraud { job_id: JobId, layer: u8, kind: MlpFraudKind },
    Settle { job_id: JobId },
}

pub fn mlp_id(client: &PubKey, nonce: u64, x0_root: &Hash, w_roots: &[Hash]) -> JobId {
    let nb = nonce.to_le_bytes();
    let mut parts: Vec<&[u8]> = vec![client, &nb, x0_root];
    for w in w_roots {
        parts.push(w);
    }
    h("vearl/mlp/v1", &parts)
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayerPost {
    pub z_root: Hash,
    pub x_root: Hash,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MlpStatus {
    Open,
    Working { worker: PubKey, bond: u64 },
    Complete { worker: PubKey, bond: u64, at: u64, at_ts: u64 },
    Settled,
    Slashed,
    Expired,
}

#[derive(Clone, Debug)]
pub struct MlpJob {
    pub client: PubKey,
    pub x0_root: Hash,
    pub w_roots: Vec<Hash>,
    pub shifts: Vec<u8>,
    pub escrow_net: u64,
    pub deadline: u64,
    pub status: MlpStatus,
    pub rebate_left: u64,
    pub layers: Vec<LayerPost>,
}

impl MlpJob {
    pub fn is_hidden(&self, layer: usize) -> bool {
        layer + 1 < self.w_roots.len()
    }
    /// (root of A = W_l, root of B = X_{l-1}) for layer `l`, if it is already unlocked.
    pub fn layer_roots(&self, l: usize) -> Option<(Hash, Hash)> {
        if l >= self.w_roots.len() || l > self.layers.len() {
            return None;
        }
        let b = if l == 0 { self.x0_root } else { self.layers[l - 1].x_root };
        Some((self.w_roots[l], b))
    }
}

/// act_l(z): arithmetic shift then saturation. Hidden: ReLU + [0,64]; last: [-64,64].
pub fn act(z: i32, shift: u8, hidden: bool) -> i8 {
    let v = z >> shift;
    (if hidden { v.clamp(0, 64) } else { v.clamp(-64, 64) }) as i8
}

pub fn requantize(z: &[i32], shift: u8, hidden: bool) -> Vec<i8> {
    z.iter().map(|&x| act(x, shift, hidden)).collect()
}

/// Merkle roots of the matrices of a job (client side): X_0 by columns, each W_l by rows.
pub fn cols_root_of(m: &[i8], n: usize) -> Hash {
    cols_tree(m, n).root()
}
pub fn rows_root_of(m: &[i8], n: usize) -> Hash {
    rows_tree(m, n).root()
}

fn bad<T>(m: &str) -> Result<T, String> {
    Err(m.to_string())
}

impl State {
    pub(crate) fn apply_mlp(&mut self, from: &PubKey, nonce: u64, m: &MlpTx, height: u64, ts: u64, p: &WorkParams) -> Result<(), String> {
        match m {
            MlpTx::Submit { x0_root, w_roots, shifts, escrow, deadline } => {
                let l = w_roots.len();
                if l == 0 || l > MAX_LAYERS || shifts.len() != l || shifts.iter().any(|s| *s > MAX_SHIFT) {
                    return bad("bad layer spec");
                }
                if *escrow < MIN_ESCROW * l as u64 {
                    return bad("escrow below floor price (per layer)");
                }
                if *deadline <= height {
                    return bad("deadline in the past");
                }
                let id = mlp_id(from, nonce, x0_root, w_roots);
                if self.mlps.contains_key(&id) {
                    return bad("duplicate job");
                }
                self.debit(from, *escrow)?;
                let burn = bps(*escrow, BURN_BPS);
                self.burned += burn;
                self.mlps.insert(
                    id,
                    MlpJob {
                        client: *from,
                        x0_root: *x0_root,
                        w_roots: w_roots.clone(),
                        shifts: shifts.clone(),
                        escrow_net: escrow - burn,
                        deadline: *deadline,
                        status: MlpStatus::Open,
                        rebate_left: burn,
                        layers: vec![],
                    },
                );
            }
            MlpTx::Claim { job_id } => {
                let (net, status, deadline) = {
                    let j = self.mlps.get(job_id).ok_or("unknown job")?;
                    (j.escrow_net, j.status.clone(), j.deadline)
                };
                if status != MlpStatus::Open || height > deadline {
                    return bad("job not claimable");
                }
                let bond = bps(net, BOND_BPS);
                self.debit(from, bond)?;
                self.mlps.get_mut(job_id).unwrap().status = MlpStatus::Working { worker: *from, bond };
            }
            MlpTx::PostLayer { job_id, layer, z_root, x_root } => {
                let j = self.mlps.get_mut(job_id).ok_or("unknown job")?;
                let MlpStatus::Working { worker, bond } = j.status.clone() else { return bad("job is not in progress") };
                if worker != *from || height > j.deadline {
                    return bad("cannot post layer");
                }
                if *layer as usize != j.layers.len() || j.layers.len() >= j.w_roots.len() {
                    return bad("layers must be posted in order");
                }
                j.layers.push(LayerPost { z_root: *z_root, x_root: *x_root });
                if j.layers.len() == j.w_roots.len() {
                    j.status = MlpStatus::Complete { worker, bond, at: height, at_ts: ts };
                }
            }
            MlpTx::Accept { job_id } => {
                let (client, status, net) = {
                    let j = self.mlps.get(job_id).ok_or("unknown job")?;
                    (j.client, j.status.clone(), j.escrow_net)
                };
                match status {
                    MlpStatus::Complete { worker, bond, .. } if client == *from => {
                        self.credit(&worker, net + bond);
                        self.rep.entry(worker).or_default().done += 1;
                        self.mlps.get_mut(job_id).unwrap().status = MlpStatus::Settled;
                    }
                    _ => return bad("cannot accept"),
                }
            }
            MlpTx::Fraud { job_id, layer, kind } => {
                let job = self.mlps.get(job_id).ok_or("unknown job")?.clone();
                let (bond, accused) = match &job.status {
                    MlpStatus::Working { bond, worker } => (*bond, *worker),
                    MlpStatus::Complete { bond, at, at_ts, worker } => {
                        if !self.timing.challenge_open(height, ts, *at, *at_ts) {
                            return bad("challenge window closed");
                        }
                        (*bond, *worker)
                    }
                    _ => return bad("no result to challenge"),
                };
                verify_mlp_fraud(&job, *layer as usize, kind, p)?;
                let reward = bond / 2;
                self.credit(from, reward);
                self.burned += bond - reward;
                self.credit(&job.client, job.escrow_net);
                self.rep.entry(accused).or_default().slashed += 1;
                self.mlps.get_mut(job_id).unwrap().status = MlpStatus::Slashed;
            }
            MlpTx::Settle { job_id } => {
                let (client, status, net, deadline) = {
                    let j = self.mlps.get(job_id).ok_or("unknown job")?;
                    (j.client, j.status.clone(), j.escrow_net, j.deadline)
                };
                match status {
                    MlpStatus::Complete { worker, bond, at, at_ts } if !self.timing.challenge_open(height, ts, at, at_ts) => {
                        self.credit(&worker, net + bond);
                        self.rep.entry(worker).or_default().done += 1;
                        self.mlps.get_mut(job_id).unwrap().status = MlpStatus::Settled;
                    }
                    MlpStatus::Open if height > deadline => {
                        self.credit(&client, net);
                        self.mlps.get_mut(job_id).unwrap().status = MlpStatus::Expired;
                    }
                    MlpStatus::Working { bond, worker } if height > deadline => {
                        self.rep.entry(worker).or_default().expired += 1;
                        self.burned += bond; // the worker did not finish: bond burned
                        self.credit(&client, net);
                        self.mlps.get_mut(job_id).unwrap().status = MlpStatus::Expired;
                    }
                    _ => return bad("cannot settle yet"),
                }
            }
        }
        Ok(())
    }

    /// Value held by the inference jobs (escrow + bonds), for the supply invariant.
    pub(crate) fn mlp_held(&self) -> u64 {
        self.mlps
            .values()
            .map(|j| match &j.status {
                MlpStatus::Open => j.escrow_net,
                MlpStatus::Working { bond, .. } | MlpStatus::Complete { bond, .. } => j.escrow_net + bond,
                _ => 0,
            })
            .sum()
    }
}

fn verify_mlp_fraud(job: &MlpJob, layer: usize, kind: &MlpFraudKind, p: &WorkParams) -> Result<(), String> {
    let n = p.n;
    if layer >= job.layers.len() {
        return bad("layer not posted");
    }
    let z_root = job.layers[layer].z_root;
    match kind {
        MlpFraudKind::Gemm { i, j, a_row, a_path, b_col, b_path, z_row, z_path } => {
            let (i, j) = (*i as usize, *j as usize);
            if i >= n || j >= n || a_row.len() != n || b_col.len() != n {
                return bad("malformed fraud proof");
            }
            if a_row.iter().chain(b_col.iter()).any(|v| (*v as i32).abs() > VAL_MAX) {
                return bad("values out of range");
            }
            let (a_root, b_root) = job.layer_roots(layer).ok_or("layer roots unavailable")?;
            if !verify_proof(&a_root, leaf_hash(i as u64, &i8_bytes(a_row)), i, a_path, p.depth()) {
                return bad("bad W opening");
            }
            if !verify_proof(&b_root, leaf_hash(j as u64, &i8_bytes(b_col)), j, b_path, p.depth()) {
                return bad("bad X opening");
            }
            if !verify_row(&z_root, i, z_row, z_path, p) {
                return bad("bad Z opening");
            }
            let dot: i32 = a_row.iter().zip(b_col).map(|(a, b)| *a as i32 * *b as i32).sum();
            if dot == z_row[j] {
                return bad("entry is correct: not a fraud");
            }
            Ok(())
        }
        MlpFraudKind::Act { i, j, z_row, z_path, x_col, x_path } => {
            let (i, j) = (*i as usize, *j as usize);
            if i >= n || j >= n || x_col.len() != n {
                return bad("malformed fraud proof");
            }
            if !verify_row(&z_root, i, z_row, z_path, p) {
                return bad("bad Z opening");
            }
            if !verify_proof(&job.layers[layer].x_root, leaf_hash(j as u64, &i8_bytes(x_col)), j, x_path, p.depth()) {
                return bad("bad X opening");
            }
            if x_col[i] == act(z_row[j], job.shifts[layer], job.is_hidden(layer)) {
                return bad("activation is correct: not a fraud");
            }
            Ok(())
        }
    }
}

/// Deterministic demonstration network (X_0 in [0,64], weights in [-8,8]): used by the CLI and the network tests.
pub fn demo_net(n: usize, layers: usize, seed: u64) -> (Vec<i8>, Vec<Vec<i8>>) {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);
    let x0 = (0..n * n).map(|_| rng.gen_range(0..=64) as i8).collect();
    let ws = (0..layers).map(|_| (0..n * n).map(|_| rng.gen_range(-8..=8) as i8).collect()).collect();
    (x0, ws)
}

/// Quantisation shift of the demonstration network.
pub const DEMO_SHIFT: u8 = 9;
