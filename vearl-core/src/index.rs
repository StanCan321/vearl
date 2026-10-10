//! Chain index for the explorer: what the state alone cannot tell (when a job was submitted, claimed, posted, settled, who mined for it,
//! which fraud proofs were accepted) and per-day statistics, both rebuilt from the blocks.
//!
//! It is derived data, not consensus: it is built block by block as the chain grows, thrown away and rebuilt if the chain below it changes
//! (reorganisation), and costs nothing to the nodes that never serve the explorer API.
use crate::chain::Chain;
use crate::crypto::*;
use crate::l2::{l2_id, L2Tx};
use crate::mlp::{mlp_id, MlpTx};
use crate::prog::{prog_id, ProgTx};
use crate::state::*;
use crate::tiled::{tiled_id, TiledTx};
use crate::types::*;
use std::collections::{BTreeMap, HashMap, HashSet};

/// Newest events kept for the activity feed.
pub const RECENT_CAP: usize = 5000;

#[derive(Clone, Debug)]
pub struct Event {
    pub height: u64,
    pub ts: u64,
    /// `submitted`, `bid`, `claimed`, `posted`, `accepted`, `settled`, `fraud_proof`, `data_requested`, `data_served`, `slash_unserved`,
    /// `work_block` (a block mined on the matrices of the job).
    pub what: &'static str,
    pub by: PubKey,
    pub tx: Option<Hash>,
}

#[derive(Default, Clone)]
pub struct Day {
    pub blocks: u64,
    pub paid_blocks: u64,
    pub txs: u64,
    pub miners: HashSet<PubKey>,
    pub jobs_submitted: u64,
    pub jobs_closed: u64,
    pub fraud_proofs: u64,
    /// Transfers only, in base units.
    pub volume: u128,
    pub fees: u128,
    /// Base block rewards, in base units.
    pub rewards: u128,
}

#[derive(Default)]
pub struct Index {
    done: usize,
    last: Hash,
    pub jobs: HashMap<JobId, Vec<Event>>,
    pub days: BTreeMap<u64, Day>,
    /// Newest last; capped at `RECENT_CAP`.
    pub recent: Vec<(JobId, &'static str, Event)>,
}

/// The job a transaction is about, and what it does to it. `None` for transfers.
fn classify(t: &SignedTx) -> Option<(JobId, &'static str, &'static str)> {
    let (from, nonce) = (&t.tx.from, t.tx.nonce);
    Some(match &t.tx.kind {
        TxKind::Transfer { .. } => return None,
        TxKind::SubmitJob { a_root, b_root, .. } | TxKind::SubmitJobAuction { a_root, b_root, .. } => (job_id(from, nonce, a_root, b_root), "matrix_product", "submitted"),
        TxKind::Bid { job_id, .. } => (*job_id, "matrix_product", "bid"),
        TxKind::ClaimJob { job_id } => (*job_id, "matrix_product", "claimed"),
        TxKind::PostResult { job_id, .. } => (*job_id, "matrix_product", "posted"),
        TxKind::Accept { job_id } => (*job_id, "matrix_product", "accepted"),
        TxKind::Fraud(f) => (f.job_id, "matrix_product", "fraud_proof"),
        TxKind::Settle { job_id } => (*job_id, "matrix_product", "settled"),
        TxKind::Fetch { job_id, .. } => (*job_id, "matrix_product", "data_requested"),
        TxKind::Serve { job_id, .. } => (*job_id, "matrix_product", "data_served"),
        TxKind::SlashUnserved { job_id } => (*job_id, "matrix_product", "slash_unserved"),
        TxKind::Prog(m) => match m {
            ProgTx::Submit { x0_root, w_roots, ops, .. } => (prog_id(from, nonce, x0_root, w_roots, ops), "tensor_program", "submitted"),
            ProgTx::Claim { job_id } => (*job_id, "tensor_program", "claimed"),
            ProgTx::PostOp { job_id, .. } | ProgTx::PostBatch { job_id, .. } => (*job_id, "tensor_program", "posted"),
            ProgTx::Accept { job_id } => (*job_id, "tensor_program", "accepted"),
            ProgTx::Fraud { job_id, .. } => (*job_id, "tensor_program", "fraud_proof"),
            ProgTx::Settle { job_id } => (*job_id, "tensor_program", "settled"),
            ProgTx::Fetch { job_id, .. } => (*job_id, "tensor_program", "data_requested"),
            ProgTx::Serve { job_id, .. } => (*job_id, "tensor_program", "data_served"),
            ProgTx::SlashUnserved { job_id } => (*job_id, "tensor_program", "slash_unserved"),
        },
        TxKind::Tiled(m) => match m {
            TiledTx::Submit { dims, a_roots, b_roots, .. } => (tiled_id(from, nonce, *dims, a_roots, b_roots), "tiled_matrix_product", "submitted"),
            TiledTx::Claim { job_id } => (*job_id, "tiled_matrix_product", "claimed"),
            TiledTx::Post { job_id, .. } => (*job_id, "tiled_matrix_product", "posted"),
            TiledTx::Accept { job_id } => (*job_id, "tiled_matrix_product", "accepted"),
            TiledTx::Fraud { job_id, .. } => (*job_id, "tiled_matrix_product", "fraud_proof"),
            TiledTx::Settle { job_id } => (*job_id, "tiled_matrix_product", "settled"),
            TiledTx::Fetch { job_id, .. } => (*job_id, "tiled_matrix_product", "data_requested"),
            TiledTx::Serve { job_id, .. } => (*job_id, "tiled_matrix_product", "data_served"),
            TiledTx::SlashUnserved { job_id } => (*job_id, "tiled_matrix_product", "slash_unserved"),
        },
        TxKind::Mlp(m) => match m {
            MlpTx::Submit { x0_root, w_roots, .. } => (mlp_id(from, nonce, x0_root, w_roots), "inference", "submitted"),
            MlpTx::Claim { job_id } => (*job_id, "inference", "claimed"),
            MlpTx::PostLayer { job_id, .. } => (*job_id, "inference", "posted"),
            MlpTx::Accept { job_id } => (*job_id, "inference", "accepted"),
            MlpTx::Fraud { job_id, .. } => (*job_id, "inference", "fraud_proof"),
            MlpTx::Settle { job_id } => (*job_id, "inference", "settled"),
        },
        TxKind::L2(m) => match m {
            L2Tx::Submit { kernel, chunks, .. } => (l2_id(from, nonce, kernel, *chunks), "level2", "submitted"),
            L2Tx::Claim { job_id } => (*job_id, "level2", "claimed"),
            L2Tx::Post { job_id, .. } => (*job_id, "level2", "posted"),
            L2Tx::Accept { job_id } => (*job_id, "level2", "accepted"),
            L2Tx::Fetch { job_id, .. } => (*job_id, "level2", "data_requested"),
            L2Tx::Serve { job_id, .. } => (*job_id, "level2", "data_served"),
            L2Tx::SlashUnserved { job_id } => (*job_id, "level2", "slash_unserved"),
            L2Tx::Fraud { job_id, .. } => (*job_id, "level2", "fraud_proof"),
            L2Tx::Settle { job_id } => (*job_id, "level2", "settled"),
        },
    })
}

impl Index {
    /// Brings the index up to the chain: only the new blocks are read; if the chain below the indexed part changed, everything is rebuilt.
    pub fn sync(&mut self, c: &Chain) {
        let len = c.blocks.len();
        let consistent = self.done <= len && (self.done == 0 || c.blocks[self.done - 1].header.hash() == self.last);
        if !consistent {
            *self = Index::default();
        }
        for b in &c.blocks[self.done..] {
            self.add(b);
        }
        self.done = len;
        if let Some(b) = c.blocks.last() {
            self.last = b.header.hash();
        }
    }

    /// Number of blocks indexed.
    pub fn len(&self) -> usize {
        self.done
    }

    fn add(&mut self, b: &Block) {
        let hd = &b.header;
        let day = self.days.entry(hd.timestamp / 86_400).or_default();
        day.blocks += 1;
        day.txs += b.txs.len() as u64;
        day.miners.insert(hd.miner);
        day.rewards += base_reward(hd.height) as u128;
        let for_job = match &hd.class {
            WorkClass::Native => None,
            WorkClass::Job(id) => Some(*id),
            WorkClass::Layer { job, .. } | WorkClass::Op { job, .. } | WorkClass::Product { job, .. } => Some(*job),
        };
        if let Some(j) = for_job {
            day.paid_blocks += 1;
            let ev = Event { height: hd.height, ts: hd.timestamp, what: "work_block", by: hd.miner, tx: None };
            self.jobs.entry(j).or_default().push(ev);
        }
        for t in &b.txs {
            day.fees += t.tx.fee as u128;
            if let TxKind::Transfer { amount, .. } = &t.tx.kind {
                day.volume += *amount as u128;
            }
            let Some((job, _kind, what)) = classify(t) else { continue };
            match what {
                "submitted" => day.jobs_submitted += 1,
                "accepted" | "settled" | "fraud_proof" | "slash_unserved" => day.jobs_closed += 1,
                _ => {}
            }
            if what == "fraud_proof" {
                day.fraud_proofs += 1;
            }
            let ev = Event { height: hd.height, ts: hd.timestamp, what, by: t.tx.from, tx: Some(t.id()) };
            self.jobs.entry(job).or_default().push(ev.clone());
            if !matches!(what, "data_requested" | "data_served" | "bid") {
                self.recent.push((job, _kind, ev));
            }
        }
        if self.recent.len() > RECENT_CAP {
            let extra = self.recent.len() - RECENT_CAP;
            self.recent.drain(..extra);
        }
    }
}
