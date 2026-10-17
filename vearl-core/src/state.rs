//! State: accounts + escrowed compute market (Fee-Bound Useful Work).
//!
//! Principle: the SECURITY reward (base_reward) is identical for every valid block,
//! so the network works with an empty market. USEFULNESS is only rewarded through a rebate
//! bounded by what was actually BURNED by a client (rebate_left <= burn):
//! fabricating a fake useful job costs more (burn) than it can bring back (rebate).
use crate::crypto::*;
use crate::mlp::MlpJob;
use crate::prog::ProgJob;
use crate::tiled::TiledJob;
use crate::types::*;
use crate::work::*;
use std::collections::BTreeMap;

pub const COIN: u64 = 100_000_000;
pub const BLOCK_TIME: u64 = 10;
pub const CHALLENGE_WINDOW: u64 = 10; // blocks
/// The window in blocks is not enough: on a fast network (or with a miner chaining blocks) 10 blocks
/// can pass in under a second. A duration in timestamp seconds is ALSO required. Timestamps are
/// bounded by the validators' clock (<= now + 60 s), so this duration cannot be "consumed"
/// by a burst of blocks: at least CHALLENGE_SECS - 60 s of real time elapse.
pub const CHALLENGE_SECS: u64 = 300;

/// Consensus durations in timestamp seconds, GENESIS PARAMETERS (tests shorten them; in production
/// they must stay long compared to the tolerable clock drift of 60 s).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Timing {
    pub challenge_secs: u64,
    pub serve_secs: u64,
}

impl Default for Timing {
    fn default() -> Self {
        Timing { challenge_secs: CHALLENGE_SECS, serve_secs: SERVE_SECS }
    }
}

impl Timing {
    pub fn challenge_open(&self, height: u64, ts: u64, at: u64, at_ts: u64) -> bool {
        !(height > at + CHALLENGE_WINDOW && ts >= at_ts + self.challenge_secs)
    }
}

/// The dispute window is open as long as one of the two durations has not elapsed.
pub fn challenge_open(height: u64, ts: u64, at: u64, at_ts: u64) -> bool {
    !(height > at + CHALLENGE_WINDOW && ts >= at_ts + CHALLENGE_SECS)
}
pub const BURN_BPS: u64 = 1_000; // 10 % of the escrow burned at job creation
pub const BOND_BPS: u64 = 1_000; // worker bond = 10 % of the net escrow
pub const MIN_ESCROW: u64 = COIN; // minimum price of a job (testnet)
/// Emission. The schedule is defined per YEAR and per second, not per block: 21,024,000 VRL in the first year, halved every
/// 365 days (3,153,600 blocks of 10 s). The reward of a block is therefore 20 VRL x 10 s / 30 s, ROUNDED DOWN to the unit so that
/// no year ever issues more than its nominal amount (6.66666666 VRL; the 31 years of halving issue 42,047,999.45 VRL, never more
/// than the 42,048,000 of the published cap).
pub const R0: u64 = 20 * COIN * BLOCK_TIME / 30;
pub const HALVING: u64 = 365 * 86_400 / BLOCK_TIME;
/// The floor of the testnet: 0.5 VRL per 30 s, i.e. 1/6 VRL per block of 10 s (mainnet plan: no floor, `docs/04`).
pub const TAIL: u64 = COIN / 2 * BLOCK_TIME / 30;
/// Native-work matrices change every 3,000 s (50 minutes), whatever the block time.
pub const EPOCH_LEN: u64 = 3_000 / BLOCK_TIME;
/// After the end of the auction, the winner has this delay (timestamp seconds) to claim; after that any worker
/// can claim at the full price.
pub const AUCTION_GRACE_SECS: u64 = 60;
pub const MAX_AUCTION_SECS: u64 = 3_600;
/// Data availability: a client can demand entries of the result ON-CHAIN. The worker has SERVE_SECS to
/// answer (with the Merkle path against c_root), otherwise it is slashed. Each request is paid by the client to the
/// worker if it answers, refunded to the client otherwise.
pub const FETCH_FEE: u64 = COIN / 20;
pub const SERVE_SECS: u64 = 120;
pub const MAX_PENDING_FETCHES: usize = 8;
pub const MAX_SERVED_KEPT: usize = 64;
/// After an answer, the dispute window stays open for at least this long (so that the client can verify).
pub const FETCH_EXTEND_SECS: u64 = 60;

pub fn base_reward(height: u64) -> u64 {
    (R0 >> (height / HALVING).min(63)).max(TAIL)
}

pub fn bps(x: u64, b: u64) -> u64 {
    ((x as u128) * (b as u128) / 10_000) as u64
}

pub fn account_count_leaf(count: usize) -> Hash {
    leaf_hash(0, &[b"count".as_slice(), &(count as u64).to_le_bytes()].concat())
}

pub fn account_leaf(i: usize, pk: &PubKey, balance: u64, nonce: u64) -> Hash {
    leaf_hash(i as u64 + 1, &[pk.as_slice(), &balance.to_le_bytes(), &nonce.to_le_bytes()].concat())
}

/// Verifies that an account (balance, nonce) appears at index `index` in an `accounts_root` of `count` accounts.
pub fn verify_account_proof(root: &Hash, pk: &PubKey, balance: u64, nonce: u64, index: usize, count: usize, path: &[Hash]) -> bool {
    index < count && verify_proof(root, account_leaf(index, pk, balance, nonce), index + 1, path, depth_for(count + 1))
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Account {
    pub balance: u64,
    pub nonce: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum JobStatus {
    Open,
    Claimed { worker: PubKey, bond: u64 },
    Posted { worker: PubKey, bond: u64, c_root: Hash, at: u64, at_ts: u64 },
    Settled,
    Slashed,
    Expired,
}

#[derive(Clone, Debug)]
pub struct Job {
    pub client: PubKey,
    pub a_root: Hash,
    pub b_root: Hash,
    pub escrow_net: u64,
    pub deadline: u64,
    pub status: JobStatus,
    pub rebate_left: u64,
    /// 0 = no auction; otherwise end of the auction (timestamp seconds).
    pub auction_end_ts: u64,
    pub best_bid: Option<(PubKey, u64)>,
    pub fetches: Vec<PendingFetch>,
    pub fetch_held: u64,
    /// Latest served entries (i, j, value, path): the client uses them to verify or prove a fraud.
    /// (i, j, served row i, path): the requested value is `row[j]`.
    pub served: Vec<(u32, u32, Vec<i32>, Vec<Hash>)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PendingFetch {
    pub i: u32,
    pub j: u32,
    pub deadline_ts: u64,
}

/// History of a worker (informational: clients can use it to choose; the protocol does not enforce it).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Reputation {
    pub done: u32,
    pub slashed: u32,
    pub expired: u32,
}

#[derive(Clone, Debug, Default)]
pub struct State {
    pub accounts: BTreeMap<PubKey, Account>,
    pub jobs: BTreeMap<JobId, Job>,
    pub mlps: BTreeMap<JobId, MlpJob>,
    pub tiled: BTreeMap<JobId, TiledJob>,
    pub progs: BTreeMap<JobId, ProgJob>,
    pub l2: BTreeMap<JobId, crate::l2::L2Job>,
    pub rep: BTreeMap<PubKey, Reputation>,
    pub timing: Timing,
    pub minted: u64,
    pub burned: u64,
}

impl State {
    /// Leaves of the account tree: leaf 0 commits the NUMBER of accounts, leaf i+1 the i-th live account (non-zero balance or nonce)
    /// in public key order. Two identical states have the same root; this is only a root of the ACCOUNTS (not of the jobs).
    fn account_leaves(&self) -> Vec<Hash> {
        let live: Vec<(&PubKey, &Account)> = self.accounts.iter().filter(|(_, a)| a.balance != 0 || a.nonce != 0).collect();
        let mut leaves = vec![account_count_leaf(live.len())];
        leaves.extend(live.iter().enumerate().map(|(i, (pk, a))| account_leaf(i, pk, a.balance, a.nonce)));
        leaves
    }

    pub fn accounts_root(&self) -> Hash {
        MerkleTree::root_of(self.account_leaves())
    }

    /// Proof of a live account: (index in the account list, number of accounts, Merkle path).
    pub fn accounts_proof(&self, pk: &PubKey) -> Option<(usize, usize, Vec<Hash>)> {
        let leaves = self.account_leaves();
        let count = leaves.len() - 1;
        let live: Vec<&PubKey> = self.accounts.iter().filter(|(_, a)| a.balance != 0 || a.nonce != 0).map(|(k, _)| k).collect();
        let i = live.iter().position(|k| *k == pk)?;
        Some((i, count, MerkleTree::new(leaves).proof(i + 1)))
    }

    pub fn balance(&self, pk: &PubKey) -> u64 {
        self.accounts.get(pk).map(|a| a.balance).unwrap_or(0)
    }
    pub fn nonce(&self, pk: &PubKey) -> u64 {
        self.accounts.get(pk).map(|a| a.nonce).unwrap_or(0)
    }
    pub fn reputation(&self, pk: &PubKey) -> Reputation {
        self.rep.get(pk).cloned().unwrap_or_default()
    }

    #[allow(clippy::too_many_arguments)]
    fn submit_job(&mut self, from: &PubKey, nonce: u64, a_root: &Hash, b_root: &Hash, escrow: u64, deadline: u64, auction_secs: u64, height: u64, ts: u64) -> Result<(), String> {
        if escrow < MIN_ESCROW {
            return Err("escrow below floor price".into());
        }
        if deadline <= height {
            return Err("deadline in the past".into());
        }
        if auction_secs > MAX_AUCTION_SECS {
            return Err("auction too long".into());
        }
        let id = job_id(from, nonce, a_root, b_root);
        if self.jobs.contains_key(&id) {
            return Err("duplicate job".into());
        }
        self.debit(from, escrow)?;
        let burn = bps(escrow, BURN_BPS);
        self.burned += burn;
        self.jobs.insert(
            id,
            Job {
                client: *from,
                a_root: *a_root,
                b_root: *b_root,
                escrow_net: escrow - burn,
                deadline,
                status: JobStatus::Open,
                rebate_left: burn, // the usefulness rebate can never exceed what was burned
                auction_end_ts: if auction_secs > 0 { ts + auction_secs } else { 0 },
                best_bid: None,
                fetches: vec![],
                fetch_held: 0,
                served: vec![],
            },
        );
        Ok(())
    }

    pub(crate) fn credit(&mut self, pk: &PubKey, amt: u64) {
        self.accounts.entry(*pk).or_default().balance += amt;
    }
    pub(crate) fn debit(&mut self, pk: &PubKey, amt: u64) -> Result<(), String> {
        let a = self.accounts.entry(*pk).or_default();
        a.balance = a.balance.checked_sub(amt).ok_or("insufficient balance")?;
        Ok(())
    }

    /// Total value held (balances + escrows + bonds). Invariant:
    /// total_value + burned == genesis_supply + minted.
    pub fn total_value(&self) -> u64 {
        let bal: u64 = self.accounts.values().map(|a| a.balance).sum();
        let held: u64 = self
            .jobs
            .values()
            .map(|j| {
                j.fetch_held
                    + match &j.status {
                        JobStatus::Open => j.escrow_net,
                        JobStatus::Claimed { bond, .. } | JobStatus::Posted { bond, .. } => j.escrow_net + bond,
                        _ => 0,
                    }
            })
            .sum();
        bal + held + self.mlp_held() + self.tiled_held() + self.prog_held() + self.l2_held()
    }

    pub fn apply_tx(&mut self, stx: &SignedTx, height: u64, ts: u64, miner: &PubKey, p: &WorkParams) -> Result<(), String> {
        let tx = &stx.tx;
        let msg = bincode::serialize(tx).map_err(|e| e.to_string())?;
        if !verify_sig(&tx.from, &msg, &stx.sig) {
            return Err("bad signature".into());
        }
        {
            let a = self.accounts.entry(tx.from).or_default();
            if a.nonce != tx.nonce {
                return Err(format!("bad nonce (expected {}, got {})", a.nonce, tx.nonce));
            }
            a.nonce += 1;
        }
        self.debit(&tx.from, tx.fee)?;
        self.credit(miner, tx.fee);

        match &tx.kind {
            TxKind::Transfer { to, amount } => {
                self.debit(&tx.from, *amount)?;
                self.credit(to, *amount);
            }
            TxKind::SubmitJob { a_root, b_root, escrow, deadline } => {
                self.submit_job(&tx.from, tx.nonce, a_root, b_root, *escrow, *deadline, 0, height, ts)?;
            }
            TxKind::SubmitJobAuction { a_root, b_root, escrow, deadline, auction_secs } => {
                if *auction_secs == 0 {
                    return Err("auction_secs must be > 0".into());
                }
                self.submit_job(&tx.from, tx.nonce, a_root, b_root, *escrow, *deadline, *auction_secs, height, ts)?;
            }
            TxKind::Bid { job_id, price } => {
                let j = self.jobs.get_mut(job_id).ok_or("unknown job")?;
                if j.status != JobStatus::Open || j.auction_end_ts == 0 || ts >= j.auction_end_ts {
                    return Err("no open auction".into());
                }
                if *price == 0 || *price > j.escrow_net {
                    return Err("bad price".into());
                }
                if let Some((_, best)) = j.best_bid {
                    if *price >= best {
                        return Err("bid is not lower than the best bid".into());
                    }
                }
                j.best_bid = Some((tx.from, *price));
            }
            TxKind::ClaimJob { job_id } => {
                let (client, net, status, deadline, end, best) = {
                    let j = self.jobs.get(job_id).ok_or("unknown job")?;
                    (j.client, j.escrow_net, j.status.clone(), j.deadline, j.auction_end_ts, j.best_bid)
                };
                if status != JobStatus::Open || height > deadline {
                    return Err("job not claimable".into());
                }
                let mut price = net;
                if end > 0 {
                    if ts < end {
                        return Err("auction still open".into());
                    }
                    match best {
                        Some((w, pr)) if w == tx.from => price = pr,
                        _ if ts >= end + AUCTION_GRACE_SECS || best.is_none() => {} // full price
                        _ => return Err("only the winning bidder can claim during the grace period".into()),
                    }
                }
                let bond = bps(price, BOND_BPS);
                self.debit(&tx.from, bond)?;
                if price < net {
                    self.credit(&client, net - price); // the client recovers the difference between escrow and the retained price
                }
                let j = self.jobs.get_mut(job_id).unwrap();
                j.escrow_net = price;
                j.status = JobStatus::Claimed { worker: tx.from, bond };
            }
            TxKind::PostResult { job_id, c_root } => {
                let j = self.jobs.get_mut(job_id).ok_or("unknown job")?;
                match j.status.clone() {
                    JobStatus::Claimed { worker, bond } if worker == tx.from && height <= j.deadline => {
                        j.status = JobStatus::Posted { worker, bond, c_root: *c_root, at: height, at_ts: ts };
                    }
                    _ => return Err("cannot post result".into()),
                }
            }
            TxKind::Accept { job_id } => {
                let (client, status, net) = {
                    let j = self.jobs.get(job_id).ok_or("unknown job")?;
                    (j.client, j.status.clone(), j.escrow_net)
                };
                match status {
                    JobStatus::Posted { worker, bond, .. } if client == tx.from => {
                        self.credit(&worker, net + bond);
                        self.rep.entry(worker).or_default().done += 1;
                        let j = self.jobs.get_mut(job_id).unwrap();
                        let held = std::mem::take(&mut j.fetch_held);
                        j.fetches.clear();
                        j.status = JobStatus::Settled;
                        self.credit(&client, held);
                    }
                    _ => return Err("cannot accept".into()),
                }
            }
            TxKind::Fraud(f) => {
                let (j, status) = {
                    let j = self.jobs.get(&f.job_id).ok_or("unknown job")?;
                    (j.clone(), j.status.clone())
                };
                let JobStatus::Posted { worker: accused, bond, c_root, at, at_ts } = status else { return Err("no posted result".into()) };
                if !self.timing.challenge_open(height, ts, at, at_ts) {
                    return Err("challenge window closed".into());
                }
                verify_fraud(&j, &c_root, f, p)?;
                // bond: half to the challenger, half burned; the client is refunded.
                let reward = bond / 2;
                self.credit(&tx.from, reward);
                self.burned += bond - reward;
                self.credit(&j.client, j.escrow_net + j.fetch_held);
                self.rep.entry(accused).or_default().slashed += 1;
                let jm = self.jobs.get_mut(&f.job_id).unwrap();
                jm.fetch_held = 0;
                jm.fetches.clear();
                jm.status = JobStatus::Slashed;
            }
            TxKind::Fetch { job_id, i, j } => {
                let n = p.n as u32;
                if *i >= n || *j >= n {
                    return Err("index out of range".into());
                }
                let jb = self.jobs.get(job_id).ok_or("unknown job")?;
                let JobStatus::Posted { at, at_ts, .. } = jb.status.clone() else { return Err("no posted result".into()) };
                if jb.client != tx.from {
                    return Err("only the client can request data".into());
                }
                if !self.timing.challenge_open(height, ts, at, at_ts) {
                    return Err("challenge window closed".into());
                }
                if jb.fetches.len() >= MAX_PENDING_FETCHES || jb.fetches.iter().any(|f| f.i == *i && f.j == *j) {
                    return Err("too many or duplicate pending requests".into());
                }
                self.debit(&tx.from, FETCH_FEE)?;
                let jb = self.jobs.get_mut(job_id).unwrap();
                jb.fetch_held += FETCH_FEE;
                jb.fetches.push(PendingFetch { i: *i, j: *j, deadline_ts: ts + self.timing.serve_secs });
            }
            TxKind::Serve { job_id, i, j, row, path } => {
                let jb = self.jobs.get(job_id).ok_or("unknown job")?;
                let JobStatus::Posted { worker, bond, c_root, at, at_ts } = jb.status.clone() else { return Err("no posted result".into()) };
                if worker != tx.from {
                    return Err("only the worker can serve".into());
                }
                let Some(pos) = jb.fetches.iter().position(|f| f.i == *i && f.j == *j) else { return Err("no such pending request".into()) };
                if !verify_row(&c_root, *i as usize, row, path, p) {
                    return Err("bad Merkle proof for the served entry".into());
                }
                let jb = self.jobs.get_mut(job_id).unwrap();
                jb.fetches.remove(pos);
                jb.fetch_held -= FETCH_FEE;
                jb.served.push((*i, *j, row.clone(), path.clone()));
                if jb.served.len() > MAX_SERVED_KEPT {
                    jb.served.remove(0);
                }
                // the client must have time to verify the served value
                let new_at_ts = (ts + FETCH_EXTEND_SECS).saturating_sub(self.timing.challenge_secs).max(at_ts);
                let new_at = (height + 2).saturating_sub(CHALLENGE_WINDOW).max(at);
                jb.status = JobStatus::Posted { worker, bond, c_root, at: new_at, at_ts: new_at_ts };
                self.credit(&worker, FETCH_FEE);
            }
            TxKind::SlashUnserved { job_id } => {
                let jb = self.jobs.get(job_id).ok_or("unknown job")?.clone();
                let JobStatus::Posted { worker, bond, .. } = jb.status else { return Err("no posted result".into()) };
                if !jb.fetches.iter().any(|f| ts > f.deadline_ts) {
                    return Err("no expired data request".into());
                }
                // same penalty as a proven fraud: half of the bond to whoever reported, half burned; client refunded
                let reward = bond / 2;
                self.credit(&tx.from, reward);
                self.burned += bond - reward;
                self.credit(&jb.client, jb.escrow_net + jb.fetch_held);
                self.rep.entry(worker).or_default().slashed += 1;
                let jm = self.jobs.get_mut(job_id).unwrap();
                jm.fetch_held = 0;
                jm.fetches.clear();
                jm.status = JobStatus::Slashed;
            }
            TxKind::Prog(m) => self.apply_prog(&tx.from, tx.nonce, m, height, ts, p)?,
            TxKind::L2(m) => self.apply_l2(&tx.from, tx.nonce, m, height, ts, p)?,
            TxKind::Tiled(m) => self.apply_tiled(&tx.from, tx.nonce, m, height, ts, p)?,
            TxKind::Mlp(m) => self.apply_mlp(&tx.from, tx.nonce, m, height, ts, p)?,
            TxKind::Settle { job_id } => {
                let (client, status, net, deadline, pending) = {
                    let j = self.jobs.get(job_id).ok_or("unknown job")?;
                    (j.client, j.status.clone(), j.escrow_net, j.deadline, !j.fetches.is_empty())
                };
                match status {
                    JobStatus::Posted { .. } if pending => return Err("data requests pending: serve or slash first".into()),
                    JobStatus::Posted { worker, bond, at, at_ts, .. } if !self.timing.challenge_open(height, ts, at, at_ts) => {
                        self.credit(&worker, net + bond);
                        self.rep.entry(worker).or_default().done += 1;
                        self.jobs.get_mut(job_id).unwrap().status = JobStatus::Settled;
                    }
                    JobStatus::Open if height > deadline => {
                        self.credit(&client, net);
                        self.jobs.get_mut(job_id).unwrap().status = JobStatus::Expired;
                    }
                    JobStatus::Claimed { worker, bond } if height > deadline => {
                        self.rep.entry(worker).or_default().expired += 1;
                        self.burned += bond; // the worker delivered nothing: bond burned
                        self.credit(&client, net);
                        self.jobs.get_mut(job_id).unwrap().status = JobStatus::Expired;
                    }
                    _ => return Err("cannot settle yet".into()),
                }
            }
        }
        Ok(())
    }

    /// Block reward: security (identical for all) + usefulness rebate bounded by the burn.
    pub fn pay_block_reward(&mut self, height: u64, miner: &PubKey, class: &WorkClass) -> u64 {
        let base = base_reward(height);
        let mut bonus = 0;
        if let WorkClass::Job(id) = class {
            if let Some(j) = self.jobs.get_mut(id) {
                bonus = (base / 2).min(j.rebate_left);
                j.rebate_left -= bonus;
            }
        }
        if let WorkClass::Op { job, .. } = class {
            if let Some(j) = self.progs.get_mut(job) {
                bonus = (base / 2).min(j.rebate_left);
                j.rebate_left -= bonus;
            }
        }
        if let WorkClass::Product { job, .. } = class {
            if let Some(j) = self.tiled.get_mut(job) {
                bonus = (base / 2).min(j.rebate_left);
                j.rebate_left -= bonus;
            }
        }
        if let WorkClass::Layer { job, .. } = class {
            if let Some(j) = self.mlps.get_mut(job) {
                bonus = (base / 2).min(j.rebate_left);
                j.rebate_left -= bonus;
            }
        }
        self.minted += base + bonus;
        self.credit(miner, base + bonus);
        base + bonus
    }
}

fn verify_fraud(job: &Job, c_root: &Hash, f: &FraudProof, p: &WorkParams) -> Result<(), String> {
    let n = p.n;
    let (i, j) = (f.i as usize, f.j as usize);
    if i >= n || j >= n || f.a_row.len() != n || f.b_col.len() != n {
        return Err("malformed fraud proof".into());
    }
    if f.a_row.iter().chain(f.b_col.iter()).any(|v| (*v as i32).abs() > VAL_MAX) {
        return Err("values out of range".into());
    }
    if !verify_proof(&job.a_root, leaf_hash(i as u64, &i8_bytes(&f.a_row)), i, &f.a_path, p.depth()) {
        return Err("bad A opening".into());
    }
    if !verify_proof(&job.b_root, leaf_hash(j as u64, &i8_bytes(&f.b_col)), j, &f.b_path, p.depth()) {
        return Err("bad B opening".into());
    }
    if !verify_row(c_root, i, &f.c_row, &f.c_path, p) {
        return Err("bad C opening".into());
    }
    let dot: i32 = f.a_row.iter().zip(&f.b_col).map(|(a, b)| *a as i32 * *b as i32).sum();
    if dot == f.c_row[j] {
        return Err("entry is correct: not a fraud".into());
    }
    Ok(())
}
