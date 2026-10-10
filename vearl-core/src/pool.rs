//! VEARL mining pool, `vpool/1` protocol (specification: docs/16-pool-protocol.md).
//!
//! The pool is the block "miner" (`header.miner` = pool key): it builds JOBS (parent, transactions, timestamp),
//! hands each worker a disjoint `extra_nonce` range and a SHARE target (vardiff), verifies each share by
//! recomputing the jackpot of the announced tile from its own native matrices, assembles and submits the block when a share
//! reaches the network target, accounts the shares in PPLNS and pays the confirmed rewards with on-chain transfers.
//!
//! Assumptions: the pool runs a full node (process `vearl-pool`); the pool's work is Native Work (paid jobs
//! stay reserved for solo workers); the pool key is a hot key; UNAUDITED PROTOTYPE, testnet without value.
//! Locks: `Inner` (accounting/sessions) and `Shared.node` are NEVER held at the same time.
use crate::chain::*;
use crate::crypto::*;
use crate::httpmini::{self, Request, Response};
use crate::net::{now_secs, select_txs, submit_block, submit_tx, Shared};
use crate::state::*;
use crate::types::*;
use crate::wallet::{address_of, parse_address};
use crate::work::*;
use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{self, BufRead, BufReader, Write};
use std::net::{IpAddr, Shutdown, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

pub const PROTO: &str = "vpool/1";
/// Maximum size of a JSON line received from a worker.
pub const MAX_LINE: usize = 4096;
/// The low 40 bits of the extra_nonce are the worker's counter; the high 24 bits are its SLOT.
pub const NONCE_BITS: u32 = 40;
pub const MAX_SLOTS: u32 = 1 << 24;
const KEEP_JOBS: usize = 8;
const STATE_VERSION: u32 = 1;

// protocol error codes
pub const E_STALE: i32 = 20;
pub const E_DUPLICATE: i32 = 21;
pub const E_LOW_DIFF: i32 = 22;
pub const E_INVALID: i32 = 23;
pub const E_UNKNOWN_JOB: i32 = 24;
pub const E_UNAUTHORIZED: i32 = 25;
pub const E_RATE: i32 = 26;
pub const E_BAD_REQUEST: i32 = 30;
pub const E_BANNED: i32 = 31;
pub const E_FULL: i32 = 32;

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

pub fn unhex32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(s.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(out)
}

/// 64-bit integer of the protocol: 16 hexadecimal digits (a JSON number cannot hold a u64 in every language).
pub fn u64hex(v: u64) -> String {
    format!("{v:016x}")
}

pub fn parse_u64hex(s: &str) -> Option<u64> {
    (s.len() == 16 && s.bytes().all(|c| c.is_ascii_hexdigit())).then(|| u64::from_str_radix(s, 16).ok()).flatten()
}

// ---------------------------------------------------------------- pure accounting (tested without a network)

/// Weight of a share of target `t`: proportional to the expected work (1/t), in 2^-96 fixed point.
pub fn share_weight(target: u64) -> u128 {
    (1u128 << 96) / (target.max(1) as u128)
}

/// New share target of a worker: `shares` shares received in `dt` seconds with target `cur`, goal one share every
/// `want_secs` seconds. Result bounded to a factor 4 per step, never harder than the network target `net` (a block must
/// also be a share). `None` if the difference is below 25 %.
pub fn retarget(cur: u64, shares: u32, dt: f64, want_secs: f64, net: u64) -> Option<u64> {
    if dt < 1.0 {
        return None;
    }
    let rate = shares as f64 / dt; // shares per second
    let want = 1.0 / want_secs.max(0.1);
    let ratio = (want / rate.max(want / 16.0)).clamp(0.25, 4.0); // > 1: too few shares, easier target
    if ratio > 0.8 && ratio < 1.25 {
        return None;
    }
    let new = ((cur as f64) * ratio).min(u64::MAX as f64) as u64;
    let new = new.max(net).max(1);
    (new != cur).then_some(new)
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum BlockStatus {
    Pending,
    Confirmed,
    Orphaned,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockRec {
    pub height: u64,
    pub hash: Hash,
    pub time: u64,
    /// gross block reward: emission + transaction fees
    pub reward: u64,
    pub pool_cut: u64,
    /// (address index, amount): frozen when the block is found (the PPLNS window of that moment)
    pub credits: Vec<(u32, u64)>,
    pub finder: u32,
    pub status: BlockStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InFlight {
    pub nonce: u64,
    pub tx: SignedTx,
    pub to: u32,
    pub amount: u64,
    /// chain height at which the transfer was seen included (None: not yet, or lost in a reorganisation)
    pub included: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PayRec {
    pub time: u64,
    pub to: u32,
    pub amount: u64,
    pub txid: Hash,
}

#[derive(Default, Serialize, Deserialize)]
pub struct Ledger {
    pub version: u32,
    pub genesis: Hash,
    pub pool_key: PubKey,
    pub addrs: Vec<PubKey>,
    #[serde(skip)]
    index: HashMap<PubKey, u32>,
    /// PPLNS window: (address, weight), from oldest to most recent
    pub window: VecDeque<(u32, u128)>,
    pub window_weight: u128,
    /// owed to the miners (confirmed blocks, not yet paid)
    pub balances: HashMap<u32, u64>,
    pub paid_total: HashMap<u32, u64>,
    pub blocks: Vec<BlockRec>,
    pub payments: VecDeque<PayRec>,
    pub in_flight: Vec<InFlight>,
    pub pool_earned: u64,
    pub shares_accepted: u64,
}

impl Ledger {
    pub fn new(genesis: Hash, pool_key: PubKey) -> Self {
        Ledger { version: STATE_VERSION, genesis, pool_key, ..Default::default() }
    }

    pub fn reindex(&mut self) {
        self.index = self.addrs.iter().enumerate().map(|(i, a)| (*a, i as u32)).collect();
    }

    pub fn idx(&mut self, a: &PubKey) -> u32 {
        if let Some(i) = self.index.get(a) {
            return *i;
        }
        let i = self.addrs.len() as u32;
        self.addrs.push(*a);
        self.index.insert(*a, i);
        i
    }

    pub fn find(&self, a: &PubKey) -> Option<u32> {
        self.index.get(a).copied()
    }

    /// Adds a share to the window; the window is bounded to `keep` (weight): beyond it, the oldest shares have no more claim.
    pub fn add_share(&mut self, a: u32, w: u128, keep: u128) {
        self.window.push_back((a, w));
        self.window_weight += w;
        self.shares_accepted += 1;
        while self.window_weight > keep && self.window.len() > 1 {
            let (_, ow) = self.window.pop_front().unwrap();
            self.window_weight -= ow;
        }
    }

    /// PPLNS split of `net` among the authors of the latest shares totalling `need` of weight (or the whole window if it is
    /// smaller). The sum of the parts is EXACTLY `net` (the rounding remainder goes to the largest holder).
    pub fn distribute(&self, need: u128, net: u64) -> Vec<(u32, u64)> {
        let mut acc: HashMap<u32, u128> = HashMap::new();
        let mut got = 0u128;
        for (a, w) in self.window.iter().rev() {
            if got >= need {
                break;
            }
            let take = (*w).min(need - got);
            *acc.entry(*a).or_default() += take;
            got += take;
        }
        if got == 0 || net == 0 {
            return vec![];
        }
        let shift = (128 - got.leading_zeros()).saturating_sub(80);
        let total = (got >> shift).max(1);
        let mut out: Vec<(u32, u64)> = acc.iter().map(|(a, w)| (*a, ((net as u128) * (*w >> shift) / total) as u64)).collect();
        out.sort_by_key(|(a, _)| *a);
        let sum: u64 = out.iter().map(|(_, v)| *v).sum();
        let rest = net.saturating_sub(sum);
        if rest > 0 {
            if let Some(big) = out.iter_mut().max_by_key(|(a, v)| (*v, std::cmp::Reverse(*a))) {
                big.1 += rest;
            }
        }
        out.retain(|(_, v)| *v > 0);
        out
    }

    pub fn credit_block(&mut self, i: usize) {
        let b = &mut self.blocks[i];
        if b.status != BlockStatus::Pending {
            return;
        }
        b.status = BlockStatus::Confirmed;
        let (credits, cut) = (b.credits.clone(), b.pool_cut);
        for (a, v) in credits {
            *self.balances.entry(a).or_default() += v;
        }
        self.pool_earned += cut;
    }

    pub fn owed(&self) -> u64 {
        self.balances.values().sum()
    }
    pub fn in_flight_total(&self) -> u64 {
        self.in_flight.iter().map(|f| f.amount).sum()
    }
    pub fn pending_total(&self) -> u64 {
        self.blocks.iter().filter(|b| b.status == BlockStatus::Pending).map(|b| b.reward - b.pool_cut).sum()
    }

    pub fn save(&self, path: &std::path::Path) -> io::Result<()> {
        let bytes = bincode::serialize(self).map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, bytes)?;
        let _ = std::fs::remove_file(path.with_extension("bak"));
        if path.exists() {
            let _ = std::fs::rename(path, path.with_extension("bak"));
        }
        std::fs::rename(&tmp, path)
    }

    pub fn load(path: &std::path::Path, genesis: &Hash, pool_key: &PubKey) -> Result<Option<Ledger>, String> {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) if e.kind() == io::ErrorKind::NotFound => match std::fs::read(path.with_extension("bak")) {
                Ok(b) => b,
                Err(_) => return Ok(None),
            },
            Err(e) => return Err(format!("{}: {e}", path.display())),
        };
        let mut l: Ledger = bincode::deserialize(&bytes).map_err(|e| format!("pool state unreadable ({}): {e}", path.display()))?;
        if l.version != STATE_VERSION {
            return Err(format!("unknown pool state version: {}", l.version));
        }
        if &l.genesis != genesis {
            return Err("the pool state belongs to another network (different genesis)".into());
        }
        if &l.pool_key != pool_key {
            return Err("the pool state belongs to another pool key".into());
        }
        l.reindex();
        Ok(Some(l))
    }
}

// ---------------------------------------------------------------- configuration, jobs, sessions

#[derive(Clone, Debug)]
pub struct PoolConfig {
    /// listening for workers (JSON lines over TCP)
    pub listen: String,
    /// HTTP listening for statistics
    pub http: Option<String>,
    /// pool fee in basis points of the gross reward (100 = 1 %)
    pub fee_bps: u64,
    /// PPLNS window = pplns_n x the weight of a share at the network target
    pub pplns_n: f64,
    /// depth (blocks on top) before a block's reward is credited
    pub confirmations: u64,
    pub min_payout: u64,
    pub payout_interval: u64,
    pub max_payouts_per_round: usize,
    /// goal: one share every `share_secs` seconds per worker
    pub share_secs: f64,
    pub retarget_secs: u64,
    /// a new job even without a new block at most every `refresh_secs` seconds
    pub refresh_secs: u64,
    /// fixed share target (tests); None: vardiff
    pub fixed_share_target: Option<u64>,
    pub max_conns: usize,
    pub max_per_ip: usize,
    pub stale_grace_secs: u64,
    pub data_dir: Option<PathBuf>,
    pub log: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig {
            listen: "127.0.0.1:3333".into(),
            http: None,
            fee_bps: 100,
            pplns_n: 2.0,
            confirmations: 10,
            min_payout: COIN,
            payout_interval: 600,
            max_payouts_per_round: 50,
            share_secs: 15.0,
            retarget_secs: 30,
            refresh_secs: 30,
            fixed_share_target: None,
            max_conns: 2000,
            max_per_ip: 64,
            stale_grace_secs: 4,
            data_dir: None,
            log: true,
        }
    }
}

struct Job {
    seq: u64,
    id: String,
    tpl: MiningTemplate,
    ts: u64,
    txs: Vec<SignedTx>,
    tx_root: Hash,
    accounts_root: Hash,
    /// emission + fees
    reward: u64,
}

struct Session {
    ip: IpAddr,
    writer: Arc<Mutex<TcpStream>>,
    login: Option<Login>,
    slot: u32,
    target: u64,
    prev_target: u64,
    prev_until: Instant,
    since: Instant,
    win_start: Instant,
    win_shares: u32,
    accepted: u64,
    rejected: u64,
    strikes: VecDeque<Instant>,
    bucket: f64,
    bucket_t: Instant,
    recent: VecDeque<(Instant, u128)>,
    last_share: Option<Instant>,
    hints: u32,
    rate_viol: u32,
    rate_win: Instant,
}

#[derive(Clone)]
struct Login {
    idx: u32,
    worker: String,
    agent: String,
}

#[derive(Default, Clone, Debug)]
pub struct Counters {
    pub rejected_stale: u64,
    pub rejected_duplicate: u64,
    pub rejected_low: u64,
    pub rejected_invalid: u64,
    pub blocks_found: u64,
    pub blocks_stale: u64,
    pub conns_total: u64,
    pub bans: u64,
}

struct Inner {
    ledger: Ledger,
    dirty: bool,
    cur: Vec<Arc<Job>>,
    grace: Vec<(Arc<Job>, Instant)>,
    tip: Hash,
    seq: u64,
    last_job: Instant,
    last_fp: (usize, Option<Hash>),
    seen: HashSet<(u64, u64, u32, u32)>,
    sess: HashMap<u64, Session>,
    slots: HashSet<u32>,
    next_slot: u32,
    per_ip: HashMap<IpAddr, usize>,
    banned: HashMap<IpAddr, Instant>,
    c: Counters,
    last_save: Instant,
    last_payout: Instant,
    last_maint: Instant,
}

pub struct Pool {
    pub cfg: PoolConfig,
    pub sh: Arc<Shared>,
    sk: SigningKey,
    pub pk: PubKey,
    inner: Mutex<Inner>,
    stop: AtomicBool,
    next_conn: std::sync::atomic::AtomicU64,
    started: Instant,
    pub addr: Mutex<String>,
    pub http_addr: Mutex<Option<String>>,
}

fn send_line(w: &Arc<Mutex<TcpStream>>, v: &Value) -> bool {
    let mut line = v.to_string();
    line.push('\n');
    match w.lock() {
        Ok(mut s) => s.write_all(line.as_bytes()).is_ok(),
        Err(_) => false,
    }
}

fn err(code: i32, msg: &str) -> (i32, String) {
    (code, msg.to_string())
}

impl Pool {
    /// Starts the pool: loads the state, listens for workers, starts the job loop and (optionally) the HTTP API.
    pub fn start(cfg: PoolConfig, sh: Arc<Shared>, sk: SigningKey) -> Result<Arc<Pool>, String> {
        let pk = pubkey(&sk);
        let genesis = sh.genesis_hash;
        let mut ledger = Ledger::new(genesis, pk);
        if let Some(d) = &cfg.data_dir {
            std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
            if let Some(l) = Ledger::load(&d.join("pool-state.bin"), &genesis, &pk)? {
                ledger = l;
            }
        }
        let listener = TcpListener::bind(&cfg.listen).map_err(|e| format!("cannot listen on {}: {e}", cfg.listen))?;
        let addr = listener.local_addr().map_err(|e| e.to_string())?.to_string();
        let now = Instant::now();
        let pool = Arc::new(Pool {
            cfg: cfg.clone(),
            sh,
            sk,
            pk,
            inner: Mutex::new(Inner {
                ledger,
                dirty: false,
                cur: vec![],
                grace: vec![],
                tip: [0xff; 32],
                seq: 0,
                last_job: now,
                last_fp: (0, None),
                seen: HashSet::new(),
                sess: HashMap::new(),
                slots: HashSet::new(),
                next_slot: rand::random::<u32>() % MAX_SLOTS,
                per_ip: HashMap::new(),
                banned: HashMap::new(),
                c: Counters::default(),
                last_save: now,
                last_payout: now,
                last_maint: now,
            }),
            stop: AtomicBool::new(false),
            next_conn: std::sync::atomic::AtomicU64::new(1),
            started: now,
            addr: Mutex::new(addr),
            http_addr: Mutex::new(None),
        });
        {
            let p = pool.clone();
            thread::spawn(move || {
                for s in listener.incoming() {
                    if p.stop.load(Ordering::SeqCst) {
                        break;
                    }
                    let Ok(s) = s else { continue };
                    let p2 = p.clone();
                    thread::spawn(move || p2.handle_conn(s));
                }
            });
        }
        {
            let p = pool.clone();
            thread::spawn(move || p.job_loop());
        }
        if let Some(h) = &cfg.http {
            let p = pool.clone();
            let bound = httpmini::serve(h, move |req| p.http(req)).map_err(|e| format!("HTTP {h}: {e}"))?;
            *pool.http_addr.lock().unwrap() = Some(bound);
        }
        Ok(pool)
    }

    pub fn local_addr(&self) -> String {
        self.addr.lock().unwrap().clone()
    }

    pub fn address(&self) -> String {
        address_of(&self.pk)
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        self.save_state(true);
        let a = self.local_addr();
        let _ = TcpStream::connect(a);
        let writers: Vec<_> = self.inner.lock().unwrap().sess.values().map(|s| s.writer.clone()).collect();
        for w in writers {
            if let Ok(s) = w.lock() {
                let _ = s.shutdown(Shutdown::Both);
            }
        }
    }

    fn log(&self, m: &str) {
        if self.cfg.log {
            eprintln!("[pool] {m}");
        }
    }

    pub fn ledger_snapshot<R>(&self, f: impl FnOnce(&Ledger) -> R) -> R {
        f(&self.inner.lock().unwrap().ledger)
    }

    pub fn counters(&self) -> Counters {
        self.inner.lock().unwrap().c.clone()
    }

    pub fn save_state(&self, force: bool) {
        let Some(d) = &self.cfg.data_dir else { return };
        let mut g = self.inner.lock().unwrap();
        if !(g.dirty || force) {
            return;
        }
        match g.ledger.save(&d.join("pool-state.bin")) {
            Ok(()) => {
                g.dirty = false;
                g.last_save = Instant::now();
            }
            Err(e) => eprintln!("[pool] cannot save the state: {e}"),
        }
    }

    // ------------------------------------------------------------ jobs

    fn make_job(&self, seq: u64) -> Result<Arc<Job>, String> {
        let (tpl, txs, ts) = {
            let mut n = self.sh.node.lock().unwrap();
            let ts = now_secs().max(n.chain.tip_ts() + 1);
            if ts > now_secs() + MAX_FUTURE - 5 {
                return Err("clock: the last block is too close to the future limit".into());
            }
            let tpl = n.chain.template(WorkClass::Native, None)?;
            let txs = select_txs(&n, &self.pk, ts);
            (tpl, txs, ts)
        };
        let root = tx_root(&txs);
        let mut post = (*tpl.state).clone();
        for t in &txs {
            post.apply_tx(t, tpl.height, ts, &self.pk, &tpl.params).map_err(|e| format!("invalid job tx: {e}"))?;
        }
        let base = post.pay_block_reward(tpl.height, &self.pk, &WorkClass::Native);
        let fees: u64 = txs.iter().map(|t| t.tx.fee).sum();
        let mut idb = [0u8; 6];
        for b in idb.iter_mut() {
            *b = rand::random();
        }
        Ok(Arc::new(Job { seq, id: hex(&idb), tpl, ts, txs, tx_root: root, accounts_root: post.accounts_root(), reward: base + fees }))
    }

    fn job_json(&self, job: &Job, slot: u32, share_target: u64, clean: bool) -> Value {
        let p = job.tpl.params;
        json!({"method": "job", "params": {
            "job_id": job.id, "clean": clean, "height": job.tpl.height, "prev": hex(&job.tpl.prev), "miner": hex(&self.pk),
            "class": "native", "timestamp": job.ts, "tx_root": hex(&job.tx_root), "epoch": job.tpl.height / EPOCH_LEN,
            "target": u64hex(job.tpl.target), "share_target": u64hex(share_target),
            "slot": slot, "nonce_start": u64hex((slot as u64) << NONCE_BITS), "nonce_end": u64hex(((slot as u64) + 1) << NONCE_BITS),
            "params": {"n": p.n, "tile": p.tile, "rank": p.rank, "fold_k": p.fold_k},
        }})
    }

    /// Initial share target: one share every `share_secs` seconds for a rate of `sigma_rate` sigma/s (hint from the worker).
    fn initial_target(&self, net: u64, tiles: usize, sigma_rate: Option<f64>) -> u64 {
        if let Some(t) = self.cfg.fixed_share_target {
            return t.max(net);
        }
        let rate = sigma_rate.filter(|r| r.is_finite() && *r > 0.0).unwrap_or(10.0).min(1e7);
        let attempts_per_share = rate * (tiles * tiles) as f64 * self.cfg.share_secs;
        let t = (u64::MAX as f64 / attempts_per_share.max(1.0)) as u64;
        t.max(net).max(1)
    }

    fn install_job(&self, job: Arc<Job>, clean: bool) {
        let mut out: Vec<(Arc<Mutex<TcpStream>>, String)> = vec![];
        {
            let mut g = self.inner.lock().unwrap();
            if clean {
                let old: Vec<Arc<Job>> = g.cur.drain(..).collect();
                let now = Instant::now();
                g.grace.extend(old.into_iter().map(|j| (j, now)));
            }
            g.cur.push(job.clone());
            while g.cur.len() > KEEP_JOBS {
                let j = g.cur.remove(0);
                g.grace.push((j, Instant::now()));
            }
            g.tip = job.tpl.prev;
            g.last_job = Instant::now();
            for s in g.sess.values_mut() {
                if s.login.is_some() {
                    let net = job.tpl.target;
                    if s.target < net {
                        s.prev_target = s.target;
                        s.prev_until = Instant::now() + Duration::from_secs(30);
                        s.target = net; // the network target became easier than the worker's: we cannot be harder than it
                    }
                    out.push((s.writer.clone(), self.job_json(&job, s.slot, s.target, clean).to_string()));
                }
            }
        }
        for (w, line) in out {
            let mut l = line;
            l.push('\n');
            let ok = w.lock().map(|mut s| s.write_all(l.as_bytes()).is_ok()).unwrap_or(false);
            if !ok {
                if let Ok(s) = w.lock() {
                    let _ = s.shutdown(Shutdown::Both);
                }
            }
        }
    }

    fn job_loop(self: Arc<Self>) {
        while !self.stop.load(Ordering::SeqCst) {
            let (tip, fp) = {
                let n = self.sh.node.lock().unwrap();
                (n.chain.tip_hash(), (n.mempool.len(), n.mempool.last().map(|t| t.id())))
            };
            let (cur_tip, age, last_fp, have) = {
                let g = self.inner.lock().unwrap();
                (g.tip, g.last_job.elapsed(), g.last_fp, !g.cur.is_empty())
            };
            let new_tip = tip != cur_tip || !have;
            let refresh = new_tip || age >= Duration::from_secs(self.cfg.refresh_secs) || (fp != last_fp && age >= Duration::from_secs(5));
            if refresh {
                let seq = {
                    let mut g = self.inner.lock().unwrap();
                    g.seq += 1;
                    g.seq
                };
                match self.make_job(seq) {
                    Ok(job) => {
                        if new_tip {
                            self.log(&format!("new job (block {}, target {}, parent {})", job.tpl.height, u64hex(job.tpl.target), &hex(&job.tpl.prev)[..8]));
                        }
                        self.inner.lock().unwrap().last_fp = fp;
                        self.install_job(job, new_tip);
                    }
                    Err(e) => {
                        self.log(&format!("cannot build job: {e}"));
                        thread::sleep(Duration::from_millis(500));
                    }
                }
            }
            let due = self.inner.lock().unwrap().last_maint.elapsed() >= Duration::from_secs(2);
            if due {
                self.inner.lock().unwrap().last_maint = Instant::now();
                self.maintenance();
            }
            thread::sleep(Duration::from_millis(100));
        }
    }

    // ------------------------------------------------------------ maintenance: vardiff, confirmations, payouts, saving

    fn maintenance(&self) {
        self.retarget_all();
        self.confirm_blocks();
        let (pay_due, save_due) = {
            let mut g = self.inner.lock().unwrap();
            let now = Instant::now();
            g.grace.retain(|(_, t)| now.duration_since(*t) < Duration::from_secs(self.cfg.stale_grace_secs));
            let live: HashSet<u64> = g.cur.iter().map(|j| j.seq).chain(g.grace.iter().map(|(j, _)| j.seq)).collect();
            g.seen.retain(|k| live.contains(&k.0));
            g.banned.retain(|_, t| now.duration_since(*t) < Duration::from_secs(600));
            for s in g.sess.values_mut() {
                while s.recent.front().map_or(false, |(t, _)| now.duration_since(*t) > Duration::from_secs(600)) {
                    s.recent.pop_front();
                }
            }
            (now.duration_since(g.last_payout) >= Duration::from_secs(self.cfg.payout_interval), now.duration_since(g.last_save) >= Duration::from_secs(30))
        };
        self.reconcile_payouts();
        if pay_due {
            self.inner.lock().unwrap().last_payout = Instant::now();
            self.run_payouts();
        }
        if save_due {
            self.save_state(false);
        }
    }

    fn retarget_all(&self) {
        if self.cfg.fixed_share_target.is_some() {
            return;
        }
        let mut notes: Vec<(Arc<Mutex<TcpStream>>, String)> = vec![];
        {
            let mut g = self.inner.lock().unwrap();
            let Some(job) = g.cur.last().cloned() else { return };
            let net = job.tpl.target;
            let now = Instant::now();
            for s in g.sess.values_mut() {
                if s.login.is_none() {
                    continue;
                }
                let dt = now.duration_since(s.win_start).as_secs_f64();
                let due = dt >= self.cfg.retarget_secs as f64 || (s.win_shares >= 30 && dt >= 2.0) || (s.accepted == 0 && s.win_shares == 0 && dt >= 10.0 && s.since.elapsed().as_secs() < 3 * self.cfg.retarget_secs);
                if !due {
                    continue;
                }
                if let Some(nt) = retarget(s.target, s.win_shares, dt, self.cfg.share_secs, net) {
                    s.prev_target = s.target;
                    s.prev_until = now + Duration::from_secs(30);
                    s.target = nt;
                    notes.push((s.writer.clone(), json!({"method": "set_target", "params": {"share_target": u64hex(nt)}}).to_string()));
                }
                s.win_start = now;
                s.win_shares = 0;
            }
        }
        for (w, line) in notes {
            let mut l = line;
            l.push('\n');
            let _ = w.lock().map(|mut s| s.write_all(l.as_bytes()));
        }
    }

    fn confirm_blocks(&self) {
        let pending: Vec<(u64, Hash)> = {
            let g = self.inner.lock().unwrap();
            g.ledger.blocks.iter().filter(|b| b.status == BlockStatus::Pending).map(|b| (b.height, b.hash)).collect()
        };
        if pending.is_empty() {
            return;
        }
        let (len, hashes): (u64, Vec<Option<Hash>>) = {
            let n = self.sh.node.lock().unwrap();
            (n.chain.height(), pending.iter().map(|(h, _)| n.chain.blocks.get(*h as usize).map(|b| b.header.hash())).collect())
        };
        let mut g = self.inner.lock().unwrap();
        for ((h, hash), onchain) in pending.iter().zip(hashes) {
            let Some(i) = g.ledger.blocks.iter().position(|b| b.height == *h && b.hash == *hash && b.status == BlockStatus::Pending) else { continue };
            match onchain {
                Some(x) if x == *hash => {
                    if len >= *h + self.cfg.confirmations {
                        g.ledger.credit_block(i);
                        g.dirty = true;
                        if self.cfg.log {
                            eprintln!("[pool] block {h} confirmed: {} to distribute", fmt(g.ledger.blocks[i].reward - g.ledger.blocks[i].pool_cut));
                        }
                    }
                }
                _ if len > *h => {
                    g.ledger.blocks[i].status = BlockStatus::Orphaned;
                    g.dirty = true;
                    if self.cfg.log {
                        eprintln!("[pool] block {h} ORPHANED (reorganisation): no reward credited");
                    }
                }
                _ => {}
            }
        }
    }

    /// Follows the transfers in flight: included (pool nonce passed), final after `confirmations` blocks, or lost (resubmitted).
    fn reconcile_payouts(&self) {
        let has_flight = !self.inner.lock().unwrap().ledger.in_flight.is_empty();
        if !has_flight {
            return;
        }
        let (chain_nonce, len, mempool_ids): (u64, u64, HashSet<Hash>) = {
            let n = self.sh.node.lock().unwrap();
            (n.chain.state.nonce(&self.pk), n.chain.height(), n.mempool.iter().map(|t| t.id()).collect())
        };
        let mut resend: Vec<SignedTx> = vec![];
        {
            let mut g = self.inner.lock().unwrap();
            let mut done: Vec<usize> = vec![];
            let now = now_secs();
            for (i, f) in g.ledger.in_flight.iter_mut().enumerate() {
                if f.nonce < chain_nonce {
                    let inc = *f.included.get_or_insert(len);
                    if len >= inc + self.cfg.confirmations {
                        done.push(i);
                    }
                } else {
                    f.included = None;
                    if !mempool_ids.contains(&f.tx.id()) {
                        resend.push(f.tx.clone());
                    }
                }
            }
            for i in done.into_iter().rev() {
                let f = g.ledger.in_flight.remove(i);
                *g.ledger.paid_total.entry(f.to).or_default() += f.amount;
                g.ledger.payments.push_back(PayRec { time: now, to: f.to, amount: f.amount, txid: f.tx.id() });
                while g.ledger.payments.len() > 1000 {
                    g.ledger.payments.pop_front();
                }
                g.dirty = true;
            }
        }
        resend.sort_by_key(|t| t.tx.nonce);
        for t in resend {
            let _ = submit_tx(&self.sh, t); // "duplicate" or inconsistent nonce: harmless, we retry on the next round
        }
    }

    fn run_payouts(&self) {
        let (chain_nonce, balance, pending_own) = {
            let n = self.sh.node.lock().unwrap();
            (n.chain.state.nonce(&self.pk), n.chain.state.balance(&self.pk), n.mempool.iter().filter(|t| t.tx.from == self.pk).count() as u64)
        };
        let mut sends: Vec<SignedTx> = vec![];
        {
            let mut g = self.inner.lock().unwrap();
            // transfers in flight are not yet debited from the chain; those included but not yet final already are
            let unspent_flight: u64 = g.ledger.in_flight.iter().filter(|f| f.nonce >= chain_nonce).map(|f| f.amount).sum();
            let mut available = balance.saturating_sub(unspent_flight);
            let mut next_nonce = chain_nonce + pending_own;
            let max_nonce = g.ledger.in_flight.iter().map(|f| f.nonce + 1).max().unwrap_or(0);
            next_nonce = next_nonce.max(max_nonce);
            let mut due: Vec<(u32, u64)> = g.ledger.balances.iter().filter(|(_, v)| **v >= self.cfg.min_payout.max(1)).map(|(a, v)| (*a, *v)).collect();
            due.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            for (a, amount) in due.into_iter().take(self.cfg.max_payouts_per_round) {
                if amount > available {
                    self.log(&format!("payout deferred: pool balance insufficient ({} available, {} owed)", fmt(available), fmt(amount)));
                    continue;
                }
                let to = g.ledger.addrs[a as usize];
                if to == self.pk {
                    g.ledger.balances.remove(&a);
                    continue;
                }
                let tx = SignedTx::new(Tx { from: self.pk, nonce: next_nonce, fee: 0, kind: TxKind::Transfer { to, amount } }, &self.sk);
                g.ledger.balances.remove(&a);
                g.ledger.in_flight.push(InFlight { nonce: next_nonce, tx: tx.clone(), to: a, amount, included: None });
                g.dirty = true;
                next_nonce += 1;
                available -= amount;
                if self.cfg.log {
                    eprintln!("[pool] payout of {} to {} (nonce {})", fmt(amount), address_of(&to), next_nonce - 1);
                }
                sends.push(tx);
            }
        }
        for t in sends {
            if let Err(e) = submit_tx(&self.sh, t) {
                self.log(&format!("transfer refused by the mempool: {e} (will be resubmitted)"));
            }
        }
    }

    // ------------------------------------------------------------ connections

    fn strike(g: &mut Inner, cid: u64, n: usize) -> bool {
        let now = Instant::now();
        let Some(s) = g.sess.get_mut(&cid) else { return false };
        for _ in 0..n {
            s.strikes.push_back(now);
        }
        while s.strikes.front().map_or(false, |t| now.duration_since(*t) > Duration::from_secs(60)) {
            s.strikes.pop_front();
        }
        s.strikes.len() >= 25
    }

    fn ban(&self, g: &mut Inner, ip: IpAddr) {
        g.c.bans += 1;
        if ip.is_loopback() {
            return;
        }
        g.banned.insert(ip, Instant::now());
        for s in g.sess.values() {
            if s.ip == ip {
                if let Ok(w) = s.writer.lock() {
                    let _ = w.shutdown(Shutdown::Both);
                }
            }
        }
    }

    fn handle_conn(self: Arc<Self>, stream: TcpStream) {
        let Ok(peer) = stream.peer_addr() else { return };
        let ip = peer.ip();
        let _ = stream.set_nodelay(true);
        let _ = stream.set_write_timeout(Some(Duration::from_secs(3)));
        let _ = stream.set_read_timeout(Some(Duration::from_secs(15))); // 15 s to connect; then `idle`
        let Ok(wr) = stream.try_clone() else { return };
        let writer = Arc::new(Mutex::new(wr));
        let cid = self.next_conn.fetch_add(1, Ordering::SeqCst);
        {
            let mut g = self.inner.lock().unwrap();
            let now = Instant::now();
            if !ip.is_loopback() && g.banned.get(&ip).map_or(false, |t| now.duration_since(*t) < Duration::from_secs(600)) {
                drop(g);
                send_line(&writer, &json!({"id": null, "error": {"code": E_BANNED, "message": "banned"}}));
                return;
            }
            if g.sess.len() >= self.cfg.max_conns || *g.per_ip.get(&ip).unwrap_or(&0) >= self.cfg.max_per_ip {
                drop(g);
                send_line(&writer, &json!({"id": null, "error": {"code": E_FULL, "message": "too many connections"}}));
                return;
            }
            *g.per_ip.entry(ip).or_default() += 1;
            g.c.conns_total += 1;
            g.sess.insert(
                cid,
                Session {
                    ip,
                    writer: writer.clone(),
                    login: None,
                    slot: 0,
                    target: u64::MAX,
                    prev_target: u64::MAX,
                    prev_until: now,
                    since: now,
                    win_start: now,
                    win_shares: 0,
                    accepted: 0,
                    rejected: 0,
                    strikes: VecDeque::new(),
                    bucket: 100.0,
                    bucket_t: now,
                    recent: VecDeque::new(),
                    last_share: None,
                    hints: 0,
                    rate_viol: 0,
                    rate_win: now,
                },
            );
        }
        self.conn_loop(cid, stream, &writer);
        let mut g = self.inner.lock().unwrap();
        if let Some(s) = g.sess.remove(&cid) {
            if s.login.is_some() {
                g.slots.remove(&s.slot);
            }
        }
        if let Some(c) = g.per_ip.get_mut(&ip) {
            *c = c.saturating_sub(1);
            if *c == 0 {
                g.per_ip.remove(&ip);
            }
        }
        drop(g);
        let _ = writer.lock().map(|s| s.shutdown(Shutdown::Both));
    }

    fn conn_loop(&self, cid: u64, stream: TcpStream, writer: &Arc<Mutex<TcpStream>>) {
        let mut rd = BufReader::new(stream);
        let mut line: Vec<u8> = Vec::with_capacity(512);
        let mut bad = 0;
        while !self.stop.load(Ordering::SeqCst) {
            line.clear();
            match read_line_limited(&mut rd, &mut line) {
                Ok(true) => {}
                _ => return, // end of connection, timeout or line too long
            }
            let v: Value = match serde_json::from_slice(&line) {
                Ok(v) => v,
                Err(_) => {
                    bad += 1;
                    send_line(writer, &json!({"id": null, "error": {"code": E_BAD_REQUEST, "message": "invalid JSON"}}));
                    if bad >= 5 {
                        return;
                    }
                    continue;
                }
            };
            let id = v.get("id").cloned().unwrap_or(Value::Null);
            let method = v.get("method").and_then(|m| m.as_str()).unwrap_or("");
            let params = v.get("params").cloned().unwrap_or(Value::Null);
            let res = match method {
                "login" => self.m_login(cid, &params),
                "submit" => self.m_submit(cid, &params),
                "hint" => self.m_hint(cid, &params),
                "ping" => Ok(json!({"pong": now_secs()})),
                "stats" => self.m_stats(cid),
                _ => Err(err(E_BAD_REQUEST, "unknown method")),
            };
            let (reply, fatal) = match res {
                Ok(r) => (json!({"id": id, "result": r}), false),
                Err((c, m)) => (json!({"id": id, "error": {"code": c, "message": m}}), c == E_BANNED),
            };
            if !send_line(writer, &reply) || fatal {
                return;
            }
            if method == "login" && reply.get("result").is_some() {
                // the worker immediately receives the current job, then the following ones by notification
                let (job, slot, target) = {
                    let g = self.inner.lock().unwrap();
                    let job = g.cur.last().cloned();
                    let s = g.sess.get(&cid);
                    (job, s.map(|s| s.slot).unwrap_or(0), s.map(|s| s.target).unwrap_or(u64::MAX))
                };
                if let Some(j) = job {
                    send_line(writer, &self.job_json(&j, slot, target.max(j.tpl.target), true));
                }
                let _ = rd.get_ref().set_read_timeout(Some(Duration::from_secs(600)));
            }
        }
    }

    fn m_login(&self, cid: u64, p: &Value) -> Result<Value, (i32, String)> {
        if p.get("proto").and_then(|x| x.as_str()) != Some(PROTO) {
            return Err(err(E_BAD_REQUEST, &format!("unsupported protocol (this pool speaks {PROTO})")));
        }
        let addr_s = p.get("address").and_then(|x| x.as_str()).ok_or_else(|| err(E_BAD_REQUEST, "address required"))?;
        let addr = parse_address(addr_s).map_err(|e| err(E_BAD_REQUEST, &e))?;
        if addr == self.pk {
            return Err(err(E_BAD_REQUEST, "the pool address cannot be a payout address"));
        }
        let worker: String = p.get("worker").and_then(|x| x.as_str()).unwrap_or("default").to_string();
        if worker.is_empty() || worker.len() > 32 || !worker.bytes().all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c)) {
            return Err(err(E_BAD_REQUEST, "worker name: 1-32 characters among letters, digits, . _ -"));
        }
        let agent: String = p.get("agent").and_then(|x| x.as_str()).unwrap_or("").chars().filter(|c| c.is_ascii_graphic() || *c == ' ').take(64).collect();
        let rate = p.get("sigma_rate").and_then(|x| x.as_f64());
        let (net, tiles) = {
            let g = self.inner.lock().unwrap();
            match g.cur.last() {
                Some(j) => (j.tpl.target, j.tpl.params.tiles()),
                None => (u64::MAX / 2048, 64),
            }
        };
        let target = self.initial_target(net, tiles, rate);
        let mut g = self.inner.lock().unwrap();
        if g.sess.get(&cid).map_or(true, |s| s.login.is_some()) {
            return Err(err(E_BAD_REQUEST, "already logged in"));
        }
        let mut slot = None;
        for _ in 0..64 {
            g.next_slot = (g.next_slot + 1) % MAX_SLOTS;
            if !g.slots.contains(&g.next_slot) {
                slot = Some(g.next_slot);
                break;
            }
        }
        let slot = slot.ok_or_else(|| err(E_FULL, "no free slot"))?;
        g.slots.insert(slot);
        let idx = g.ledger.idx(&addr);
        g.dirty = true;
        let s = g.sess.get_mut(&cid).unwrap();
        s.slot = slot;
        s.target = target;
        s.prev_target = target;
        s.win_start = Instant::now();
        s.login = Some(Login { idx, worker: worker.clone(), agent });
        if self.cfg.log {
            eprintln!("[pool] connection {worker}@{} (slot {slot}, share target {})", &addr_s[..addr_s.len().min(12)], u64hex(target));
        }
        Ok(json!({
            "proto": PROTO, "slot": slot, "share_target": u64hex(target), "pool_address": address_of(&self.pk),
            "fee_bps": self.cfg.fee_bps, "min_payout": self.cfg.min_payout, "payout_interval": self.cfg.payout_interval,
            "confirmations": self.cfg.confirmations, "pplns_n": self.cfg.pplns_n, "nonce_bits": NONCE_BITS,
            "share_secs": self.cfg.share_secs,
        }))
    }

    /// The worker announces its measured rate (sigma/s): the share target jumps straight to the right scale instead of converging
    /// in x4 steps (a GPU starting from a small CPU's target would flood the pool for about fifteen seconds).
    fn m_hint(&self, cid: u64, p: &Value) -> Result<Value, (i32, String)> {
        let rate = p.get("sigma_rate").and_then(|x| x.as_f64()).filter(|r| r.is_finite() && *r > 0.0).ok_or_else(|| err(E_BAD_REQUEST, "sigma_rate: positive number"))?;
        let mut g = self.inner.lock().unwrap();
        let (net, tiles) = g.cur.last().map_or((u64::MAX / 2048, 64), |j| (j.tpl.target, j.tpl.params.tiles()));
        let new = self.initial_target(net, tiles, Some(rate));
        let now = Instant::now();
        let s = g.sess.get_mut(&cid).ok_or_else(|| err(E_BAD_REQUEST, "gone"))?;
        if s.login.is_none() {
            return Err(err(E_UNAUTHORIZED, "login first"));
        }
        if s.hints >= 3 {
            return Err(err(E_RATE, "too many hints"));
        }
        s.hints += 1;
        if new != s.target && self.cfg.fixed_share_target.is_none() {
            s.prev_target = s.target;
            s.prev_until = now + Duration::from_secs(30);
            s.target = new;
            s.win_start = now;
            s.win_shares = 0;
            let line = json!({"method": "set_target", "params": {"share_target": u64hex(new)}});
            send_line(&s.writer, &line);
        }
        Ok(json!({"share_target": u64hex(s.target)}))
    }

    fn m_stats(&self, cid: u64) -> Result<Value, (i32, String)> {
        let g = self.inner.lock().unwrap();
        let s = g.sess.get(&cid).ok_or_else(|| err(E_BAD_REQUEST, "gone"))?;
        let l = s.login.as_ref().ok_or_else(|| err(E_UNAUTHORIZED, "login first"))?;
        let bal = g.ledger.balances.get(&l.idx).copied().unwrap_or(0);
        let paid = g.ledger.paid_total.get(&l.idx).copied().unwrap_or(0);
        Ok(json!({"accepted": s.accepted, "rejected": s.rejected, "share_target": u64hex(s.target), "balance": bal, "paid": paid}))
    }

    fn m_submit(&self, cid: u64, p: &Value) -> Result<Value, (i32, String)> {
        let job_id = p.get("job_id").and_then(|x| x.as_str()).unwrap_or("").to_string();
        let en = p.get("extra_nonce").and_then(|x| x.as_str()).and_then(parse_u64hex).ok_or_else(|| err(E_BAD_REQUEST, "extra_nonce: 16 hex digits"))?;
        let ti = p.get("ti").and_then(|x| x.as_u64()).ok_or_else(|| err(E_BAD_REQUEST, "ti required"))?;
        let tj = p.get("tj").and_then(|x| x.as_u64()).ok_or_else(|| err(E_BAD_REQUEST, "tj required"))?;
        // phase 1 (lock): authorisation, rate, job, nonce range, duplicate
        let (job, grace, login, targets, ip) = {
            let mut g = self.inner.lock().unwrap();
            let now = Instant::now();
            let (slot, target, prev, ip, login, limited) = {
                let s = g.sess.get_mut(&cid).ok_or_else(|| err(E_BAD_REQUEST, "gone"))?;
                let login = s.login.clone().ok_or_else(|| err(E_UNAUTHORIZED, "login first"))?;
                // token bucket: 50 shares/s in steady state, burst of 100
                s.bucket = (s.bucket + now.duration_since(s.bucket_t).as_secs_f64() * 50.0).min(100.0);
                s.bucket_t = now;
                let limited = s.bucket < 1.0;
                if !limited {
                    s.bucket -= 1.0;
                }
                (s.slot, s.target, if now < s.prev_until { Some(s.prev_target) } else { None }, s.ip, login, limited)
            };
            if limited {
                // no strike: a fast GPU whose starting target is too easy would exceed the rate before its `hint` and be banned wrongly;
                // we only disconnect a sustained abnormal rate
                let s = g.sess.get_mut(&cid).unwrap();
                if s.rate_win.elapsed() > Duration::from_secs(60) {
                    s.rate_win = now;
                    s.rate_viol = 0;
                }
                s.rate_viol += 1;
                if s.rate_viol > 5000 {
                    if let Ok(w) = s.writer.lock() {
                        let _ = w.shutdown(Shutdown::Both);
                    }
                }
                return Err(err(E_RATE, "too many submissions"));
            }
            let found = g.cur.iter().find(|j| j.id == job_id).map(|j| (j.clone(), false)).or_else(|| {
                g.grace.iter().find(|(j, t)| j.id == job_id && now.duration_since(*t) < Duration::from_secs(self.cfg.stale_grace_secs)).map(|(j, _)| (j.clone(), true))
            });
            let Some((job, grace)) = found else {
                g.c.rejected_stale += 1;
                if let Some(s) = g.sess.get_mut(&cid) {
                    s.rejected += 1;
                }
                return Err(err(E_STALE, "stale or unknown job"));
            };
            let tl = job.tpl.params.tiles() as u64;
            if (en >> NONCE_BITS) as u32 != slot || ti >= tl || tj >= tl {
                g.c.rejected_invalid += 1;
                if let Some(s) = g.sess.get_mut(&cid) {
                    s.rejected += 1;
                }
                if Self::strike(&mut g, cid, 5) {
                    self.ban(&mut g, ip);
                }
                return Err(err(E_INVALID, "extra_nonce outside your slot, or tile out of range"));
            }
            if !g.seen.insert((job.seq, en, ti as u32, tj as u32)) {
                g.c.rejected_duplicate += 1;
                if let Some(s) = g.sess.get_mut(&cid) {
                    s.rejected += 1;
                }
                if Self::strike(&mut g, cid, 2) {
                    self.ban(&mut g, ip);
                }
                return Err(err(E_DUPLICATE, "duplicate share"));
            }
            (job, grace, login, (target, prev), ip)
        };
        // phase 2 (outside the lock): we RECOMPUTE the tile jackpot from our own matrices
        let sg = sigma(&job.tpl.prev, &self.pk, &WorkClass::Native, en, &job.tx_root, job.ts);
        let jp = tile_jackpot(&job.tpl.cm, &sg, ti as usize, tj as usize, &job.tpl.params).ok_or_else(|| err(E_INVALID, "tile out of range"))?;
        let is_block = meets_target(&jp, job.tpl.target);
        // the share counts at the weight of the hardest assigned target it satisfies
        let mut cands = vec![targets.0];
        cands.extend(targets.1);
        cands.retain(|t| meets_target(&jp, *t));
        let credited = cands.into_iter().min().or(is_block.then_some(job.tpl.target));
        let Some(credited) = credited else {
            let mut g = self.inner.lock().unwrap();
            g.c.rejected_low += 1;
            if let Some(s) = g.sess.get_mut(&cid) {
                s.rejected += 1;
            }
            if Self::strike(&mut g, cid, 5) {
                self.ban(&mut g, ip);
            }
            return Err(err(E_LOW_DIFF, "share above the share target (jackpot not low enough)"));
        };
        let w = share_weight(credited);
        let need = (share_weight(job.tpl.target) as f64 * self.cfg.pplns_n) as u128;
        {
            let mut g = self.inner.lock().unwrap();
            g.ledger.add_share(login.idx, w, need.saturating_mul(4).max(w));
            g.dirty = true;
            if let Some(s) = g.sess.get_mut(&cid) {
                s.accepted += 1;
                s.win_shares += 1;
                s.last_share = Some(Instant::now());
                s.recent.push_back((Instant::now(), w));
            }
        }
        let mut block_hash = None;
        if is_block {
            block_hash = self.try_block(&job, en, ti as u32, tj as u32, jp, &login, need, grace);
        }
        Ok(json!({"accepted": true, "block": block_hash.is_some(), "weight": w.to_string()}))
    }

    /// A share reaches the network target: we assemble the block (header + tile proof + job transactions) and submit it.
    fn try_block(&self, job: &Job, en: u64, ti: u32, tj: u32, jp: Hash, finder: &Login, need: u128, _grace: bool) -> Option<Hash> {
        let header = Header {
            version: 2,
            height: job.tpl.height,
            prev: job.tpl.prev,
            timestamp: job.ts,
            miner: self.pk,
            class: WorkClass::Native,
            extra_nonce: en,
            ti,
            tj,
            jackpot: jp,
            tx_root: job.tx_root,
            target: job.tpl.target,
            a_root: job.tpl.cm.a_root(),
            b_root: job.tpl.cm.b_root(),
            accounts_root: job.accounts_root,
        };
        let hash = header.hash();
        let proof = build_proof(&job.tpl.cm, ti as usize, tj as usize, &job.tpl.params);
        match submit_block(&self.sh, Block { header, proof, txs: job.txs.clone() }) {
            Ok(()) => {
                let mut g = self.inner.lock().unwrap();
                let cut = ((job.reward as u128) * (self.cfg.fee_bps.min(10_000) as u128) / 10_000) as u64;
                let credits = g.ledger.distribute(need, job.reward - cut);
                let distributed: u64 = credits.iter().map(|(_, v)| *v).sum();
                g.ledger.blocks.push(BlockRec {
                    height: job.tpl.height,
                    hash,
                    time: now_secs(),
                    reward: job.reward,
                    pool_cut: cut + (job.reward - cut - distributed), // if the window is empty (should not happen), the pool keeps the rest
                    credits,
                    finder: finder.idx,
                    status: BlockStatus::Pending,
                });
                if g.ledger.blocks.len() > 5000 {
                    if let Some(i) = g.ledger.blocks.iter().position(|b| b.status != BlockStatus::Pending) {
                        g.ledger.blocks.remove(i);
                    }
                }
                g.c.blocks_found += 1;
                g.dirty = true;
                drop(g);
                self.log(&format!("BLOCK {} found by {} ({}); gross reward {}", job.tpl.height, finder.worker, &hex(&hash)[..8], fmt(job.reward)));
                Some(hash)
            }
            Err(e) => {
                self.inner.lock().unwrap().c.blocks_stale += 1;
                self.log(&format!("block {} refused: {e}", job.tpl.height));
                None
            }
        }
    }

    // ------------------------------------------------------------ statistics

    fn est_rate(recent: &VecDeque<(Instant, u128)>, since: Instant, tiles: usize) -> f64 {
        let now = Instant::now();
        let sum: u128 = recent.iter().map(|(_, w)| *w).sum();
        let dt = now.duration_since(since).as_secs_f64().clamp(30.0, 600.0);
        // a weight w = 2^96 / t is worth 2^64 / t = w / 2^32 tile attempts; a sigma = tiles^2 attempts
        sum as f64 / 4294967296.0 / (tiles * tiles) as f64 / dt
    }

    pub fn stats_json(&self) -> Value {
        let (len, target, nbal, cnonce) = {
            let n = self.sh.node.lock().unwrap();
            (n.chain.height(), n.chain.target, n.chain.state.balance(&self.pk), n.chain.state.nonce(&self.pk))
        };
        let g = self.inner.lock().unwrap();
        let tiles = g.cur.last().map_or(64, |j| j.tpl.params.tiles());
        let logged: Vec<&Session> = g.sess.values().filter(|s| s.login.is_some()).collect();
        let rate: f64 = logged.iter().map(|s| Self::est_rate(&s.recent, s.since.max(Instant::now() - Duration::from_secs(600)), tiles)).sum();
        let miners: HashSet<u32> = logged.iter().filter_map(|s| s.login.as_ref().map(|l| l.idx)).collect();
        let l = &g.ledger;
        let count = |st: BlockStatus| l.blocks.iter().filter(|b| b.status == st).count();
        // funds needed: owed + transfers not yet debited from the chain + net share of the pending blocks (already in the on-chain balance)
        let unspent: u64 = l.in_flight.iter().filter(|f| f.nonce >= cnonce).map(|f| f.amount).sum();
        let obligations = l.owed() + unspent + l.pending_total();
        json!({
            "proto": PROTO, "pool_address": address_of(&self.pk), "uptime_s": self.started.elapsed().as_secs(),
            "network": {"height": len, "target": u64hex(target), "work_per_block": block_work(target).to_string()},
            "fee_bps": self.cfg.fee_bps, "pplns_n": self.cfg.pplns_n, "confirmations": self.cfg.confirmations,
            "min_payout": self.cfg.min_payout, "payout_interval": self.cfg.payout_interval,
            "workers": logged.len(), "miners": miners.len(), "hashrate_sigma_per_s": rate,
            "shares": {"accepted": l.shares_accepted, "rejected_stale": g.c.rejected_stale, "rejected_duplicate": g.c.rejected_duplicate,
                       "rejected_low": g.c.rejected_low, "rejected_invalid": g.c.rejected_invalid},
            "blocks": {"found": g.c.blocks_found, "stale_submissions": g.c.blocks_stale, "pending": count(BlockStatus::Pending),
                       "confirmed": count(BlockStatus::Confirmed), "orphaned": count(BlockStatus::Orphaned)},
            "money": {"owed_to_miners": l.owed(), "in_flight": l.in_flight_total(), "pending_blocks": l.pending_total(),
                      "pool_fees_earned": l.pool_earned, "pool_wallet_balance": nbal, "obligations_confirmed": l.owed() + l.in_flight_total(),
                      "solvent": nbal >= obligations},
            "unit": "1 VRL = 100000000",
        })
    }

    pub fn miner_json(&self, addr: &PubKey) -> Value {
        let g = self.inner.lock().unwrap();
        let tiles = g.cur.last().map_or(64, |j| j.tpl.params.tiles());
        let Some(i) = g.ledger.find(addr) else {
            return json!({"address": address_of(addr), "known": false});
        };
        let l = &g.ledger;
        let mine: u128 = l.window.iter().filter(|(a, _)| *a == i).map(|(_, w)| *w).sum();
        let pending: u64 = l.blocks.iter().filter(|b| b.status == BlockStatus::Pending).flat_map(|b| b.credits.iter()).filter(|(a, _)| *a == i).map(|(_, v)| *v).sum();
        let workers: Vec<Value> = g
            .sess
            .values()
            .filter_map(|s| s.login.as_ref().filter(|l| l.idx == i).map(|lg| (s, lg)))
            .map(|(s, lg)| {
                json!({"worker": lg.worker, "agent": lg.agent, "accepted": s.accepted, "rejected": s.rejected, "share_target": u64hex(s.target),
                       "hashrate_sigma_per_s": Self::est_rate(&s.recent, s.since.max(Instant::now() - Duration::from_secs(600)), tiles),
                       "connected_s": s.since.elapsed().as_secs(), "last_share_s": s.last_share.map(|t| t.elapsed().as_secs())})
            })
            .collect();
        let pays: Vec<Value> = l.payments.iter().rev().filter(|p| p.to == i).take(20).map(|p| json!({"time": p.time, "amount": p.amount, "txid": hex(&p.txid)})).collect();
        json!({
            "address": address_of(addr), "known": true, "balance": l.balances.get(&i).copied().unwrap_or(0), "pending_confirmation": pending,
            "in_flight": l.in_flight.iter().filter(|f| f.to == i).map(|f| f.amount).sum::<u64>(),
            "paid_total": l.paid_total.get(&i).copied().unwrap_or(0),
            "window_share": if l.window_weight == 0 { 0.0 } else { mine as f64 / l.window_weight as f64 },
            "workers": workers, "payments": pays,
        })
    }

    fn blocks_json(&self) -> Value {
        let g = self.inner.lock().unwrap();
        let v: Vec<Value> = g
            .ledger
            .blocks
            .iter()
            .rev()
            .take(50)
            .map(|b| {
                json!({"height": b.height, "hash": hex(&b.hash), "time": b.time, "reward": b.reward, "pool_fee": b.pool_cut,
                       "status": format!("{:?}", b.status).to_lowercase(), "finder": address_of(&g.ledger.addrs[b.finder as usize])})
            })
            .collect();
        json!({"blocks": v})
    }

    fn payments_json(&self) -> Value {
        let g = self.inner.lock().unwrap();
        let v: Vec<Value> = g.ledger.payments.iter().rev().take(100).map(|p| json!({"time": p.time, "to": address_of(&g.ledger.addrs[p.to as usize]), "amount": p.amount, "txid": hex(&p.txid)})).collect();
        json!({"payments": v})
    }

    fn http(&self, req: Request) -> Response {
        if req.method != "GET" {
            return Response::error(405, "GET only");
        }
        match req.path.as_str() {
            "/" | "/index.html" => Response::html(POOL_PAGE),
            "/health" => Response::json(200, &json!({"ok": true})),
            "/api/stats" => Response::json(200, &self.stats_json()),
            "/api/blocks" => Response::json(200, &self.blocks_json()),
            "/api/payments" => Response::json(200, &self.payments_json()),
            p if p.starts_with("/api/miner/") => match parse_address(&p["/api/miner/".len()..]) {
                Ok(a) => Response::json(200, &self.miner_json(&a)),
                Err(e) => Response::error(400, &e),
            },
            _ => Response::error(404, "not found"),
        }
    }
}

fn fmt(v: u64) -> String {
    format!("{}.{:08} VRL", v / COIN, v % COIN)
}

/// Reads a line (terminated by \n) of at most MAX_LINE bytes. Ok(false): end of stream.
pub fn read_line_limited<R: BufRead>(r: &mut R, buf: &mut Vec<u8>) -> io::Result<bool> {
    loop {
        let avail = r.fill_buf()?;
        if avail.is_empty() {
            return Ok(false);
        }
        if let Some(pos) = avail.iter().position(|b| *b == b'\n') {
            if buf.len() + pos > MAX_LINE {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "line too long"));
            }
            buf.extend_from_slice(&avail[..pos]);
            r.consume(pos + 1);
            return Ok(true);
        }
        if buf.len() + avail.len() > MAX_LINE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "line too long"));
        }
        buf.extend_from_slice(avail);
        let l = avail.len();
        r.consume(l);
    }
}

const POOL_PAGE: &str = r#"<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vearl pool</title>
<style>body{font:15px/1.5 system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#e8ecf4;background:#0b0e17}
h1{font-size:1.4rem}table{border-collapse:collapse;width:100%}td,th{padding:.35rem .5rem;border-bottom:1px solid #243;text-align:left;font-size:.9rem}
code{background:#182033;padding:.1rem .3rem;border-radius:4px}input{width:100%;padding:.5rem;background:#10172a;color:inherit;border:1px solid #345;border-radius:6px}
.k{color:#8aa}</style>
<h1>Vearl pool <small class="k">vpool/1 &middot; testnet, tokens have no value</small></h1>
<div id="s">loading...</div>
<h2>Your miner</h2><input id="a" placeholder="your vrl1... address"><div id="m"></div>
<h2>Blocks</h2><div id="b"></div>
<script>
const f=(v)=>(v/1e8).toFixed(4)+" VRL";
async function j(u){const r=await fetch(u);return r.json()}
async function s(){try{const d=await j("/api/stats");
document.getElementById("s").innerHTML="<table><tr><td>Pool address</td><td><code>"+d.pool_address+"</code></td></tr><tr><td>Height</td><td>"+d.network.height+"</td></tr><tr><td>Workers / miners</td><td>"+d.workers+" / "+d.miners+"</td></tr><tr><td>Hashrate</td><td>"+d.hashrate_sigma_per_s.toFixed(2)+" sigma/s</td></tr><tr><td>Fee</td><td>"+d.fee_bps/100+" % (PPLNS x"+d.pplns_n+", "+d.confirmations+" confirmations)</td></tr><tr><td>Blocks found</td><td>"+d.blocks.found+" ("+d.blocks.confirmed+" confirmed, "+d.blocks.pending+" pending, "+d.blocks.orphaned+" orphaned)</td></tr><tr><td>Minimum payout</td><td>"+f(d.min_payout)+"</td></tr></table>";
const b=await j("/api/blocks");document.getElementById("b").innerHTML="<table><tr><th>Height</th><th>Reward</th><th>Status</th></tr>"+b.blocks.slice(0,15).map(x=>"<tr><td>"+x.height+"</td><td>"+f(x.reward)+"</td><td>"+x.status+"</td></tr>").join("")+"</table>";
const a=document.getElementById("a").value.trim();if(a){const m=await j("/api/miner/"+a);document.getElementById("m").innerHTML=m.known?"<table><tr><td>Balance</td><td>"+f(m.balance)+"</td></tr><tr><td>Pending confirmation</td><td>"+f(m.pending_confirmation)+"</td></tr><tr><td>Paid</td><td>"+f(m.paid_total)+"</td></tr><tr><td>Share of the PPLNS window</td><td>"+(m.window_share*100).toFixed(2)+" %</td></tr><tr><td>Workers</td><td>"+m.workers.map(w=>w.worker+" ("+w.hashrate_sigma_per_s.toFixed(2)+" sigma/s)").join(", ")+"</td></tr></table>":"unknown address (no share yet)"}}catch(e){document.getElementById("s").textContent="pool unreachable"}}
s();setInterval(s,10000);document.getElementById("a").onchange=s;
</script>"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn pk(b: u8) -> PubKey {
        [b; 32]
    }

    #[test]
    fn weights_are_inversely_proportional_to_the_target() {
        assert_eq!(share_weight(1 << 40), 2 * share_weight(1 << 41));
        assert!(share_weight(u64::MAX) >= 1);
    }

    #[test]
    fn distribution_is_exact_and_proportional() {
        let mut l = Ledger::new([0; 32], pk(9));
        let (a, b, c) = (l.idx(&pk(1)), l.idx(&pk(2)), l.idx(&pk(3)));
        for _ in 0..30 {
            l.add_share(a, 1 << 40, u128::MAX);
        }
        for _ in 0..10 {
            l.add_share(b, 1 << 40, u128::MAX);
        }
        l.add_share(c, 1 << 40, u128::MAX);
        let d = l.distribute(41 << 40, 1_000_000_007);
        assert_eq!(d.iter().map(|(_, v)| *v).sum::<u64>(), 1_000_000_007, "the distributed total is exactly the net reward");
        let va = d.iter().find(|(x, _)| *x == a).unwrap().1 as f64;
        let vb = d.iter().find(|(x, _)| *x == b).unwrap().1 as f64;
        assert!((va / vb - 3.0).abs() < 0.001);
    }

    #[test]
    fn the_window_only_counts_the_latest_shares_and_prorates_the_boundary() {
        let mut l = Ledger::new([0; 32], pk(9));
        let (a, b) = (l.idx(&pk(1)), l.idx(&pk(2)));
        l.add_share(a, 100, u128::MAX); // too old
        l.add_share(b, 100, u128::MAX);
        l.add_share(a, 100, u128::MAX);
        let d = l.distribute(150, 300); // 100 of a + 50 of b
        assert_eq!(d.iter().find(|(x, _)| *x == a).unwrap().1, 200);
        assert_eq!(d.iter().find(|(x, _)| *x == b).unwrap().1, 100);
        // window smaller than the need: everybody shares
        let d = l.distribute(10_000, 300);
        assert_eq!(d.iter().map(|(_, v)| *v).sum::<u64>(), 300);
    }

    #[test]
    fn the_window_is_bounded() {
        let mut l = Ledger::new([0; 32], pk(9));
        let a = l.idx(&pk(1));
        for _ in 0..1000 {
            l.add_share(a, 10, 100);
        }
        assert!(l.window.len() <= 11 && l.window_weight <= 100);
    }

    #[test]
    fn confirmation_credits_once_and_orphans_credit_nothing() {
        let mut l = Ledger::new([0; 32], pk(9));
        let a = l.idx(&pk(1));
        l.blocks.push(BlockRec { height: 5, hash: [1; 32], time: 0, reward: 100, pool_cut: 10, credits: vec![(a, 90)], finder: a, status: BlockStatus::Pending });
        l.blocks.push(BlockRec { height: 6, hash: [2; 32], time: 0, reward: 100, pool_cut: 10, credits: vec![(a, 90)], finder: a, status: BlockStatus::Orphaned });
        l.credit_block(0);
        l.credit_block(0); // second call has no effect
        l.credit_block(1); // orphaned: no effect
        assert_eq!(l.balances[&a], 90);
        assert_eq!(l.pool_earned, 10);
    }

    #[test]
    fn vardiff_converges_towards_the_wanted_share_rate_and_is_bounded() {
        // too many shares: harder target; at most x4 per step
        let t = retarget(1 << 50, 600, 30.0, 15.0, 1 << 20).unwrap();
        assert!(t < 1 << 50 && t >= (1 << 50) / 4);
        // too few: easier
        let t = retarget(1 << 50, 0, 30.0, 15.0, 1 << 20).unwrap();
        assert_eq!(t, (1u64 << 50) * 4);
        // within the band: unchanged
        assert_eq!(retarget(1 << 50, 2, 30.0, 15.0, 1 << 20), None);
        // never harder than the network target
        assert_eq!(retarget(1 << 21, 10_000, 30.0, 15.0, 1 << 21), None);
        let t = retarget(1 << 22, 10_000, 30.0, 15.0, 1 << 21).unwrap();
        assert!(t >= 1 << 21);
    }

    #[test]
    fn state_roundtrip_refuses_a_foreign_network() {
        let dir = std::env::temp_dir().join(format!("vpool-state-{}", rand::random::<u32>()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pool-state.bin");
        let mut l = Ledger::new([7; 32], pk(9));
        let a = l.idx(&pk(1));
        l.add_share(a, 5, 100);
        l.balances.insert(a, 42);
        l.save(&path).unwrap();
        l.save(&path).unwrap(); // second write: the old one becomes .bak
        let back = Ledger::load(&path, &[7; 32], &pk(9)).unwrap().unwrap();
        assert_eq!(back.balances[&a], 42);
        assert_eq!(back.find(&pk(1)), Some(a));
        assert!(Ledger::load(&path, &[8; 32], &pk(9)).is_err());
        assert!(Ledger::load(&path, &[7; 32], &pk(10)).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn hex_helpers() {
        assert_eq!(parse_u64hex(&u64hex(0x0123_4567_89ab_cdef)), Some(0x0123_4567_89ab_cdef));
        assert_eq!(parse_u64hex("123"), None);
        assert_eq!(parse_u64hex("zzzzzzzzzzzzzzzz"), None);
        assert_eq!(unhex32(&hex(&[7u8; 32])), Some([7u8; 32]));
    }
}
