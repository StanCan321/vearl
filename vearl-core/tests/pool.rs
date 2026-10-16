//! Pool tests (vpool/1): end to end with real CPU miners on a real node, hostile protocol, persistence.
use ed25519_dalek::SigningKey;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use vearl_core::chain::*;
use vearl_core::crypto::*;
use vearl_core::net::*;
use vearl_core::pool::*;
use vearl_core::poolminer::{parse_job, run, JobSpec, MinerConfig, MinerStats};
use vearl_core::state::*;
use vearl_core::types::*;
use vearl_core::wallet::address_of;
use vearl_core::work::*;

const SHARE_TARGET: u64 = u64::MAX / 1024;

fn genesis() -> GenesisConfig {
    let mut g = devnet_genesis();
    g.target = u64::MAX / 4096; // one block every ~65 sigma attempts: fast, but the shares are more frequent
    g.retarget = Retarget::Fixed;
    g
}

fn node(dir: Option<std::path::PathBuf>) -> NodeHandle {
    spawn_node(NodeConfig { genesis: genesis(), listen: "127.0.0.1:0".into(), peers: vec![], mine_key: None, use_gpu: false, log: false, worker: None, data_dir: dir }).unwrap()
}

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("vpool-{tag}-{}", rand::random::<u32>()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn cfg(dir: Option<std::path::PathBuf>) -> PoolConfig {
    PoolConfig {
        listen: "127.0.0.1:0".into(),
        fee_bps: 200,
        confirmations: 2,
        min_payout: 1,
        payout_interval: 1,
        fixed_share_target: Some(SHARE_TARGET),
        data_dir: dir,
        log: false,
        ..PoolConfig::default()
    }
}

fn wait(what: &str, secs: u64, mut f: impl FnMut() -> bool) {
    let t = Instant::now();
    while t.elapsed() < Duration::from_secs(secs) {
        if f() {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("timeout: {what}");
}

fn miner(pool: &Pool, seed: u8, worker: &str, threads: usize, hint: Option<f64>) -> (Arc<MinerStats>, Arc<AtomicBool>, String) {
    let address = address_of(&pubkey(&SigningKey::from_bytes(&[seed; 32])));
    let stats = Arc::new(MinerStats::default());
    let stop = Arc::new(AtomicBool::new(false));
    let c = MinerConfig { pool: pool.local_addr(), address: address.clone(), worker: worker.into(), threads, gpu: false, rate_hint: hint, log: false, receipts: None };
    let (s, st) = (stats.clone(), stop.clone());
    std::thread::spawn(move || run(c, s, st));
    (stats, stop, address)
}

/// Raw protocol client, for the hostile tests.
struct Raw {
    r: BufReader<TcpStream>,
    w: TcpStream,
    id: u64,
    jobs: Vec<Value>,
}

impl Raw {
    fn connect(addr: &str) -> Raw {
        let s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        Raw { r: BufReader::new(s.try_clone().unwrap()), w: s, id: 0, jobs: vec![] }
    }
    fn line(&mut self) -> Option<Value> {
        let mut l = String::new();
        match self.r.read_line(&mut l) {
            Ok(0) | Err(_) => None,
            Ok(_) => serde_json::from_str(&l).ok(),
        }
    }
    fn send_raw(&mut self, bytes: &[u8]) {
        let _ = self.w.write_all(bytes);
    }
    fn call(&mut self, method: &str, params: Value) -> Value {
        self.id += 1;
        let id = self.id;
        self.send_raw(format!("{}\n", json!({"id": id, "method": method, "params": params})).as_bytes());
        loop {
            let v = self.line().expect("connection closed while waiting for a reply");
            if v.get("id").and_then(|x| x.as_u64()) == Some(id) {
                return v;
            }
            if v.get("method").and_then(|m| m.as_str()) == Some("job") {
                self.jobs.push(v);
            }
        }
    }
    fn next_job(&mut self) -> JobSpec {
        loop {
            if let Some(j) = self.jobs.pop() {
                return parse_job(&j["params"]).unwrap();
            }
            let v = self.line().expect("no job");
            if v.get("method").and_then(|m| m.as_str()) == Some("job") {
                self.jobs.push(v);
            }
        }
    }
}

fn code(v: &Value) -> i64 {
    v["error"]["code"].as_i64().unwrap_or(0)
}

fn login(c: &mut Raw, seed: u8) -> Value {
    c.call("login", json!({"proto": PROTO, "address": address_of(&pubkey(&SigningKey::from_bytes(&[seed; 32]))), "worker": "t", "sigma_rate": 1000.0}))
}

/// Looks for a share (and a tile that is NOT one) in the worker's range.
fn find_shares(j: &JobSpec, share_target: u64) -> ((u64, usize), (u64, usize)) {
    let (a, b) = native_matrices(j.epoch, &j.params);
    let (mut good, mut bad) = (None, None);
    let mut en = j.nonce_start;
    while good.is_none() || bad.is_none() {
        let sg = sigma(&j.prev, &j.miner, &WorkClass::Native, en, &j.tx_root, j.timestamp);
        let (out, _) = run_noisy_gemm(&a, &b, &sg, &j.params);
        for (idx, jp) in out.jackpots.iter().enumerate() {
            if meets_target(jp, share_target) {
                good.get_or_insert((en, idx));
            } else {
                bad.get_or_insert((en, idx));
            }
        }
        en += 1;
        assert!(en < j.nonce_start + 100_000);
    }
    (good.unwrap(), bad.unwrap())
}

#[test]
fn tile_jackpot_matches_the_full_run() {
    let p = WorkParams::TESTNET;
    let (a, b) = native_matrices(3, &p);
    let cm = CommittedMatrices::new(a.clone(), b.clone(), &p);
    let sg = sigma(&[1; 32], &[2; 32], &WorkClass::Native, 77, &[0; 32], 1_700_000_123);
    let (out, _) = run_noisy_gemm(&a, &b, &sg, &p);
    let tl = p.tiles();
    for (ti, tj) in [(0, 0), (0, tl - 1), (tl - 1, 0), (3, 5), (tl - 1, tl - 1)] {
        let jp = tile_jackpot(&cm, &sg, ti, tj, &p).unwrap();
        assert_eq!(jp, out.jackpots[ti * tl + tj], "tile ({ti},{tj})");
        let proof = build_proof(&cm, ti, tj, &p);
        assert_eq!(verify_tile_proof(&proof, &sg, &cm.a_root(), &cm.b_root(), &p), Some(jp));
    }
    assert!(tile_jackpot(&cm, &sg, tl, 0, &p).is_none());
}

#[test]
fn pool_mines_blocks_splits_the_rewards_and_pays_on_chain() {
    let nd = node(None);
    let pool = Pool::start(cfg(None), nd.shared.clone(), SigningKey::from_bytes(&[150; 32])).unwrap();
    let pk = pool.pk;
    // a transaction with a fee: the pool's reward must include it (fees = to the block's miner)
    {
        let faucet = SigningKey::from_bytes(&[100; 32]);
        let tx = {
            let n = nd.shared.node.lock().unwrap();
            make_tx(&n.chain, &faucet, 5_000, TxKind::Transfer { to: [9; 32], amount: 7 * COIN })
        };
        submit_tx(&nd.shared, tx).unwrap();
    }
    let (s1, stop1, a1) = miner(&pool, 201, "rig1", 1, Some(1000.0));
    let (s2, stop2, a2) = miner(&pool, 202, "rig2", 1, Some(1000.0));
    let (p1, p2) = (parse_address(&a1), parse_address(&a2));
    wait("blocks found and confirmed, shares from both miners", 180, || {
        let st = pool.stats_json();
        st["blocks"]["confirmed"].as_u64().unwrap() >= 6 && s1.accepted.load(Ordering::Relaxed) > 5 && s2.accepted.load(Ordering::Relaxed) > 5
    });
    // both miners eventually receive REAL funds on the chain
    wait("on-chain payouts to both miners", 120, || {
        let n = nd.shared.node.lock().unwrap();
        n.chain.state.balance(&p1) > 0 && n.chain.state.balance(&p2) > 0
    });
    stop1.store(true, Ordering::SeqCst);
    stop2.store(true, Ordering::SeqCst);
    // let the last payouts/blocks in flight settle before comparing the accounts
    std::thread::sleep(Duration::from_secs(3));

    // 1. each reward of the registry = emission + fees of the real block of the chain
    let n = nd.shared.node.lock().unwrap();
    let mut chain_total = 0u64;
    pool.ledger_snapshot(|l| {
        for b in &l.blocks {
            let on = n.chain.blocks.get(b.height as usize).filter(|x| x.header.hash() == b.hash);
            if let Some(blk) = on {
                let fees: u64 = blk.txs.iter().map(|t| t.tx.fee).sum();
                assert_eq!(b.reward, base_reward(b.height) + fees, "gross reward of block {}", b.height);
                assert_eq!(blk.header.miner, pk);
                assert_eq!(b.credits.iter().map(|(_, v)| *v).sum::<u64>() + b.pool_cut, b.reward, "the split of block {} is exact", b.height);
            } else {
                assert_ne!(b.status, BlockStatus::Confirmed, "a confirmed block must be in the chain");
            }
        }
        // 2. conservation: everything that was confirmed is either owed, in flight, paid, or the pool's fees
        let confirmed: u64 = l.blocks.iter().filter(|b| b.status == BlockStatus::Confirmed).map(|b| b.reward).sum();
        let paid: u64 = l.paid_total.values().sum();
        assert_eq!(confirmed, l.owed() + l.in_flight_total() + paid + l.pool_earned, "conservation of the pool's funds");
        assert!(l.pool_earned > 0, "the pool takes its fees");
        // the fees are about 2 % (rounded per block)
        assert!((l.pool_earned as f64 / confirmed as f64 - 0.02).abs() < 0.005);
    });
    // 3. the chain: pool balance = mined rewards - included transfers
    for b in n.chain.blocks.iter().filter(|b| b.header.miner == pk) {
        chain_total += base_reward(b.header.height) + b.txs.iter().map(|t| t.tx.fee).sum::<u64>();
    }
    let sent: u64 = n
        .chain
        .blocks
        .iter()
        .flat_map(|b| b.txs.iter())
        .filter(|t| t.tx.from == pk)
        .map(|t| match &t.tx.kind {
            TxKind::Transfer { amount, .. } => *amount,
            _ => 0,
        })
        .sum();
    assert!(sent > 0, "at least one payout is included in the chain");
    assert_eq!(n.chain.state.balance(&pk), chain_total - sent, "on-chain balance of the pool");
    assert!(chain_total > base_reward(0) * 6, "the transaction fees are in the rewards");
    // 4. the miners received what they are owed, nothing more
    let (b1, b2) = (n.chain.state.balance(&p1), n.chain.state.balance(&p2));
    pool.ledger_snapshot(|l| {
        let (i1, i2) = (l.find(&p1).unwrap(), l.find(&p2).unwrap());
        let paid1 = l.paid_total.get(&i1).copied().unwrap_or(0);
        let paid2 = l.paid_total.get(&i2).copied().unwrap_or(0);
        assert!(b1 >= paid1 && b2 >= paid2);
        assert!(b1 <= paid1 + l.in_flight.iter().filter(|f| f.to == i1).map(|f| f.amount).sum::<u64>());
        assert!(b2 <= paid2 + l.in_flight.iter().filter(|f| f.to == i2).map(|f| f.amount).sum::<u64>());
    });
    drop(n);
    let st = pool.stats_json();
    assert_eq!(st["money"]["solvent"], true);
    // 5. the statistics APIs answer with the same figures
    let m = pool.miner_json(&p1);
    assert_eq!(m["known"], true);
    assert!(m["window_share"].as_f64().unwrap() > 0.0);
    assert_eq!(pool.miner_json(&[0xee; 32])["known"], false);
    pool.stop();
    nd.stop();
}

#[test]
fn hostile_clients_cannot_corrupt_the_accounting() {
    let nd = node(None);
    let pool = Pool::start(cfg(None), nd.shared.clone(), SigningKey::from_bytes(&[151; 32])).unwrap();
    let addr = pool.local_addr();

    // submission before login, unknown method, invalid JSON
    let mut c = Raw::connect(&addr);
    assert_eq!(code(&c.call("submit", json!({"job_id": "x", "extra_nonce": "0000000000000000", "ti": 0, "tj": 0}))), E_UNAUTHORIZED as i64);
    assert_eq!(code(&c.call("frobnicate", json!({}))), E_BAD_REQUEST as i64);
    c.send_raw(b"{not json\n");
    assert_eq!(code(&c.line().unwrap()), E_BAD_REQUEST as i64);
    // wrong protocol, invalid address, invalid worker name
    assert_eq!(code(&c.call("login", json!({"proto": "vpool/9", "address": "x"}))), E_BAD_REQUEST as i64);
    assert_eq!(code(&c.call("login", json!({"proto": PROTO, "address": "vrl1notanaddress"}))), E_BAD_REQUEST as i64);
    let good_addr = address_of(&pubkey(&SigningKey::from_bytes(&[1; 32])));
    assert_eq!(code(&c.call("login", json!({"proto": PROTO, "address": good_addr, "worker": "bad name!"}))), E_BAD_REQUEST as i64);
    assert_eq!(code(&c.call("login", json!({"proto": PROTO, "address": address_of(&pool.pk)}))), E_BAD_REQUEST as i64, "the pool address is not a payout address");

    // correct login, then second login refused
    let r = login(&mut c, 1);
    assert!(r["result"]["slot"].is_u64(), "{r}");
    let slot = r["result"]["slot"].as_u64().unwrap();
    assert_eq!(code(&login(&mut c, 1)), E_BAD_REQUEST as i64);
    let job = c.next_job();
    assert_eq!(job.nonce_start >> NONCE_BITS, slot);
    let ((en_ok, idx_ok), (en_bad, idx_bad)) = find_shares(&job, job.share_target);
    let tl = job.params.tiles();
    let sub = |c: &mut Raw, job_id: &str, en: u64, idx: usize| c.call("submit", json!({"job_id": job_id, "extra_nonce": u64hex(en), "ti": idx / tl, "tj": idx % tl}));

    // valid share: accepted only once
    let r = sub(&mut c, &job.id, en_ok, idx_ok);
    assert_eq!(r["result"]["accepted"], true, "{r}");
    assert_eq!(code(&sub(&mut c, &job.id, en_ok, idx_ok)), E_DUPLICATE as i64);
    // unknown job
    assert_eq!(code(&sub(&mut c, "deadbeefcafe", en_ok, idx_ok)), E_STALE as i64);
    // nonce outside our range (that of another worker): refused even if the jackpot would be good
    let foreign = en_ok ^ (1u64 << 62);
    assert_eq!(code(&sub(&mut c, &job.id, foreign, idx_ok)), E_INVALID as i64);
    // tile out of bounds
    assert_eq!(code(&c.call("submit", json!({"job_id": job.id, "extra_nonce": u64hex(en_ok + 1), "ti": 9999, "tj": 0}))), E_INVALID as i64);
    // jackpot too high: the share is not one
    assert_eq!(code(&sub(&mut c, &job.id, en_bad, idx_bad)), E_LOW_DIFF as i64);
    // a valid share is not transferable to another worker or to another address: only the author of the session is credited
    let st = pool.stats_json();
    assert_eq!(st["shares"]["accepted"], 1);
    assert_eq!(st["shares"]["rejected_duplicate"], 1);
    assert_eq!(st["shares"]["rejected_low"], 1);
    assert_eq!(st["shares"]["rejected_stale"], 1);
    assert_eq!(st["shares"]["rejected_invalid"], 2);
    assert_eq!(pool.ledger_snapshot(|l| l.window.len()), 1);

    // a 10 KB line without an end of line cuts the connection
    let mut big = Raw::connect(&addr);
    big.send_raw(&vec![b'a'; 10_000]);
    assert!(big.line().is_none(), "the connection must be closed");
    // five invalid lines: disconnection
    let mut junk = Raw::connect(&addr);
    for _ in 0..6 {
        junk.send_raw(b"}{\n");
    }
    let mut n = 0;
    while junk.line().is_some() {
        n += 1;
        assert!(n < 20);
    }
    // ping works and the pool still holds
    let mut ok = Raw::connect(&addr);
    assert!(ok.call("ping", json!({}))["result"]["pong"].is_u64());
    pool.stop();
    nd.stop();
}

#[test]
fn per_ip_limit_is_enforced() {
    let nd = node(None);
    let mut c = cfg(None);
    c.max_per_ip = 3;
    let pool = Pool::start(c, nd.shared.clone(), SigningKey::from_bytes(&[152; 32])).unwrap();
    let addr = pool.local_addr();
    let mut keep: Vec<Raw> = (0..3).map(|_| Raw::connect(&addr)).collect();
    for k in keep.iter_mut() {
        assert!(k.call("ping", json!({}))["result"]["pong"].is_u64());
    }
    let mut over = Raw::connect(&addr);
    let v = over.line().unwrap();
    assert_eq!(code(&v), E_FULL as i64);
    pool.stop();
    nd.stop();
}

#[test]
fn vardiff_moves_the_share_target_towards_the_wanted_rate() {
    let nd = node(None);
    let mut c = cfg(None);
    c.fixed_share_target = None;
    c.share_secs = 1.0;
    c.retarget_secs = 3;
    let pool = Pool::start(c, nd.shared.clone(), SigningKey::from_bytes(&[153; 32])).unwrap();
    let (stats, stop, a) = miner(&pool, 203, "rig", 1, None);
    let pa = parse_address(&a);
    let mut first = None;
    let t = Instant::now();
    let mut moved = false;
    while t.elapsed() < Duration::from_secs(60) && !moved {
        std::thread::sleep(Duration::from_millis(300));
        let m = pool.miner_json(&pa);
        if let Some(w) = m["workers"].as_array().and_then(|w| w.first()) {
            let tgt = parse_u64hex(w["share_target"].as_str().unwrap()).unwrap();
            match first {
                None => first = Some(tgt),
                Some(f) => moved = tgt != f,
            }
        }
    }
    assert!(moved, "the share target should have been adjusted");
    assert!(stats.accepted.load(Ordering::Relaxed) > 0);
    stop.store(true, Ordering::SeqCst);
    pool.stop();
    nd.stop();
}

#[test]
fn state_and_balances_survive_a_restart_and_a_foreign_network_is_refused() {
    let (ndir, pdir) = (tmpdir("node"), tmpdir("pool"));
    let nd = node(Some(ndir.clone()));
    let key = SigningKey::from_bytes(&[154; 32]);
    let pool = Pool::start(cfg(Some(pdir.clone())), nd.shared.clone(), key.clone()).unwrap();
    let (s, stop, a) = miner(&pool, 204, "rig", 1, Some(1000.0));
    wait("three confirmed blocks", 180, || pool.stats_json()["blocks"]["confirmed"].as_u64().unwrap() >= 3 && s.accepted.load(Ordering::Relaxed) > 3);
    stop.store(true, Ordering::SeqCst);
    std::thread::sleep(Duration::from_secs(1));
    pool.stop(); // saves the state
    let before = pool.ledger_snapshot(|l| (l.blocks.len(), l.shares_accepted, l.addrs.clone(), l.window.len(), l.pool_earned + l.owed() + l.in_flight_total() + l.paid_total.values().sum::<u64>()));
    nd.stop();
    drop(nd);

    // same network, same keys: everything is found again
    let nd2 = node(Some(ndir.clone()));
    let pool2 = Pool::start(cfg(Some(pdir.clone())), nd2.shared.clone(), key).unwrap();
    let after = pool2.ledger_snapshot(|l| (l.blocks.len(), l.shares_accepted, l.addrs.clone(), l.window.len(), l.pool_earned + l.owed() + l.in_flight_total() + l.paid_total.values().sum::<u64>()));
    assert_eq!(before, after);
    assert!(after.2.contains(&parse_address(&a)));
    pool2.stop();
    nd2.stop();

    // another pool key: flat refusal, no overwriting of the state
    let nd3 = node(None);
    let r = Pool::start(cfg(Some(pdir.clone())), nd3.shared.clone(), SigningKey::from_bytes(&[155; 32]));
    assert!(r.is_err() && r.err().unwrap().contains("another pool key"));
    nd3.stop();
    let _ = std::fs::remove_dir_all(ndir);
    let _ = std::fs::remove_dir_all(pdir);
}

#[test]
fn a_measured_rate_hint_moves_the_share_target_at_once() {
    let nd = node(None);
    let mut c = cfg(None);
    c.fixed_share_target = None;
    c.share_secs = 0.5; // on this tiny devnet, the initial target (10 sigma/s) stays easier than the network's
    let pool = Pool::start(c, nd.shared.clone(), SigningKey::from_bytes(&[156; 32])).unwrap();
    let mut w = Raw::connect(&pool.local_addr());
    assert_eq!(code(&w.call("hint", json!({"sigma_rate": 100.0}))), E_UNAUTHORIZED as i64);
    let a5 = address_of(&pubkey(&SigningKey::from_bytes(&[5; 32])));
    let r = w.call("login", json!({"proto": PROTO, "address": a5, "worker": "t"}));
    let t0 = parse_u64hex(r["result"]["share_target"].as_str().unwrap()).unwrap();
    // wait for the first job: until the pool has one, a hint is computed with default parameters and could equal the login target
    // (on a slow machine the pool is not ready yet when the test logs in), which made this test fail now and then
    let _ = w.next_job();
    assert_eq!(code(&w.call("hint", json!({"sigma_rate": "fast"}))), E_BAD_REQUEST as i64);
    assert_eq!(code(&w.call("hint", json!({"sigma_rate": -3.0}))), E_BAD_REQUEST as i64);
    let r = w.call("hint", json!({"sigma_rate": 100000.0}));
    let t1 = parse_u64hex(r["result"]["share_target"].as_str().unwrap()).unwrap();
    assert!(t1 < t0, "a huge rate gives a harder target: {t1:016x} < {t0:016x}");
    assert!(t1 >= u64::MAX / 4096, "never harder than the network target");
    assert!(w.call("hint", json!({"sigma_rate": 10.0}))["result"].is_object());
    assert!(w.call("hint", json!({"sigma_rate": 20.0}))["result"].is_object());
    assert_eq!(code(&w.call("hint", json!({"sigma_rate": 10.0}))), E_RATE as i64, "at most 3 hints per session");
    pool.stop();
    nd.stop();
}

#[test]
fn a_burst_of_submissions_is_throttled_but_never_bans() {
    let nd = node(None);
    let pool = Pool::start(cfg(None), nd.shared.clone(), SigningKey::from_bytes(&[157; 32])).unwrap();
    let mut w = Raw::connect(&pool.local_addr());
    assert!(login(&mut w, 6)["result"].is_object());
    let mut limited = 0;
    for _ in 0..400 {
        let r = w.call("submit", json!({"job_id": "000000000000", "extra_nonce": "0000000000000000", "ti": 0, "tj": 0}));
        if code(&r) == E_RATE as i64 {
            limited += 1;
        } else {
            assert_eq!(code(&r), E_STALE as i64);
        }
    }
    assert!(limited > 100, "the token bucket must limit the burst ({limited} refusals)");
    assert_eq!(pool.counters().bans, 0, "a too fast rate is not a banning fault");
    assert!(w.call("ping", json!({}))["result"]["pong"].is_u64(), "the connection stays open");
    pool.stop();
    nd.stop();
}

fn parse_address(s: &str) -> PubKey {
    vearl_core::wallet::parse_address(s).unwrap()
}
