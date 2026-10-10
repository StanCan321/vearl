//! vearl-pool: Vearl mining pool (vpool/1 protocol, docs/16-pool-protocol.md). Runs a full node (without mining of its own)
//! and the pool server.
//!
//! vearl-pool --wallet POOL.json [--account 0] [--password-file F]   (or --insecure-devnet-key N: PUBLIC seed key, devnet)
//!            [--genesis FILE] [--listen 0.0.0.0:9000] [--peer HOST:PORT]... [--pool-listen 0.0.0.0:3333] [--http 127.0.0.1:8650]
//!            [--data-dir DIR] [--fee 1.0] [--min-payout 1] [--payout-interval 600] [--confirmations 10] [--pplns 2.0]
//!            [--share-secs 15] [--refresh-secs 30] [--max-conns 2000] [--max-per-ip 64] [--api ADDR [--api-public]]
//!            [--fixed-share-target HEX16]   (tests: disables vardiff)
//! The pool wallet receives the block rewards and signs the payouts: DEDICATED wallet, with few funds of its own.
use ed25519_dalek::SigningKey;
use vearl_core::api::spawn_api;
use vearl_core::chain::devnet_genesis;
use vearl_core::genesis::parse_genesis;
use vearl_core::net::*;
use vearl_core::pool::{parse_u64hex, Pool, PoolConfig};
use vearl_core::wallet::*;
use std::time::Duration;

fn die(m: &str) -> ! {
    eprintln!("error: {m}");
    std::process::exit(1)
}

const HELP: &str = r#"vearl-pool: a Vearl mining pool (protocol vpool/1). It runs a full node (without mining of its own) and the pool server.

  vearl-pool --wallet POOL.json [--genesis genesis.json] [--peer HOST:PORT]... [--listen 0.0.0.0:9000] [--pool-listen 0.0.0.0:3333]
             [--http 127.0.0.1:8650] [--data-dir DIR] [--fee 1.0] [--min-payout 1] [--payout-interval 600] [--confirmations 10]
             [--password-file FILE]

The pool wallet receives the block rewards and signs the payouts: use a DEDICATED wallet with few funds of its own.
All the options and the protocol are in docs/16."#;

fn main() {
    vearl_core::launcher::help_or_continue(HELP);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let all = |name: &str| -> Vec<String> { args.iter().enumerate().filter(|(_, a)| *a == name).filter_map(|(i, _)| args.get(i + 1).cloned()).collect() };
    let num = |name: &str, d: f64| -> f64 { arg(name).map_or(d, |v| v.parse().unwrap_or_else(|_| die(&format!("{name}: number expected")))) };
    for a in &args {
        if a.starts_with("--") && !KNOWN.contains(&a.as_str()) {
            die(&format!("unknown option: {a}"));
        }
    }

    let sk: SigningKey = if let Some(f) = arg("--wallet") {
        let pw = read_password("Pool wallet password: ", arg("--password-file").as_deref()).unwrap_or_else(|e| die(&e));
        load_signing_key(std::path::Path::new(&f), &pw, arg("--account").map_or(0, |v| v.parse().unwrap_or_else(|_| die("--account: integer")))).unwrap_or_else(|e| die(&e))
    } else if let Some(s) = arg("--insecure-devnet-key") {
        eprintln!("[pool] PUBLIC SEED key (devnet only, funds without value)");
        SigningKey::from_bytes(&[s.parse::<u8>().unwrap_or_else(|_| die("--insecure-devnet-key: 0..255")); 32])
    } else {
        die("--wallet FILE (or --insecure-devnet-key N for a devnet) required");
    };
    let genesis = match arg("--genesis") {
        Some(f) => {
            let txt = std::fs::read_to_string(&f).unwrap_or_else(|e| die(&format!("{f}: {e}")));
            let (name, g) = parse_genesis(&txt).unwrap_or_else(|e| die(&format!("invalid genesis: {e}")));
            eprintln!("[pool] network '{name}' (genesis {f})");
            g
        }
        None => {
            eprintln!("[pool] built-in DEVNET genesis");
            devnet_genesis()
        }
    };
    let data_dir = arg("--data-dir").map(std::path::PathBuf::from);
    let cfg = PoolConfig {
        listen: arg("--pool-listen").unwrap_or_else(|| "127.0.0.1:3333".into()),
        http: arg("--http"),
        fee_bps: (num("--fee", 1.0) * 100.0).round() as u64,
        pplns_n: num("--pplns", 2.0),
        confirmations: num("--confirmations", 10.0) as u64,
        min_payout: parse_amount(&arg("--min-payout").unwrap_or_else(|| "1".into())).unwrap_or_else(|e| die(&e)),
        payout_interval: num("--payout-interval", 600.0) as u64,
        share_secs: num("--share-secs", 15.0),
        refresh_secs: num("--refresh-secs", 30.0) as u64,
        fixed_share_target: arg("--fixed-share-target").map(|v| parse_u64hex(&v).unwrap_or_else(|| die("--fixed-share-target: 16 hexadecimal digits"))),
        max_conns: num("--max-conns", 2000.0) as usize,
        max_per_ip: num("--max-per-ip", 64.0) as usize,
        data_dir: data_dir.as_ref().map(|d| d.join("pool")),
        ..PoolConfig::default()
    };
    if cfg.fee_bps > 10_000 || cfg.pplns_n < 0.1 || cfg.share_secs < 0.5 {
        die("parameters out of bounds (--fee 0..100, --pplns >= 0.1, --share-secs >= 0.5)");
    }
    let handle = spawn_node(NodeConfig {
        genesis,
        listen: arg("--listen").unwrap_or_else(|| "127.0.0.1:9000".into()),
        peers: all("--peer"),
        mine_key: None,
        use_gpu: false,
        log: true,
        worker: None,
        data_dir: data_dir.as_ref().map(|d| d.join("node")),
    })
    .unwrap_or_else(|e| die(&format!("node: {e}")));
    eprintln!("[pool] node on {} (height {})", handle.addr, handle.height());
    if let Some(a) = arg("--api") {
        let loopback = a.parse::<std::net::SocketAddr>().map(|s| s.ip().is_loopback()).unwrap_or(false);
        if !loopback && !flag("--api-public") {
            die("--api on a non-local address: add --api-public (and put a TLS reverse proxy in front)");
        }
        let bound = spawn_api(handle.shared.clone(), &a).unwrap_or_else(|e| die(&format!("API: {e}")));
        eprintln!("[pool] API + explorer on http://{bound}/");
    }
    let pool = Pool::start(cfg.clone(), handle.shared.clone(), sk).unwrap_or_else(|e| die(&e));
    eprintln!("[pool] pool address {}; workers on {}; fee {} %; PPLNS x{}; {} confirmations", pool.address(), pool.local_addr(), cfg.fee_bps as f64 / 100.0, cfg.pplns_n, cfg.confirmations);
    if let Some(h) = pool.http_addr.lock().unwrap().clone() {
        eprintln!("[pool] statistics on http://{h}/");
    }
    if !cfg.listen.starts_with("127.") && !cfg.listen.starts_with("[::1]") {
        eprintln!("[pool] WARNING: the protocol is not encrypted; put a TLS proxy (stunnel / nginx stream) in front of {} for public use", cfg.listen);
    }
    loop {
        std::thread::sleep(Duration::from_secs(30));
        let s = pool.stats_json();
        eprintln!(
            "[pool] height={} workers={} rate={:.2} sigma/s shares={} blocks found={} (pending {}, confirmed {}, orphaned {}) owed to miners={}",
            s["network"]["height"], s["workers"], s["hashrate_sigma_per_s"].as_f64().unwrap_or(0.0), s["shares"]["accepted"], s["blocks"]["found"], s["blocks"]["pending"], s["blocks"]["confirmed"], s["blocks"]["orphaned"], fmt_amount(s["money"]["owed_to_miners"].as_u64().unwrap_or(0))
        );
        if s["money"]["solvent"] == false {
            eprintln!("[pool] WARNING: the pool wallet balance is lower than the amounts owed to miners");
        }
    }
}

const KNOWN: &[&str] = &[
    "--wallet", "--account", "--password-file", "--insecure-devnet-key", "--genesis", "--listen", "--peer", "--pool-listen", "--http", "--data-dir",
    "--fee", "--min-payout", "--payout-interval", "--confirmations", "--pplns", "--share-secs", "--refresh-secs", "--max-conns", "--max-per-ip", "--api",
    "--api-public", "--fixed-share-target",
];
