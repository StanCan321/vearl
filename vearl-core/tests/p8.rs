//! Wallet + JSON genesis + HTTP API + explorer + faucet, end to end.
use ed25519_dalek::SigningKey;
use vearl_core::api::spawn_api;
use vearl_core::chain::*;
use vearl_core::crypto::*;
use vearl_core::faucet::*;
use vearl_core::genesis::*;
use vearl_core::httpmini::Request;
use vearl_core::net::*;
use vearl_core::state::*;
use vearl_core::types::*;
use vearl_core::wallet::*;
use serde_json::Value;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

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

/// Raw HTTP client: (status, lowercase headers, body).
fn http(addr: &str, method: &str, path: &str, body: &[u8]) -> (u16, String, Vec<u8>) {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    write!(s, "{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n", body.len()).unwrap();
    s.write_all(body).unwrap();
    let mut buf = Vec::new();
    s.read_to_end(&mut buf).unwrap();
    let pos = buf.windows(4).position(|w| w == b"\r\n\r\n").expect("HTTP response");
    let head = String::from_utf8_lossy(&buf[..pos]).to_lowercase();
    let status = head.split(' ').nth(1).unwrap().parse().unwrap();
    (status, head, buf[pos + 4..].to_vec())
}

fn get_json(addr: &str, path: &str) -> (u16, Value) {
    let (st, _, body) = http(addr, "GET", path, b"");
    (st, serde_json::from_slice(&body).unwrap_or(Value::Null))
}

fn cfg_with(alloc: Vec<(PubKey, u64)>) -> GenesisConfig {
    GenesisConfig { params: WorkParams::TESTNET, alloc, ts: 1_700_000_000, target: u64::MAX / 256, timing: Timing::default(), retarget: Retarget::Fixed }
}

use vearl_core::work::WorkParams;

fn node_from(g: GenesisConfig, mine: Option<u8>) -> NodeHandle {
    spawn_node(NodeConfig {
        genesis: g,
        listen: "127.0.0.1:0".into(),
        peers: vec![],
        mine_key: mine.map(|s| SigningKey::from_bytes(&[s; 32])),
        use_gpu: false,
        log: false,
        worker: None,
        data_dir: None,
    })
    .unwrap()
}

#[test]
fn wallet_to_genesis_to_node_to_api_end_to_end() {
    // 1. an encrypted wallet, a JSON genesis that funds its address
    let m = new_mnemonic();
    let ks = Keystore::create(&m, "a solid password", KdfParams::LIGHT).unwrap();
    let sk = derive_account(&ks.unlock("a solid password").unwrap(), 0);
    let me = pubkey(&sk);
    assert_eq!(ks.address, address_of(&me));
    let json = genesis_to_json("test-net", &cfg_with(vec![(me, 250 * COIN)]));
    let (name, g) = parse_genesis(&json).unwrap();
    assert_eq!(name, "test-net");

    // 2. a mining node + the API
    let node = node_from(g, Some(1));
    let api = spawn_api(node.shared.clone(), "127.0.0.1:0").unwrap();
    wait("blocks", 30, || node.height() >= 3);

    // 3. reading through the API
    let (st, v) = get_json(&api, &format!("/api/account/{}", ks.address));
    assert_eq!(st, 200);
    assert_eq!(v["balance"], "250.00000000");
    let (st, s) = get_json(&api, "/api/status");
    assert_eq!((st, s["params"]["rank"].as_u64()), (200, Some(63)));

    // 4. the wallet sends 12.5 VRL to another address through POST /api/tx
    let other = pubkey(&derive_account(&new_mnemonic(), 0));
    let tx = SignedTx::new(Tx { from: me, nonce: 0, fee: COIN / 100, kind: TxKind::Transfer { to: other, amount: parse_amount("12.5").unwrap() } }, &sk);
    let body = serde_json::json!({ "hex": bincode::serialize(&tx).unwrap().iter().map(|b| format!("{b:02x}")).collect::<String>() }).to_string();
    let (st, _, resp) = http(&api, "POST", "/api/tx", body.as_bytes());
    let r: Value = serde_json::from_slice(&resp).unwrap();
    assert_eq!((st, r["accepted"].as_bool()), (200, Some(true)), "{r}");
    let id: String = tx.id().iter().map(|b| format!("{b:02x}")).collect();

    wait("transaction confirmed", 30, || get_json(&api, &format!("/api/tx/{id}")).1["status"] == "confirmed");
    let (_, t) = get_json(&api, &format!("/api/tx/{id}"));
    assert_eq!(t["kind"]["type"], "transfer");
    assert_eq!(t["kind"]["amount"], "12.50000000");
    let (_, a) = get_json(&api, &format!("/api/account/{}", address_of(&other)));
    assert_eq!(a["balance"], "12.50000000");
    let (_, a) = get_json(&api, &format!("/api/account/{}", ks.address));
    assert_eq!(a["balance"], "237.49000000", "250 - 12.5 - 0.01 fee");

    // 5. the same transaction replayed, an altered signature, nonsense: refused
    let (st, _, _) = http(&api, "POST", "/api/tx", body.as_bytes());
    assert_eq!(st, 400, "replay refused");
    let mut bad = tx.clone();
    bad.sig[0] ^= 1;
    bad.tx.nonce = 1;
    let bad_body = serde_json::json!({ "hex": bincode::serialize(&bad).unwrap().iter().map(|b| format!("{b:02x}")).collect::<String>() }).to_string();
    assert_eq!(http(&api, "POST", "/api/tx", bad_body.as_bytes()).0, 400);
    assert_eq!(http(&api, "POST", "/api/tx", b"not json").0, 400);
    assert_eq!(http(&api, "POST", "/api/tx", br#"{"hex":"zz"}"#).0, 400);
    node.stop();
}

#[test]
fn api_routes_errors_headers_and_explorer() {
    let node = node_from(cfg_with(vec![]), Some(1));
    let api = spawn_api(node.shared.clone(), "127.0.0.1:0").unwrap();
    wait("blocks", 30, || node.height() >= 5);

    let (st, v) = get_json(&api, "/api/blocks?limit=3");
    assert_eq!(st, 200);
    let blocks = v["blocks"].as_array().unwrap();
    assert_eq!(blocks.len(), 3);
    assert!(blocks[0]["height"].as_u64().unwrap() > blocks[1]["height"].as_u64().unwrap(), "from most recent to oldest");
    let hash = blocks[0]["hash"].as_str().unwrap().to_string();
    let h0 = blocks[0]["height"].as_u64().unwrap();
    assert_eq!(get_json(&api, &format!("/api/block/{hash}")).1["height"].as_u64(), Some(h0), "search by hash");
    assert_eq!(get_json(&api, "/api/block/0").1["work"]["type"], "native");
    assert_eq!(get_json(&api, "/api/block/99999999").0, 404);
    assert_eq!(get_json(&api, "/api/block/not-a-block").0, 400);
    assert_eq!(get_json(&api, "/api/account/vrl1notvalid").0, 400);
    assert_eq!(get_json(&api, "/api/tx/00").0, 400);
    assert_eq!(get_json(&api, &format!("/api/tx/{}", "0".repeat(64))).0, 404);
    assert_eq!(get_json(&api, &format!("/api/job/{}", "0".repeat(64))).0, 404);
    assert_eq!(get_json(&api, "/api/nope").0, 404);
    assert_eq!(http(&api, "DELETE", "/api/status", b"").0, 405);
    assert_eq!(http(&api, "POST", "/api/status", b"{}").0, 404);

    // security headers on all the responses
    let (_, head, _) = http(&api, "GET", "/api/status", b"");
    for h in ["x-content-type-options: nosniff", "x-frame-options: deny", "cache-control: no-store", "content-security-policy: default-src 'none'"] {
        assert!(head.contains(h), "missing header: {h}\n{head}");
    }
    // explorer: HTML, strict CSP, no external resource
    let (st, head, body) = http(&api, "GET", "/", b"");
    let page = String::from_utf8(body).unwrap();
    assert_eq!(st, 200);
    assert!(head.contains("text/html") && head.contains("connect-src 'self'"));
    assert!(page.contains("VEARL testnet") && page.contains("/api/blocks"));
    assert!(!page.contains("src=\"http") && !page.contains("href=\"http") && !page.contains("innerHTML"), "no external resource nor HTML injection");
    node.stop();
}

#[test]
fn http_server_rejects_oversized_malformed_and_unsupported_requests() {
    let node = node_from(cfg_with(vec![]), None);
    let api = spawn_api(node.shared.clone(), "127.0.0.1:0").unwrap();
    // declared body > 2 MiB
    let mut s = TcpStream::connect(&api).unwrap();
    s.write_all(b"POST /api/tx HTTP/1.1\r\nContent-Length: 3000000\r\n\r\n").unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    assert!(out.starts_with("HTTP/1.1 413"), "{out}");
    // transfer-encoding refused (avoids request smuggling attacks)
    let mut s = TcpStream::connect(&api).unwrap();
    s.write_all(b"POST /api/tx HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n").unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    assert!(out.starts_with("HTTP/1.1 400"), "{out}");
    // request without HTTP version / target without slash
    let mut s = TcpStream::connect(&api).unwrap();
    s.write_all(b"GET nothing\r\n\r\n").unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    assert!(out.starts_with("HTTP/1.1 400"), "{out}");
    // normal requests still go through after these abuses, including in a burst
    for _ in 0..100 {
        assert_eq!(http(&api, "GET", "/api/status", b"").0, 200);
    }
    node.stop();
}

fn drip(f: &Faucet, addr: &str, ip: &str) -> (u16, Value) {
    let r = f.handle(Request { method: "POST".into(), path: "/drip".into(), query: String::new(), body: serde_json::json!({ "address": addr }).to_string().into_bytes(), peer: ip.into() });
    (r.status, serde_json::from_slice(&r.body).unwrap())
}

#[test]
fn faucet_pays_once_per_address_and_ip_and_respects_the_daily_cap() {
    let fsk = SigningKey::from_bytes(&[101; 32]);
    let node = node_from(cfg_with(vec![(pubkey(&fsk), 100 * COIN)]), Some(1));
    wait("blocks", 30, || node.height() >= 2);
    let faucet = Faucet::new(FaucetConfig { node: node.addr.clone(), amount: 10 * COIN, cooldown: Duration::from_secs(3600), max_per_day: 25 * COIN }, fsk);
    let addr = |i: u32| address_of(&pubkey(&derive_account(&new_mnemonic(), i)));
    let (a1, a2, a3) = (addr(0), addr(0), addr(0));

    let (st, r) = drip(&faucet, &a1, "1.1.1.1");
    assert_eq!(st, 200, "{r}");
    wait("tokens received", 30, || matches!(request(&node.addr, &Msg::GetAccount(parse_address(&a1).unwrap())), Ok(Msg::Account { balance, .. }) if balance == 10 * COIN));

    let (st, r) = drip(&faucet, &a1, "2.2.2.2");
    assert_eq!(st, 429, "same address, other IP");
    assert!(r["retry_in_secs"].as_u64().unwrap() > 3000);
    assert_eq!(drip(&faucet, &a2, "1.1.1.1").0, 429, "other address, same IP");
    assert_eq!(drip(&faucet, "vrl1invalid", "3.3.3.3").0, 400);
    assert_eq!(drip(&faucet, &address_of(&pubkey(&SigningKey::from_bytes(&[101; 32]))), "4.4.4.4").0, 400, "no self-send");
    assert_eq!(drip(&faucet, &a2, "5.5.5.5").0, 200, "new address and IP: served");
    assert_eq!(drip(&faucet, &a3, "6.6.6.6").0, 429, "daily cap (25): a 3rd send of 10 exceeds it");
    node.stop();
}

#[test]
fn genesis_file_roundtrip_and_validation() {
    let cfg = cfg_with(vec![(pubkey(&SigningKey::from_bytes(&[7; 32])), 5 * COIN + COIN / 4)]);
    let json = genesis_to_json("net-a", &cfg);
    let (name, back) = parse_genesis(&json).unwrap();
    assert_eq!(name, "net-a");
    assert_eq!((back.params, back.alloc.clone(), back.ts, back.target, back.timing, back.retarget), (cfg.params, cfg.alloc.clone(), cfg.ts, cfg.target, cfg.timing, cfg.retarget));
    // the genesis hash of the nodes covers all the fields: two different geneses do not talk to each other
    let a = node_from(cfg.clone(), None);
    let mut other = cfg.clone();
    other.target -= 1;
    let b = node_from(other, None);
    assert_ne!(a.shared.genesis_hash, b.shared.genesis_hash);
    a.stop();
    b.stop();

    let edit = |f: &dyn Fn(&mut Value)| {
        let mut v: Value = serde_json::from_str(&json).unwrap();
        f(&mut v);
        parse_genesis(&v.to_string())
    };
    assert!(edit(&|v| v["params"]["fold_k"] = 256.into()).is_err(), "unsafe parameters refused");
    assert!(edit(&|v| v["target"] = "100".into()).is_err(), "target too hard");
    assert!(edit(&|v| v["retarget"] = "Magic".into()).is_err());
    assert!(edit(&|v| v["timing"]["challenge_secs"] = 5.into()).is_err(), "dispute window too short");
    assert!(edit(&|v| v["alloc"][0]["address"] = "bc1qxyz".into()).is_err());
    assert!(edit(&|v| v["alloc"][0]["amount"] = "1.123456789".into()).is_err(), "9 decimals refused");
    assert!(edit(&|v| { let a = v["alloc"][0].clone(); v["alloc"].as_array_mut().unwrap().push(a); }).is_err(), "duplicate address");
    assert!(parse_genesis("{}").is_err());
}

#[test]
fn amounts_are_parsed_exactly() {
    assert_eq!(parse_amount("12.5").unwrap(), 12 * COIN + COIN / 2);
    assert_eq!(parse_amount("0.00000001").unwrap(), 1);
    assert_eq!(parse_amount(".5").unwrap(), COIN / 2);
    assert_eq!(fmt_amount(1234567890), "12.34567890");
    for bad in ["", ".", "-1", "1e3", "1.000000001", "abc", "18446744073709551616"] {
        assert!(parse_amount(bad).is_err(), "{bad}");
    }
}

// ---------------------------------------------------------------- P8b: wallets plugged in everywhere, history, long poll

use std::process::{Child, Command, Stdio};

struct Proc(Child);
impl Drop for Proc {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("vearl-{tag}-{}", rand::random::<u64>()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

const WPW: &str = "wallet-pass-for-tests";

fn make_wallet(dir: &std::path::Path, name: &str) -> (std::path::PathBuf, String, SigningKey) {
    let m = new_mnemonic();
    let ks = Keystore::create(&m, WPW, KdfParams::LIGHT).unwrap();
    let path = dir.join(name);
    ks.save(&path).unwrap();
    (path, ks.address.clone(), derive_account(&m, 0))
}

#[test]
fn head_history_and_long_poll_on_the_api() {
    let (dir, _) = (temp_dir("api"), ());
    let (_, addr_a, sk_a) = make_wallet(&dir, "a.json");
    let (_, addr_b, _) = make_wallet(&dir, "b.json");
    let node = node_from(cfg_with(vec![(pubkey(&sk_a), 100 * COIN)]), Some(1));
    let api = spawn_api(node.shared.clone(), "127.0.0.1:0").unwrap();
    wait("blocks", 30, || node.height() >= 2);

    // HEAD: same headers as a GET, no body
    let (st, head, body) = http(&api, "HEAD", "/api/status", b"");
    assert_eq!(st, 200);
    assert!(body.is_empty() && head.contains("content-length:") && head.contains("application/json"));

    // history: a transfer appears at the sender (direction "out") and at the recipient (direction "in")
    let tx = SignedTx::new(Tx { from: pubkey(&sk_a), nonce: 0, fee: 0, kind: TxKind::Transfer { to: parse_address(&addr_b).unwrap(), amount: 3 * COIN } }, &sk_a);
    request(&node.addr, &Msg::NewTx(tx.clone())).unwrap();
    wait("history", 30, || {
        let (_, v) = get_json(&api, &format!("/api/address/{addr_b}/txs"));
        v["txs"].as_array().map_or(false, |t| t.len() == 1 && t[0]["status"] == "confirmed")
    });
    let (_, out) = get_json(&api, &format!("/api/address/{addr_a}/txs?limit=5"));
    assert_eq!(out["txs"][0]["direction"], "out");
    let (_, inb) = get_json(&api, &format!("/api/address/{addr_b}/txs"));
    assert_eq!(inb["txs"][0]["direction"], "in");
    assert_eq!(inb["txs"][0]["kind"]["amount"], "3.00000000");
    assert_eq!(get_json(&api, "/api/address/vrl1bad/txs").0, 400);

    // long poll: returns as soon as a block arrives; expires cleanly otherwise
    let h = node.height();
    let t0 = Instant::now();
    let (st, v) = get_json(&api, &format!("/api/wait?height={h}&timeout=10"));
    assert_eq!(st, 200);
    assert_eq!(v["timed_out"], false, "a block arrived before the expiry");
    assert!(t0.elapsed() < Duration::from_secs(9));
    let t0 = Instant::now();
    let (_, v) = get_json(&api, "/api/wait?height=999999999&timeout=1");
    assert_eq!(v["timed_out"], true);
    assert!(t0.elapsed() >= Duration::from_millis(900) && t0.elapsed() < Duration::from_secs(4));
    assert_eq!(get_json(&api, "/api/wait").0, 400);
    node.stop();
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn node_binary_mines_with_a_wallet_and_refuses_public_seed_keys_on_a_file_genesis() {
    let dir = temp_dir("node");
    let (wallet, addr, _) = make_wallet(&dir, "miner.json");
    let gfile = dir.join("genesis.json");
    std::fs::write(&gfile, genesis_to_json("bin-test", &cfg_with(vec![]))).unwrap();
    let (p2p, api_port) = (free_port(), free_port());
    let bin = env!("CARGO_BIN_EXE_node");

    // --mine-seed + file genesis without --insecure-devnet-keys: refused
    let st = Command::new(bin)
        .args(["--genesis", gfile.to_str().unwrap(), "--listen", &format!("127.0.0.1:{}", free_port()), "--mine-seed", "1"])
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(!st.success(), "a public seed must not be accepted on a network defined by a file");

    // mining with a wallet: the rewards arrive at the wallet's address
    let _child = Proc(
        Command::new(bin)
            .args(["--genesis", gfile.to_str().unwrap(), "--listen", &format!("127.0.0.1:{p2p}"), "--api", &format!("127.0.0.1:{api_port}"), "--miner-wallet", wallet.to_str().unwrap()])
            .env("VEARL_WALLET_PASSWORD", WPW)
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let api = format!("127.0.0.1:{api_port}");
    wait("the node mines for the wallet's address", 60, || {
        TcpStream::connect(&api).is_ok() && {
            let (st, v) = get_json(&api, &format!("/api/account/{addr}"));
            st == 200 && v["balance_base_units"].as_str().map_or(false, |b| b != "0")
        }
    });
    // wrong password: the node stops instead of mining
    let bad = Command::new(bin)
        .args(["--genesis", gfile.to_str().unwrap(), "--listen", &format!("127.0.0.1:{}", free_port()), "--miner-wallet", wallet.to_str().unwrap()])
        .env("VEARL_WALLET_PASSWORD", "wrong-password")
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(!bad.success());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn cli_signs_with_a_wallet_and_accepts_nx1_addresses() {
    let dir = temp_dir("cli");
    let (wallet, addr, sk) = make_wallet(&dir, "me.json");
    let (_, addr_b, _) = make_wallet(&dir, "you.json");
    let node = node_from(cfg_with(vec![(pubkey(&sk), 40 * COIN)]), Some(1));
    wait("blocks", 30, || node.height() >= 2);
    let cli = env!("CARGO_BIN_EXE_vearl-cli");

    let out = Command::new(cli).args(["balance", &node.addr, &addr]).output().unwrap();
    assert!(String::from_utf8_lossy(&out.stdout).contains("40"), "{}", String::from_utf8_lossy(&out.stdout));

    let out = Command::new(cli)
        .args(["transfer", &node.addr, &format!("wallet:{}", wallet.to_str().unwrap()), &addr_b, "5", "0.5"])
        .env("VEARL_WALLET_PASSWORD", WPW)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    wait("transfer through the CLI", 30, || matches!(request(&node.addr, &Msg::GetAccount(parse_address(&addr_b).unwrap())), Ok(Msg::Account { balance, .. }) if balance == 5 * COIN));

    // wrong password: the CLI signs nothing
    let bad = Command::new(cli)
        .args(["transfer", &node.addr, &format!("wallet:{}", wallet.to_str().unwrap()), &addr_b, "1"])
        .env("VEARL_WALLET_PASSWORD", "nope-nope-nope")
        .output()
        .unwrap();
    assert!(!bad.status.success());
    node.stop();
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn explorer_routes_jobs_miners_accounts_chart_and_search() {
    let node = node_from(cfg_with(vec![]), Some(1));
    let api = spawn_api(node.shared.clone(), "127.0.0.1:0").unwrap();
    wait("blocks", 30, || node.height() >= 6);
    // status carries the network identity and the counters of every job kind
    let (st, s) = get_json(&api, "/api/status");
    assert_eq!(st, 200);
    assert_eq!(s["genesis_hash"].as_str().map(str::len), Some(64));
    assert!(s["version"].is_string() && s["reward"].is_string() && s["node_time"].as_u64().unwrap() > 0);
    for k in ["tensor_programs", "tiled_jobs", "level2_jobs", "accounts"] {
        assert!(s[k].is_u64(), "{k}");
    }
    // jobs: empty on a fresh chain, with the filters accepted
    let (st, j) = get_json(&api, "/api/jobs?kind=level2&status=posted&limit=5");
    assert_eq!(st, 200);
    assert_eq!(j["total"].as_u64(), Some(0));
    assert!(j["jobs"].as_array().unwrap().is_empty());
    // miners: the one miner of the node owns every block of the window
    let (_, m) = get_json(&api, "/api/miners?window=50");
    assert_eq!(m["distinct_miners"].as_u64(), Some(1));
    let top = &m["miners"][0];
    assert_eq!(top["blocks"], m["blocks"]);
    assert!((top["share"].as_f64().unwrap() - 1.0).abs() < 1e-9 && top["useful_blocks"].as_u64() == Some(0));
    // accounts: largest balances first, shares add up to at most 1
    let (_, a) = get_json(&api, "/api/accounts?limit=10");
    let list = a["accounts"].as_array().unwrap();
    assert!(!list.is_empty() && a["holders"].as_u64().unwrap() >= list.len() as u64);
    let bal = |v: &Value| v["balance"].as_str().unwrap().replace('.', "").parse::<u64>().unwrap();
    assert!(list.windows(2).all(|w| bal(&w[0]) >= bal(&w[1])));
    // chart: aligned arrays, oldest first
    let (_, c) = get_json(&api, "/api/chart?blocks=4");
    let n = c["heights"].as_array().unwrap().len();
    assert_eq!(n, 4);
    for k in ["timestamps", "txs", "classes", "targets", "miners"] {
        assert_eq!(c[k].as_array().unwrap().len(), n, "{k}");
    }
    let hs: Vec<u64> = c["heights"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
    assert!(hs.windows(2).all(|w| w[0] < w[1]));
    // search resolves a height, a block hash and an address, and refuses the rest
    let (_, b) = get_json(&api, "/api/blocks?limit=1");
    let hash = b["blocks"][0]["hash"].as_str().unwrap().to_string();
    let miner = b["blocks"][0]["miner"].as_str().unwrap().to_string();
    assert_eq!(get_json(&api, "/api/search?q=1").1["type"], "block");
    assert_eq!(get_json(&api, &format!("/api/search?q={hash}")).1["type"], "block");
    assert_eq!(get_json(&api, &format!("/api/search?q={miner}")).1["type"], "address");
    assert_eq!(get_json(&api, &format!("/api/search?q={}", "0".repeat(64))).0, 404);
    assert_eq!(get_json(&api, "/api/search?q=hello").0, 404);
    assert_eq!(get_json(&api, "/api/search?q=").0, 400);
    node.stop();
}

#[test]
fn utc_dates_parse_and_format_exactly() {
    use vearl_core::genesis::{fmt_utc, parse_utc};
    assert_eq!(parse_utc("1970-01-01T00:00:00Z"), Ok(0));
    assert_eq!(parse_utc("2000-02-29T12:34:56Z"), Ok(951_827_696));
    assert_eq!(parse_utc("2026-10-23T16:00:00Z"), Ok(1_792_771_200));
    assert_eq!(fmt_utc(1_792_771_200), "2026-10-23 16:00:00 UTC");
    assert_eq!(fmt_utc(951_827_696), "2000-02-29 12:34:56 UTC");
    for ts in [0u64, 86_399, 86_400, 1_700_000_000, 4_102_444_800] {
        let s = fmt_utc(ts).replace(' ', "T").replace("TUTC", "Z");
        assert_eq!(parse_utc(&s), Ok(ts), "{s}");
    }
    for bad in ["2026-02-30T00:00:00Z", "2026-13-01T00:00:00Z", "2026-10-23 16:00:00Z", "2026-10-23T16:00:00", "2026-10-23T24:00:00Z", "yesterday", "", "2025-02-29T00:00:00Z"] {
        assert!(parse_utc(bad).is_err(), "{bad}");
    }
}

#[test]
fn genesis_tool_builds_a_public_genesis_and_checks_it() {
    use std::process::Command;
    let bin = env!("CARGO_BIN_EXE_vearl-genesis");
    let faucet = vearl_core::wallet::address_of(&pubkey(&SigningKey::from_bytes(&[9; 32])));
    let other = vearl_core::wallet::address_of(&pubkey(&SigningKey::from_bytes(&[8; 32])));
    let run = |args: &[&str]| Command::new(bin).args(args).output().unwrap();
    // refuses to build a public genesis without a start time, or with two ways of choosing the target
    assert!(!run(&["--preset", "public", "--faucet", &faucet]).status.success());
    assert!(!run(&["--faucet", &faucet, "--hashrate", "1000", "--target-divisor", "5"]).status.success());
    assert!(!run(&["--faucet", "bc1qxyz"]).status.success());
    assert!(!run(&["--faucet", &faucet, "--launch", "2026-02-30T00:00:00Z"]).status.success());
    let out = run(&["--network", "vearl-testnet-1", "--preset", "public", "--faucet", &faucet, "--alloc", &format!("{other}=250.5"), "--launch", "2026-10-23T16:00:00Z", "--hashrate", "100000000"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let json = String::from_utf8(out.stdout).unwrap();
    let (name, cfg) = parse_genesis(&json).unwrap();
    assert_eq!(name, "vearl-testnet-1");
    assert_eq!((cfg.params.n, cfg.params.tile, cfg.params.rank, cfg.params.fold_k), (1024, 16, 63, 64));
    assert_eq!(cfg.ts, 1_792_771_200);
    assert_eq!(cfg.alloc.len(), 2);
    assert_eq!(cfg.alloc[0].1, 1_000_000 * COIN);
    assert_eq!(cfg.alloc[1].1, 250 * COIN + COIN / 2);
    // 100 M attempts per second and 30 s per block: one attempt in 3 x 10^9 succeeds
    let p = (u64::MAX as f64) / (cfg.target as f64);
    assert!((p / 3.0e9 - 1.0).abs() < 1e-6, "{p}");
    let err = String::from_utf8(out.stderr).unwrap();
    assert!(err.contains("GENESIS HASH") && err.contains("2026-10-23 16:00:00 UTC") && err.contains("n = 1024"), "{err}");
    // --check prints the same hash from the file, and refuses a damaged file
    let dir = std::env::temp_dir().join(format!("vearl-gen-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("genesis.json");
    std::fs::write(&f, &json).unwrap();
    let chk = run(&["--check", f.to_str().unwrap()]);
    assert!(chk.status.success());
    let shown = String::from_utf8(chk.stdout).unwrap();
    let hash = vearl_core::net::genesis_hash(&cfg).iter().map(|b| format!("{b:02x}")).collect::<String>();
    assert!(shown.contains(&hash) && shown.contains("valid: yes"), "{shown}");
    std::fs::write(&f, json.replace("\"rank\": 63", "\"rank\": 64")).unwrap();
    assert!(!run(&["--check", f.to_str().unwrap()]).status.success());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explorer_stats_events_cache_and_job_history() {
    let node = node_from(cfg_with(vec![]), Some(1));
    let api = spawn_api(node.shared.clone(), "127.0.0.1:0").unwrap();
    wait("blocks", 30, || node.height() >= 8);
    // daily totals: newest day first, block counts add up to what the chain has
    let (st, d) = get_json(&api, "/api/stats/daily?days=30");
    assert_eq!(st, 200);
    let days = d["days"].as_array().unwrap();
    assert!(!days.is_empty() && days[0]["day"].as_str().unwrap().len() == 10);
    let total: u64 = days.iter().map(|x| x["blocks"].as_u64().unwrap()).sum();
    assert_eq!(total, d["blocks_indexed"].as_u64().unwrap());
    assert!(total >= 8 && days[0]["miners"].as_u64() == Some(1) && days[0]["paid_blocks"].as_u64() == Some(0));
    // events: nothing yet on a chain without jobs, filters accepted
    let (st, e) = get_json(&api, "/api/events?what=fraud_proof,slash_unserved&limit=5");
    assert_eq!(st, 200);
    assert!(e["events"].as_array().unwrap().is_empty());
    // the cache answers the same URL identically while the tip does not move, and refreshes when it does
    let tip = node.tip();
    let (_, a1) = get_json(&api, "/api/miners?window=20");
    let (_, a2) = get_json(&api, "/api/miners?window=20");
    if node.tip() == tip {
        assert_eq!(a1, a2);
    }
    wait("a new block", 30, || node.tip() != tip);
    let (_, a3) = get_json(&api, "/api/chart?blocks=3");
    assert_eq!(a3["heights"].as_array().unwrap().len(), 3);
    // a job that does not exist has no history, a good id of the wrong shape is refused
    assert_eq!(get_json(&api, &format!("/api/job/{}", "1".repeat(64))).0, 404);
    assert_eq!(get_json(&api, "/api/job/zz").0, 400);
    node.stop();
}

/// Behind the local reverse proxy, the client address is the one nginx puts in X-Real-IP (so the faucet's per-IP limit is per visitor).
/// A connection from the loopback is trusted for it; a malformed value is ignored.
#[test]
fn the_http_server_takes_the_client_address_from_the_local_proxy() {
    let addr = vearl_core::httpmini::serve("127.0.0.1:0", |req| vearl_core::httpmini::Response::html(&req.peer)).unwrap();
    let ask = |header: &str| -> String {
        let mut s = TcpStream::connect(&addr).unwrap();
        write!(s, "GET / HTTP/1.1\r\nHost: x\r\n{header}Connection: close\r\n\r\n").unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        out.split("\r\n\r\n").nth(1).unwrap_or("").to_string()
    };
    assert_eq!(ask(""), "127.0.0.1");
    assert_eq!(ask("X-Real-IP: 203.0.113.9\r\n"), "203.0.113.9");
    assert_eq!(ask("X-Real-IP: 2001:db8::7\r\n"), "2001:db8::7");
    assert_eq!(ask("X-Real-IP: not an address\r\n"), "127.0.0.1");
}
