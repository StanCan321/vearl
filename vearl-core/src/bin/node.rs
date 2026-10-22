//! Vearl testnet node.
//! node --listen 0.0.0.0:9000 [--genesis FILE] [--api 127.0.0.1:8545 [--api-public]] [--peer host:port]... [--miner-wallet FILE [--miner-account N]] [--mine-seed N --insecure-devnet-keys] [--worker [--cheat]] [--gpu] [--mess] [--data-dir DIR]
//! --worker: claims and mines the jobs whose data the node knows (deposited by `vearl-cli job-submit`)
//! --cheat: TEST ONLY, the worker publishes a wrong result to exercise the fraud proofs
//! --log-file PATH: UTC-timestamped copy of the whole log into PATH (the node runs as a child process, its log goes through this process)
//! --no-native: with --worker, never mine Native blocks: only work on the jobs claimed (the miner's pool mode with paid jobs)
//! --max-jobs N: level 2 jobs claimed at the same time (default 1): a busy worker leaves the queue to the other miners
//! --ask VRL: asking price for auction jobs (the worker bids this price if it beats the best bid)
//! --no-serve / --cheat-garbage: TEST ONLY (worker that withholds the data / publishes an invented result)
//! --qtc-miner PATH --qtc-address qz... [--qtc-pool HOST:PORT --qtc-cert SHA256 --qtc-args "..."] [--partner-share 80] [--partner-cycle 60] : the node starts the official Quantus miner (OPERATOR's address) and freezes it on a cycle; with no job to compute the GPU is shared (80 % Quantus / 20 % Vearl by default). 10 % of the Quantus time goes to the Vearl development address (a constant of the program, displayed at every start, docs/15 section 4 bis)
//! --partner-cmd "CMD": (`devtools` build only) same thing with a free command, WITHOUT fee (tests)
//! --qtc-engine cuda|vulkan: engine of the Quantus miner (fixed arguments); --collect-fee-beacons: seed that receives the audit beacons (docs/17)
//! --relay: relay node (the seed): workers that dial it from behind a router are reachable through it for the audit of their jobs (docs/21)
//! --max-inbound N / --max-per-ip N / --max-per-subnet N / --max-conns N: admission limits (default 24 peers, 2 per IP, 4 per /16, 256 connections of every kind); a public seed raises them (deploy/systemd). Wallets and tools are not peers: only --max-conns and the per-IP connection limit apply to them
//! --cheat-act: TEST ONLY, wrong activation at layer 0 of an inference job
//! (keys derived from PUBLIC seeds: testnet only, never real funds)
use ed25519_dalek::SigningKey;
use vearl_core::api::spawn_api;
use vearl_core::chain::devnet_genesis;
use vearl_core::genesis::parse_genesis;
use vearl_core::wallet::{load_signing_key, read_password};
use vearl_core::net::*;
use std::time::Duration;

/// UTC date/time "YYYY-MM-DD HH:MM:SS" without a dependency (H. Hinnant's civil algorithm).
fn utc_stamp() -> String {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
    let (days, rem) = (t.div_euclid(86_400), t.rem_euclid(86_400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02}", rem / 3600, rem % 3600 / 60, rem % 60)
}

/// `--log-file`: restarts this program as a child process whose log (stderr) is read line by line, timestamped, displayed and appended to the file.
fn run_with_log_file(path: &str) -> ! {
    use std::io::{BufRead, Write};
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut child_args = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--log-file" {
            it.next();
        } else {
            child_args.push(a.clone());
        }
    }
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap_or_else(|e| {
        eprintln!("--log-file {path}: {e}");
        std::process::exit(2)
    });
    let mut cmd = std::process::Command::new(std::env::current_exe().expect("program path"));
    cmd.args(&child_args).env("VEARL_LOG_CHILD", "1").stderr(std::process::Stdio::piped());
    #[cfg(target_os = "linux")]
    {
        // the real node stops when this logging parent dies (for example when the launcher is killed and this process receives its SIGTERM): without
        // it the node would go on alone, with the wallet it unlocked
        use std::os::unix::process::CommandExt;
        extern "C" {
            fn prctl(option: i32, arg2: u64, arg3: u64, arg4: u64, arg5: u64) -> i32;
        }
        unsafe {
            cmd.pre_exec(|| {
                prctl(1, 15, 0, 0, 0); // PR_SET_PDEATHSIG, SIGTERM
                Ok(())
            });
        }
    }
    let mut child = cmd
        .spawn()
        .unwrap_or_else(|e| {
            eprintln!("could not start the node: {e}");
            std::process::exit(2)
        });
    let _ = writeln!(file, "{} [log] ---- start ({}) ----", utc_stamp(), args.join(" ").replace(path, "<log>"));
    let rd = std::io::BufReader::new(child.stderr.take().expect("stderr"));
    for line in rd.lines().map_while(Result::ok) {
        let l = format!("{} {line}", utc_stamp());
        eprintln!("{l}");
        let _ = writeln!(file, "{l}");
    }
    let code = child.wait().ok().and_then(|s| s.code()).unwrap_or(1);
    let _ = writeln!(file, "{} [log] ---- stop (code {code}) ----", utc_stamp());
    std::process::exit(code)
}

const HELP: &str = r#"node: the Vearl node, the server program of the network. Most people do not need to open it: double-click vearl-miner.exe to mine.

  node --genesis genesis.json --peer HOST:PORT@KEY --listen 0.0.0.0:9000 --data-dir data
       [--miner-wallet wallet.json] [--gpu] [--worker] [--api 127.0.0.1:8545] [--log-file node.log]

The seed nodes (HOST:PORT@KEY) are in network.json. All the options are in docs/06 and docs/15."#;

fn main() {
    vearl_core::launcher::explain_if_double_clicked(HELP);
    if std::env::var("VEARL_LOG_CHILD").is_err() {
        let a: Vec<String> = std::env::args().collect();
        if let Some(i) = a.iter().position(|x| x == "--log-file") {
            run_with_log_file(a.get(i + 1).unwrap_or_else(|| {
                eprintln!("--log-file PATH");
                std::process::exit(2)
            }));
        }
    }
    let (mut listen, mut peers, mut seed, mut gpu) = ("127.0.0.1:9000".to_string(), vec![], None, false);
    let (mut worker, mut cheat, mut cheat_act, mut data_dir, mut ask) = (false, false, false, None, None::<u64>);
    let (mut max_jobs, mut no_native) = (1usize, false);
    let (mut no_serve, mut cheat_garbage, mut cheat_sum, mut cheat_elem, mut cheat_col, mut cheat_l2) = (false, false, false, false, false, false);
    let (mut genesis_path, mut api_addr, mut api_public) = (None::<String>, None::<String>, false);
    let (mut miner_wallet, mut miner_account, mut password_file, mut insecure) = (None::<String>, 0u32, None::<String>, false);
    #[allow(unused_mut)]
    let (mut partner_cmd, mut partner_share, mut partner_cycle) = (None::<String>, 80u32, 60u64);
    let (mut qtc_miner, mut qtc_address, mut qtc_pool, mut qtc_cert) = (None::<String>, None::<String>, vearl_core::dual::QTC_POOL_ADDR.to_string(), vearl_core::dual::QTC_POOL_CERT.to_string());
    // Release builds: the Quantus miner always runs with fixed arguments (the engine is the only choice); free-form arguments and
    // --partner-cmd (a miner without the development fee) exist only in builds made with `--features devtools` (docs/17).
    let mut qtc_args: Option<String> = None;
    let mut qtc_engine = "cuda".to_string();
    let mut collect_beacons = false;
    let mut relay = false;
    // admission limits (0 = the default of net.rs: 24 peers, 2 per IP, 4 per /16, 256 connections of every kind)
    let (mut max_inbound, mut max_per_ip, mut max_per_subnet, mut max_conns) = (0usize, 0usize, 0usize, 0usize);
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--log-file" => {
                it.next(); // handled by the parent process (run_with_log_file); here the node runs without this flag
            }
            "--listen" => listen = it.next().expect("--listen ADDR"),
            "--peer" => peers.push(it.next().expect("--peer ADDR")),
            "--mine-seed" => seed = Some(it.next().expect("--mine-seed N").parse::<u8>().expect("u8")),
            "--gpu" => gpu = true,
            #[cfg(feature = "devtools")]
            "--partner-cmd" => partner_cmd = Some(it.next().expect("--partner-cmd \"COMMAND\"")),
            "--collect-fee-beacons" => collect_beacons = true,
            "--relay" => relay = true,
            "--qtc-engine" => qtc_engine = it.next().expect("--qtc-engine cuda|vulkan"),
            "--partner-share" => partner_share = it.next().expect("--partner-share PERCENT").parse().expect("integer 0..100"),
            "--partner-cycle" => partner_cycle = it.next().expect("--partner-cycle SECONDS").parse().expect("integer"),
            "--qtc-miner" => qtc_miner = Some(it.next().expect("--qtc-miner PATH")),
            "--qtc-address" => qtc_address = Some(it.next().expect("--qtc-address qz...")),
            "--qtc-pool" => qtc_pool = it.next().expect("--qtc-pool HOST:PORT"),
            "--qtc-cert" => qtc_cert = it.next().expect("--qtc-cert SHA256"),
            #[cfg(feature = "devtools")]
            "--qtc-args" => qtc_args = Some(it.next().expect("--qtc-args \"...\"")),
            "--max-inbound" => max_inbound = it.next().expect("--max-inbound N").parse().expect("integer"),
            "--max-per-ip" => max_per_ip = it.next().expect("--max-per-ip N").parse().expect("integer"),
            "--max-per-subnet" => max_per_subnet = it.next().expect("--max-per-subnet N").parse().expect("integer"),
            "--max-conns" => max_conns = it.next().expect("--max-conns N").parse().expect("integer"),
            "--mess" => std::env::set_var("VEARL_MESS", "1"), // slowdown of deep reorganisations (docs/13 section 3), optional
            "--miner-wallet" => miner_wallet = Some(it.next().expect("--miner-wallet FILE")),
            "--miner-account" => miner_account = it.next().expect("--miner-account N").parse().expect("u32"),
            "--password-file" => password_file = Some(it.next().expect("--password-file FILE")),
            "--insecure-devnet-keys" => insecure = true,
            "--genesis" => genesis_path = Some(it.next().expect("--genesis FILE")),
            "--api" => api_addr = Some(it.next().expect("--api ADDR")),
            "--api-public" => api_public = true,
            "--worker" => worker = true,
            "--cheat" => cheat = true,
            "--cheat-act" => cheat_act = true,
            "--no-serve" => no_serve = true,
            "--cheat-sum" => cheat_sum = true,
            "--cheat-elem" => cheat_elem = true,
            "--cheat-col" => cheat_col = true,
            "--cheat-l2" => cheat_l2 = true,
            "--cheat-garbage" => cheat_garbage = true,
            "--no-native" => no_native = true,
            "--max-jobs" => max_jobs = it.next().expect("--max-jobs N").parse().expect("integer >= 1"),
            "--ask" => ask = Some((it.next().expect("--ask VRL").parse::<f64>().expect("number") * vearl_core::state::COIN as f64) as u64),
            "--data-dir" => data_dir = Some(std::path::PathBuf::from(it.next().expect("--data-dir DIR"))),
            _ => panic!("unknown argument: {a}"),
        }
    }
    assert!(!(seed.is_some() && miner_wallet.is_some()), "--mine-seed and --miner-wallet are mutually exclusive");
    assert!(partner_cmd.is_none() || seed.is_some() || miner_wallet.is_some(), "--partner-cmd requires a mining node (--miner-wallet)");
    assert!(qtc_miner.is_none() || seed.is_some() || miner_wallet.is_some(), "--qtc-miner requires a mining node (--miner-wallet)");
    assert!(qtc_miner.is_some() == qtc_address.is_some(), "--qtc-miner and --qtc-address go together");
    assert!(qtc_miner.is_none() || partner_cmd.is_none(), "--qtc-miner and --partner-cmd are mutually exclusive");
    if let Some(a) = &qtc_address {
        assert!(vearl_core::dual::valid_qtc_address(a), "--qtc-address: Quantus address expected (qz + letters and digits)");
    }
    assert!(partner_share <= 100, "--partner-share: 0 to 100");
    assert!(!worker || seed.is_some() || miner_wallet.is_some(), "--worker requires --miner-wallet (worker key, with funds for the bond)");
    assert!(!no_native || worker, "--no-native goes with --worker (a node that neither mines Native blocks nor works on jobs does nothing)");
    if seed.is_some() && genesis_path.is_some() && !insecure {
        panic!("--mine-seed derives the key from a PUBLIC seed: with a genesis file, use --miner-wallet (or --insecure-devnet-keys if the network has no value)");
    }
    let mine_key = if let Some(f) = &miner_wallet {
        let pw = read_password("Password of the mining wallet: ", password_file.as_deref()).unwrap_or_else(|e| panic!("{e}"));
        let sk = load_signing_key(std::path::Path::new(f), &pw, miner_account).unwrap_or_else(|e| panic!("{e}"));
        eprintln!("[node] mining active, rewards to {}", vearl_core::wallet::address_of(&vearl_core::crypto::pubkey(&sk)));
        Some(sk)
    } else {
        if let Some(s) = seed {
            eprintln!("[node] mining active with a PUBLIC SEED key (seed:{s}): devnet only");
        }
        seed.map(|s| SigningKey::from_bytes(&[s; 32]))
    };
    let genesis = match &genesis_path {
        Some(f) => {
            let txt = std::fs::read_to_string(f).unwrap_or_else(|e| panic!("{f}: {e}"));
            let (name, g) = parse_genesis(&txt).unwrap_or_else(|e| panic!("invalid genesis: {e}"));
            eprintln!("[node] network '{name}' (genesis {f})");
            g
        }
        None => {
            eprintln!("[node] built-in DEVNET genesis (faucet keys derived from PUBLIC seeds)");
            devnet_genesis()
        }
    };
    if gpu {
        #[cfg(feature = "gpu")]
        match vearl_core::gpu::check_supported() {
            Ok(d) => eprintln!("[node] GPU: {d}"),
            Err(e) => {
                eprintln!("[node] --gpu unavailable: {e}");
                std::process::exit(2);
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            eprintln!("[node] --gpu: this binary was compiled WITHOUT the gpu feature (cargo build --release --features gpu)");
            std::process::exit(2);
        }
    }
    vearl_core::work::set_gpu_exact(gpu && worker);   // the USEFUL computation of the jobs (exact GEMM) runs on the tensor cores
    let mine_sk = mine_key.clone();
    let want_peers = !peers.is_empty();
    let seed_specs: Vec<String> = peers.clone();
    vearl_core::net::set_inbound_limits(max_inbound, max_per_ip, max_per_subnet, max_conns);
    let handle = spawn_node(NodeConfig {
        genesis,
        listen,
        peers,
        mine_key,
        use_gpu: gpu,
        log: true,
        worker: worker.then_some(WorkerCfg { cheat, cheat_act, ask, no_serve, cheat_garbage, cheat_sum, cheat_elem, cheat_col, cheat_l2, max_jobs, no_native }),
        data_dir,
    })
    .expect("bind");
    eprintln!("[node] listening on {} (gpu={gpu}, worker={worker}, height at start={})", handle.addr, handle.height());
    eprintln!("[node] network public key (to be pinned by peers: --peer HOST:PORT@KEY): {}", handle.id_hex());
    if relay {
        handle.set_relay(true);
        eprintln!("[node] RELAY: workers that dial this node from behind a router can be audited through it (docs/21)");
    }
    if collect_beacons {
        vearl_core::feeguard::set_collecting(true);
        eprintln!("[node] collecting fee beacons from miners (local audit API: /api/fee/beacons on the loopback API)");
    }
    if let (Some(m), Some(addr)) = (&qtc_miner, &qtc_address) {
        let id: String = vearl_core::crypto::pubkey(mine_sk.as_ref().expect("mining key")).iter().take(4).map(|b| format!("{b:02x}")).collect();
        // CUDA: no --gpu-devices, the Quantus miner then uses EVERY CUDA GPU of the machine (one worker per GPU; CUDA_VISIBLE_DEVICES
        // restricts it, and it applies to this node too). Vulkan (fallback engine): one device, as before.
        let fixed = if qtc_engine == "vulkan" { "--gpu-devices 1 --cpu-workers 0" } else { "--cuda-gpu --cpu-workers 0" };
        let args_line = qtc_args.clone().unwrap_or_else(|| fixed.to_string());
        let extra = vearl_core::dual::parse_cmd(&args_line);
        // 1. the fee constants of this program are intact, 2. the Quantus miner is exactly the one pinned in this build
        let guard = if !vearl_core::feeguard::fee_constants_ok() {
            Err("the fee constants of this program were altered".to_string())
        } else {
            vearl_core::feeguard::verify_quantus_binary(std::path::Path::new(m)).map(|_| ())
        };
        let allow_unpinned = cfg!(feature = "devtools") && std::env::var("VEARL_ALLOW_UNPINNED_QUANTUS").is_ok();
        match guard {
            Err(e) if !allow_unpinned => {
                eprintln!("[qtc] Quantus mining is OFF: {e}. Vearl keeps mining. Use the official release (vearl.org).");
            }
            _ => {
                let (cmd, fee_cmd) = vearl_core::dual::qtc_commands(m, addr, &id, &qtc_pool, &qtc_cert, &extra);
                eprintln!("[qtc] Quantus mining for YOUR address {addr} (worker vrl-{id}) on {qtc_pool}");
                eprintln!("[qtc] Vearl development fee: {} % of the Quantus mining time goes to {} (worker vrl-fee-{id}); see docs/15 and docs/17", vearl_core::dual::qtc_fee_pct(), vearl_core::dual::dev_qtc_address());
                eprintln!("[qtc] fee audit: a signed beacon (Vearl account, Quantus address, versions) is sent to the seed nodes every 5 minutes");
                let fee = Some(vearl_core::dual::FeeCfg { cmd: fee_cmd, pct: vearl_core::dual::qtc_fee_pct() });
                let beacon = Some(vearl_core::dual::BeaconCfg { key: mine_sk.clone().expect("mining key"), quantus: addr.clone(), seeds: seed_specs.clone() });
                vearl_core::dual::spawn_scheduler(handle.shared.clone(), vearl_core::dual::DualCfg { cmd, fee, share_pct: partner_share, cycle_s: partner_cycle, quantus_bin: Some(m.clone()), beacon });
            }
        }
    } else if let Some(c) = partner_cmd {
        let cmd = vearl_core::dual::parse_cmd(&c);
        assert!(!cmd.is_empty(), "--partner-cmd is empty");
        vearl_core::dual::spawn_scheduler(handle.shared.clone(), vearl_core::dual::DualCfg { cmd, fee: None, share_pct: partner_share, cycle_s: partner_cycle, quantus_bin: None, beacon: None });
    }
    if let Some(a) = api_addr {
        let loopback = a.parse::<std::net::SocketAddr>().map(|s| s.ip().is_loopback()).unwrap_or(false);
        if !loopback && !api_public {
            panic!("--api on a non-local address: add --api-public (and put a TLS reverse proxy in front)");
        }
        let bound = spawn_api(handle.shared.clone(), &a).unwrap_or_else(|e| panic!("API: {e}"));
        eprintln!("[node] API + explorer on http://{bound}/");
    }
    eprintln!("[node] vearl {} ; status every 10 s{}", env!("CARGO_PKG_VERSION"), if mine_sk.is_some() { " (line [miner]: rate, blocks, balance)" } else { "" });
    let me = mine_sk.as_ref().map(vearl_core::crypto::pubkey);
    let mut last_active = 0u64;
    let (mut last_t, mut last_sigma, mut last_pairs) = (std::time::Instant::now(), vearl_core::chain::SIGMA_DONE.load(std::sync::atomic::Ordering::Relaxed), usize::MAX);
    let mut ema: f64 = 0.0;
    loop {
        std::thread::sleep(Duration::from_secs(10));
        let (height, peers) = (handle.height(), handle.peer_count());
        if want_peers && peers == 0 && last_pairs != 0 {
            eprintln!("[node] WARNING: no peer connected (this node is isolated: check --peer, the firewall and the listening port)");
        }
        last_pairs = peers;
        let Some(me) = me else {
            eprintln!("[node] height={} peers={} work={}", height, peers, handle.work());
            continue;
        };
        // rate measured over the elapsed window, smoothed (moving average) so as not to jump at each GPU batch
        let now_sigma = vearl_core::chain::SIGMA_DONE.load(std::sync::atomic::Ordering::Relaxed);
        // rate measured over the time the GPU works for Vearl (in dual mining, the Quantus windows do not count)
        let now_active = vearl_core::dual::vearl_active_ms();
        let dual_on = vearl_core::dual::DUAL_PHASE.load(std::sync::atomic::Ordering::Relaxed) != 0;
        let dt = if dual_on { (now_active.saturating_sub(last_active)) as f64 / 1000.0 } else { last_t.elapsed().as_secs_f64() }.max(0.001);
        last_active = now_active;
        let inst = (now_sigma - last_sigma) as f64 / dt;
        ema = if ema == 0.0 { inst } else { 0.7 * ema + 0.3 * inst };
        last_t = std::time::Instant::now();
        last_sigma = now_sigma;
        let (mined, last_ts, balance, tiles, target, mempool) = {
            let n = handle.shared.node.lock().unwrap();
            let mine: Vec<&vearl_core::types::Block> = n.chain.blocks.iter().filter(|b| b.header.miner == me).collect();
            let tl = n.chain.params.tiles() as f64;
            (mine.len(), mine.last().map(|b| b.header.timestamp), n.chain.state.balance(&me), tl * tl, n.chain.target, n.mempool.len())
        };
        let tile_rate = ema * tiles;
        let ago = last_ts.map(|t| {
            let d = vearl_core::net::now_secs().saturating_sub(t);
            if d < 120 { format!("{d} s ago") } else if d < 7200 { format!("{} min ago", d / 60) } else { format!("{} h ago", d / 3600) }
        });
        // average time to find a block ALONE at this rate: 2^64 / (target x tiles per sigma) sigma per block
        let eta = if ema > 0.5 { Some((18446744073709551616.0 / (target as f64 * tiles)) / ema) } else { None };
        let eta_s = eta.map_or("n/a".to_string(), |s| if s < 120.0 { format!("{s:.0} s") } else if s < 7200.0 { format!("{:.0} min", s / 60.0) } else { format!("{:.1} h", s / 3600.0) });
        eprintln!(
            "[miner] height={} peers={} | {:.1} sigma/s ({:.2} M tiles/s) | blocks found={}{} | balance={} | mempool={} | one block alone every ~{} at this rate",
            height,
            peers,
            ema,
            tile_rate / 1e6,
            mined,
            ago.map_or(String::new(), |a| format!(" (last {a})")),
            vearl_core::wallet::fmt_amount(balance),
            mempool,
            eta_s
        );
        if let Some(d) = vearl_core::dual::dual_summary() {
            eprintln!("[dual mining] {d}");
        }
        if ema < 0.5 && height > 0 && !dual_on {
            eprintln!("[miner] nothing is being computed: the node is following the chain (syncing) or working on a paid job");
        }
    }
}
