//! vearl-poolminer: miner for a Vearl pool (vpool/1, docs/16-pool-protocol.md).
//!
//! vearl-poolminer --pool HOST:PORT --address vrl1... [--worker NAME] [--threads N] [--gpu] [--rate-hint SIGMA_PER_S]
//! The address is the one that will receive the payouts: no private key is needed here.
use vearl_core::poolminer::{run, MinerConfig, MinerStats};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn die(m: &str) -> ! {
    eprintln!("error: {m}");
    std::process::exit(1)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    for a in &args {
        if a.starts_with("--") && !["--pool", "--address", "--worker", "--threads", "--gpu", "--rate-hint", "--quiet"].contains(&a.as_str()) {
            die(&format!("unknown option: {a}"));
        }
    }
    let pool = arg("--pool").unwrap_or_else(|| die("--pool HOST:PORT required"));
    let address = arg("--address").unwrap_or_else(|| die("--address vrl1... required (payout address)"));
    if let Err(e) = vearl_core::wallet::parse_address(&address) {
        die(&e);
    }
    let gpu = args.iter().any(|a| a == "--gpu");
    #[cfg(feature = "gpu")]
    if gpu {
        match vearl_core::gpu::check_supported() {
            Ok(d) => eprintln!("[poolminer] GPU: {d}"),
            Err(e) => die(&format!("--gpu unavailable: {e}")),
        }
    }
    #[cfg(not(feature = "gpu"))]
    if gpu {
        die("--gpu: this binary was compiled WITHOUT the gpu feature (cargo build --release --features gpu)");
    }
    let threads = arg("--threads").map_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()), |v| v.parse().unwrap_or_else(|_| die("--threads: integer")));
    let cfg = MinerConfig {
        pool,
        address,
        worker: arg("--worker").unwrap_or_else(|| "default".into()),
        threads,
        gpu,
        rate_hint: arg("--rate-hint").map(|v| v.parse().unwrap_or_else(|_| die("--rate-hint: number"))),
        log: !args.iter().any(|a| a == "--quiet"),
    };
    let stats = Arc::new(MinerStats::default());
    let stop = Arc::new(AtomicBool::new(false));
    {
        let (c, s, st) = (cfg.clone(), stats.clone(), stop.clone());
        std::thread::spawn(move || run(c, s, st));
    }
    let (mut last, mut last_sigmas) = (Instant::now(), 0u64);
    loop {
        std::thread::sleep(Duration::from_secs(10));
        let sg = stats.sigmas.load(Ordering::Relaxed);
        let dt = last.elapsed().as_secs_f64();
        eprintln!(
            "[poolminer] {:.1} sigma/s; shares accepted {} refused {}; blocks {}; {}",
            (sg - last_sigmas) as f64 / dt,
            stats.accepted.load(Ordering::Relaxed),
            stats.rejected.load(Ordering::Relaxed),
            stats.blocks.load(Ordering::Relaxed),
            if stats.connected.load(Ordering::SeqCst) { "connected" } else { "disconnected" }
        );
        last = Instant::now();
        last_sigmas = sg;
    }
}
