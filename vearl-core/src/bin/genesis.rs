//! vearl-genesis: generates (or checks) the JSON genesis file of a testnet.
//!   vearl-genesis --network vearl-testnet-1 --preset public --faucet vrl1... [--faucet-amount 1000000] [--alloc vrl1...=500]...
//!                 --launch 2026-10-23T16:00:00Z --hashrate 100000000 [--retarget Asert|Lwma|Ema|Fixed] > genesis.json
//! vearl-genesis --check genesis.json      (validates the file, prints the summary and the GENESIS HASH)
//! --preset devnet (default): n = 128, runs fast on CPU (trials, tests). --preset public: n = 1024 (size of the models and jobs), rank 63, fold_k 64.
//! --launch / --timestamp: the START instant of the network (required with `public`): ASERT retargeting starts from that instant, a genesis
//!   dated too early would make the first blocks almost free.
//! --hashrate H: total expected rate at launch, in attempts (tiles) per second; the initial target aims at one block per BLOCK_TIME (10 s) at that rate.
//!   Or --target-divisor D (target = 2^64 / D). One or the other; with neither: D = 2048 (devnet).
//! All the nodes of the network must use the SAME file (their genesis hash must be identical). The summary goes to the error output.
use vearl_core::chain::{GenesisConfig, Retarget};
use vearl_core::genesis::{fmt_utc, genesis_to_json, parse_genesis, parse_utc};
use vearl_core::net::genesis_hash;
use vearl_core::state::{Timing, BLOCK_TIME};
use vearl_core::wallet::{fmt_amount, parse_address, parse_amount};
use vearl_core::work::WorkParams;

fn die(m: &str) -> ! {
    eprintln!("error: {m}");
    std::process::exit(1)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Readable summary of a genesis: what an operator must reread before distributing it.
fn summary(name: &str, c: &GenesisConfig) -> String {
    let mut s = String::new();
    s += &format!("network           : {name}\n");
    s += &format!("GENESIS HASH      : {}\n", hex(&genesis_hash(c)));
    s += &format!("start             : {} (timestamp {})\n", fmt_utc(c.ts), c.ts);
    s += &format!("work              : n = {}, tile {}, rank {}, fold_k {}\n", c.params.n, c.params.tile, c.params.rank, c.params.fold_k);
    let total: u64 = c.alloc.iter().map(|a| a.1).sum();
    s += &format!("initial supply    : {} VRL in {} allocation(s)\n", fmt_amount(total), c.alloc.len());
    for (pk, a) in &c.alloc {
        s += &format!("                    {} VRL -> {}\n", fmt_amount(*a), vearl_core::wallet::address_of(pk));
    }
    let attempts = (u128::from(u64::MAX) + 1) / (u128::from(c.target) + 1);
    s += &format!("initial target    : {} (one attempt in {attempts} succeeds)\n", c.target);
    let line: Vec<String> = [49_000_000u64, 100_000_000, 500_000_000, 2_000_000_000]
        .iter()
        .map(|h| format!("{:.0} s at {} M/s", attempts as f64 / *h as f64, h / 1_000_000))
        .collect();
    s += &format!("first block due   : {}\n", line.join("; "));
    s += &format!("retargeting       : {:?} (target {BLOCK_TIME} s per block); windows: dispute {} s, service {} s\n", c.retarget, c.timing.challenge_secs, c.timing.serve_secs);
    s
}

const HELP: &str = r#"vearl-genesis: creates or checks the genesis file of a Vearl network (every node of a network must use the same file).

  vearl-genesis --network NAME --preset public --faucet vrl1... --launch 2026-10-23T16:00:00Z --hashrate 100000000 > genesis.json
        [--faucet-amount 1000000] [--alloc vrl1...=500]... [--retarget Asert|Lwma|Ema|Fixed]
  vearl-genesis --check genesis.json        validate a file, print its summary and the GENESIS HASH

--preset devnet (default) is small and fast, for tests. --preset public is the real size (n = 1024) and needs --launch.
The faucet address must come from a DEDICATED wallet created offline (vearl-wallet new): only its address is needed here.
Details: docs/06."#;

fn main() {
    vearl_core::launcher::help_or_continue(HELP);
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(i) = args.iter().position(|a| a == "--check") {
        let f = args.get(i + 1).unwrap_or_else(|| die("--check FILE"));
        let txt = std::fs::read_to_string(f).unwrap_or_else(|e| die(&format!("{f}: {e}")));
        let (name, cfg) = parse_genesis(&txt).unwrap_or_else(|e| die(&format!("invalid genesis: {e}")));
        print!("{}", summary(&name, &cfg));
        println!("valid: yes");
        return;
    }
    let (mut network, mut faucet, mut famount, mut divisor, mut hashrate, mut ts, mut retarget, mut preset) =
        ("vearl-testnet".to_string(), None, "1000000".to_string(), None::<u64>, None::<u64>, None::<u64>, Retarget::Asert, "devnet".to_string());
    let mut allocs: Vec<(String, String)> = vec![];
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().unwrap_or_else(|| die(&format!("missing value for {a}")));
        match a.as_str() {
            "--network" => network = val(),
            "--preset" => preset = val(),
            "--faucet" => faucet = Some(val()),
            "--faucet-amount" => famount = val(),
            "--alloc" => {
                let v = val();
                let (ad, am) = v.split_once('=').unwrap_or_else(|| die("--alloc ADDRESS=AMOUNT"));
                allocs.push((ad.to_string(), am.to_string()));
            }
            "--target-divisor" => divisor = Some(val().parse().unwrap_or_else(|_| die("invalid divisor"))),
            "--hashrate" => hashrate = Some(val().replace('_', "").parse().unwrap_or_else(|_| die("invalid rate (attempts per second, integer)"))),
            "--timestamp" => ts = Some(val().parse().unwrap_or_else(|_| die("invalid timestamp"))),
            "--launch" => ts = Some(parse_utc(&val()).unwrap_or_else(|e| die(&e))),
            "--retarget" => {
                retarget = match val().as_str() {
                    "Lwma" => Retarget::Lwma,
                    "Ema" => Retarget::Ema,
                    "Fixed" => Retarget::Fixed,
                    "Asert" => Retarget::Asert,
                    o => die(&format!("unknown retargeting '{o}'")),
                }
            }
            o => die(&format!("unknown argument {o}")),
        }
    }
    let params = match preset.as_str() {
        "devnet" => WorkParams::TESTNET,
        "public" => WorkParams { n: 1024, tile: 16, rank: 63, fold_k: 64 },
        o => die(&format!("unknown preset '{o}' (devnet or public)")),
    };
    if preset == "public" && ts.is_none() {
        die("--launch YYYY-MM-DDTHH:MM:SSZ (or --timestamp) is required with --preset public: it is the start instant of the network");
    }
    if divisor.is_some() && hashrate.is_some() {
        die("--target-divisor and --hashrate are mutually exclusive");
    }
    let faucet = faucet.unwrap_or_else(|| die("--faucet ADDRESS required (DEDICATED faucet wallet, created OFFLINE: only its address is needed here)"));
    let mut alloc = vec![(parse_address(&faucet).unwrap_or_else(|e| die(&e)), parse_amount(&famount).unwrap_or_else(|e| die(&e)))];
    for (ad, am) in allocs {
        alloc.push((parse_address(&ad).unwrap_or_else(|e| die(&e)), parse_amount(&am).unwrap_or_else(|e| die(&e))));
    }
    let target = match (divisor, hashrate) {
        (Some(d), _) => u64::MAX / d.max(1),
        (None, Some(h)) => {
            if h == 0 {
                die("--hashrate must be > 0");
            }
            ((u128::from(u64::MAX) + 1) / (u128::from(h) * u128::from(BLOCK_TIME))).min(u128::from(u64::MAX)) as u64
        }
        (None, None) => u64::MAX / 2048,
    };
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let cfg = GenesisConfig { params, alloc, ts: ts.unwrap_or(now), target, timing: Timing::default(), retarget };
    let json = genesis_to_json(&network, &cfg);
    if let Err(e) = parse_genesis(&json) {
        die(&format!("the generated genesis is invalid: {e}"));
    }
    eprint!("{}", summary(&network, &cfg));
    println!("{json}");
}
