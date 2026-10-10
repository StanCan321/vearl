//! vearl-netfuzz ADDRESS [TRIALS=2000] [SEED=1]: fuzz of a running node over the network (see src/netfuzz.rs).
//! Exit code 1 if the node stops answering or no longer advances.
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let addr = a.first().expect("usage: vearl-netfuzz ADDRESS [TRIALS] [SEED]");
    let iters = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(2000);
    let seed = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(1);
    let r = vearl_core::netfuzz::run(addr, iters, seed, true);
    println!(
        "NETFUZZ: {} trials, {} messages sent, {} replies, {} liveness checks; height {} -> {}; {} failure(s)",
        r.iterations, r.sent, r.replies, r.liveness_checks, r.height_before, r.height_after, r.failures.len()
    );
    for f in &r.failures {
        println!("  FAILURE: {f}");
    }
    std::process::exit(if r.failures.is_empty() && r.height_after > r.height_before { 0 } else { 1 });
}
