//! Consensus safeguard: the test vectors (work layer, header, sigma, Merkle) are FROZEN in
//! tests/vectors/consensus-work-v1.json. Any change to the code that alters them alters the consensus: this test
//! fails, and then either the code must be fixed, or a new version of the protocol must be published (and the file regenerated
//! with `vearl-vectors`, after updating docs/07 and tools/spec_ref.py).
use std::process::Command;

#[test]
fn consensus_vectors_are_frozen() {
    let out = Command::new(env!("CARGO_BIN_EXE_vearl-vectors")).output().unwrap();
    assert!(out.status.success());
    let got: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let want: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("tests/vectors/consensus-work-v1.json").unwrap()).unwrap();
    assert_eq!(got, want, "the consensus vectors changed: the protocol is no longer compatible with the frozen v1");
}

#[test]
fn chain_vectors_are_frozen_and_python_mined_blocks_are_accepted() {
    // 1. the chain produced by the reference node is identical to the frozen chain (no consensus drift)
    let out = Command::new(env!("CARGO_BIN_EXE_vearl-chain-vectors")).output().unwrap();
    assert!(out.status.success());
    let got: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let want: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("tests/vectors/chain-native-v1.json").unwrap()).unwrap();
    assert_eq!(got, want, "the reference chain changed: consensus modified");
    // 2. the chain extended by 6 blocks MINED IN PYTHON (tools/chain_ref.py) is accepted by the Rust validator
    let v = Command::new(env!("CARGO_BIN_EXE_vearl-chain-vectors")).args(["verify", "tests/vectors/chain-native-v1.ext.json"]).output().unwrap();
    assert!(v.status.success(), "{}", String::from_utf8_lossy(&v.stdout));
    assert!(String::from_utf8_lossy(&v.stdout).contains("336 blocks accepted"));
}

#[test]
fn market_vectors_are_frozen_and_python_mined_market_blocks_are_accepted() {
    // 1. reference market chain (auction, useful blocks + rebate, fetch/serve, fraud, expiry, SlashUnserved, settlement)
    let out = Command::new(env!("CARGO_BIN_EXE_vearl-market-vectors")).output().unwrap();
    assert!(out.status.success());
    let got: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let want: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("tests/vectors/market-v1.json").unwrap()).unwrap();
    assert_eq!(got, want, "the reference market chain changed: consensus modified");
    // 2. sequence mined IN PYTHON (2 jobs: a paid one, a false one proven by a proof built in Python) accepted by the Rust validator
    let v = Command::new(env!("CARGO_BIN_EXE_vearl-market-vectors")).args(["verify", "tests/vectors/market-v1.ext.json"]).output().unwrap();
    let txt = String::from_utf8_lossy(&v.stdout).to_string();
    assert!(v.status.success(), "{txt}");
    assert!(txt.contains("62 blocks accepted") && txt.contains("Slashed") && txt.contains("Settled"), "{txt}");
}

#[test]
fn prog_vectors_are_frozen_and_python_authored_programs_are_accepted() {
    // 1. reference chain of programs (CNN, transformer, 4 kinds of fraud proofs) identical to the frozen chain
    let out = Command::new(env!("CARGO_BIN_EXE_vearl-prog-vectors")).output().unwrap();
    assert!(out.status.success());
    let got: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let want: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("tests/vectors/prog-v1.json").unwrap()).unwrap();
    assert_eq!(got, want, "the reference program chain changed: consensus modified");
    // 2. PYTHON-AUTHORED sequence (program defined, computed, mined, published and proven wrong by the Python validator) accepted by the Rust one
    let v = Command::new(env!("CARGO_BIN_EXE_vearl-prog-vectors")).args(["verify", "tests/vectors/prog-v1.ext.json"]).output().unwrap();
    let txt = String::from_utf8_lossy(&v.stdout).to_string();
    assert!(v.status.success(), "{txt}");
    assert!(txt.contains("616 blocks accepted") && txt.contains("Slashed") && txt.contains("Settled"), "{txt}");
}

#[test]
fn level2_kernel_vectors_are_frozen() {
    // chunk outputs (Monte Carlo, rendering) frozen: verified byte for byte by tools/l2_ref.py
    let out = Command::new(env!("CARGO_BIN_EXE_vearl-l2")).arg("vectors").output().unwrap();
    assert!(out.status.success());
    let got: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let want: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("tests/vectors/l2-kernels-v1.json").unwrap()).unwrap();
    assert_eq!(got, want, "the level 2 kernels changed: a node could no longer judge another's fraud proofs");
}

#[test]
fn tiled_vectors_are_frozen_and_python_authored_jobs_are_accepted() {
    // 1. reference chain of block-wise jobs (wrong product, wrong sum, Accept, settlement by window) identical to the frozen chain
    let out = Command::new(env!("CARGO_BIN_EXE_vearl-tiled-vectors")).output().unwrap();
    assert!(out.status.success());
    let got: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let want: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("tests/vectors/tiled-v1.json").unwrap()).unwrap();
    assert_eq!(got, want, "the reference chain of block-wise jobs changed: consensus modified");
    // 2. PYTHON-AUTHORED sequence (jobs defined, computed, mined, published, wrong sum proven by the Python validator) accepted by the Rust one
    let v = Command::new(env!("CARGO_BIN_EXE_vearl-tiled-vectors")).args(["verify", "tests/vectors/tiled-v1.ext.json"]).output().unwrap();
    let txt = String::from_utf8_lossy(&v.stdout).to_string();
    assert!(v.status.success(), "{txt}");
    assert!(txt.contains("73 blocks accepted") && txt.contains("Slashed") && txt.contains("Settled"), "{txt}");
}
