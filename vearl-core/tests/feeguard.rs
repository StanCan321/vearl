//! Fee audit (feeguard.rs): signed beacon of the miner -> collecting seed -> local API; refusal of forged beacons.
//! A single sequential test: the registry and the "collect" flag are global to the process.
use ed25519_dalek::SigningKey;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use vearl_core::api::spawn_api;
use vearl_core::chain::*;
use vearl_core::dual::*;
use vearl_core::feeguard::*;
use vearl_core::net::*;

const Q: &str = "qzm7tR2aEJEdMFajwDspjiWefz2F4iDzsys32gE1DK8Dxc59P";

fn http_get(addr: &str, path: &str, headers: &str) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    write!(s, "GET {path} HTTP/1.1\r\nHost: x\r\n{headers}Connection: close\r\n\r\n").unwrap();
    let mut out = String::new();
    let _ = s.read_to_string(&mut out);
    let code = out.split(' ').nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
    (code, out.split("\r\n\r\n").nth(1).unwrap_or("").to_string())
}

#[test]
fn fee_beacons_flow_from_the_miner_to_the_seed_and_forgeries_are_refused() {
    let mut g = devnet_genesis();
    g.retarget = Retarget::Fixed;
    let seed = spawn_node(NodeConfig { genesis: g, listen: "127.0.0.1:0".into(), peers: vec![], mine_key: None, use_gpu: false, log: false, worker: None, data_dir: None }).unwrap();
    let api = spawn_api(seed.shared.clone(), "127.0.0.1:0").unwrap();
    let spec = format!("{}@{}", seed.addr, seed.id_hex());
    let k = SigningKey::from_bytes(&[41; 32]);
    let beacon = |ts: u64| FeeBeacon::new(&k, Q, 80, "a".repeat(64), "b".repeat(64), ts);

    // 1. a seed that does not collect refuses, and the API route does not exist
    set_collecting(false);
    let e = send_beacon(&spec, &beacon(now_secs())).unwrap_err();
    assert!(e.contains("does not collect"), "{e}");
    assert_eq!(http_get(&api, "/api/fee/beacons", "").0, 404);

    // 2. a collecting seed stores the beacon; the local auditor reads it
    set_collecting(true);
    send_beacon(&spec, &beacon(now_secs())).unwrap();
    let (code, body) = http_get(&api, "/api/fee/beacons", "");
    assert_eq!(code, 200, "{body}");
    let j: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(j["count"], 1);
    assert_eq!(j["beacons"][0]["quantus"], Q);
    assert_eq!(j["beacons"][0]["fee_pct"], 10);
    assert_eq!(j["beacons"][0]["worker_id"], FeeBeacon::worker_id_of(&vearl_core::crypto::pubkey(&k)));

    // 3. never through the public proxy (it sets X-Real-IP): 404
    assert_eq!(http_get(&api, "/api/fee/beacons", "X-Real-IP: 8.8.8.8\r\n").0, 404);

    // 4. forgeries: wrong pinned network key, tampered fee, stale, other account's signature
    let wrong = format!("{}@{}", seed.addr, "00".repeat(32));
    assert!(send_beacon(&wrong, &beacon(now_secs())).unwrap_err().contains("does not match"));
    let mut cheat = beacon(now_secs() + 1);
    cheat.fee_pct = 0;
    assert!(send_beacon(&spec, &cheat).is_err(), "a beacon that declares no fee (or any change) is refused: bad signature");
    assert!(send_beacon(&spec, &beacon(now_secs().saturating_sub(3600))).is_err(), "stale");
    let mut other = beacon(now_secs() + 2);
    other.account = vearl_core::crypto::pubkey(&SigningKey::from_bytes(&[42; 32]));
    assert!(send_beacon(&spec, &other).is_err());
    assert_eq!(registry_len(), 1);

    // 5. the scheduler of a combined miner sends its own beacon (first one 20 s after the start)
    clear_registry();
    let k2 = SigningKey::from_bytes(&[43; 32]);
    let sleeper: Vec<String> = if cfg!(windows) { vec!["ping".into(), "-n".into(), "120".into(), "127.0.0.1".into()] } else { vec!["sleep".into(), "120".into()] };
    spawn_scheduler(
        seed.shared.clone(),
        DualCfg { cmd: sleeper, fee: None, share_pct: 80, cycle_s: 10, quantus_bin: None, beacon: Some(BeaconCfg { key: k2.clone(), quantus: Q.to_string(), seeds: vec![spec.clone()] }) },
    );
    let t = Instant::now();
    while registry_len() == 0 && t.elapsed() < Duration::from_secs(45) {
        std::thread::sleep(Duration::from_millis(200));
    }
    assert_eq!(registry_len(), 1, "the scheduler must send a beacon");
    let j = beacons_json(now_secs());
    assert_eq!(j["beacons"][0]["share_pct"], 80);
    assert_eq!(j["beacons"][0]["fee_pct"], 10);
    assert_eq!(j["beacons"][0]["node_hash"].as_str().unwrap().len(), 64);
    seed.shared.stop.store(true, Ordering::SeqCst);
    seed.stop();
    set_collecting(false);
    clear_registry();
    std::thread::sleep(Duration::from_millis(600));
    let _ = std::fs::remove_file("external-miner.log");
}
