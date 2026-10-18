//! Admission of inbound connections (net.rs): a wallet or a command-line tool is not a peer, so it must be served even when the peer slots of
//! the node are full; a peer that does not fit is told other addresses before it is closed. This test binary sets its own limits (the limits
//! are per process), so it does not disturb the other network tests.
use vearl_core::chain::*;
use vearl_core::net::*;
use std::time::{Duration, Instant};

fn genesis() -> GenesisConfig {
    let mut g = devnet_genesis();
    g.target = u64::MAX / 256;
    g.retarget = Retarget::Fixed;
    g
}

fn node(peers: Vec<String>) -> NodeHandle {
    spawn_node(NodeConfig { genesis: genesis(), listen: "127.0.0.1:0".into(), peers, mine_key: None, use_gpu: false, log: false, worker: None, data_dir: None }).unwrap()
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

#[test]
fn a_wallet_is_served_when_the_peer_slots_are_full_and_a_refused_peer_learns_other_addresses() {
    set_inbound_limits(2, 0, 0, 0); // two peers at most; wallets and tools are counted apart
    let a = node(vec![]);
    let b = node(vec![a.addr.clone()]);
    let c = node(vec![a.addr.clone()]);
    wait("a has its two peers", 20, || a.peer_count() == 2);

    // the peer slots of `a` are full: a wallet (a client without Hello) is still answered, as many times as it likes
    for _ in 0..6 {
        match request(&a.addr, &Msg::GetStatus) {
            Ok(Msg::Status { .. }) => {}
            other => panic!("a wallet request must be served while the peer slots are full: {other:?}"),
        }
    }
    assert_eq!(a.peer_count(), 2, "requests do not take peer slots");

    // a third peer does not fit in `a`, but `a` gives it the addresses of the others: it ends up connected to one of them
    let d = node(vec![a.addr.clone()]);
    std::thread::sleep(Duration::from_secs(4));
    assert_eq!(a.peer_count(), 2, "the third peer is not admitted by a");
    assert!(d.peer_count() >= 1, "the refused peer found another node through the list it received");

    d.stop();
    c.stop();
    b.stop();
    a.stop();
}
