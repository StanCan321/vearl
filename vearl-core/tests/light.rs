//! Header-based light client: follows the chain without the state, verifies the work of each header and a balance through a Merkle path.
use ed25519_dalek::SigningKey;
use vearl_core::chain::*;
use vearl_core::crypto::*;
use vearl_core::net::*;
use vearl_core::state::*;
use vearl_core::types::*;
use vearl_core::work::*;
use std::time::{Duration, Instant};

fn key(s: u8) -> SigningKey {
    SigningKey::from_bytes(&[s; 32])
}

fn cfg(retarget: Retarget) -> GenesisConfig {
    let mut g = devnet_genesis();
    g.alloc = (1..4u8).map(|s| (pubkey(&key(s)), 1_000 * COIN)).collect();
    g.target = u64::MAX / 64;
    g.retarget = retarget;
    g
}

/// Complete chain of `count` blocks with irregular timestamps (the target moves) and transfers.
fn full_chain(g: &GenesisConfig, count: u64) -> Chain {
    let mut c = Chain::from_config(g);
    let mut nonce = 0;
    for h in 0..count {
        let txs = if h % 3 == 0 {
            nonce += 1;
            vec![SignedTx::new(Tx { from: pubkey(&key(1)), nonce: nonce - 1, fee: 10, kind: TxKind::Transfer { to: pubkey(&key(9)), amount: (h + 1) * COIN } }, &key(1))]
        } else {
            vec![]
        };
        let t = c.tip_ts() + [3, 50, 12, 90, 30, 1][(h % 6) as usize];
        let out = c.mine(&key(2 + (h % 2) as u8), WorkClass::Native, None, txs, t).unwrap();
        c.validate_and_apply(out.block).unwrap();
    }
    c
}

#[test]
fn light_client_follows_headers_and_verifies_a_balance_for_every_difficulty_rule() {
    for rule in [Retarget::Asert, Retarget::Lwma, Retarget::Ema, Retarget::Fixed] {
        let g = cfg(rule);
        let c = full_chain(&g, 45);
        let mut lc = LightClient::new(&g);
        for b in &c.blocks {
            lc.apply(&b.header, &b.proof, None).unwrap_or_else(|e| panic!("{rule:?} height {}: {e}", b.header.height));
        }
        assert_eq!((lc.height(), lc.target, lc.work), (c.height(), c.target, c.total_work()), "{rule:?}: same target and same work as the full chain");
        let pk = pubkey(&key(9));
        let (idx, count, path) = c.state.accounts_proof(&pk).unwrap();
        let (bal, nonce) = (c.state.balance(&pk), c.state.nonce(&pk));
        assert!(bal > 0);
        assert!(lc.verify_account(&pk, bal, nonce, idx as u64, count as u64, &path), "{rule:?}: balance proven");
        assert!(!lc.verify_account(&pk, bal + 1, nonce, idx as u64, count as u64, &path), "{rule:?}: inflated balance refused");
    }
}

#[test]
fn light_client_rejects_forged_headers() {
    let g = cfg(Retarget::Asert);
    let c = full_chain(&g, 12);
    let prefix = |k: usize| {
        let mut lc = LightClient::new(&g);
        for b in &c.blocks[..k] {
            lc.apply(&b.header, &b.proof, None).unwrap();
        }
        lc
    };
    let b = &c.blocks[8];
    let cases: Vec<(&str, Box<dyn Fn(&mut Header, &mut TileProof)>)> = vec![
        ("falsified jackpot", Box::new(|h, _| h.jackpot[5] ^= 1)),
        ("target too easy", Box::new(|h, _| h.target = h.target.saturating_mul(2))),
        ("invented native work root", Box::new(|h, _| h.a_root[0] ^= 1)),
        ("wrong parent", Box::new(|h, _| h.prev[0] ^= 1)),
        ("timestamp going backwards", Box::new(|h, _| h.timestamp = 1)),
        ("miner changed (sigma changes)", Box::new(|h, _| h.miner = pubkey(&key(7)))),
        ("row of A altered in the proof", Box::new(|_, p| p.a_rows[0][0] ^= 1)),
        ("version 1", Box::new(|h, _| h.version = 1)),
    ];
    for (name, f) in cases {
        let mut lc = prefix(8);
        let (mut h, mut p) = (b.header.clone(), b.proof.clone());
        f(&mut h, &mut p);
        assert!(lc.apply(&h, &p, None).is_err(), "{name}: must be refused");
        lc.apply(&b.header, &b.proof, None).expect("the authentic header stays acceptable after a refusal");
    }
    // future too far ahead of the local clock
    let mut lc = prefix(8);
    assert!(lc.apply(&b.header, &b.proof, Some(b.header.timestamp - MAX_FUTURE - 1)).is_err());
}

#[test]
fn a_running_node_serves_headers_and_account_proofs_over_the_network() {
    let mut g = cfg(Retarget::Asert);
    g.target = u64::MAX / 256;
    let node = spawn_node(NodeConfig { genesis: g.clone(), listen: "127.0.0.1:0".into(), peers: vec![], mine_key: Some(key(2)), use_gpu: false, log: false, worker: None, data_dir: None }).unwrap();
    let t = Instant::now();
    while node.height() < 20 {
        assert!(t.elapsed() < Duration::from_secs(60));
        std::thread::sleep(Duration::from_millis(100));
    }
    let mut lc = LightClient::new(&g);
    while lc.height() < 20 {
        let Msg::Headers(hs) = request(&node.addr, &Msg::GetHeaders { from: lc.height() }).unwrap() else { panic!() };
        for (h, p) in &hs {
            lc.apply(h, p, None).unwrap();
        }
    }
    // the miner (key 2) has a balance: proof served by the network, verified if the client has the same tip
    for _ in 0..50 {
        let Msg::AccountProof { height, tip, balance, nonce, index, count, path } = request(&node.addr, &Msg::GetAccountProof(pubkey(&key(2)))).unwrap() else { panic!() };
        while lc.height() < height {
            let Msg::Headers(hs) = request(&node.addr, &Msg::GetHeaders { from: lc.height() }).unwrap() else { panic!() };
            for (h, p) in &hs {
                lc.apply(h, p, None).unwrap();
            }
        }
        if lc.height() == height && lc.tip_header().map(|h| h.hash()) == Some(tip) {
            assert!(balance > 0 && lc.verify_account(&pubkey(&key(2)), balance, nonce, index, count, &path));
            node.stop();
            return;
        }
    }
    panic!("never synchronised on the tip of the proof");
}
