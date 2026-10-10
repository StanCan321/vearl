//! Fuzz of BLOCKS and NETWORK MESSAGES: valid blocks and messages are mutated byte by byte; the decoder and the
//! validator must NEVER panic (a hostile peer must not be able to crash a node), and a mutated block must only
//! be accepted if it is really valid (in which case the chain advances and we rebuild it).
use bincode::Options;
use ed25519_dalek::SigningKey;
use vearl_core::chain::*;
use vearl_core::crypto::*;
use vearl_core::net::*;
use vearl_core::state::*;
use vearl_core::types::*;
use vearl_core::work::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::panic::{catch_unwind, AssertUnwindSafe};

const P: WorkParams = WorkParams::TESTNET;

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

fn genesis() -> GenesisConfig {
    let alloc: Vec<(PubKey, u64)> = [10u8, 11, 12].iter().map(|s| (pubkey(&key(*s)), 1000 * COIN)).collect();
    GenesisConfig { params: P, alloc, ts: 1_000, target: u64::MAX / 16, timing: Timing::default(), retarget: Retarget::Fixed }
}

/// A small chain: Native blocks with transfers, one job mined by its worker.
fn build() -> Vec<Block> {
    let mut c = Chain::from_config(&genesis());
    let mut out = vec![];
    let mut push = |c: &mut Chain, who: u8, class: WorkClass, cm: Option<&CommittedMatrices>, txs: Vec<SignedTx>| {
        let ts = c.tip_ts() + BLOCK_TIME;
        let o = c.mine(&key(who), class, cm, txs, ts).unwrap();
        c.validate_and_apply(o.block.clone()).unwrap();
        out.push(o.block);
    };
    let mut rng = StdRng::seed_from_u64(5);
    let mut m = || (0..P.n * P.n).map(|_| rng.gen_range(-64..=64) as i8).collect::<Vec<i8>>();
    let cm = CommittedMatrices::new(m(), m(), &P);
    let id = job_id(&pubkey(&key(10)), 0, &cm.a_root(), &cm.b_root());
    let t = make_tx(&c, &key(10), 3, TxKind::SubmitJob { a_root: cm.a_root(), b_root: cm.b_root(), escrow: 20 * COIN, deadline: 500 });
    push(&mut c, 12, WorkClass::Native, None, vec![t]);
    let t = make_tx(&c, &key(11), 0, TxKind::ClaimJob { job_id: id });
    push(&mut c, 12, WorkClass::Native, None, vec![t]);
    push(&mut c, 11, WorkClass::Job(id), Some(&cm), vec![]);
    let t = make_tx(&c, &key(12), 2, TxKind::Transfer { to: pubkey(&key(10)), amount: COIN });
    push(&mut c, 12, WorkClass::Native, None, vec![t]);
    push(&mut c, 12, WorkClass::Native, None, vec![]);
    out
}

fn mutate(bytes: &mut Vec<u8>, rng: &mut StdRng) {
    for _ in 0..rng.gen_range(1..=3) {
        let len = bytes.len();
        if len == 0 {
            return;
        }
        match rng.gen_range(0..7) {
            0 => bytes[rng.gen_range(0..len)] ^= 1 << rng.gen_range(0..8),
            1 => bytes[rng.gen_range(0..len)] = rng.gen(),
            2 => {
                let p = rng.gen_range(0..len);
                for b in bytes.iter_mut().skip(p).take(8) {
                    *b = 0xFF;
                }
            }
            3 => {
                let p = rng.gen_range(0..len);
                for b in bytes.iter_mut().skip(p).take(8) {
                    *b = 0;
                }
            }
            4 => bytes.truncate(rng.gen_range(0..len)),
            5 => {
                let extra: Vec<u8> = (0..rng.gen_range(1..64)).map(|_| rng.gen()).collect();
                bytes.extend(extra);
            }
            _ => {
                let p = rng.gen_range(0..len);
                bytes[p] = bytes[p].wrapping_add(rng.gen_range(1..=3));
            }
        }
    }
}

fn replay(blocks: &[Block], upto: usize) -> Chain {
    let mut c = Chain::from_config(&genesis());
    for b in &blocks[..upto] {
        c.validate_and_apply(b.clone()).unwrap();
    }
    c
}

#[test]
fn mutated_blocks_and_messages_never_panic() {
    let blocks = build();
    let iters: usize = std::env::var("FUZZ_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(150);
    let mut rng = StdRng::seed_from_u64(std::env::var("FUZZ_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(9));
    let codec = bincode::options().with_fixint_encoding().with_limit(16 << 20);
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if std::env::var("FUZZ_VERBOSE").is_err() {
            return;
        }
        default_hook(info);
    }));
    let (mut tried, mut decoded, mut accepted, mut panics) = (0usize, 0usize, 0usize, vec![]);
    // 1. blocks: mutation of the next block, validation on the chain at the right height
    for k in 0..blocks.len() {
        let mut chain = replay(&blocks, k);
        let orig = bincode::serialize(&blocks[k]).unwrap();
        for m in 0..iters {
            let mut bytes = orig.clone();
            if m > 0 {
                mutate(&mut bytes, &mut rng);
            }
            tried += 1;
            let Ok(b) = codec.deserialize::<Block>(&bytes) else { continue };
            decoded += 1;
            match catch_unwind(AssertUnwindSafe(|| chain.validate_and_apply(b))) {
                Err(_) => panics.push(format!("block {k}: PANIC (mutant #{m})")),
                Ok(Ok(())) => {
                    accepted += 1;
                    if bytes != orig {
                        panics.push(format!("block {k}: a MUTATED block (different bytes) was ACCEPTED (mutant #{m})"));
                    }
                    chain = replay(&blocks, k); // an equivalent mutant was accepted: start again from the initial state
                }
                Ok(Err(_)) => {}
            }
        }
    }
    // 2. network messages: decoding of mutated valid messages
    let msgs = vec![
        Msg::Hello { node_id: 7, genesis: [1; 32], height: 5, work: 12345, listen: Some("127.0.0.1:9000".into()) },
        Msg::GetBlocks { from: 3 },
        Msg::Blocks(blocks.clone()),
        Msg::NewBlock(blocks[2].clone()),
        Msg::NewTx(blocks[0].txs[0].clone()),
        Msg::Peers(vec!["a:1".into(), "b:2".into()]),
        Msg::PutJob { a: vec![1; 300], b: vec![2; 300] },
        Msg::OpData { z: Some(vec![1, 2, 3]), x: vec![4, 5] },
    ];
    for msg in &msgs {
        let orig = encode(msg).unwrap();
        for m in 0..iters * 4 {
            let mut bytes = orig.clone();
            if m > 0 {
                mutate(&mut bytes, &mut rng);
            }
            tried += 1;
            match catch_unwind(AssertUnwindSafe(|| decode_msg(&bytes))) {
                Err(_) => panics.push(format!("message {:?}: PANIC while decoding", format!("{msg:?}").chars().take(40).collect::<String>())),
                Ok(Ok(_)) => decoded += 1,
                Ok(Err(_)) => {}
            }
        }
    }
    // 3. purely random bytes
    for _ in 0..iters * 40 {
        let len = rng.gen_range(0..600);
        let bytes: Vec<u8> = (0..len).map(|_| rng.gen()).collect();
        tried += 1;
        if catch_unwind(AssertUnwindSafe(|| decode_msg(&bytes))).is_err() {
            panics.push("random bytes: PANIC while decoding".into());
        }
    }
    println!("network fuzz: {tried} inputs, {decoded} decoded, {accepted} blocks accepted (the originals and the mutations without effect), {} panics", panics.len());
    assert!(panics.is_empty(), "{:?}", panics.iter().take(5).collect::<Vec<_>>());
    assert!(decoded > tried / 20, "too few decodable inputs: test of little significance");
}

/// Fuzz of a RUNNING node, over the network (active worker to exercise the deposit handlers): it must stay alive
/// and keep mining. FUZZ_NET_ITERS for a longer campaign (default 600).
#[test]
fn a_running_node_survives_network_fuzzing() {
    let mut g = genesis();
    g.target = u64::MAX / 256;
    let node = spawn_node(NodeConfig {
        genesis: g,
        listen: "127.0.0.1:0".into(),
        peers: vec![],
        mine_key: Some(key(11)),
        use_gpu: false,
        log: false,
        worker: Some(WorkerCfg::default()),
        data_dir: None,
    })
    .unwrap();
    std::thread::sleep(std::time::Duration::from_secs(1));
    let iters = std::env::var("FUZZ_NET_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(600);
    let r = vearl_core::netfuzz::run(&node.addr, iters, 3, false);
    println!("{r:?}");
    assert!(r.failures.is_empty(), "{:?}", r.failures);
    assert!(r.replies > 50, "the fuzz must get replies (valid requests or accepted mutated ones)");
    assert!(r.height_after > r.height_before, "the node keeps mining during and after the fuzz");
    assert!(node.shared.node.lock().is_ok(), "node lock not poisoned");
    node.stop();
}
