//! Relay (docs/21): a worker that nobody can dial (behind a home router) is audited through the seed it is connected to.
//! The "router" is simulated by never giving the client the worker's address: the client only knows the seed.
use ed25519_dalek::SigningKey;
use std::time::{Duration, Instant};
use vearl_core::chain::*;
use vearl_core::crypto::*;
use vearl_core::l2::*;
use vearl_core::net::*;
use vearl_core::state::*;
use vearl_core::types::*;

fn genesis() -> GenesisConfig {
    let mut g = devnet_genesis();
    g.target = u64::MAX / 256;
    g.retarget = Retarget::Fixed;
    g.timing.serve_secs = 6;
    g
}

fn node_with(peers: Vec<String>, seed: Option<u8>, worker: Option<WorkerCfg>) -> NodeHandle {
    spawn_node(NodeConfig { genesis: genesis(), listen: "127.0.0.1:0".into(), peers, mine_key: seed.map(|s| SigningKey::from_bytes(&[s; 32])), use_gpu: false, log: false, worker, data_dir: None }).unwrap()
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

fn client_tx(node: &NodeHandle, client: &SigningKey, kind: TxKind) -> Msg {
    let nonce = match request(&node.addr, &Msg::GetAccount(pubkey(client))).unwrap() {
        Msg::Account { next_nonce, .. } => next_nonce,
        m => panic!("{m:?}"),
    };
    request(&node.addr, &Msg::NewTx(SignedTx::new(Tx { from: pubkey(client), nonce, fee: 0, kind }, client))).unwrap()
}

fn l2_status(node: &str, id: JobId) -> String {
    match request(node, &Msg::GetL2(id)).unwrap() {
        Msg::L2Info { status, .. } => status,
        _ => "unknown".into(),
    }
}

fn hex32(b: &[u8; 32]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// seed (relay, mines) <- worker (dials the seed with the pinned key, never dialed by anybody)
fn setup(relay_on: bool) -> (NodeHandle, NodeHandle) {
    let seed = node_with(vec![], Some(1), None);
    seed.set_relay(relay_on);
    let worker = node_with(vec![format!("{}@{}", seed.addr, seed.id_hex())], Some(101), Some(WorkerCfg::default()));
    wait("worker connected to the seed", 30, || seed.peer_count() >= 1 && worker.peer_count() >= 1);
    (seed, worker)
}

fn submit_job(seed: &NodeHandle, chunks: u32) -> (JobId, Kernel) {
    let client = SigningKey::from_bytes(&[100; 32]);
    let kernel = Kernel::MonteCarlo(McParams { seed: [3; 32], dim: 2, samples: 1024, out: 6, program: mc_pi_program() });
    let nonce = match request(&seed.addr, &Msg::GetAccount(pubkey(&client))).unwrap() {
        Msg::Account { next_nonce, .. } => next_nonce,
        m => panic!("{m:?}"),
    };
    let id = l2_id(&pubkey(&client), nonce, &kernel, chunks);
    let submit = TxKind::L2(L2Tx::Submit { kernel: kernel.clone(), chunks, escrow: 10 * COIN, deadline: seed.height() + 500 });
    assert!(matches!(client_tx(seed, &client, submit), Msg::Ack { ok: true, .. }));
    (id, kernel)
}

#[test]
fn a_worker_nobody_can_dial_is_audited_through_the_seed() {
    let (seed, worker) = setup(true);
    wait("the seed knows a route to the worker", 30, || seed.relay_routes() == 1);
    let (id, kernel) = submit_job(&seed, 24);
    wait("result published", 60, || l2_status(&seed.addr, id).starts_with("Posted"));
    let Msg::L2Info { root: Some(root), chunks, .. } = request(&seed.addr, &Msg::GetL2(id)).unwrap() else { panic!("root") };
    let depth = depth_for(chunks as usize);

    // by the level 2 job: the seed itself finds who holds it
    let by_job = format!("relay://{}/l2:{}", seed.addr, hex32(&id));
    // by the account: the worker key of this test
    let by_acct = format!("relay://{}/acct:{}", seed.addr, hex32(&pubkey(&SigningKey::from_bytes(&[101; 32]))));
    for addr in [&by_job, &by_acct] {
        for i in 0..chunks {
            let Msg::L2ChunkData { out, path } = request(addr, &Msg::GetL2Chunk { job: id, chunk: i }).unwrap() else { panic!("chunk {i} through {addr}") };
            assert!(verify_proof(&root, chunk_leaf(i, &out), i as usize, &path, depth), "the relayed chunk opens against the on-chain root");
            assert_eq!(kernel.chunk(i), out, "and it is the right output");
        }
    }

    // a connection kept open for several requests (what `l2-finish` does)
    let mut c = Conn::connect(&by_job).unwrap();
    c.set_timeout(Duration::from_secs(10));
    for i in 0..3 {
        c.send(&Msg::GetL2Chunk { job: id, chunk: i }).unwrap();
        let reply = loop {
            match c.recv().unwrap() {
                Msg::Hello { .. } => continue, // every node greets a new connection first
                r => break r,
            }
        };
        assert!(matches!(reply, Msg::L2ChunkData { .. }), "{reply:?}");
    }
    // 8 clients at once
    std::thread::scope(|s| {
        for t in 0..8u32 {
            let a = by_job.clone();
            s.spawn(move || {
                for k in 0..chunks {
                    let r = request(&a, &Msg::GetL2Chunk { job: id, chunk: (k + t) % chunks });
                    assert!(matches!(r, Ok(Msg::L2ChunkData { .. })), "{r:?}");
                }
            });
        }
    });

    // what writes, or relays again, does not travel
    let put = Msg::PutJob { a: vec![0; 4], b: vec![0; 4] };
    // requests whose reply is a matrix do not travel either (they need a direct connection)
    for big in [Msg::GetOp { job: id, op: 0 }, Msg::GetOpsBox { job: id, ops: vec![0] }, Msg::GetResult(id), Msg::GetTiledSum { job: id, i: 0, j: 0 }] {
        match request(&by_job, &big).unwrap() {
            Msg::Ack { ok: false, info } => assert!(info.contains("not relayable"), "{info}"),
            r => panic!("{r:?}"),
        }
    }
    let nest = Msg::RelayRequest { to: RelayTo::L2Job(id), req: encode(&Msg::GetStatus).unwrap() };
    for m in [put, nest] {
        match request(&by_job, &m).unwrap() {
            Msg::Ack { ok: false, info } => assert!(info.contains("not relayable"), "{info}"),
            r => panic!("{r:?}"),
        }
    }
    // an unknown worker, a job nobody holds
    let ghost = format!("relay://{}/acct:{}", seed.addr, hex32(&[7; 32]));
    assert!(matches!(request(&ghost, &Msg::GetL2(id)).unwrap(), Msg::Ack { ok: false, info } if info.contains("no relay route")));
    let nojob = format!("relay://{}/l2:{}", seed.addr, hex32(&[9; 32]));
    assert!(matches!(request(&nojob, &Msg::GetL2(id)).unwrap(), Msg::Ack { ok: false, info } if info.contains("no worker")));

    // the audit then settles the job like a direct one
    let client = SigningKey::from_bytes(&[100; 32]);
    assert!(matches!(client_tx(&seed, &client, TxKind::L2(L2Tx::Accept { job_id: id })), Msg::Ack { ok: true, .. }));
    wait("settled", 60, || l2_status(&seed.addr, id) == "Settled");

    // the worker leaves: the route goes away at once and a request is refused, not left hanging
    worker.stop();
    drop(c);
    wait("route removed", 30, || seed.relay_routes() == 0);
    assert!(matches!(request(&by_acct, &Msg::GetL2(id)).unwrap(), Msg::Ack { ok: false, .. }));
    assert!(seed.shared.node.lock().unwrap().chain.check_supply_invariant());
    seed.stop();
}

#[test]
fn a_node_that_is_not_a_relay_refuses_and_keeps_no_route() {
    let (seed, worker) = setup(false);
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(seed.relay_routes(), 0, "announcements are ignored unless --relay");
    let addr = format!("relay://{}/acct:{}", seed.addr, hex32(&pubkey(&SigningKey::from_bytes(&[101; 32]))));
    assert!(matches!(request(&addr, &Msg::GetStatus).unwrap(), Msg::Ack { ok: false, info } if info.contains("not a relay")));
    worker.stop();
    seed.stop();
}

#[test]
fn a_forged_announcement_does_not_create_or_steal_a_route() {
    let (seed, worker) = setup(true);
    wait("route", 30, || seed.relay_routes() == 1);
    let real = SigningKey::from_bytes(&[101; 32]);
    // an attacker connects as a peer (Hello) and announces the worker's account with the worker's own genuine signature:
    // that signature is bound to the WORKER's connection key, not to the attacker's, so it must be refused
    let mut c = Conn::connect(&seed.addr).unwrap();
    c.send(&Msg::Hello { node_id: 42, genesis: seed.shared.genesis_hash, height: 0, work: 0, listen: None }).unwrap();
    let sig = sign(&real, &relay_msg(&seed.shared.genesis_hash, &worker.shared.keys.public));
    c.send(&Msg::RelayAnnounce { account: pubkey(&real), sig }).unwrap();
    // and one with a random signature
    c.send(&Msg::RelayAnnounce { account: pubkey(&real), sig: vec![1; 64] }).unwrap();
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(seed.relay_routes(), 1, "still only the real route");
    // the real worker still answers
    let acct = format!("relay://{}/acct:{}", seed.addr, hex32(&pubkey(&real)));
    assert!(matches!(request(&acct, &Msg::GetStatus).unwrap(), Msg::Status { .. }), "the route of the real worker was not stolen");
    drop(c);
    worker.stop();
    seed.stop();
}

#[test]
fn relay_address_syntax() {
    let id = [0xabu8; 32];
    let a = format!("relay://seed.example.org:9000/l2:{}", hex32(&id));
    assert_eq!(parse_relay_addr(&a), Some(("seed.example.org:9000".to_string(), RelayTo::L2Job(id))));
    let b = format!("relay://127.0.0.1:9000/acct:{}", hex32(&id));
    assert_eq!(parse_relay_addr(&b), Some(("127.0.0.1:9000".to_string(), RelayTo::Account(id))));
    for bad in ["127.0.0.1:9000", "relay://host/l2:zz", "relay://host/xx:00", "relay://host", "relay://host/l2:abcd"] {
        assert_eq!(parse_relay_addr(bad), None, "{bad}");
    }
}
