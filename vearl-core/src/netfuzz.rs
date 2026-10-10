//! FUZZING OF A RUNNING NODE, over the network: raw bytes before the handshake, truncated handshakes, random application
//! messages on a valid Noise channel, and MUTATED VALID messages (all the client and peer requests, with boundary
//! values), sent as a client (without Hello) or as a peer (after Hello). After each slice, we check that the node
//! still answers (`GetStatus`); at the end, that it kept mining. A panic in a handler that holds the node lock
//! would poison it and kill the node: this is what this fuzz looks for.
use crate::crypto::*;
use crate::net::*;
use crate::types::*;
use crate::work::TileProof;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::io::Write;
use std::net::TcpStream;
use std::time::{Duration, Instant};

#[derive(Debug, Default)]
pub struct FuzzReport {
    pub iterations: usize,
    pub sent: usize,
    pub replies: usize,
    pub liveness_checks: usize,
    pub failures: Vec<String>,
    pub height_before: u64,
    pub height_after: u64,
}

/// Byte mutation (same family as tests/fuzz_net.rs): bit, byte, 8 x 0xFF (giant lengths), 8 x 0, truncation, append, +1..3.
pub fn mutate(bytes: &mut Vec<u8>, rng: &mut StdRng) {
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
                bytes.iter_mut().skip(p).take(8).for_each(|b| *b = 0xFF);
            }
            3 => {
                let p = rng.gen_range(0..len);
                bytes.iter_mut().skip(p).take(8).for_each(|b| *b = 0);
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

fn status(addr: &str) -> Option<u64> {
    match request(addr, &Msg::GetStatus) {
        Ok(Msg::Status { height, .. }) => Some(height),
        _ => None,
    }
}

/// Corpus: one example of EACH request (and a few replies, which a node must ignore), with boundary values.
fn corpus(n: usize, genesis: Hash, rng: &mut StdRng) -> Vec<Msg> {
    let id: Hash = rng.gen();
    let pk: PubKey = rng.gen();
    let mut m = |len: usize| (0..len).map(|_| rng.gen_range(-64i8..=64)).collect::<Vec<i8>>();
    let nn = n * n;
    let hdr = Header { version: 2, height: 1, prev: [0; 32], timestamp: 5, miner: pk, class: WorkClass::Native, extra_nonce: 0, ti: 0, tj: 0, jackpot: [0; 32], tx_root: [0; 32], target: 1, a_root: [0; 32], b_root: [0; 32], accounts_root: [0; 32] };
    let proof = TileProof { ti: 0, tj: 0, a_rows: vec![vec![0; 4]], a_paths: vec![vec![[0; 32]]], b_cols: vec![], b_paths: vec![] };
    let block = Block { header: hdr.clone(), proof: proof.clone(), txs: vec![] };
    let tx = SignedTx { tx: Tx { from: pk, nonce: 0, fee: 0, kind: TxKind::Transfer { to: pk, amount: 1 } }, sig: vec![0; 64] };
    vec![
        Msg::Hello { node_id: 1, genesis, height: u64::MAX, work: u128::MAX, listen: Some("127.0.0.1:1".into()) },
        Msg::GetBlocks { from: u64::MAX },
        Msg::GetBlocks { from: 0 },
        Msg::Blocks(vec![block.clone()]),
        Msg::NewBlock(block),
        Msg::NewTx(tx),
        Msg::GetPeers,
        Msg::Peers(vec!["x".repeat(70), "127.0.0.1:9".into()]),
        Msg::GetStatus,
        Msg::GetAccount(pk),
        Msg::Ack { ok: true, info: "x".into() },
        Msg::PutJob { a: m(nn), b: m(nn) },
        Msg::PutJob { a: m(3), b: m(nn) },
        Msg::GetJob(id),
        Msg::GetServed(id),
        Msg::GetRep(pk),
        Msg::GetResult(id),
        Msg::PutMlp { x0: m(nn), ws: vec![m(nn), m(7)] },
        Msg::GetMlp(id),
        Msg::GetLayer { job: id, layer: 255 },
        Msg::PutTiled { a: m(nn * 2), b: m(nn * 2), dims: (2, 1, 2) },
        Msg::PutTiled { a: m(10), b: m(10), dims: (0, 255, 9) },
        Msg::GetTiled(id),
        Msg::GetTiledGroup { job: id, i: 255, j: 255 },
        Msg::PutProg { x0: m(nn), ws: vec![m(nn)] },
        Msg::PutProg { x0: vec![], ws: vec![] },
        Msg::GetProg(id),
        Msg::GetOp { job: id, op: 255 },
        Msg::GetL2(id),
        Msg::GetL2Chunk { job: id, chunk: u32::MAX },
        Msg::PutProgMore { x0_root: id, ws: vec![m(nn)] },
        Msg::PutModel { ws: vec![m(nn), vec![]] },
        Msg::HasModel { roots: vec![id; 100] },
        Msg::PutProgRef { x0: m(nn), w_roots: vec![id; 3] },
        Msg::PutTiledStage { key: id, which: 0, dims: (8, 8, 8), row_start: u32::MAX, data: m(n) },
        Msg::PutTiledStage { key: id, which: 7, dims: (1, 1, 1), row_start: 0, data: m(nn) },
        Msg::PutTiledStage { key: id, which: 1, dims: (2, 2, 2), row_start: (2 * n - 1) as u32, data: m(2 * n * 2) },
        Msg::PutTiledDone { key: id },
        Msg::GetTiledProduct { job: id, i: 0, j: 0, k: 255 },
        Msg::GetTiledSum { job: id, i: 255, j: 0 },
        Msg::GetHeaders { from: u64::MAX },
        Msg::GetHeaders { from: 0 },
        Msg::GetAccountProof(pk),
        Msg::Headers(vec![(hdr, proof)]),
        Msg::Status { height: 0, tip: id, work: 0, mempool: 0, peers: 0, burned: 0, minted: 0, params: (0, 0, 0, 0) },
        Msg::OpData { z: None, x: vec![] },
        Msg::GetOpsBox { job: id, ops: vec![255; 300] },
        Msg::GetOpsBox { job: id, ops: vec![] },
        Msg::GetOpsBox { job: id, ops: vec![0, 1, 2] },
        Msg::OpsBox(vec![BoxedOp { x_dims: (u32::MAX, u32::MAX), x: vec![], z_dims: Some((5, 5)), z: vec![] }]),
    ]
}

/// Runs `iters` trials against the node `addr`. `seed` makes the campaign reproducible.
pub fn run(addr: &str, iters: usize, seed: u64, log: bool) -> FuzzReport {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut rep = FuzzReport { height_before: status(addr).unwrap_or(0), ..Default::default() };
    let n = match request(addr, &Msg::GetStatus) {
        Ok(Msg::Status { params, .. }) => params.0 as usize,
        _ => 128,
    };
    // the node announces its Hello (with the genesis hash) right from the handshake
    let genesis = Conn::connect(addr).ok().and_then(|mut c| {
        c.set_timeout(Duration::from_secs(3));
        match c.recv() {
            Ok(Msg::Hello { genesis, .. }) => Some(genesis),
            _ => None,
        }
    });
    let Some(genesis) = genesis else {
        rep.failures.push("no Hello from the node at the start".into());
        return rep;
    };
    let corpus: Vec<Vec<u8>> = corpus(n, genesis, &mut rng).iter().map(|m| encode(m).unwrap()).collect();
    let sa: std::net::SocketAddr = addr.parse().expect("ip:port address");
    let t0 = Instant::now();
    for it in 0..iters {
        rep.iterations += 1;
        let strategy = rng.gen_range(0..100);
        let res: std::io::Result<()> = (|| {
            if strategy < 10 {
                // raw bytes before any handshake
                let mut s = TcpStream::connect_timeout(&sa, Duration::from_secs(2))?;
                let junk: Vec<u8> = (0..rng.gen_range(0..4096)).map(|_| rng.gen()).collect();
                let _ = s.write_all(&junk);
                rep.sent += 1;
                return Ok(());
            }
            if strategy < 18 {
                // handshake frame announced long then truncated
                let mut s = TcpStream::connect_timeout(&sa, Duration::from_secs(2))?;
                let _ = s.write_all(&[0xFF, 0xFF, 1, 2, 3]);
                rep.sent += 1;
                return Ok(());
            }
            let mut c = Conn::connect(addr)?;
            c.set_timeout(Duration::from_millis(400));
            let _ = c.recv(); // Hello of the node
            let as_peer = strategy >= 70;
            if as_peer {
                c.send(&Msg::Hello { node_id: rng.gen(), genesis, height: 0, work: 0, listen: None })?;
            }
            if strategy < 30 {
                // random application message (not a valid Msg), sometimes very long
                let len = if rng.gen_bool(0.1) { rng.gen_range(1 << 16..1 << 20) } else { rng.gen_range(0..512) };
                let junk: Vec<u8> = (0..len).map(|_| rng.gen()).collect();
                c.send_raw(&junk)?;
                rep.sent += 1;
            } else {
                for _ in 0..rng.gen_range(1..=4) {
                    let mut b = corpus[rng.gen_range(0..corpus.len())].clone();
                    if rng.gen_bool(0.8) {
                        mutate(&mut b, &mut rng);
                    }
                    c.send_raw(&b)?;
                    rep.sent += 1;
                }
            }
            while c.recv().is_ok() {
                rep.replies += 1;
            }
            Ok(())
        })();
        let _ = res; // a connection refused or closed by the node is normal behaviour
        if (it + 1) % 50 == 0 || it + 1 == iters {
            rep.liveness_checks += 1;
            if status(addr).is_none() {
                std::thread::sleep(Duration::from_millis(500));
                if status(addr).is_none() {
                    rep.failures.push(format!("the node no longer answers after trial {} (strategy {strategy})", it + 1));
                    break;
                }
            }
            if log {
                eprintln!("[netfuzz] {} trials, {} messages, {} replies, {:.0} s", it + 1, rep.sent, rep.replies, t0.elapsed().as_secs_f64());
            }
        }
    }
    rep.height_after = status(addr).unwrap_or(0);
    rep
}
