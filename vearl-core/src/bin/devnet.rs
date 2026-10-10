//! Single-node devnet: replays a complete scenario (empty market -> honest job -> fraud).
use ed25519_dalek::SigningKey;
use vearl_core::chain::*;
use vearl_core::crypto::*;
use vearl_core::state::*;
use vearl_core::types::*;
use vearl_core::work::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

const P: WorkParams = WorkParams::TESTNET;

fn vrl(v: u64) -> String {
    format!("{:.2}", v as f64 / COIN as f64)
}

fn show(chain: &Chain, names: &[(&str, &SigningKey)]) {
    let line: Vec<String> = names.iter().map(|(n, k)| format!("{n}={}", vrl(chain.state.balance(&pubkey(k))))).collect();
    println!("    balances: {} | burned={} | minted={} | supply OK={}", line.join(" "), vrl(chain.state.burned), vrl(chain.state.minted), chain.check_supply_invariant());
}

fn mine(chain: &mut Chain, sk: &SigningKey, class: WorkClass, cm: Option<&CommittedMatrices>, txs: Vec<SignedTx>, label: &str) -> Vec<i32> {
    let t = chain.tip_ts() + BLOCK_TIME;
    let start = Instant::now();
    let out = chain.mine(sk, class, cm, txs, t).expect("mining");
    let c = out.c.clone();
    let (h, tile) = (out.block.header.height, (out.block.header.ti, out.block.header.tj));
    let ntx = out.block.txs.len();
    chain.validate_and_apply(out.block).expect("block valid");
    println!("  block #{h:<3} {label:<28} tile={tile:?} txs={ntx} sigma_trials={} ({:.0} ms)", out.attempts_sigma, start.elapsed().as_secs_f64() * 1e3);
    c
}

fn main() {
    let (alice, bob, carol, mallory) = (SigningKey::from_bytes(&[10; 32]), SigningKey::from_bytes(&[11; 32]), SigningKey::from_bytes(&[12; 32]), SigningKey::from_bytes(&[13; 32]));
    let names = [("alice", &alice), ("bob", &bob), ("carol", &carol), ("mallory", &mallory)];
    let alloc = [(pubkey(&alice), 1000 * COIN), (pubkey(&bob), 100 * COIN), (pubkey(&mallory), 100 * COIN)];
    let mut chain = Chain::genesis(P, &alloc, 1_000, u64::MAX / 64);

    println!("== VEARL devnet: n={} tile={} rank={} | initial target p~1/64 per tile ==", P.n, P.tile, P.rank);

    println!("\n[1] Empty market: carol mines Native Work (pure security, no usefulness claimed)");
    for _ in 0..3 {
        mine(&mut chain, &carol, WorkClass::Native, None, vec![], "native");
    }
    show(&chain, &names);

    println!("\n[2] Alice buys compute: matrices A,B committed (Merkle), escrow 50 VRL (10 % burned)");
    let mut rng = StdRng::seed_from_u64(42);
    let mut m = || (0..P.n * P.n).map(|_| rng.gen_range(-64..=64) as i8).collect::<Vec<i8>>();
    let (a, b) = (m(), m());
    let cm = CommittedMatrices::new(a, b, &P);
    let nonce = chain.state.nonce(&pubkey(&alice));
    let id = job_id(&pubkey(&alice), nonce, &cm.a_root(), &cm.b_root());
    let submit = make_tx(&chain, &alice, 0, TxKind::SubmitJob { a_root: cm.a_root(), b_root: cm.b_root(), escrow: 50 * COIN, deadline: chain.height() + 200 });
    mine(&mut chain, &carol, WorkClass::Native, None, vec![submit], "native + SubmitJob");
    let claim = make_tx(&chain, &bob, 0, TxKind::ClaimJob { job_id: id });
    mine(&mut chain, &carol, WorkClass::Native, None, vec![claim], "native + ClaimJob(bob)");
    show(&chain, &names);

    println!("\n[3] Bob mines WHILE computing Alice's real product A*B (duplex: useful work + lottery)");
    let c = mine(&mut chain, &bob, WorkClass::Job(id), Some(&cm), vec![], "job-backed");
    let ok = freivalds(&cm.a, &cm.b, &c, P.n, 20, &[1u8; 32]);
    println!("    Alice verifies the result by Freivalds (20 rounds, error <= 2^-20): {ok}");
    let post = make_tx(&chain, &bob, 0, TxKind::PostResult { job_id: id, c_root: c_tree(&c, P.n).root() });
    mine(&mut chain, &bob, WorkClass::Job(id), Some(&cm), vec![post], "job-backed + PostResult");
    let accept = make_tx(&chain, &alice, 0, TxKind::Accept { job_id: id });
    mine(&mut chain, &carol, WorkClass::Native, None, vec![accept], "native + Accept(alice)");
    show(&chain, &names);

    println!("\n[4] Mallory cheats: posts a wrong result on ONE entry; Alice publishes an O(n log n) fraud proof");
    let nonce = chain.state.nonce(&pubkey(&alice));
    let id2 = job_id(&pubkey(&alice), nonce, &cm.a_root(), &cm.b_root());
    let submit = make_tx(&chain, &alice, 0, TxKind::SubmitJob { a_root: cm.a_root(), b_root: cm.b_root(), escrow: 50 * COIN, deadline: chain.height() + 200 });
    mine(&mut chain, &carol, WorkClass::Native, None, vec![submit], "native + SubmitJob #2");
    let claim = make_tx(&chain, &mallory, 0, TxKind::ClaimJob { job_id: id2 });
    mine(&mut chain, &carol, WorkClass::Native, None, vec![claim], "native + ClaimJob(mallory)");
    let mut c_bad = plain_gemm(&cm.a, &cm.b, P.n);
    c_bad[17 * P.n + 3] += 1;
    let bad_tree = c_tree(&c_bad, P.n);
    let post = make_tx(&chain, &mallory, 0, TxKind::PostResult { job_id: id2, c_root: bad_tree.root() });
    mine(&mut chain, &carol, WorkClass::Native, None, vec![post], "native + PostResult(wrong)");
    let (i, j) = find_bad_entry(&cm.a, &cm.b, &c_bad, P.n, &[2u8; 32]).expect("fraud detected");
    println!("    Freivalds locates the wrong entry: C[{i}][{j}]");
    let proof = FraudProof {
        job_id: id2,
        i: i as u32,
        j: j as u32,
        a_row: cm.a[i * P.n..(i + 1) * P.n].to_vec(),
        a_path: cm.a_tree.proof(i),
        b_col: cm.col(j, P.n),
        b_path: cm.b_tree.proof(j),
        c_row: row_of(&c_bad, P.n, i),
        c_path: bad_tree.proof(i),
    };
    let size = bincode::serialize(&proof).unwrap().len();
    let fraud = make_tx(&chain, &alice, 0, TxKind::Fraud(proof));
    mine(&mut chain, &carol, WorkClass::Native, None, vec![fraud], "native + Fraud(alice)");
    println!("    fraud proof size: {size} bytes; job status: {:?}", chain.state.jobs[&id2].status);
    show(&chain, &names);

    println!("\nChain: {} blocks, final target = {:.3e} (p={:.4})", chain.height(), chain.target as f64, chain.target as f64 / u64::MAX as f64);
}
