use ed25519_dalek::SigningKey;
use vearl_core::chain::*;
use vearl_core::crypto::*;
use vearl_core::state::*;
use vearl_core::types::*;
use vearl_core::work::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

const P: WorkParams = WorkParams::TESTNET;

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

fn new_chain(alloc: &[(&SigningKey, u64)]) -> Chain {
    let a: Vec<(PubKey, u64)> = alloc.iter().map(|(k, v)| (pubkey(k), v * COIN)).collect();
    Chain::genesis(P, &a, 1_000, u64::MAX / 64)
}

fn ts(chain: &Chain) -> u64 {
    chain.tip_ts() + BLOCK_TIME
}

fn random_job_matrices(seed: u64) -> CommittedMatrices {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut m = || (0..P.n * P.n).map(|_| rng.gen_range(-64..=64) as i8).collect::<Vec<i8>>();
    let a = m();
    let b = m();
    CommittedMatrices::new(a, b, &P)
}

fn mine_native(chain: &mut Chain, sk: &SigningKey, txs: Vec<SignedTx>) {
    let t = ts(chain);
    let out = chain.mine(sk, WorkClass::Native, None, txs, t).unwrap();
    chain.validate_and_apply(out.block).unwrap();
}

fn mine_job(chain: &mut Chain, sk: &SigningKey, id: JobId, cm: &CommittedMatrices, txs: Vec<SignedTx>) -> Vec<i32> {
    let t = ts(chain);
    let out = chain.mine(sk, WorkClass::Job(id), Some(cm), txs, t).unwrap();
    let c = out.c.clone();
    chain.validate_and_apply(out.block).unwrap();
    c
}

fn submit_job(chain: &Chain, client: &SigningKey, cm: &CommittedMatrices, escrow: u64) -> (SignedTx, JobId) {
    let nonce = chain.state.nonce(&pubkey(client));
    let id = job_id(&pubkey(client), nonce, &cm.a_root(), &cm.b_root());
    let deadline = chain.height() + 200;
    (make_tx(chain, client, 0, TxKind::SubmitJob { a_root: cm.a_root(), b_root: cm.b_root(), escrow: escrow * COIN, deadline }), id)
}

#[test]
fn decode_recovers_true_product() {
    let cm = random_job_matrices(1);
    let sg = [7u8; 32];
    let (out, noise) = run_noisy_gemm(&cm.a, &cm.b, &sg, &P);
    assert_ne!(out.c_noised, plain_gemm(&cm.a, &cm.b, P.n), "noised product must differ from the plain one");
    assert_eq!(decode(&out.c_noised, &cm.a, &cm.b, &noise, &P), plain_gemm(&cm.a, &cm.b, P.n));
}

#[test]
fn tile_proof_verifies_and_rejects_tampering() {
    let cm = random_job_matrices(2);
    let sg = [9u8; 32];
    let (out, _) = run_noisy_gemm(&cm.a, &cm.b, &sg, &P);
    let (ti, tj) = (3, 5);
    let proof = build_proof(&cm, ti, tj, &P);
    let jp = verify_tile_proof(&proof, &sg, &cm.a_root(), &cm.b_root(), &P).expect("valid proof");
    assert_eq!(jp, out.jackpots[ti * P.tiles() + tj], "verifier tile hash == miner tile hash");

    let mut bad = proof.clone();
    bad.a_rows[0][0] = bad.a_rows[0][0].wrapping_add(1);
    assert!(verify_tile_proof(&bad, &sg, &cm.a_root(), &cm.b_root(), &P).is_none(), "tampered row must fail Merkle check");

    let other = random_job_matrices(3);
    assert!(verify_tile_proof(&proof, &sg, &other.a_root(), &other.b_root(), &P).is_none(), "wrong commitments must fail");
}

#[test]
fn freivalds_detects_wrong_result_and_localizes_it() {
    let cm = random_job_matrices(4);
    let c = plain_gemm(&cm.a, &cm.b, P.n);
    let seed = [1u8; 32];
    assert!(freivalds(&cm.a, &cm.b, &c, P.n, 20, &seed));
    let mut bad = c.clone();
    bad[5 * P.n + 7] += 1;
    assert!(!freivalds(&cm.a, &cm.b, &bad, P.n, 20, &seed));
    assert_eq!(find_bad_entry(&cm.a, &cm.b, &bad, P.n, &seed), Some((5, 7)));
}

#[test]
fn native_block_roundtrip_and_replay_is_rejected() {
    let (miner, other) = (key(1), key(2));
    let mut chain = new_chain(&[]);
    mine_native(&mut chain, &miner, vec![]);
    assert_eq!(chain.height(), 1);
    assert_eq!(chain.state.balance(&pubkey(&miner)), base_reward(0));
    assert!(chain.check_supply_invariant());

    // A third party cannot take over another's work by changing the beneficiary.
    let t = ts(&chain);
    let mut out = chain.mine(&miner, WorkClass::Native, None, vec![], t).unwrap();
    out.block.header.miner = pubkey(&other);
    assert!(chain.validate_and_apply(out.block).is_err());

    // Nor change the transactions afterwards (tx_root is bound to sigma).
    let t = ts(&chain);
    let mut out = chain.mine(&miner, WorkClass::Native, None, vec![], t).unwrap();
    let junk = make_tx(&chain, &other, 0, TxKind::Transfer { to: pubkey(&other), amount: 0 });
    out.block.txs = vec![junk];
    assert!(chain.validate_and_apply(out.block).is_err());
}

#[test]
fn honest_job_lifecycle_pays_worker_and_conserves_supply() {
    let (alice, bob, carol) = (key(10), key(11), key(12));
    let mut chain = new_chain(&[(&alice, 1000), (&bob, 100)]);
    let cm = random_job_matrices(20);
    let (submit, id) = submit_job(&chain, &alice, &cm, 50);
    mine_native(&mut chain, &carol, vec![submit]);
    let burn = bps(50 * COIN, BURN_BPS);
    assert_eq!(chain.state.burned, burn);

    let claim = make_tx(&chain, &bob, 0, TxKind::ClaimJob { job_id: id });
    mine_native(&mut chain, &carol, vec![claim]);

    let bob_before = chain.state.balance(&pubkey(&bob));
    let h = chain.height();
    let c = mine_job(&mut chain, &bob, id, &cm, vec![]);
    assert_eq!(c, plain_gemm(&cm.a, &cm.b, P.n), "the mined GEMM yields the client's real product");
    // security (base) + usefulness rebate (bounded by the burn)
    let rebate = (base_reward(h) / 2).min(burn);
    assert_eq!(chain.state.balance(&pubkey(&bob)), bob_before + base_reward(h) + rebate);

    let c_root = c_tree(&c, P.n).root();
    let post = make_tx(&chain, &bob, 0, TxKind::PostResult { job_id: id, c_root });
    mine_job(&mut chain, &bob, id, &cm, vec![post]);

    // Too early: Settle refused.
    let early = make_tx(&chain, &carol, 0, TxKind::Settle { job_id: id });
    let t = ts(&chain);
    let out = chain.mine(&carol, WorkClass::Native, None, vec![early], t).unwrap();
    assert!(chain.validate_and_apply(out.block).is_err());

    for _ in 0..CHALLENGE_WINDOW + 1 {
        mine_native(&mut chain, &carol, vec![]);
    }
    let before = chain.state.balance(&pubkey(&bob));
    let settle = make_tx(&chain, &carol, 0, TxKind::Settle { job_id: id });
    mine_native(&mut chain, &carol, vec![settle]);
    let net = 50 * COIN - burn;
    assert_eq!(chain.state.balance(&pubkey(&bob)), before + net + bps(net, BOND_BPS));
    assert_eq!(chain.state.jobs[&id].status, JobStatus::Settled);
    assert!(chain.check_supply_invariant());
}

#[test]
fn only_the_claimed_worker_can_mine_job_blocks() {
    let (alice, bob, mallory) = (key(10), key(11), key(13));
    let mut chain = new_chain(&[(&alice, 1000), (&bob, 100), (&mallory, 100)]);
    let cm = random_job_matrices(21);
    let (submit, id) = submit_job(&chain, &alice, &cm, 10);
    mine_native(&mut chain, &bob, vec![submit]);
    // Unclaimed: no miner can declare itself a worker.
    let t = ts(&chain);
    let out = chain.mine(&mallory, WorkClass::Job(id), Some(&cm), vec![], t).unwrap();
    assert!(chain.validate_and_apply(out.block).is_err());
    let claim = make_tx(&chain, &bob, 0, TxKind::ClaimJob { job_id: id });
    mine_native(&mut chain, &bob, vec![claim]);
    let t = ts(&chain);
    let out = chain.mine(&mallory, WorkClass::Job(id), Some(&cm), vec![], t).unwrap();
    assert!(chain.validate_and_apply(out.block).is_err(), "mallory is not the worker");
}

#[test]
fn fraudulent_result_is_slashed_with_a_cheap_proof() {
    let (alice, mallory, carol) = (key(10), key(13), key(12));
    let mut chain = new_chain(&[(&alice, 1000), (&mallory, 100)]);
    let cm = random_job_matrices(22);
    let (submit, id) = submit_job(&chain, &alice, &cm, 50);
    mine_native(&mut chain, &carol, vec![submit]);
    let claim = make_tx(&chain, &mallory, 0, TxKind::ClaimJob { job_id: id });
    mine_native(&mut chain, &carol, vec![claim]);

    let mut c_bad = plain_gemm(&cm.a, &cm.b, P.n);
    c_bad[5 * P.n + 7] += 1; // the worker cheats on ONE entry
    let bad_tree = c_tree(&c_bad, P.n);
    let post = make_tx(&chain, &mallory, 0, TxKind::PostResult { job_id: id, c_root: bad_tree.root() });
    mine_native(&mut chain, &carol, vec![post]);

    // The client verifies (Freivalds), locates, and publishes an O(n log n) proof.
    let (i, j) = find_bad_entry(&cm.a, &cm.b, &c_bad, P.n, &[3u8; 32]).expect("fraud detected");
    let proof = FraudProof {
        job_id: id,
        i: i as u32,
        j: j as u32,
        a_row: cm.a[i * P.n..(i + 1) * P.n].to_vec(),
        a_path: cm.a_tree.proof(i),
        b_col: cm.col(j, P.n),
        b_path: cm.b_tree.proof(j),
        c_row: row_of(&c_bad, P.n, i),
        c_path: bad_tree.proof(i),
    };
    let alice_before = chain.state.balance(&pubkey(&alice));
    let fraud = make_tx(&chain, &alice, 0, TxKind::Fraud(proof.clone()));
    mine_native(&mut chain, &carol, vec![fraud]);
    let net = 50 * COIN - bps(50 * COIN, BURN_BPS);
    let bond = bps(net, BOND_BPS);
    assert_eq!(chain.state.balance(&pubkey(&alice)), alice_before + net + bond / 2);
    assert_eq!(chain.state.jobs[&id].status, JobStatus::Slashed);
    assert!(chain.check_supply_invariant());

    // A false accusation (correct entry) is rejected.
    let (alice2, carol2, mallory2) = (key(10), key(12), key(13));
    let mut chain2 = new_chain(&[(&alice2, 1000), (&mallory2, 100)]);
    let (s, id2) = submit_job(&chain2, &alice2, &cm, 50);
    mine_native(&mut chain2, &carol2, vec![s]);
    let cl = make_tx(&chain2, &mallory2, 0, TxKind::ClaimJob { job_id: id2 });
    mine_native(&mut chain2, &carol2, vec![cl]);
    let good = plain_gemm(&cm.a, &cm.b, P.n);
    let good_tree = c_tree(&good, P.n);
    let post = make_tx(&chain2, &mallory2, 0, TxKind::PostResult { job_id: id2, c_root: good_tree.root() });
    mine_native(&mut chain2, &carol2, vec![post]);
    let mut false_accusation = proof;
    false_accusation.job_id = id2;
    false_accusation.c_row = row_of(&good, P.n, i);
    false_accusation.c_path = good_tree.proof(i);
    let fr = make_tx(&chain2, &alice2, 0, TxKind::Fraud(false_accusation));
    let t = ts(&chain2);
    let out = chain2.mine(&carol2, WorkClass::Native, None, vec![fr], t).unwrap();
    assert!(chain2.validate_and_apply(out.block).is_err(), "correct entry is not fraud");
}

#[test]
fn wash_trading_a_fake_job_is_not_profitable() {
    // An attacker is client AND worker of its own job to collect the usefulness rebate.
    let x = key(20);
    let mut chain = new_chain(&[(&x, 100)]);
    let start = chain.state.balance(&pubkey(&x));
    let cm = random_job_matrices(23);
    let mut base_sum = 0;

    let (submit, id) = submit_job(&chain, &x, &cm, 10);
    base_sum += base_reward(chain.height());
    mine_native(&mut chain, &x, vec![submit]);
    let claim = make_tx(&chain, &x, 0, TxKind::ClaimJob { job_id: id });
    base_sum += base_reward(chain.height());
    mine_native(&mut chain, &x, vec![claim]);
    let mut c = vec![];
    for _ in 0..3 {
        base_sum += base_reward(chain.height());
        c = mine_job(&mut chain, &x, id, &cm, vec![]);
    }
    let post = make_tx(&chain, &x, 0, TxKind::PostResult { job_id: id, c_root: c_tree(&c, P.n).root() });
    base_sum += base_reward(chain.height());
    mine_job(&mut chain, &x, id, &cm, vec![post]);
    let accept = make_tx(&chain, &x, 0, TxKind::Accept { job_id: id });
    base_sum += base_reward(chain.height());
    mine_native(&mut chain, &x, vec![accept]);

    let end = chain.state.balance(&pubkey(&x));
    assert!(end <= start + base_sum, "faking usefulness must not beat plain mining: end={end} start+base={}", start + base_sum);
    assert!(chain.check_supply_invariant());
}

#[test]
fn difficulty_moves_in_the_right_direction() {
    let t = u64::MAX / 64;
    assert!(next_target(t, 5) < t, "blocks too fast -> harder (smaller target)");
    assert!(next_target(t, 300) > t, "blocks too slow -> easier");
    assert_eq!(next_target(t, BLOCK_TIME), t);
}

#[test]
fn a_burst_of_fast_blocks_cannot_exhaust_the_challenge_window() {
    // Devnet / attacker: dozens of blocks in a short time (timestamps +1 s) must NOT close
    // the dispute window: it is also defined in timestamp seconds.
    let (alice, mallory, carol) = (key(10), key(13), key(12));
    let mut chain = new_chain(&[(&alice, 1000), (&mallory, 100)]);
    chain.retarget = Retarget::Fixed; // 43 blocks at +1 s: we test the dispute window, not the difficulty
    let cm = random_job_matrices(31);
    let (submit, id) = submit_job(&chain, &alice, &cm, 50);
    mine_native(&mut chain, &carol, vec![submit]);
    let claim = make_tx(&chain, &mallory, 0, TxKind::ClaimJob { job_id: id });
    mine_native(&mut chain, &carol, vec![claim]);
    let mut c_bad = plain_gemm(&cm.a, &cm.b, P.n);
    c_bad[3 * P.n + 9] += 1;
    let bad_tree = c_tree(&c_bad, P.n);
    let post = make_tx(&chain, &mallory, 0, TxKind::PostResult { job_id: id, c_root: bad_tree.root() });
    mine_native(&mut chain, &carol, vec![post]);

    // 40 blocks, one per timestamp second: far more than CHALLENGE_WINDOW blocks, but 40 s << CHALLENGE_SECS
    for _ in 0..40 {
        let t = chain.tip_ts() + 1;
        let out = chain.mine(&carol, WorkClass::Native, None, vec![], t).unwrap();
        chain.validate_and_apply(out.block).unwrap();
    }
    let settle = make_tx(&chain, &carol, 0, TxKind::Settle { job_id: id });
    let t = chain.tip_ts() + 1;
    let out = chain.mine(&carol, WorkClass::Native, None, vec![settle], t).unwrap();
    assert!(chain.validate_and_apply(out.block).is_err(), "the cheater cannot get paid by a burst of blocks");

    // and the fraud proof is still receivable
    let (i, j) = find_bad_entry(&cm.a, &cm.b, &c_bad, P.n, &[3u8; 32]).unwrap();
    let proof = FraudProof {
        job_id: id,
        i: i as u32,
        j: j as u32,
        a_row: cm.a[i * P.n..(i + 1) * P.n].to_vec(),
        a_path: cm.a_tree.proof(i),
        b_col: cm.col(j, P.n),
        b_path: cm.b_tree.proof(j),
        c_row: row_of(&c_bad, P.n, i),
        c_path: bad_tree.proof(i),
    };
    let fraud = make_tx(&chain, &alice, 0, TxKind::Fraud(proof));
    let t = chain.tip_ts() + 1;
    let out = chain.mine(&carol, WorkClass::Native, None, vec![fraud], t).unwrap();
    chain.validate_and_apply(out.block).unwrap();
    assert_eq!(chain.state.jobs[&id].status, JobStatus::Slashed);
    assert!(chain.check_supply_invariant());
}

// ---------------------------------------------------------------- LWMA-1 (difficulty)

fn fake_block(target: u64, ts: u64) -> Block {
    Block {
        header: Header { version: 1, height: 0, prev: [0; 32], timestamp: ts, miner: [0; 32], class: WorkClass::Native, extra_nonce: 0, ti: 0, tj: 0, jackpot: [0; 32], tx_root: [0; 32], target, a_root: [0; 32], b_root: [0; 32], accounts_root: [0; 32] },
        proof: TileProof { ti: 0, tj: 0, a_rows: vec![], a_paths: vec![], b_cols: vec![], b_paths: vec![] },
        txs: vec![],
    }
}

fn series(target: u64, solvetime: u64, n: usize) -> Vec<Block> {
    (0..n).map(|i| fake_block(target, 1_000 + i as u64 * solvetime)).collect()
}

#[test]
fn lwma_keeps_the_target_when_blocks_arrive_on_schedule() {
    let t = u64::MAX / 1000;
    let blocks = series(t, BLOCK_TIME, 40);
    assert_eq!(lwma_next_target(&blocks, 7), t, "block time = target => target unchanged");
    assert_eq!(lwma_next_target(&blocks[..10], 7), 7, "incomplete window: fallback to the starting rule");
}

#[test]
fn lwma_reacts_in_the_right_direction_and_within_bounds() {
    let t = u64::MAX / 1000;
    let slow = lwma_next_target(&series(t, 6 * BLOCK_TIME, 40), 0);
    assert!(slow > t && slow <= 2 * t, "blocks too slow => easier target (at most x2 per block)");
    let fast = lwma_next_target(&series(t, 1, 40), 0);
    assert!(fast < t && fast >= t / 2, "blocks too fast => harder target (at most /2 per block)");
    // extreme values: no overflow, lower bound respected
    assert_eq!(lwma_next_target(&series(u64::MAX, 6 * BLOCK_TIME, 40), 0), u64::MAX);
    assert!(lwma_next_target(&series(1 << 16, 1, 40), 0) >= 1 << 16);
    // a timestamp that goes backwards does not blow up the computation (solve time bounded to >= 1)
    let mut weird = series(t, BLOCK_TIME, 40);
    weird[35].header.timestamp = 500;
    let _ = lwma_next_target(&weird, 0);
}

#[test]
fn lwma_recovers_after_a_hashrate_collapse() {
    // Poisson process: the power drops by 95 % at block 60; the protocol's EXACT rule must bring back blocks
    // of about 30 s within a few dozen blocks (the EMA needed > 100).
    let mut rng = StdRng::seed_from_u64(7);
    let mut blocks: Vec<Block> = vec![];
    let (mut target, mut t_real, mut last_ts) = (u64::MAX / 30_000, 0.0f64, 1_000u64);
    let hash0 = 1000.0; // attempts/s; p_block/s = hash * target / 2^64; equilibrium: target = 2^64 / (hash*30)
    let mut intervals = vec![];
    for i in 0..400usize {
        let hash = if i >= 60 { hash0 * 0.05 } else { hash0 };
        let rate = hash * target as f64 / 18446744073709551616.0;
        let dt = -(1.0 - rng.gen::<f64>()).ln() / rate;
        t_real += dt;
        let ts = (1_000.0 + t_real) as u64;
        let ts = ts.max(last_ts + 1);
        intervals.push(dt);
        blocks.push(fake_block(target, ts));
        let ema = next_target(target, ts - last_ts);
        target = lwma_next_target(&blocks, ema);
        last_ts = ts;
    }
    let tail: f64 = intervals[350..].iter().sum::<f64>() / 50.0;
    assert!((18.0..=48.0).contains(&tail), "after 290 blocks the average time is back near 30 s (got {tail:.1} s)");
}
