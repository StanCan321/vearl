//! Quantised multi-layer inference jobs: life cycle, fraud proofs (GEMM and activation), access controls.
use ed25519_dalek::SigningKey;
use vearl_core::chain::*;
use vearl_core::crypto::*;
use vearl_core::mlp::*;
use vearl_core::state::*;
use vearl_core::types::*;
use vearl_core::work::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

const P: WorkParams = WorkParams::TESTNET;
const SHIFT: u8 = 9;

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

struct Net {
    x0: Vec<i8>,
    ws: Vec<Vec<i8>>,
}

fn make_net(layers: usize, seed: u64) -> Net {
    let mut rng = StdRng::seed_from_u64(seed);
    let x0 = (0..P.n * P.n).map(|_| rng.gen_range(0..=64) as i8).collect();
    let ws = (0..layers).map(|_| (0..P.n * P.n).map(|_| rng.gen_range(-8..=8) as i8).collect()).collect();
    Net { x0, ws }
}

fn new_chain() -> Chain {
    let alloc = [(pubkey(&key(10)), 1000 * COIN), (pubkey(&key(11)), 100 * COIN), (pubkey(&key(13)), 100 * COIN)];
    Chain::genesis(P, &alloc, 1_000, u64::MAX / 64)
}

fn ts(c: &Chain) -> u64 {
    c.tip_ts() + BLOCK_TIME
}

fn mine(chain: &mut Chain, sk: &SigningKey, class: WorkClass, cm: Option<&CommittedMatrices>, txs: Vec<SignedTx>) -> Vec<i32> {
    let t = ts(chain);
    let out = chain.mine(sk, class, cm, txs, t).unwrap();
    let c = out.c.clone();
    chain.validate_and_apply(out.block).unwrap();
    c
}

fn try_mine(chain: &mut Chain, sk: &SigningKey, class: WorkClass, cm: Option<&CommittedMatrices>, txs: Vec<SignedTx>) -> Result<(), String> {
    let t = ts(chain);
    let out = chain.mine(sk, class, cm, txs, t)?;
    chain.validate_and_apply(out.block)
}

fn mtx(chain: &Chain, sk: &SigningKey, m: MlpTx) -> SignedTx {
    make_tx(chain, sk, 0, TxKind::Mlp(m))
}

/// What the worker computes (and possibly falsifies) for each layer.
#[derive(Clone, Copy, PartialEq)]
enum Tamper {
    None,
    Gemm { layer: usize },
    Act { layer: usize },
}

struct Run {
    chain: Chain,
    id: JobId,
    client: SigningKey,
    worker: SigningKey,
    zs: Vec<Vec<i32>>,
    xs: Vec<Vec<i8>>, // X_1..X_L published (possibly falsified)
    net: Net,
}

fn submit_and_claim(layers: usize, seed: u64) -> (Chain, JobId, SigningKey, SigningKey, Net) {
    let (client, worker, carol) = (key(10), key(11), key(12));
    let mut chain = new_chain();
    let net = make_net(layers, seed);
    let x0_root = cols_root_of(&net.x0, P.n);
    let w_roots: Vec<Hash> = net.ws.iter().map(|w| rows_root_of(w, P.n)).collect();
    let nonce = chain.state.nonce(&pubkey(&client));
    let id = mlp_id(&pubkey(&client), nonce, &x0_root, &w_roots);
    let submit = mtx(&chain, &client, MlpTx::Submit { x0_root, w_roots, shifts: vec![SHIFT; layers], escrow: 30 * COIN * layers as u64, deadline: 500 });
    mine(&mut chain, &carol, WorkClass::Native, None, vec![submit]);
    let claim = mtx(&chain, &worker, MlpTx::Claim { job_id: id });
    mine(&mut chain, &carol, WorkClass::Native, None, vec![claim]);
    (chain, id, client, worker, net)
}

/// The worker executes all the layers (one Layer block per layer + PostLayer), with the possible falsification.
fn run_worker(layers: usize, seed: u64, tamper: Tamper) -> Run {
    let (mut chain, id, client, worker, net) = submit_and_claim(layers, seed);
    let mut xs: Vec<Vec<i8>> = vec![];
    let mut zs: Vec<Vec<i32>> = vec![];
    for l in 0..layers {
        let xprev = if l == 0 { net.x0.clone() } else { xs[l - 1].clone() };
        let cm = CommittedMatrices::new(net.ws[l].clone(), xprev.clone(), &P);
        let mut z = mine(&mut chain, &worker, WorkClass::Layer { job: id, layer: l as u8 }, Some(&cm), vec![]);
        assert_eq!(z, plain_gemm(&net.ws[l], &xprev, P.n), "the Layer block produces the true Z_l = W_l * X_(l-1)");
        let mut x = requantize(&z, SHIFT, l + 1 < layers);
        if tamper == (Tamper::Gemm { layer: l }) {
            z[5 * P.n + 7] += 1;
            x = requantize(&z, SHIFT, l + 1 < layers); // consistent with the falsified Z: only the GEMM is wrong
        }
        if tamper == (Tamper::Act { layer: l }) {
            let idx = x.iter().position(|v| *v != 63 && *v != 64).unwrap();
            x[idx] = x[idx].wrapping_add(1); // correct Z, wrong activation
        }
        let post = mtx(&chain, &worker, MlpTx::PostLayer { job_id: id, layer: l as u8, z_root: c_tree(&z, P.n).root(), x_root: cols_root_of(&x, P.n) });
        mine(&mut chain, &worker, WorkClass::Layer { job: id, layer: l as u8 }, Some(&cm), vec![post]);
        zs.push(z);
        xs.push(x);
    }
    Run { chain, id, client, worker, zs, xs, net }
}

#[test]
fn honest_inference_job_runs_layer_by_layer_and_is_paid() {
    let mut r = run_worker(3, 1, Tamper::None);
    assert!(matches!(r.chain.state.mlps[&r.id].status, MlpStatus::Complete { .. }));
    // the client verifies each layer by Freivalds + activation, then accepts
    let mut xprev = r.net.x0.clone();
    for l in 0..3 {
        assert!(freivalds(&r.net.ws[l], &xprev, &r.zs[l], P.n, 24, &[7u8; 32]), "GEMM of layer {l} correct");
        assert_eq!(r.xs[l], requantize(&r.zs[l], SHIFT, l + 1 < 3), "activation of layer {l} correct");
        xprev = r.xs[l].clone();
    }
    let before = r.chain.state.balance(&pubkey(&r.worker));
    let accept = mtx(&r.chain, &r.client, MlpTx::Accept { job_id: r.id });
    mine(&mut r.chain, &key(12), WorkClass::Native, None, vec![accept]);
    assert_eq!(r.chain.state.mlps[&r.id].status, MlpStatus::Settled);
    let net = 90 * COIN - bps(90 * COIN, BURN_BPS);
    assert_eq!(r.chain.state.balance(&pubkey(&r.worker)), before + net + bps(net, BOND_BPS));
    assert!(r.chain.check_supply_invariant());
}

#[test]
fn gemm_fraud_in_a_middle_layer_is_proven_with_one_entry() {
    let mut r = run_worker(3, 2, Tamper::Gemm { layer: 1 });
    let l = 1;
    let xprev = &r.xs[0];
    let (i, j) = find_bad_entry(&r.net.ws[l], xprev, &r.zs[l], P.n, &[3u8; 32]).expect("fraud detected");
    assert_eq!((i, j), (5, 7));
    let (wt, xt, zt) = (rows_tree(&r.net.ws[l], P.n), cols_tree(xprev, P.n), c_tree(&r.zs[l], P.n));
    let kind = MlpFraudKind::Gemm {
        i: i as u32,
        j: j as u32,
        a_row: r.net.ws[l][i * P.n..(i + 1) * P.n].to_vec(),
        a_path: wt.proof(i),
        b_col: (0..P.n).map(|k| xprev[k * P.n + j]).collect(),
        b_path: xt.proof(j),
        z_row: row_of(&r.zs[l], P.n, i),
        z_path: zt.proof(i),
    };
    let alice_before = r.chain.state.balance(&pubkey(&r.client));
    let fraud = mtx(&r.chain, &r.client, MlpTx::Fraud { job_id: r.id, layer: l as u8, kind });
    mine(&mut r.chain, &key(12), WorkClass::Native, None, vec![fraud]);
    assert_eq!(r.chain.state.mlps[&r.id].status, MlpStatus::Slashed);
    let net = 90 * COIN - bps(90 * COIN, BURN_BPS);
    assert_eq!(r.chain.state.balance(&pubkey(&r.client)), alice_before + net + bps(net, BOND_BPS) / 2);
    assert!(r.chain.check_supply_invariant());
}

#[test]
fn activation_fraud_is_proven_with_one_entry() {
    let mut r = run_worker(3, 3, Tamper::Act { layer: 0 });
    let l = 0;
    // Z_0 is correct (Freivalds passes), it is the published activation that is wrong
    assert!(freivalds(&r.net.ws[0], &r.net.x0, &r.zs[0], P.n, 24, &[7u8; 32]));
    let want = requantize(&r.zs[l], SHIFT, true);
    let idx = (0..want.len()).find(|&k| want[k] != r.xs[l][k]).expect("activation falsified");
    let (i, j) = (idx / P.n, idx % P.n);
    let (zt, xt) = (c_tree(&r.zs[l], P.n), cols_tree(&r.xs[l], P.n));
    let kind = MlpFraudKind::Act {
        i: i as u32,
        j: j as u32,
        z_row: row_of(&r.zs[l], P.n, (idx) / P.n),
        z_path: zt.proof((idx) / P.n),
        x_col: (0..P.n).map(|k| r.xs[l][k * P.n + j]).collect(),
        x_path: xt.proof(j),
    };
    let fraud = mtx(&r.chain, &r.client, MlpTx::Fraud { job_id: r.id, layer: l as u8, kind });
    mine(&mut r.chain, &key(12), WorkClass::Native, None, vec![fraud]);
    assert_eq!(r.chain.state.mlps[&r.id].status, MlpStatus::Slashed);
    assert!(r.chain.check_supply_invariant());
}

#[test]
fn false_accusations_are_rejected() {
    let mut r = run_worker(2, 4, Tamper::None);
    // false GEMM accusation: correct entry
    let (wt, xt, zt) = (rows_tree(&r.net.ws[1], P.n), cols_tree(&r.xs[0], P.n), c_tree(&r.zs[1], P.n));
    let (i, j) = (3usize, 9usize);
    let gemm = MlpFraudKind::Gemm {
        i: i as u32,
        j: j as u32,
        a_row: r.net.ws[1][i * P.n..(i + 1) * P.n].to_vec(),
        a_path: wt.proof(i),
        b_col: (0..P.n).map(|k| r.xs[0][k * P.n + j]).collect(),
        b_path: xt.proof(j),
        z_row: row_of(&r.zs[1], P.n, i),
        z_path: zt.proof(i),
    };
    let f = mtx(&r.chain, &r.client, MlpTx::Fraud { job_id: r.id, layer: 1, kind: gemm });
    assert!(try_mine(&mut r.chain, &key(12), WorkClass::Native, None, vec![f]).is_err());
    // false activation accusation: correct activation
    let (zt0, xt0) = (c_tree(&r.zs[0], P.n), cols_tree(&r.xs[0], P.n));
    let idx = 5 * P.n + 5;
    let act_kind = MlpFraudKind::Act {
        i: 5,
        j: 5,
        z_row: row_of(&r.zs[0], P.n, (idx) / P.n),
        z_path: zt0.proof((idx) / P.n),
        x_col: (0..P.n).map(|k| r.xs[0][k * P.n + 5]).collect(),
        x_path: xt0.proof(5),
    };
    let f = mtx(&r.chain, &r.client, MlpTx::Fraud { job_id: r.id, layer: 0, kind: act_kind });
    assert!(try_mine(&mut r.chain, &key(12), WorkClass::Native, None, vec![f]).is_err());
    assert!(matches!(r.chain.state.mlps[&r.id].status, MlpStatus::Complete { .. }));
}

#[test]
fn access_control_and_layer_ordering() {
    let (mut chain, id, _client, worker, net) = submit_and_claim(3, 5);
    let cm0 = CommittedMatrices::new(net.ws[0].clone(), net.x0.clone(), &P);
    // another miner cannot mine this job
    assert!(try_mine(&mut chain, &key(13), WorkClass::Layer { job: id, layer: 0 }, Some(&cm0), vec![]).is_err());
    // layer 1 is not unlocked as long as X_1 is not published
    let cm_fake = CommittedMatrices::new(net.ws[1].clone(), net.x0.clone(), &P);
    assert!(try_mine(&mut chain, &worker, WorkClass::Layer { job: id, layer: 1 }, Some(&cm_fake), vec![]).is_err());
    // layer 0: accepted, with the right matrices only
    let bad = CommittedMatrices::new(net.ws[1].clone(), net.x0.clone(), &P);
    assert!(try_mine(&mut chain, &worker, WorkClass::Layer { job: id, layer: 0 }, Some(&bad), vec![]).is_err());
    mine(&mut chain, &worker, WorkClass::Layer { job: id, layer: 0 }, Some(&cm0), vec![]);
    // out-of-order publication refused
    let z = plain_gemm(&net.ws[0], &net.x0, P.n);
    let x = requantize(&z, SHIFT, true);
    let out_of_order = mtx(&chain, &worker, MlpTx::PostLayer { job_id: id, layer: 1, z_root: c_tree(&z, P.n).root(), x_root: cols_root_of(&x, P.n) });
    assert!(try_mine(&mut chain, &key(12), WorkClass::Native, None, vec![out_of_order]).is_err());
}

#[test]
fn inference_rebate_is_bounded_by_the_burn_like_plain_jobs() {
    let r = run_worker(3, 6, Tamper::None);
    let job = &r.chain.state.mlps[&r.id];
    assert!(job.rebate_left <= bps(90 * COIN, BURN_BPS), "the rebate never exceeds what was burned");
    assert!(r.chain.check_supply_invariant());
}

#[test]
fn spec_validation_and_price_floor() {
    let (client, carol) = (key(10), key(12));
    let mut chain = new_chain();
    let net = make_net(2, 7);
    let x0_root = cols_root_of(&net.x0, P.n);
    let w_roots: Vec<Hash> = net.ws.iter().map(|w| rows_root_of(w, P.n)).collect();
    // escrow below the floor per layer
    let cheap = mtx(&chain, &client, MlpTx::Submit { x0_root, w_roots: w_roots.clone(), shifts: vec![SHIFT; 2], escrow: COIN, deadline: 500 });
    assert!(try_mine(&mut chain, &carol, WorkClass::Native, None, vec![cheap]).is_err());
    // shift out of bounds
    let shifty = mtx(&chain, &client, MlpTx::Submit { x0_root, w_roots: w_roots.clone(), shifts: vec![30, 9], escrow: 10 * COIN, deadline: 500 });
    assert!(try_mine(&mut chain, &carol, WorkClass::Native, None, vec![shifty]).is_err());
    // inconsistent lengths
    let bad = mtx(&chain, &client, MlpTx::Submit { x0_root, w_roots, shifts: vec![9], escrow: 10 * COIN, deadline: 500 });
    assert!(try_mine(&mut chain, &carol, WorkClass::Native, None, vec![bad]).is_err());
}
