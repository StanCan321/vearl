//! Block-wise matrix products (rectangular shapes): splitting, life cycle, fraud proofs, access controls.
use ed25519_dalek::SigningKey;
use vearl_core::chain::*;
use vearl_core::crypto::*;
use vearl_core::state::*;
use vearl_core::tiled::*;
use vearl_core::types::*;
use vearl_core::work::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

const P: WorkParams = WorkParams::TESTNET;
const N: usize = 128;

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

fn new_chain() -> Chain {
    let alloc: Vec<(PubKey, u64)> = [10u8, 11, 13].iter().map(|s| (pubkey(&key(*s)), 1000 * COIN)).collect();
    Chain::genesis(P, &alloc, 1_000, u64::MAX / 64)
}

fn rand_matrix(rows: usize, cols: usize, seed: u64) -> Vec<i8> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..rows * cols).map(|_| rng.gen_range(-64..=64) as i8).collect()
}

fn mine_at(c: &mut Chain, sk: &SigningKey, class: WorkClass, data: Option<&CommittedMatrices>, txs: Vec<SignedTx>) -> Result<Vec<i32>, String> {
    let t = c.tip_ts() + BLOCK_TIME;
    let out = c.mine(sk, class, data, txs, t)?;
    let r = out.c.clone();
    c.validate_and_apply(out.block)?;
    Ok(r)
}

fn ttx(c: &Chain, who: u8, m: TiledTx) -> SignedTx {
    make_tx(c, &key(who), 0, TxKind::Tiled(m))
}

struct Job {
    c: Chain,
    id: JobId,
    dims: (usize, usize, usize),
    a: Vec<i8>,
    b: Vec<i8>,
    products: Vec<Vec<i32>>,
    sums: Vec<Vec<i32>>,
}

/// Submits and has bob claim an (I,K,J) job; does not publish the result yet.
fn setup(dims: (usize, usize, usize), seed: u64) -> Job {
    let (bi, bk, bj) = dims;
    let (a, b) = (rand_matrix(bi * N, bk * N, seed), rand_matrix(bk * N, bj * N, seed + 1));
    let (ar, br) = (a_block_roots(&a, bk * N, bi, bk, N), b_block_roots(&b, bj * N, bk, bj, N));
    let mut c = new_chain();
    let nonce = c.state.nonce(&pubkey(&key(10)));
    let d8 = (bi as u8, bk as u8, bj as u8);
    let id = tiled_id(&pubkey(&key(10)), nonce, d8, &ar, &br);
    let submit = ttx(&c, 10, TiledTx::Submit { dims: d8, a_roots: ar, b_roots: br, escrow: 10 * COIN * (bi * bk * bj) as u64, deadline: 900 });
    mine_at(&mut c, &key(12), WorkClass::Native, None, vec![submit]).unwrap();
    let claim = ttx(&c, 11, TiledTx::Claim { job_id: id });
    mine_at(&mut c, &key(12), WorkClass::Native, None, vec![claim]).unwrap();
    let (products, sums) = compute_all(&a, &b, dims, N);
    Job { c, id, dims, a, b, products, sums }
}

fn roots_of(j: &Job) -> (Vec<Hash>, Vec<Hash>) {
    (j.products.iter().map(|p| c_tree(p, N).root()).collect(), j.sums.iter().map(|s| c_tree(s, N).root()).collect())
}

fn post(j: &mut Job) {
    let (pr, cr) = roots_of(j);
    let tx = ttx(&j.c, 11, TiledTx::Post { job_id: j.id, product_roots: pr, c_roots: cr });
    mine_at(&mut j.c, &key(12), WorkClass::Native, None, vec![tx]).unwrap();
}

#[test]
fn tiling_matches_a_naive_rectangular_product() {
    // rectangular shape 1 x 3 x 2 blocks: A (128 x 384), B (384 x 256)
    let dims = (1usize, 3usize, 2usize);
    let (m, k, n_total) = (N, 3 * N, 2 * N);
    let (a, b) = (rand_matrix(m, k, 5), rand_matrix(k, n_total, 6));
    let (_, sums) = compute_all(&a, &b, dims, N);
    let c = assemble(&sums, dims.0, dims.2, N);
    let mut naive = vec![0i32; m * n_total];
    for i in 0..m {
        for kk in 0..k {
            for j in 0..n_total {
                naive[i * n_total + j] += a[i * k + kk] as i32 * b[kk * n_total + j] as i32;
            }
        }
    }
    assert_eq!(c, naive, "the split into blocks gives back exactly the unsplit product A*B");
}

#[test]
fn honest_tiled_job_is_paid_and_blocks_can_be_mined_on_any_product() {
    let mut j = setup((2, 2, 2), 1);
    // a worker miner can mine ANY product of the job (never a product outside the job, nor another miner)
    let (i, jj, k) = (1usize, 0usize, 1usize);
    let cm = CommittedMatrices::new(get_block(&j.a, 2 * N, i, k, N), get_block(&j.b, 2 * N, k, jj, N), &P);
    let z = mine_at(&mut j.c, &key(11), WorkClass::Product { job: j.id, i: i as u8, j: jj as u8, k: k as u8 }, Some(&cm), vec![]).unwrap();
    let pidx = (i * 2 + jj) * 2 + k;
    assert_eq!(z, j.products[pidx], "the mined block does compute the client's P_ijk = A_ik * B_kj");

    post(&mut j);
    // the client verifies: Freivalds on each product + sums
    for (idx, p) in j.products.iter().enumerate() {
        let (bi, rest) = (idx / (2 * 2), idx % 4);
        let (bj, bk) = (rest / 2, rest % 2);
        let (ab, bb) = (get_block(&j.a, 2 * N, bi, bk, N), get_block(&j.b, 2 * N, bk, bj, N));
        assert!(freivalds(&ab, &bb, p, N, 24, &[7u8; 32]), "product {idx}");
    }
    let before = j.c.state.balance(&pubkey(&key(11)));
    let acc = ttx(&j.c, 10, TiledTx::Accept { job_id: j.id });
    mine_at(&mut j.c, &key(12), WorkClass::Native, None, vec![acc]).unwrap();
    let net = 80 * COIN - bps(80 * COIN, BURN_BPS);
    assert_eq!(j.c.state.balance(&pubkey(&key(11))), before + net + bps(net, BOND_BPS));
    assert_eq!(j.c.state.reputation(&pubkey(&key(11))).done, 1);
    assert!(j.c.check_supply_invariant());
    assert!(j.c.state.tiled[&j.id].rebate_left <= bps(80 * COIN, BURN_BPS), "the rebate does not exceed the burn");
}

#[test]
fn a_wrong_product_entry_is_proven_with_one_entry() {
    let mut j = setup((2, 2, 2), 2);
    let (i, jj, k) = (1usize, 1usize, 0usize);
    let pidx = (i * 2 + jj) * 2 + k;
    j.products[pidx][9 * N + 4] += 3; // the product is wrong ...
    // ... and the sums are recomputed from the published products (only the block's GEMM is wrong)
    let mut sum = vec![0i32; N * N];
    for kk in 0..2 {
        for (x, v) in sum.iter_mut().zip(&j.products[(i * 2 + jj) * 2 + kk]) {
            *x += *v;
        }
    }
    j.sums[i * 2 + jj] = sum;
    post(&mut j);

    let (ab, bb) = (get_block(&j.a, 2 * N, i, k, N), get_block(&j.b, 2 * N, k, jj, N));
    assert!(!freivalds(&ab, &bb, &j.products[pidx], N, 24, &[7u8; 32]));
    let (r, c) = find_bad_entry(&ab, &bb, &j.products[pidx], N, &[7u8; 32]).unwrap();
    assert_eq!((r, c), (9, 4));
    let kind = TiledFraudKind::Product {
        i: i as u8,
        j: jj as u8,
        k: k as u8,
        r: r as u32,
        c: c as u32,
        a_row: ab[r * N..(r + 1) * N].to_vec(),
        a_path: rows_tree(&ab, N).proof(r),
        b_col: (0..N).map(|t| bb[t * N + c]).collect(),
        b_path: cols_tree(&bb, N).proof(c),
        z_row: row_of(&j.products[pidx], N, r),
        z_path: c_tree(&j.products[pidx], N).proof(r),
    };
    let alice0 = j.c.state.balance(&pubkey(&key(10)));
    let f = ttx(&j.c, 10, TiledTx::Fraud { job_id: j.id, kind });
    mine_at(&mut j.c, &key(12), WorkClass::Native, None, vec![f]).unwrap();
    assert_eq!(j.c.state.tiled[&j.id].status, TiledStatus::Slashed);
    let net = 80 * COIN - bps(80 * COIN, BURN_BPS);
    assert_eq!(j.c.state.balance(&pubkey(&key(10))), alice0 + net + bps(net, BOND_BPS) / 2);
    assert!(j.c.check_supply_invariant());
}

#[test]
fn a_wrong_sum_entry_is_proven_with_the_k_products_and_one_sum_opening() {
    let mut j = setup((2, 3, 2), 3);
    let (i, jj) = (0usize, 1usize);
    j.sums[i * 2 + jj][20 * N + 21] -= 1; // correct products, wrong sum
    post(&mut j);
    let (r, c) = (20usize, 21usize);
    let bk = 3;
    let kind = TiledFraudKind::Sum {
        i: i as u8,
        j: jj as u8,
        r: r as u32,
        c: c as u32,
        z_rows: (0..bk).map(|k| row_of(&j.products[(i * 2 + jj) * bk + k], N, r)).collect(),
        z_paths: (0..bk).map(|k| c_tree(&j.products[(i * 2 + jj) * bk + k], N).proof(r)).collect(),
        c_row: row_of(&j.sums[i * 2 + jj], N, r),
        c_path: c_tree(&j.sums[i * 2 + jj], N).proof(r),
    };
    let f = ttx(&j.c, 10, TiledTx::Fraud { job_id: j.id, kind });
    mine_at(&mut j.c, &key(12), WorkClass::Native, None, vec![f]).unwrap();
    assert_eq!(j.c.state.tiled[&j.id].status, TiledStatus::Slashed);
    assert!(j.c.check_supply_invariant());
}

#[test]
fn false_accusations_are_rejected() {
    let mut j = setup((2, 2, 2), 4);
    post(&mut j);
    // correct product
    let (i, jj, k, r, c) = (0usize, 1usize, 1usize, 3usize, 5usize);
    let (ab, bb) = (get_block(&j.a, 2 * N, i, k, N), get_block(&j.b, 2 * N, k, jj, N));
    let pidx = (i * 2 + jj) * 2 + k;
    let kind = TiledFraudKind::Product {
        i: i as u8,
        j: jj as u8,
        k: k as u8,
        r: r as u32,
        c: c as u32,
        a_row: ab[r * N..(r + 1) * N].to_vec(),
        a_path: rows_tree(&ab, N).proof(r),
        b_col: (0..N).map(|t| bb[t * N + c]).collect(),
        b_path: cols_tree(&bb, N).proof(c),
        z_row: row_of(&j.products[pidx], N, r),
        z_path: c_tree(&j.products[pidx], N).proof(r),
    };
    let f = ttx(&j.c, 10, TiledTx::Fraud { job_id: j.id, kind });
    assert!(mine_at(&mut j.c, &key(12), WorkClass::Native, None, vec![f]).is_err());
    // correct sum
    let idx = 7 * N + 7;
    let kind = TiledFraudKind::Sum {
        i: 1,
        j: 0,
        r: 7,
        c: 7,
        z_rows: (0..2).map(|k| row_of(&j.products[(1 * 2 + 0) * 2 + k], N, idx / N)).collect(),
        z_paths: (0..2).map(|k| c_tree(&j.products[(1 * 2 + 0) * 2 + k], N).proof(idx / N)).collect(),
        c_row: row_of(&j.sums[2], N, idx / N),
        c_path: c_tree(&j.sums[2], N).proof(idx / N),
    };
    let f = ttx(&j.c, 10, TiledTx::Fraud { job_id: j.id, kind });
    assert!(mine_at(&mut j.c, &key(12), WorkClass::Native, None, vec![f]).is_err());
    assert!(matches!(j.c.state.tiled[&j.id].status, TiledStatus::Posted { .. }), "no penalty on a false accusation");
}

#[test]
fn access_control_spec_validation_and_price_floor() {
    let mut j = setup((2, 2, 2), 5);
    let cm = |i: usize, jj: usize, k: usize| CommittedMatrices::new(get_block(&j.a, 2 * N, i, k, N), get_block(&j.b, 2 * N, k, jj, N), &P);
    let ok = cm(0, 0, 0);
    // another miner cannot mine this job
    assert!(mine_at(&mut j.c, &key(13), WorkClass::Product { job: j.id, i: 0, j: 0, k: 0 }, Some(&ok), vec![]).is_err());
    // index outside the job
    assert!(mine_at(&mut j.c, &key(11), WorkClass::Product { job: j.id, i: 2, j: 0, k: 0 }, Some(&ok), vec![]).is_err());
    // data of ANOTHER block than the announced one
    let other = cm(1, 1, 1);
    assert!(mine_at(&mut j.c, &key(11), WorkClass::Product { job: j.id, i: 0, j: 0, k: 0 }, Some(&other), vec![]).is_err());
    // the right block passes
    assert!(mine_at(&mut j.c, &key(11), WorkClass::Product { job: j.id, i: 0, j: 0, k: 0 }, Some(&ok), vec![]).is_ok());
    // publication: wrong number of roots, or by a third party
    let bad = ttx(&j.c, 11, TiledTx::Post { job_id: j.id, product_roots: vec![[0; 32]; 3], c_roots: vec![[0; 32]; 4] });
    assert!(mine_at(&mut j.c, &key(12), WorkClass::Native, None, vec![bad]).is_err());
    let (pr, cr) = roots_of(&j);
    let thief = ttx(&j.c, 13, TiledTx::Post { job_id: j.id, product_roots: pr, c_roots: cr });
    assert!(mine_at(&mut j.c, &key(12), WorkClass::Native, None, vec![thief]).is_err());

    // specification: dimensions, lengths, price floor PER PRODUCT
    let mut c = new_chain();
    let (a, b) = (rand_matrix(N, N, 1), rand_matrix(N, N, 2));
    let (ar, br) = (a_block_roots(&a, N, 1, 1, N), b_block_roots(&b, N, 1, 1, N));
    let cheap = ttx(&c, 10, TiledTx::Submit { dims: (1, 1, 1), a_roots: ar.clone(), b_roots: br.clone(), escrow: COIN / 2, deadline: 500 });
    assert!(mine_at(&mut c, &key(12), WorkClass::Native, None, vec![cheap]).is_err(), "below the floor");
    let wrong_len = ttx(&c, 10, TiledTx::Submit { dims: (2, 1, 1), a_roots: ar.clone(), b_roots: br.clone(), escrow: 10 * COIN, deadline: 500 });
    assert!(mine_at(&mut c, &key(12), WorkClass::Native, None, vec![wrong_len]).is_err(), "inconsistent number of roots");
    let too_big = ttx(&c, 10, TiledTx::Submit { dims: (9, 1, 1), a_roots: vec![[0; 32]; 9], b_roots: vec![[0; 32]; 1], escrow: 100 * COIN, deadline: 500 });
    assert!(mine_at(&mut c, &key(12), WorkClass::Native, None, vec![too_big]).is_err(), "beyond MAX_BLOCKS");
    let zero = ttx(&c, 10, TiledTx::Submit { dims: (0, 1, 1), a_roots: vec![], b_roots: br, escrow: 10 * COIN, deadline: 500 });
    assert!(mine_at(&mut c, &key(12), WorkClass::Native, None, vec![zero]).is_err());
    assert!(c.check_supply_invariant());
}
