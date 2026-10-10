//! Data availability: the client can demand entries of the result ON-CHAIN; the worker must answer
//! (Merkle path against c_root) on pain of slash. Prevents being paid for a root without data.
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

fn mine_at(c: &mut Chain, sk: &SigningKey, txs: Vec<SignedTx>, t: u64) -> Result<(), String> {
    let out = c.mine(sk, WorkClass::Native, None, txs, t)?;
    c.validate_and_apply(out.block)
}

fn tx(c: &Chain, who: u8, kind: TxKind) -> SignedTx {
    make_tx(c, &key(who), 0, kind)
}

fn cm(seed: u64) -> CommittedMatrices {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut m = || (0..P.n * P.n).map(|_| rng.gen_range(-64..=64) as i8).collect::<Vec<i8>>();
    let (a, b) = (m(), m());
    CommittedMatrices::new(a, b, &P)
}

struct Setup {
    c: Chain,
    id: JobId,
    data: CommittedMatrices,
    posted: Vec<i32>, // the data the worker committed to (true or invented)
    tree: MerkleTree,
}

/// alice (10) client, bob (11) worker, carol (12) miner, dave (14) third party. The worker posts `posted` as the result.
fn posted_job(seed: u64, garbage: bool) -> Setup {
    let alloc: Vec<(PubKey, u64)> = [10u8, 11, 13, 14].iter().map(|s| (pubkey(&key(*s)), 1000 * COIN)).collect();
    let mut c = Chain::genesis(P, &alloc, 1_000, u64::MAX / 64);
    let data = cm(seed);
    let nonce = c.state.nonce(&pubkey(&key(10)));
    let id = job_id(&pubkey(&key(10)), nonce, &data.a_root(), &data.b_root());
    let submit = tx(&c, 10, TxKind::SubmitJob { a_root: data.a_root(), b_root: data.b_root(), escrow: 50 * COIN, deadline: 1000 });
    mine_at(&mut c, &key(12), vec![submit], 1_010).unwrap();
    let claim = tx(&c, 11, TxKind::ClaimJob { job_id: id });
    mine_at(&mut c, &key(12), vec![claim], 1_020).unwrap();
    let mut posted = plain_gemm(&data.a, &data.b, P.n);
    if garbage {
        for (k, v) in posted.iter_mut().enumerate() {
            *v = (k as i32).wrapping_mul(97) ^ 4242;
        }
    }
    let tree = c_tree(&posted, P.n);
    let post = tx(&c, 11, TxKind::PostResult { job_id: id, c_root: tree.root() });
    mine_at(&mut c, &key(12), vec![post], 1_030).unwrap();
    Setup { c, id, data, posted, tree }
}

fn fetch(s: &mut Setup, i: u32, j: u32, t: u64) -> Result<(), String> {
    let f = tx(&s.c, 10, TxKind::Fetch { job_id: s.id, i, j });
    mine_at(&mut s.c, &key(12), vec![f], t)
}

fn serve(s: &mut Setup, who: u8, i: u32, j: u32, row: Vec<i32>, path: Vec<Hash>, t: u64) -> Result<(), String> {
    let sv = tx(&s.c, who, TxKind::Serve { job_id: s.id, i, j, row, path });
    mine_at(&mut s.c, &key(12), vec![sv], t)
}

fn honest_serve(s: &mut Setup, i: u32, j: u32, t: u64) -> Result<(), String> {
    let (row, p) = (row_of(&s.posted, P.n, i as usize), s.tree.proof(i as usize));
    serve(s, 11, i, j, row, p, t)
}

fn fill(c: &mut Chain, count: u64, start_ts: u64, step: u64) -> u64 {
    let mut t = start_ts;
    for _ in 0..count {
        mine_at(c, &key(12), vec![], t).unwrap();
        t += step;
    }
    t
}

#[test]
fn served_request_pays_the_worker_extends_the_window_and_blocks_settlement_until_answered() {
    let mut s = posted_job(1, false);
    let (bob0, alice0) = (s.c.state.balance(&pubkey(&key(11))), s.c.state.balance(&pubkey(&key(10))));
    fetch(&mut s, 5, 9, 1_040).unwrap();
    assert_eq!(s.c.state.balance(&pubkey(&key(10))), alice0 - FETCH_FEE, "the client pays the request");
    assert_eq!(s.c.state.jobs[&s.id].fetches.len(), 1);
    assert!(s.c.check_supply_invariant());

    // the ORIGINAL dispute window closes (blocks AND seconds) while the request stays unanswered
    let t = fill(&mut s.c, 12, 1_045, 30);
    let settle = tx(&s.c, 12, TxKind::Settle { job_id: s.id });
    assert!(mine_at(&mut s.c, &key(12), vec![settle], t).is_err(), "request pending: no settlement");

    // the (late) answer reopens the window so that the client can verify
    honest_serve(&mut s, 5, 9, t + 10).unwrap();
    assert_eq!(s.c.state.balance(&pubkey(&key(11))), bob0 + FETCH_FEE, "the worker receives the served request");
    assert!(s.c.state.jobs[&s.id].fetches.is_empty());
    let (i, j, row, _) = s.c.state.jobs[&s.id].served[0].clone();
    let v = row[j as usize];
    assert_eq!((i, j, v), (5, 9, s.posted[5 * P.n + 9]));
    let expected: i32 = (0..P.n).map(|k| s.data.a[5 * P.n + k] as i32 * s.data.b[k * P.n + 9] as i32).sum();
    assert_eq!(expected, v, "the client verifies the served value against A and B in O(n)");
    let settle = tx(&s.c, 12, TxKind::Settle { job_id: s.id });
    assert!(mine_at(&mut s.c, &key(12), vec![settle], t + 20).is_err(), "window reopened after the answer");

    // then, once the extended window has elapsed, settlement becomes possible again
    let t2 = fill(&mut s.c, 4, t + 30, 30);
    let settle = tx(&s.c, 12, TxKind::Settle { job_id: s.id });
    mine_at(&mut s.c, &key(12), vec![settle], t2 + 30).unwrap();
    assert_eq!(s.c.state.jobs[&s.id].status, JobStatus::Settled);
    assert!(s.c.check_supply_invariant());
}

#[test]
fn a_worker_that_committed_a_root_without_the_data_is_slashed_for_not_serving() {
    // the worker posts the root of a tree it cannot serve: it answers wrong/nothing
    let mut s = posted_job(2, false);
    let alice0 = s.c.state.balance(&pubkey(&key(10)));
    fetch(&mut s, 3, 3, 1_040).unwrap();
    let slash = tx(&s.c, 14, TxKind::SlashUnserved { job_id: s.id });
    assert!(mine_at(&mut s.c, &key(12), vec![slash.clone()], 1_050).is_err(), "before the deadline: no penalty");
    // an invalid answer does not release the worker
    let wrong_path = s.tree.proof(3);
    let mut wrong_row = row_of(&s.posted, P.n, 3);
    wrong_row[3] = 123456;
    assert!(serve(&mut s, 11, 3, 3, wrong_row, wrong_path, 1_060).is_err(), "row inconsistent with the path");
    let dave0 = s.c.state.balance(&pubkey(&key(14)));
    let slash = tx(&s.c, 14, TxKind::SlashUnserved { job_id: s.id });
    mine_at(&mut s.c, &key(12), vec![slash], 1_040 + SERVE_SECS + 5).unwrap();
    assert_eq!(s.c.state.jobs[&s.id].status, JobStatus::Slashed);
    let net = 50 * COIN - bps(50 * COIN, BURN_BPS);
    let bond = bps(net, BOND_BPS);
    assert_eq!(s.c.state.balance(&pubkey(&key(14))), dave0 + bond / 2, "the reporter receives half of the bond");
    assert_eq!(s.c.state.balance(&pubkey(&key(10))), alice0 + net, "client refunded, request fee included");
    assert_eq!(s.c.state.reputation(&pubkey(&key(11))).slashed, 1);
    assert!(s.c.check_supply_invariant());
}

#[test]
fn invalid_fetch_and_serve_attempts_are_rejected() {
    let mut s = posted_job(3, false);
    let bad_client = tx(&s.c, 14, TxKind::Fetch { job_id: s.id, i: 0, j: 0 });
    assert!(mine_at(&mut s.c, &key(12), vec![bad_client], 1_040).is_err(), "only the client can request");
    assert!(fetch(&mut s, P.n as u32, 0, 1_040).is_err(), "index out of range");
    fetch(&mut s, 1, 1, 1_040).unwrap();
    assert!(fetch(&mut s, 1, 1, 1_050).is_err(), "duplicate");
    // only the worker serves
    let (row, p) = (row_of(&s.posted, P.n, 1), s.tree.proof(1));
    assert!(serve(&mut s, 14, 1, 1, row, p, 1_060).is_err(), "a third party cannot serve");
    // no answer to a nonexistent request
    assert!(honest_serve(&mut s, 2, 2, 1_060).is_err());
    // cap on simultaneous requests
    for k in 0..(MAX_PENDING_FETCHES as u32 - 1) {
        fetch(&mut s, 10 + k, 10, 1_070 + k as u64).unwrap();
    }
    assert!(fetch(&mut s, 99, 99, 1_200).is_err(), "too many pending requests");
    assert!(s.c.check_supply_invariant());
}

#[test]
fn an_invented_but_consistently_served_result_is_caught_by_sampling_and_slashed() {
    // The worker has "the data" (an invented result) and serves it consistently with its root. The client
    // samples ONE entry, recomputes it against A and B, notes the gap and proves the fraud with the served path.
    let mut s = posted_job(5, true);
    fetch(&mut s, 7, 11, 1_040).unwrap();
    honest_serve(&mut s, 7, 11, 1_045).unwrap(); // served consistently with the (invented) root
    let (i, j, row, path) = s.c.state.jobs[&s.id].served[0].clone();
    let (iu, ju) = (i as usize, j as usize);
    let v = row[ju];
    let expected: i32 = (0..P.n).map(|t| s.data.a[iu * P.n + t] as i32 * s.data.b[t * P.n + ju] as i32).sum();
    assert_ne!(expected, v, "the served entry is wrong");
    let proof = FraudProof {
        job_id: s.id,
        i,
        j,
        a_row: s.data.a[iu * P.n..(iu + 1) * P.n].to_vec(),
        a_path: s.data.a_tree.proof(iu),
        b_col: s.data.col(ju, P.n),
        b_path: s.data.b_tree.proof(ju),
        c_row: row,
        c_path: path,
    };
    let fraud = tx(&s.c, 10, TxKind::Fraud(proof));
    mine_at(&mut s.c, &key(12), vec![fraud], 1_050).unwrap();
    assert_eq!(s.c.state.jobs[&s.id].status, JobStatus::Slashed);
    assert_eq!(s.c.state.jobs[&s.id].fetch_held, 0);
    assert!(s.c.check_supply_invariant());
}

#[test]
fn accepting_with_a_pending_request_refunds_the_request_fee() {
    let mut s = posted_job(6, false);
    let alice0 = s.c.state.balance(&pubkey(&key(10)));
    fetch(&mut s, 2, 2, 1_040).unwrap();
    let accept = tx(&s.c, 10, TxKind::Accept { job_id: s.id });
    mine_at(&mut s.c, &key(12), vec![accept], 1_050).unwrap();
    assert_eq!(s.c.state.balance(&pubkey(&key(10))), alice0, "request fees refunded at acceptance");
    assert_eq!(s.c.state.jobs[&s.id].fetch_held, 0);
    assert!(s.c.check_supply_invariant());
}
