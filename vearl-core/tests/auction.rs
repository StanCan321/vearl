//! Reverse auctions (the lowest price wins), grace period, refund of the difference, reputation.
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

fn chain() -> Chain {
    let alloc: Vec<(PubKey, u64)> = [10u8, 11, 13, 14].iter().map(|s| (pubkey(&key(*s)), 1000 * COIN)).collect();
    Chain::genesis(P, &alloc, 1_000, u64::MAX / 64)
}

fn cm(seed: u64) -> CommittedMatrices {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut m = || (0..P.n * P.n).map(|_| rng.gen_range(-64..=64) as i8).collect::<Vec<i8>>();
    let (a, b) = (m(), m());
    CommittedMatrices::new(a, b, &P)
}

fn mine_at(c: &mut Chain, sk: &SigningKey, class: WorkClass, data: Option<&CommittedMatrices>, txs: Vec<SignedTx>, t: u64) -> Result<Vec<i32>, String> {
    let out = c.mine(sk, class, data, txs, t)?;
    let res = out.c.clone();
    c.validate_and_apply(out.block)?;
    Ok(res)
}

fn tx(c: &Chain, sk: &SigningKey, kind: TxKind) -> SignedTx {
    make_tx(c, sk, 0, kind)
}

/// Submits an auction job (duration in timestamp seconds); returns (id, timestamp of the end of the auction).
fn submit_auction(c: &mut Chain, data: &CommittedMatrices, secs: u64, deadline: u64) -> (JobId, u64) {
    let alice = key(10);
    let nonce = c.state.nonce(&pubkey(&alice));
    let id = job_id(&pubkey(&alice), nonce, &data.a_root(), &data.b_root());
    let t = c.tip_ts() + 10;
    let submit = tx(c, &alice, TxKind::SubmitJobAuction { a_root: data.a_root(), b_root: data.b_root(), escrow: 50 * COIN, deadline, auction_secs: secs });
    mine_at(c, &key(12), WorkClass::Native, None, vec![submit], t).unwrap();
    (id, t + secs)
}

fn bid(c: &mut Chain, who: u8, id: JobId, price: u64, t: u64) -> Result<(), String> {
    let b = tx(c, &key(who), TxKind::Bid { job_id: id, price: price * COIN });
    mine_at(c, &key(12), WorkClass::Native, None, vec![b], t).map(|_| ())
}

fn claim(c: &mut Chain, who: u8, id: JobId, t: u64) -> Result<(), String> {
    let cl = tx(c, &key(who), TxKind::ClaimJob { job_id: id });
    mine_at(c, &key(12), WorkClass::Native, None, vec![cl], t).map(|_| ())
}

#[test]
fn lowest_bid_wins_client_gets_the_difference_and_worker_is_paid_the_bid() {
    let mut c = chain();
    let data = cm(1);
    let (id, end) = submit_auction(&mut c, &data, 100, 300);
    let net = 50 * COIN - bps(50 * COIN, BURN_BPS); // 45 VRL
    let t = c.tip_ts() + 5;

    assert!(bid(&mut c, 11, id, 40, t).is_ok());
    assert!(bid(&mut c, 13, id, 45, t + 5).is_err(), "a higher bid is refused");
    assert!(bid(&mut c, 13, id, 40, t + 5).is_err(), "an equal bid is refused (strictly lower)");
    assert!(bid(&mut c, 13, id, 30, t + 5).is_ok());
    assert!(bid(&mut c, 14, id, 0, t + 10).is_err(), "zero price refused");
    assert!(bid(&mut c, 14, id, 46, t + 10).is_err(), "price above the net escrow refused");
    assert_eq!(c.state.jobs[&id].best_bid, Some((pubkey(&key(13)), 30 * COIN)));

    // during the auction, nobody can claim
    assert!(claim(&mut c, 11, id, t + 15).is_err(), "auction still open");
    // after the end: only the winner, during the grace period
    let after = end + 1;
    assert!(claim(&mut c, 11, id, after).is_err(), "bob did not win");
    let alice_before = c.state.balance(&pubkey(&key(10)));
    assert!(claim(&mut c, 13, id, after + 1).is_ok(), "the winner claims");
    assert_eq!(c.state.balance(&pubkey(&key(10))), alice_before + (net - 30 * COIN), "the difference escrow - price goes back to the client");
    assert_eq!(c.state.jobs[&id].escrow_net, 30 * COIN);

    // the winner works (mines WHILE computing the product), publishes, the client accepts
    let mallory = key(13);
    let t = c.tip_ts() + 5;
    let cres = mine_at(&mut c, &mallory, WorkClass::Job(id), Some(&data), vec![], t).unwrap();
    let post = tx(&c, &mallory, TxKind::PostResult { job_id: id, c_root: c_tree(&cres, P.n).root() });
    let t = c.tip_ts() + 5;
    mine_at(&mut c, &mallory, WorkClass::Job(id), Some(&data), vec![post], t).unwrap();
    let before = c.state.balance(&pubkey(&mallory));
    let accept = tx(&c, &key(10), TxKind::Accept { job_id: id });
    let t = c.tip_ts() + 5;
    mine_at(&mut c, &key(12), WorkClass::Native, None, vec![accept], t).unwrap();
    assert_eq!(c.state.balance(&pubkey(&mallory)), before + 30 * COIN + bps(30 * COIN, BOND_BPS), "paid at the price of its bid + bond returned");
    assert_eq!(c.state.reputation(&pubkey(&mallory)).done, 1);
    assert_eq!(c.state.reputation(&pubkey(&key(11))), Reputation::default(), "a loser has no history");
    assert!(c.check_supply_invariant());
}

#[test]
fn if_the_winner_does_not_claim_anyone_can_after_the_grace_period_at_full_price() {
    let mut c = chain();
    let data = cm(2);
    let (id, end) = submit_auction(&mut c, &data, 50, 400);
    let net = 50 * COIN - bps(50 * COIN, BURN_BPS);
    let t = c.tip_ts() + 5;
    bid(&mut c, 13, id, 20, t).unwrap();
    // during the grace period, a third party cannot claim
    assert!(claim(&mut c, 14, id, end + 5).is_err());
    // after end + AUCTION_GRACE_SECS: open to all, at the FULL price (the failing winner's bid is lost)
    assert!(claim(&mut c, 14, id, end + AUCTION_GRACE_SECS + 5).is_ok());
    assert_eq!(c.state.jobs[&id].escrow_net, net);
    assert!(matches!(c.state.jobs[&id].status, JobStatus::Claimed { worker, .. } if worker == pubkey(&key(14))));
    assert!(c.check_supply_invariant());
}

#[test]
fn an_auction_without_bids_can_be_claimed_at_full_price_once_it_ends() {
    let mut c = chain();
    let data = cm(3);
    let (id, end) = submit_auction(&mut c, &data, 40, 400);
    assert!(claim(&mut c, 11, id, end - 5).is_err(), "before the end");
    assert!(claim(&mut c, 11, id, end + 1).is_ok());
    assert_eq!(c.state.jobs[&id].escrow_net, 50 * COIN - bps(50 * COIN, BURN_BPS));
}

#[test]
fn bids_are_rejected_after_the_end_and_on_fixed_price_jobs() {
    let mut c = chain();
    let data = cm(4);
    let (id, end) = submit_auction(&mut c, &data, 30, 400);
    assert!(bid(&mut c, 11, id, 10, end + 1).is_err(), "auction finished");

    // fixed-price job: no auction
    let alice = key(10);
    let nonce = c.state.nonce(&pubkey(&alice));
    let data2 = cm(5);
    let id2 = job_id(&pubkey(&alice), nonce, &data2.a_root(), &data2.b_root());
    let t = c.tip_ts() + 5;
    let submit = tx(&c, &alice, TxKind::SubmitJob { a_root: data2.a_root(), b_root: data2.b_root(), escrow: 50 * COIN, deadline: 400 });
    mine_at(&mut c, &key(12), WorkClass::Native, None, vec![submit], t).unwrap();
    let t = c.tip_ts() + 5;
    assert!(bid(&mut c, 11, id2, 10, t).is_err(), "no auction on a fixed-price job");
    // and the fixed-price job stays claimable immediately
    let t = c.tip_ts() + 5;
    assert!(claim(&mut c, 11, id2, t).is_ok());
}

#[test]
fn zero_length_auctions_and_absurd_durations_are_refused() {
    let mut c = chain();
    let data = cm(6);
    let alice = key(10);
    let bad0 = tx(&c, &alice, TxKind::SubmitJobAuction { a_root: data.a_root(), b_root: data.b_root(), escrow: 50 * COIN, deadline: 400, auction_secs: 0 });
    let t = c.tip_ts() + 5;
    assert!(mine_at(&mut c, &key(12), WorkClass::Native, None, vec![bad0], t).is_err());
    let long = tx(&c, &alice, TxKind::SubmitJobAuction { a_root: data.a_root(), b_root: data.b_root(), escrow: 50 * COIN, deadline: 400, auction_secs: MAX_AUCTION_SECS + 1 });
    let t = c.tip_ts() + 5;
    assert!(mine_at(&mut c, &key(12), WorkClass::Native, None, vec![long], t).is_err());
}

#[test]
fn reputation_counts_slashes_and_expiries() {
    let mut c = chain();
    let data = cm(7);
    let alice = key(10);
    // fixed-price job, short deadline; a worker claims it then disappears -> expiry
    let nonce = c.state.nonce(&pubkey(&alice));
    let id = job_id(&pubkey(&alice), nonce, &data.a_root(), &data.b_root());
    let height = c.height();
    let submit = tx(&c, &alice, TxKind::SubmitJob { a_root: data.a_root(), b_root: data.b_root(), escrow: 50 * COIN, deadline: height + 6 });
    let t = c.tip_ts() + 5;
    mine_at(&mut c, &key(12), WorkClass::Native, None, vec![submit], t).unwrap();
    let t = c.tip_ts() + 5;
    claim(&mut c, 14, id, t).unwrap();
    for _ in 0..8 {
        let t = c.tip_ts() + 5;
        mine_at(&mut c, &key(12), WorkClass::Native, None, vec![], t).unwrap();
    }
    let settle = tx(&c, &key(12), TxKind::Settle { job_id: id });
    let t = c.tip_ts() + 5;
    mine_at(&mut c, &key(12), WorkClass::Native, None, vec![settle], t).unwrap();
    assert_eq!(c.state.reputation(&pubkey(&key(14))), Reputation { done: 0, slashed: 0, expired: 1 });
    assert!(c.check_supply_invariant());
}
