//! Post-quantum accounts (SLH-DSA-SHAKE-128s): deterministic derivation, signature, refusal of alterations, on-chain transfers.
use ed25519_dalek::SigningKey;
use vearl_core::chain::*;
use vearl_core::crypto::*;
use vearl_core::pq::*;
use vearl_core::state::*;
use vearl_core::types::*;
use vearl_core::work::*;

#[test]
fn derivation_is_deterministic_and_seed_dependent() {
    let (a, b, c) = (PqKey::from_seed(&[1; 32]), PqKey::from_seed(&[1; 32]), PqKey::from_seed(&[2; 32]));
    assert_eq!(a.pubkey(), b.pubkey());
    assert_ne!(a.pubkey(), c.pubkey());
}

#[test]
fn signature_verifies_and_every_alteration_is_refused() {
    let k = PqKey::from_seed(&[3; 32]);
    let pk = k.pubkey();
    let msg = b"test transfer";
    let sig = k.sign(msg);
    assert_eq!(sig.len(), PQ_SIG_LEN);
    assert!(verify_sig(&pk, msg, &sig), "dispatch by length: SLH-DSA");
    assert!(!verify_sig(&pk, b"test transfer tesT", &sig), "altered message");
    for pos in [0usize, 17, 4000, PQ_SIG_LEN - 1] {
        let mut bad = sig.clone();
        bad[pos] ^= 1;
        assert!(!verify_sig(&pk, msg, &bad), "signature altered at {pos}");
    }
    assert!(!verify_sig(&pk, msg, &sig[..PQ_SIG_LEN - 1]), "incorrect length");
    let mut long = sig.clone();
    long.push(0);
    assert!(!verify_sig(&pk, msg, &long));
    assert!(!verify_sig(&PqKey::from_seed(&[4; 32]).pubkey(), msg, &sig), "wrong public key");
    assert!(!verify_sig(&pk, msg, &[0u8; 64]), "a null Ed25519 signature does not validate a post-quantum key");
}

#[test]
fn a_signature_made_in_another_context_is_refused() {
    use fips205::slh_dsa_shake_128s as slh;
    use fips205::traits::{KeyGen, SerDes, Signer};
    // same key, different context: domain separation prevents replay
    let (pk, sk) = slh::KG::keygen_with_seeds(&[1u8; slh::N], &[2u8; slh::N], &[3u8; slh::N]);
    let sig = sk.try_sign(b"m", b"other/context", false).unwrap();
    assert!(!pq_verify(&pk.into_bytes(), b"m", &sig));
}

fn ed(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

#[test]
fn a_post_quantum_account_transfers_on_chain_and_replay_is_refused() {
    let pq = PqKey::from_seed(&[9; 32]);
    let (miner, dest) = (ed(2), ed(3));
    let alloc = vec![(pq.pubkey(), 1_000 * COIN), (pubkey(&miner), 10 * COIN)];
    let mut chain = Chain::genesis(WorkParams::TESTNET, &alloc, 1_000, u64::MAX / 64);
    let tx = SignedTx::new_pq(Tx { from: pq.pubkey(), nonce: 0, fee: COIN / 100, kind: TxKind::Transfer { to: pubkey(&dest), amount: 5 * COIN } }, &pq);
    let t = chain.tip_ts() + BLOCK_TIME;
    let out = chain.mine(&miner, WorkClass::Native, None, vec![tx.clone()], t).unwrap();
    chain.validate_and_apply(out.block).expect("block with a post-quantum transaction");
    assert_eq!(chain.state.balance(&pubkey(&dest)), 5 * COIN);
    assert_eq!(chain.state.balance(&pq.pubkey()), 995 * COIN - COIN / 100);
    assert_eq!(chain.state.nonce(&pq.pubkey()), 1);
    assert!(chain.check_supply_invariant());
    // replay of the same transaction: nonce already consumed
    let t = chain.tip_ts() + BLOCK_TIME;
    let out = chain.mine(&miner, WorkClass::Native, None, vec![tx], t).unwrap();
    let err = chain.validate_and_apply(out.block).unwrap_err();
    assert!(err.contains("bad nonce"), "{err}");
    // altered signature: refused by the state validation
    let mut bad = SignedTx::new_pq(Tx { from: pq.pubkey(), nonce: 1, fee: 0, kind: TxKind::Transfer { to: pubkey(&dest), amount: COIN } }, &pq);
    bad.sig[100] ^= 1;
    let t = chain.tip_ts() + BLOCK_TIME;
    let out = chain.mine(&miner, WorkClass::Native, None, vec![bad], t).unwrap();
    let err = chain.validate_and_apply(out.block).unwrap_err();
    assert!(err.contains("bad signature"), "{err}");
}
