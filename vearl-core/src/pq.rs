//! POST-QUANTUM signatures: SLH-DSA-SHAKE-128s (FIPS 205, formerly SPHINCS+), `fips205` crate 0.4.1 (pure Rust, no `unsafe`, UNAUDITED).
//!
//! No home-made primitive: the standard building block is composed. What is chosen here:
//! - the public key (32 bytes) serves as the ADDRESS, like the Ed25519 key: an account is therefore post-quantum or classical depending on the key;
//! - the signature is 7,856 bytes (`PQ_SIG_LEN`); it is its LENGTH that distinguishes the scheme (64 = Ed25519): `verify_sig` dispatches;
//! - domain separation: the signed message is `bincode(tx)` with the FIPS 205 CONTEXT `vearl/tx/pq/v1`, so a transaction signature
//!   cannot be replayed in another context;
//! - "hedged" signature (system randomness) at signing; verification does not depend on the randomness;
//! - key derived from a 32-byte seed (blake3 in XOF mode -> the three FIPS 205 seeds SK.seed, SK.prf, PK.seed): deterministic
//!   restoration from the wallet phrase.
//!
//! What this protects: a quantum computer able to run Shor could forge Ed25519 signatures from a public key;
//! not SLH-DSA signatures (security based on hash functions). What it does NOT protect: the P2P channel (Noise/X25519) is
//! not post-quantum; the signature costs 7.8 KiB per transaction and about 1.6 s of computation (parameter `s`); the crate has no external audit.
use crate::crypto::PubKey;
use fips205::slh_dsa_shake_128s as slh;
use fips205::traits::{KeyGen, SerDes, Signer, Verifier};

pub const PQ_SIG_LEN: usize = slh::SIG_LEN;
pub const PQ_CTX: &[u8] = b"vearl/tx/pq/v1";

pub struct PqKey {
    sk: slh::PrivateKey,
    pk: PubKey,
}

impl PqKey {
    /// Deterministic key from a 32-byte seed.
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        let mut xof = blake3::Hasher::new_derive_key("vearl/pq/keygen/v1");
        xof.update(seed);
        let mut out = [0u8; 3 * slh::N];
        xof.finalize_xof().fill(&mut out);
        let mut s = [[0u8; slh::N]; 3];
        for (i, chunk) in out.chunks(slh::N).enumerate() {
            s[i].copy_from_slice(chunk);
        }
        let (pk, sk) = slh::KG::keygen_with_seeds(&s[0], &s[1], &s[2]);
        PqKey { sk, pk: pk.into_bytes() }
    }

    pub fn pubkey(&self) -> PubKey {
        self.pk
    }

    /// "Hedged" signature of the message with the VEARL context.
    pub fn sign(&self, msg: &[u8]) -> Vec<u8> {
        self.sk.try_sign(msg, PQ_CTX, true).expect("system randomness unavailable").to_vec()
    }
}

pub fn pq_verify(pk: &PubKey, msg: &[u8], sig: &[u8]) -> bool {
    let Ok(pk) = slh::PublicKey::try_from_bytes(pk) else { return false };
    let Ok(sig) = <&[u8; slh::SIG_LEN]>::try_from(sig) else { return false };
    pk.verify(msg, sig, PQ_CTX)
}
