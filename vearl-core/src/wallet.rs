//! Wallet: recovery phrase, encrypted keystore, readable addresses.
//!
//! No home-made primitive: BIP-39 (`bip39`), Argon2id (`argon2`), ChaCha20-Poly1305 (`chacha20poly1305`),
//! Bech32m (`bech32`), Ed25519 (`ed25519-dalek`), blake3 for account derivation.
//!
//! What is protected: the private key only exists inside the encrypted keystore (password -> Argon2id ->
//! AEAD key); a stolen file is as hard to crack as the password is strong. What is NOT: a compromised
//! machine (keylogger, memory), a weak password, and the account derivation is NOT SLIP-10:
//! a VEARL phrase is therefore not interoperable with other wallets (a deliberate choice for a testnet).
use crate::crypto::{pubkey, PubKey};
use crate::state::COIN;
use argon2::{Algorithm, Argon2, Params, Version};
use bech32::{Bech32m, Hrp};
use bip39::{Language, Mnemonic};
use chacha20poly1305::aead::{Aead, KeyInit};
use chacha20poly1305::{ChaCha20Poly1305, Nonce};
use ed25519_dalek::SigningKey;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const ADDRESS_HRP: &str = "vrl";
/// Prefix of POST-QUANTUM addresses (SLH-DSA, pq.rs): a different prefix avoids confusing the two signature schemes.
pub const PQ_ADDRESS_HRP: &str = "vrlq";

/// Cost of key derivation: production uses `KdfParams::STRONG`; tests `LIGHT`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct KdfParams {
    pub m_kib: u32,
    pub t: u32,
    pub p: u32,
}

impl KdfParams {
    pub const STRONG: KdfParams = KdfParams { m_kib: 64 * 1024, t: 3, p: 1 };
    pub const LIGHT: KdfParams = KdfParams { m_kib: 1024, t: 1, p: 1 };
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Keystore {
    pub version: u32,
    pub kdf: String,
    pub kdf_params: KdfParams,
    pub salt: String,
    pub cipher: String,
    pub nonce: String,
    pub ciphertext: String,
    /// Address of account 0 (informational: allows recognising the file without the password).
    pub address: String,
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unhex(s: &str) -> Result<Vec<u8>, String> {
    if s.len() % 2 != 0 {
        return Err("invalid hex".into());
    }
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|_| "invalid hex".to_string())).collect()
}

fn derive_key(password: &str, salt: &[u8], kp: KdfParams) -> Result<[u8; 32], String> {
    let params = Params::new(kp.m_kib, kp.t, kp.p, Some(32)).map_err(|e| e.to_string())?;
    let mut out = [0u8; 32];
    Argon2::new(Algorithm::Argon2id, Version::V0x13, params).hash_password_into(password.as_bytes(), salt, &mut out).map_err(|e| e.to_string())?;
    Ok(out)
}

/// New 24-word phrase (256 bits of system entropy).
pub fn new_mnemonic() -> Mnemonic {
    let mut entropy = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut entropy);
    Mnemonic::from_entropy_in(Language::English, &entropy).expect("32 bytes = 24 words")
}

pub fn parse_mnemonic(phrase: &str) -> Result<Mnemonic, String> {
    Mnemonic::parse_in_normalized(Language::English, phrase.trim()).map_err(|e| format!("invalid phrase: {e}"))
}

/// Key of account `index` derived from the phrase (blake3 in derive_key mode on the BIP-39 seed; NOT SLIP-10).
pub fn derive_account(mnemonic: &Mnemonic, index: u32) -> SigningKey {
    let seed = mnemonic.to_seed("");
    let mut input = seed.to_vec();
    input.extend_from_slice(&index.to_le_bytes());
    SigningKey::from_bytes(&crate::crypto::h("vearl/wallet/account/v1", &[&input]))
}

/// Post-quantum account `index` derived from the same phrase (derivation domain distinct from Ed25519's).
pub fn derive_pq_account(mnemonic: &Mnemonic, index: u32) -> crate::pq::PqKey {
    let seed = mnemonic.to_seed("");
    let mut input = seed.to_vec();
    input.extend_from_slice(&index.to_le_bytes());
    crate::pq::PqKey::from_seed(&crate::crypto::h("vearl/wallet/pq-account/v1", &[&input]))
}

pub fn pq_address_of(pk: &PubKey) -> String {
    bech32::encode::<Bech32m>(Hrp::parse(PQ_ADDRESS_HRP).unwrap(), pk).expect("32 bytes")
}

/// Readable address: Bech32m("vrl", public key); the checksum detects typing mistakes.
pub fn address_of(pk: &PubKey) -> String {
    bech32::encode::<Bech32m>(Hrp::parse(ADDRESS_HRP).unwrap(), pk).expect("32 bytes")
}

pub fn parse_address(s: &str) -> Result<PubKey, String> {
    let (hrp, data) = bech32::decode(s.trim()).map_err(|e| format!("invalid address: {e}"))?;
    if hrp.as_str() != ADDRESS_HRP && hrp.as_str() != PQ_ADDRESS_HRP {
        return Err(format!("unexpected prefix '{}' (expected '{ADDRESS_HRP}' or '{PQ_ADDRESS_HRP}')", hrp.as_str()));
    }
    <PubKey>::try_from(data.as_slice()).map_err(|_| "invalid address length".to_string())
}

impl Keystore {
    /// Encrypts the recovery phrase (its 32 bytes of entropy) with the password.
    pub fn create(mnemonic: &Mnemonic, password: &str, kp: KdfParams) -> Result<Keystore, String> {
        if password.len() < 8 {
            return Err("password too short (8 characters minimum)".into());
        }
        let entropy = mnemonic.to_entropy();
        let (mut salt, mut nonce) = ([0u8; 16], [0u8; 12]);
        rand::rngs::OsRng.fill_bytes(&mut salt);
        rand::rngs::OsRng.fill_bytes(&mut nonce);
        let key = derive_key(password, &salt, kp)?;
        let ct = ChaCha20Poly1305::new((&key).into())
            .encrypt(Nonce::from_slice(&nonce), entropy.as_slice())
            .map_err(|_| "encryption failed".to_string())?;
        Ok(Keystore {
            version: 1,
            kdf: "argon2id".into(),
            kdf_params: kp,
            salt: hex(&salt),
            cipher: "chacha20poly1305".into(),
            nonce: hex(&nonce),
            ciphertext: hex(&ct),
            address: address_of(&pubkey(&derive_account(mnemonic, 0))),
        })
    }

    /// Decrypts. A wrong password or a modified file fail (AEAD authentication tag).
    pub fn unlock(&self, password: &str) -> Result<Mnemonic, String> {
        if self.version != 1 || self.kdf != "argon2id" || self.cipher != "chacha20poly1305" {
            return Err("keystore format not supported".into());
        }
        // anti-abuse bounds: a hostile file must not make us allocate gigabytes
        if self.kdf_params.m_kib > 1 << 21 || self.kdf_params.t > 20 || self.kdf_params.p > 8 {
            return Err("KDF parameters out of bounds".into());
        }
        let (salt, nonce, ct) = (unhex(&self.salt)?, unhex(&self.nonce)?, unhex(&self.ciphertext)?);
        if nonce.len() != 12 {
            return Err("invalid nonce".into());
        }
        let key = derive_key(password, &salt, self.kdf_params)?;
        let entropy = ChaCha20Poly1305::new((&key).into())
            .decrypt(Nonce::from_slice(&nonce), ct.as_slice())
            .map_err(|_| "wrong password or modified file".to_string())?;
        Mnemonic::from_entropy_in(Language::English, &entropy).map_err(|e| e.to_string())
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if path.exists() {
            return Err(format!("{} already exists: refusing to overwrite a wallet", path.display()));
        }
        std::fs::write(path, serde_json::to_string_pretty(self).unwrap()).map_err(|e| e.to_string())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Keystore, String> {
        let txt = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_str(&txt).map_err(|e| format!("unreadable keystore: {e}"))
    }
}

/// "12.5" -> base units, without floats; at most 8 decimals.
pub fn parse_amount(s: &str) -> Result<u64, String> {
    let (int, frac) = s.split_once('.').unwrap_or((s, ""));
    if (int.is_empty() && frac.is_empty()) || frac.len() > 8 || !int.chars().chain(frac.chars()).all(|c| c.is_ascii_digit()) {
        return Err(format!("invalid amount '{s}' (digits, at most 8 decimals)"));
    }
    let i: u64 = if int.is_empty() { 0 } else { int.parse().map_err(|_| "amount too large".to_string())? };
    let f: u64 = if frac.is_empty() { 0 } else { format!("{frac:0<8}").parse().unwrap() };
    i.checked_mul(COIN).and_then(|v| v.checked_add(f)).ok_or_else(|| "amount too large".to_string())
}

pub fn fmt_amount(v: u64) -> String {
    format!("{}.{:08}", v / COIN, v % COIN)
}

/// Password: variable VEARL_WALLET_PASSWORD, otherwise file `VEARL_WALLET_PASSWORD_FILE` / `file`, otherwise masked prompt
/// on a terminal, otherwise one line on standard input. NEVER as a command-line argument (visible in `ps`).
pub fn read_password(prompt: &str, file: Option<&str>) -> Result<String, String> {
    use std::io::{BufRead, IsTerminal};
    if let Ok(p) = std::env::var("VEARL_WALLET_PASSWORD") {
        return Ok(p);
    }
    let file = file.map(str::to_string).or_else(|| std::env::var("VEARL_WALLET_PASSWORD_FILE").ok());
    if let Some(f) = file {
        return std::fs::read_to_string(&f).map(|s| s.trim_end_matches(['\n', '\r']).to_string()).map_err(|e| format!("{f}: {e}"));
    }
    if std::io::stdin().is_terminal() {
        return hidden_prompt(prompt);
    }
    let mut l = String::new();
    std::io::stdin().lock().read_line(&mut l).map_err(|e| e.to_string())?;
    Ok(l.trim_end_matches(['\n', '\r']).to_string())
}

/// Asks a secret on the terminal without echoing it. Nothing appears while typing (not even stars), which looks like a frozen prompt: the
/// first time, a line says so.
pub fn hidden_prompt(prompt: &str) -> Result<String, String> {
    use std::sync::atomic::{AtomicBool, Ordering};
    static TOLD: AtomicBool = AtomicBool::new(false);
    if !TOLD.swap(true, Ordering::SeqCst) {
        eprintln!("(Your typing is hidden: nothing appears on the screen, not even stars. Type it, then press Enter. If nothing is accepted, switch your keyboard to English letters first.)");
    }
    rpassword::prompt_password(prompt).map_err(|e| e.to_string())
}

/// Key of account `account` of a wallet (file): unlocked with the supplied password.
pub fn load_signing_key(path: &Path, password: &str, account: u32) -> Result<SigningKey, String> {
    Ok(derive_account(&Keystore::load(path)?.unlock(password)?, account))
}

/// Domain separation of signed messages: a signature made for a message can never be valid as a transaction signature.
const MESSAGE_DOMAIN: &[u8] = b"vearl/message/v1\0";

/// Signs a text message with an account key. Used to prove that one owns an address (testnet rewards registration, docs/19).
pub fn sign_message(sk: &SigningKey, message: &[u8]) -> [u8; 64] {
    use ed25519_dalek::Signer;
    sk.sign(&[MESSAGE_DOMAIN, message].concat()).to_bytes()
}

/// Checks a message signature made by [`sign_message`] against a public key.
pub fn verify_message(pk: &PubKey, message: &[u8], sig: &[u8]) -> bool {
    crate::crypto::verify_sig(pk, &[MESSAGE_DOMAIN, message].concat(), sig)
}
