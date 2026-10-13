//! Demo data: used by the picture tool (feature `devshots`) and by `vearl-wallet-gui --demo`, a mode for recordings that shows a
//! wallet full of made-up figures, touches no network and sends nothing. The fixed phrase below belongs to NO real wallet: never use it.
use crate::app::*;
use crate::core::*;
use std::path::PathBuf;
use std::time::Instant;
use vearl_core::crypto::pubkey;
use vearl_core::state::COIN;
use vearl_core::wallet::{address_of, derive_account, derive_pq_account, pq_address_of};

pub fn mnemonic(seed: u8) -> bip39::Mnemonic {
    bip39::Mnemonic::from_entropy_in(bip39::Language::English, &[seed; 32]).unwrap()
}

/// The phrase a wallet created in demo mode gets (so that a recording can be repeated and the check of the words typed by hand).
pub fn creation_mnemonic() -> bip39::Mnemonic {
    let h = blake3::hash(b"vearl wallet demo recording");
    bip39::Mnemonic::from_entropy_in(bip39::Language::English, h.as_bytes()).unwrap()
}

pub fn entry(name: &str, seed: u8) -> WalletEntry {
    WalletEntry { path: PathBuf::from(format!("C:\\Users\\you\\AppData\\Roaming\\Vearl\\wallets\\{name}.json")), name: name.into(), address: address_of(&pubkey(&derive_account(&mnemonic(seed), 0))) }
}

pub fn entries() -> Vec<WalletEntry> {
    vec![entry("main", 7), entry("mining-rig", 5)]
}

pub fn rows() -> Vec<TxRow> {
    let (a, b) = (entry("x", 9).address, entry("y", 3).address);
    vec![
        TxRow { id: "a1".repeat(32), direction: "in".into(), label: "Transfer".into(), counterparty: Some(a.clone()), amount: Some(250 * COIN), fee: 0, pending: true, height: None, confirmations: None },
        TxRow { id: "b2".repeat(32), direction: "out".into(), label: "Transfer".into(), counterparty: Some(b), amount: Some(12 * COIN + 50_000_000), fee: 1_000_000, pending: false, height: Some(20812), confirmations: Some(5) },
        TxRow { id: "c3".repeat(32), direction: "in".into(), label: "Transfer".into(), counterparty: Some(a), amount: Some(1000 * COIN), fee: 0, pending: false, height: Some(20790), confirmations: Some(27) },
        TxRow { id: "d4".repeat(32), direction: "out".into(), label: "Submit job".into(), counterparty: None, amount: Some(5 * COIN), fee: 0, pending: false, height: Some(20400), confirmations: Some(417) },
    ]
}

pub fn status() -> NodeStatus {
    NodeStatus { node: "seed.vearl.org:9000".into(), height: 20817, peers: 6, mempool: 1, genesis_ok: Some(true) }
}

pub fn standing() -> Standing {
    Standing { rank: 12, blocks: 41, useful_blocks: 6, points: 58.75, active_days: 2, eligible: false, reasons: vec!["active on 2 of the 3 required days".into(), "not registered".into()] }
}

/// Fills an open session with the demo figures.
pub fn fill_session(s: &mut Session) {
    let pq = pq_address_of(&derive_pq_account(&mnemonic(7), 0).pubkey());
    s.status = Some(Ok(status()));
    s.account = Some(Account { balance: 1_234 * COIN + 56_789_000, nonce: 3, next_nonce: 4 });
    s.pq_address = Some(pq);
    s.pq_account = Some(Account { balance: 42 * COIN + 500_000, nonce: 0, next_nonce: 0 });
    s.activity = Some(Ok(rows()));
    s.registered = Some(Ok(false));
    s.standing = Some(Ok(Some(standing())));
    s.last_refresh = Some(Instant::now());
}

/// What a confirmed action returns in demo mode.
pub fn done_for(action: &Action) -> Done {
    match action {
        Action::RevealPq => Done::Pq(pq_address_of(&derive_pq_account(&mnemonic(7), 0).pubkey())),
        Action::ShowPhrase => Done::Phrase(mnemonic(7).words().map(str::to_string).collect()),
        Action::Send { to, amount, fee, .. } => Done::Sent { txid: "e5".repeat(32), from: entry("main", 7).address, to: to.clone(), amount: *amount, fee: *fee },
        Action::Register { contact } => {
            let address = entry("main", 7).address;
            Done::Registration(RegLines { message: format!("Vearl testnet rewards registration. Address: {address}. Contact: {contact}. I accept the terms in docs/19."), address, signature: "9f".repeat(64) })
        }
        Action::Sign { message } => Done::Signed(RegLines { address: entry("main", 7).address, message: message.clone(), signature: "3c".repeat(64) }),
        Action::ChangePassword { .. } => Done::PasswordChanged,
    }
}
