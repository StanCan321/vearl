use vearl_core::crypto::pubkey;
use vearl_core::wallet::*;

const PW: &str = "correct horse battery staple";

#[test]
fn mnemonic_has_24_words_and_restores_the_same_account() {
    let m = new_mnemonic();
    assert_eq!(m.to_string().split_whitespace().count(), 24);
    let restored = parse_mnemonic(&m.to_string()).unwrap();
    assert_eq!(pubkey(&derive_account(&m, 0)), pubkey(&derive_account(&restored, 0)));
    assert_ne!(pubkey(&derive_account(&m, 0)), pubkey(&derive_account(&m, 1)), "distinct accounts per index");
    // a modified phrase (changed word) is rejected thanks to the BIP-39 checksum
    let mut words: Vec<&str> = m.to_string().leak().split_whitespace().collect();
    words[0] = if words[0] == "abandon" { "ability" } else { "abandon" };
    assert!(parse_mnemonic(&words.join(" ")).is_err() || pubkey(&derive_account(&parse_mnemonic(&words.join(" ")).unwrap(), 0)) != pubkey(&derive_account(&m, 0)));
}

#[test]
fn keystore_roundtrip_wrong_password_and_tampering() {
    let m = new_mnemonic();
    let ks = Keystore::create(&m, PW, KdfParams::LIGHT).unwrap();
    assert_eq!(ks.unlock(PW).unwrap().to_string(), m.to_string());
    assert!(ks.unlock("not the password").is_err(), "wrong password");

    let mut bad = ks.clone();
    let mut ct = bad.ciphertext.clone().into_bytes();
    ct[0] = if ct[0] == b'0' { b'1' } else { b'0' };
    bad.ciphertext = String::from_utf8(ct).unwrap();
    assert!(bad.unlock(PW).is_err(), "modified file: invalid AEAD tag");

    // the ciphertext contains neither the phrase nor the entropy in the clear
    let json = serde_json::to_string(&ks).unwrap();
    // (we only test the VALUES: a BIP-39 word like "salt" or "address" coincides with a JSON field name)
    let values: Vec<String> = serde_json::from_str::<serde_json::Value>(&json).unwrap().as_object().unwrap().values().filter_map(|v| v.as_str().map(str::to_string)).collect();
    for w in m.to_string().split_whitespace() {
        assert!(!values.iter().any(|v| v == w), "phrase word in the clear: {w}");
    }
}

#[test]
fn keystore_refuses_weak_passwords_hostile_parameters_and_overwriting() {
    let m = new_mnemonic();
    assert!(Keystore::create(&m, "short", KdfParams::LIGHT).is_err());
    let mut ks = Keystore::create(&m, PW, KdfParams::LIGHT).unwrap();
    ks.kdf_params.m_kib = u32::MAX; // hostile file: must not allocate gigabytes
    assert!(ks.unlock(PW).is_err());

    let dir = std::env::temp_dir().join(format!("vearl-wallet-{}", rand::random::<u64>()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("w.json");
    let ks = Keystore::create(&m, PW, KdfParams::LIGHT).unwrap();
    ks.save(&path).unwrap();
    assert!(ks.save(&path).is_err(), "never overwrite an existing wallet");
    let loaded = Keystore::load(&path).unwrap();
    assert_eq!(loaded.address, ks.address);
    assert_eq!(loaded.unlock(PW).unwrap().to_string(), m.to_string());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn addresses_are_checksummed_and_typos_are_detected() {
    let pk = pubkey(&derive_account(&new_mnemonic(), 0));
    let a = address_of(&pk);
    assert!(a.starts_with("vrl1"));
    assert_eq!(parse_address(&a).unwrap(), pk);
    // a changed letter is detected
    let mut chars: Vec<char> = a.chars().collect();
    let last = chars.len() - 3;
    chars[last] = if chars[last] == 'q' { 'p' } else { 'q' };
    assert!(parse_address(&chars.into_iter().collect::<String>()).is_err());
    // wrong prefix (address of another network) refused
    let other = bech32::encode::<bech32::Bech32m>(bech32::Hrp::parse("bc").unwrap(), &pk).unwrap();
    assert!(parse_address(&other).is_err());
    assert!(parse_address("vrl1short").is_err());
}

#[test]
fn post_quantum_account_is_restored_from_the_same_phrase_and_has_its_own_address_prefix() {
    let m = new_mnemonic();
    let restored = parse_mnemonic(&m.to_string()).unwrap();
    let (a, b) = (derive_pq_account(&m, 0), derive_pq_account(&restored, 0));
    assert_eq!(a.pubkey(), b.pubkey(), "deterministic restoration");
    assert_ne!(a.pubkey(), derive_pq_account(&m, 1).pubkey());
    assert_ne!(a.pubkey(), pubkey(&derive_account(&m, 0)), "post-quantum key distinct from the Ed25519 key");
    let addr = pq_address_of(&a.pubkey());
    assert!(addr.starts_with("vrlq1"), "{addr}");
    assert_eq!(parse_address(&addr).unwrap(), a.pubkey());
    assert!(parse_address("vrl1qqqqqq").is_err());
}
