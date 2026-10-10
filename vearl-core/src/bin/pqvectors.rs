//! Post-quantum signature vectors (JSON) for the independent Python verifier `tools/pq_ref.py`.
use vearl_core::pq::*;

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn main() {
    let mut cases = vec![];
    for (i, seed) in [[11u8; 32], [12u8; 32]].iter().enumerate() {
        let k = PqKey::from_seed(seed);
        let pk = k.pubkey();
        let msg = format!("test message {i}").into_bytes();
        let sig = k.sign(&msg);
        let mut push = |name: &str, pk: &[u8], msg: &[u8], sig: &[u8]| {
            let valid = pk.len() == 32 && pq_verify(&<[u8; 32]>::try_from(pk).unwrap(), msg, sig);
            cases.push(format!(r#"{{"name":"{name} (key {i})","pk":"{}","msg":"{}","sig":"{}","valid":{valid}}}"#, hex(pk), hex(msg), hex(sig)));
        };
        push("valid signature", &pk, &msg, &sig);
        push("altered message", &pk, b"altered message", &sig);
        for (label, pos) in [("R", 3usize), ("FORS", 500), ("WOTS+ chain", 3500), ("authentication path", 7000), ("last layer", PQ_SIG_LEN - 2)] {
            let mut bad = sig.clone();
            bad[pos] ^= 0x40;
            push(&format!("bit altered in {label}"), &pk, &msg, &bad);
        }
        let other = PqKey::from_seed(&[99; 32]).pubkey();
        push("wrong public key", &other, &msg, &sig);
    }
    println!(r#"{{"cases":[{}]}}"#, cases.join(","));
}
