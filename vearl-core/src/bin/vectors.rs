//! vearl-vectors: consensus test vectors (work layer), in JSON format, for a second implementer.
//! Normative specification: docs/07-work-specification.md. Independent verifier: tools/spec_ref.py.
use vearl_core::crypto::*;
use vearl_core::types::*;
use vearl_core::work::*;
use serde_json::json;

fn hx(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn i8b(v: &[i8]) -> Vec<u8> {
    v.iter().map(|x| *x as u8).collect()
}

fn i32b(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn hb(b: &[u8]) -> String {
    hx(blake3::hash(b).as_bytes())
}

fn main() {
    // --- Merkle ---
    let leaves: Vec<Hash> = (0..5u64).map(|i| leaf_hash(i, &[i as u8; 7])).collect();
    let tree = MerkleTree::new(leaves.clone());
    let m8: Vec<i8> = (0..64).map(|i| ((i * 7 % 129) as i32 - 64) as i8).collect();
    let merkle = json!({
        "leaves": leaves.iter().map(|l| hx(l)).collect::<Vec<_>>(),
        "root5": hx(&tree.root()),
        "proof_idx3": tree.proof(3).iter().map(|p| hx(p)).collect::<Vec<_>>(),
        "matrix8": m8.iter().map(|x| *x as i32).collect::<Vec<_>>(),
        "rows_root": hx(&rows_tree(&m8, 8).root()),
        "cols_root": hx(&cols_tree(&m8, 8).root()),
    });

    // --- work on reduced parameters (n=32, rank 8, fold_k 16) and on the testnet parameters ---
    let mut work = vec![];
    for (p, epoch, sg) in [
        (WorkParams { n: 32, tile: 16, rank: 8, fold_k: 16 }, 7u64, [0x11u8; 32]),
        (WorkParams { n: 32, tile: 16, rank: 8, fold_k: 32 }, 8, [0x22; 32]),
        (WorkParams::TESTNET, 9, [0x33; 32]),
    ] {
        let (a, b) = native_matrices(epoch, &p);
        let (out, noise) = run_noisy_gemm(&a, &b, &sg, &p);
        let c = decode(&out.c_noised, &a, &b, &noise, &p);
        assert_eq!(c, plain_gemm(&a, &b, p.n));
        let to8 = |v: &Vec<i32>| v.iter().map(|x| *x as i8).collect::<Vec<i8>>();
        work.push(json!({
            "params": {"n": p.n, "tile": p.tile, "rank": p.rank, "fold_k": p.fold_k},
            "epoch": epoch,
            "sigma": hx(&sg),
            "a_blake3": hb(&i8b(&a)), "b_blake3": hb(&i8b(&b)),
            "a_head": a[..16].iter().map(|x| *x as i32).collect::<Vec<_>>(),
            "noise_blake3": {"el": hb(&i8b(&to8(&noise.el))), "er": hb(&i8b(&to8(&noise.er))), "fl": hb(&i8b(&to8(&noise.fl))), "fr": hb(&i8b(&to8(&noise.fr)))},
            "c_noised_blake3": hb(&i32b(&out.c_noised)),
            "c_blake3": hb(&i32b(&c)),
            "jackpots": out.jackpots.iter().map(|j| hx(j)).collect::<Vec<_>>(),
        }));
    }

    // --- sigma and header ---
    let (prev, miner, txr) = ([0xaau8; 32], [0xbbu8; 32], [0xccu8; 32]);
    let jid = [0xddu8; 32];
    let classes = [WorkClass::Native, WorkClass::Job(jid), WorkClass::Layer { job: jid, layer: 3 }, WorkClass::Product { job: jid, i: 1, j: 2, k: 3 }, WorkClass::Op { job: jid, op: 5 }];
    let sigmas: Vec<_> = classes
        .iter()
        .enumerate()
        .map(|(i, c)| {
            json!({
                "class_bincode": hx(&bincode::serialize(c).unwrap()),
                "extra_nonce": 1000 + i as u64,
                "timestamp": 1_700_000_000u64 + i as u64,
                "sigma": hx(&sigma(&prev, &miner, c, 1000 + i as u64, &txr, 1_700_000_000 + i as u64)),
            })
        })
        .collect();
    let hd = Header { version: 2, height: 42, prev, timestamp: 1_700_000_042, miner, class: WorkClass::Job(jid), extra_nonce: 77, ti: 3, tj: 5, jackpot: [0xeeu8; 32], tx_root: txr, target: 0x1234_5678_9abc_def0, a_root: [0x11u8; 32], b_root: [0x22u8; 32], accounts_root: [0x33u8; 32] };
    let header = json!({"bincode": hx(&bincode::serialize(&hd).unwrap()), "hash": hx(&hd.hash())});
    let target = json!([
        {"jackpot": hx(&[0x00, 0, 0, 0, 0, 0, 0, 0x05, 9, 9]), "target": 6u64, "meets": meets_target(&{ let mut j = [0u8; 32]; j[7] = 5; j }, 6)},
        {"jackpot": hx(&{ let mut j = [0u8; 32]; j[7] = 6; j }), "target": 6u64, "meets": meets_target(&{ let mut j = [0u8; 32]; j[7] = 6; j }, 6)},
    ]);

    println!("{}", serde_json::to_string_pretty(&json!({"merkle": merkle, "work": work, "sigma": sigmas, "header": header, "target": target})).unwrap());
}
