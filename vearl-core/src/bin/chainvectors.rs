//! vearl-chain-vectors: a small complete chain (Native Work + transfers, LWMA difficulty, 2 epochs of native matrices)
//! produced by the reference node, in JSON. A second implementer must re-validate it block by block (docs/07 section 10) and
//! find exactly the final state. Independent verifier: tools/chain_ref.py.
use ed25519_dalek::SigningKey;
use vearl_core::chain::*;
use vearl_core::crypto::*;
use vearl_core::state::*;
use vearl_core::types::*;
use vearl_core::work::*;
use serde_json::json;

fn hx(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

fn unhex<const N: usize>(s: &str) -> [u8; N] {
    unhex_vec(s).try_into().unwrap_or_else(|_| panic!("hex length"))
}

fn unhex_vec(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
}

/// `vearl-chain-vectors verify FILE`: replays a chain (JSON) with the Rust REFERENCE VALIDATOR. Used to check
/// that blocks produced by another implementation (tools/chain_ref.py) are accepted.
fn verify(path: &str) {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let g = &v["genesis"];
    let gp = |k: &str| g["params"][k].as_u64().unwrap() as usize;
    let p = WorkParams { n: gp("n"), tile: gp("tile"), rank: gp("rank"), fold_k: gp("fold_k") };
    let alloc: Vec<(PubKey, u64)> = g["alloc"].as_array().unwrap().iter().map(|a| (unhex::<32>(a["key"].as_str().unwrap()), a["amount"].as_u64().unwrap())).collect();
    let cfg = GenesisConfig { params: p, alloc, ts: g["timestamp"].as_u64().unwrap(), target: g["target"].as_u64().unwrap(), timing: Timing::default(), retarget: Retarget::Lwma };
    let mut chain = Chain::from_config(&cfg);
    let blocks = v["blocks"].as_array().unwrap();
    for (k, b) in blocks.iter().enumerate() {
        let h = &b["header"];
        let (ti, tj) = (h["ti"].as_u64().unwrap() as usize, h["tj"].as_u64().unwrap() as usize);
        let u = |f: &str| h[f].as_u64().unwrap();
        let header = Header {
            version: u("version") as u32,
            height: u("height"),
            prev: unhex(h["prev"].as_str().unwrap()),
            timestamp: u("timestamp"),
            miner: unhex(h["miner"].as_str().unwrap()),
            class: WorkClass::Native,
            extra_nonce: u("extra_nonce"),
            ti: ti as u32,
            tj: tj as u32,
            jackpot: unhex(h["jackpot"].as_str().unwrap()),
            tx_root: unhex(h["tx_root"].as_str().unwrap()),
            target: u("target"),
            a_root: unhex(h["a_root"].as_str().unwrap()),
            b_root: unhex(h["b_root"].as_str().unwrap()),
            accounts_root: unhex(h["accounts_root"].as_str().unwrap()),
        };
        let txs: Vec<SignedTx> = b["txs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| SignedTx {
                tx: Tx { from: unhex(t["from"].as_str().unwrap()), nonce: t["nonce"].as_u64().unwrap(), fee: t["fee"].as_u64().unwrap(), kind: TxKind::Transfer { to: unhex(t["to"].as_str().unwrap()), amount: t["amount"].as_u64().unwrap() } },
                sig: unhex_vec(t["sig"].as_str().unwrap()),
            })
            .collect();
        let epoch = header.height / EPOCH_LEN;
        let proof = build_proof(chain.native_committed(epoch), ti, tj, &p);
        if let Err(e) = chain.validate_and_apply(Block { header, proof, txs }) {
            println!("REFUSED at block {k}: {e}");
            std::process::exit(1);
        }
    }
    println!("OK: {} blocks accepted by the Rust validator, height {}", blocks.len(), chain.height());
}

/// `vearl-chain-vectors asert`: lines `anchor first_ts tip_ts tip_height ideal half_life` on standard input -> ASERT target
/// (same anchoring as `asert_next_target`); used for the differential check tools/asert_ref.py.
fn asert_diff() {
    use std::io::BufRead;
    for line in std::io::stdin().lock().lines() {
        let v: Vec<i128> = line.unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect();
        if v.len() != 6 {
            continue;
        }
        println!("{}", asert_core(v[0] as u64, v[2] - (v[1] - v[4]), v[3] as u64, v[4] as u64, v[5] as u64));
    }
}

fn main() {
    if let (Some("verify"), Some(p)) = (std::env::args().nth(1).as_deref(), std::env::args().nth(2)) {
        return verify(&p);
    }
    if std::env::args().nth(1).as_deref() == Some("asert") {
        return asert_diff();
    }
    let alloc: Vec<(PubKey, u64)> = (1u8..=3).map(|s| (pubkey(&key(s)), 1000 * COIN * s as u64)).collect();
    let cfg = GenesisConfig { params: WorkParams::TESTNET, alloc: alloc.clone(), ts: 1_700_000_000, target: u64::MAX / 16, timing: Timing::default(), retarget: Retarget::Lwma };
    let mut chain = Chain::from_config(&cfg);
    let mut nonces = [0u64; 4];
    let mut blocks = vec![];
    let mut ts = cfg.ts;
    let sink = [0x77u8; 32]; // address without a key: receives transfers
    for h in 0..EPOCH_LEN + 30 {
        // deterministic timestamps, with a few extreme gaps to exercise the retargeting bounds
        let dt = match h % 23 {
            5 => 1,
            11 => 13 * BLOCK_TIME,
            r => BLOCK_TIME * 2 / 5 + (r * 7 + h * 3) % (BLOCK_TIME * 4 / 3 + 1),
        };
        ts += dt;
        let miner = key(1 + (h % 3) as u8);
        let mut txs = vec![];
        if h % 4 == 1 {
            let from_seed = 1 + ((h / 4) % 3) as u8;
            let to = if h % 8 == 1 { pubkey(&key(1 + ((h / 4 + 1) % 3) as u8)) } else { sink };
            let amount = (h + 1) * COIN / 7;
            let fee = (h % 5) * 1000;
            let tx = Tx { from: pubkey(&key(from_seed)), nonce: nonces[from_seed as usize], fee, kind: TxKind::Transfer { to, amount } };
            nonces[from_seed as usize] += 1;
            txs.push(SignedTx::new(tx, &key(from_seed)));
        }
        let out = chain.mine(&miner, WorkClass::Native, None, txs, ts).expect("mine");
        let b = out.block.clone();
        chain.validate_and_apply(out.block).expect("validate");
        let hd = &b.header;
        blocks.push(json!({
            "header": {"version": hd.version, "height": hd.height, "prev": hx(&hd.prev), "timestamp": hd.timestamp, "miner": hx(&hd.miner),
                       "extra_nonce": hd.extra_nonce, "ti": hd.ti, "tj": hd.tj, "jackpot": hx(&hd.jackpot), "tx_root": hx(&hd.tx_root), "target": hd.target, "a_root": hx(&hd.a_root), "b_root": hx(&hd.b_root), "accounts_root": hx(&hd.accounts_root)},
            "hash": hx(&hd.hash()),
            "txs": b.txs.iter().map(|t| match &t.tx.kind {
                TxKind::Transfer { to, amount } => json!({"from": hx(&t.tx.from), "nonce": t.tx.nonce, "fee": t.tx.fee, "to": hx(to), "amount": amount, "sig": hx(&t.sig), "id": hx(&t.id())}),
                _ => unreachable!(),
            }).collect::<Vec<_>>(),
        }));
    }
    let mut accounts: Vec<PubKey> = alloc.iter().map(|(p, _)| *p).collect();
    accounts.push(sink);
    let fin = json!({
        "height": chain.height(), "next_target": chain.target, "minted": chain.state.minted, "burned": chain.state.burned,
        "total_work": chain.total_work().to_string(),
        "accounts": accounts.iter().map(|p| json!({"key": hx(p), "balance": chain.state.balance(p), "nonce": chain.state.nonce(p)})).collect::<Vec<_>>(),
    });
    let genesis = json!({
        "params": {"n": cfg.params.n, "tile": cfg.params.tile, "rank": cfg.params.rank, "fold_k": cfg.params.fold_k},
        "alloc": alloc.iter().map(|(p, a)| json!({"key": hx(p), "amount": a})).collect::<Vec<_>>(),
        "timestamp": cfg.ts, "target": cfg.target, "retarget": "Lwma",
        "constants": {"block_time": BLOCK_TIME, "epoch_len": EPOCH_LEN, "r0": R0, "halving": HALVING, "tail": TAIL, "lwma_n": LWMA_N, "max_future": MAX_FUTURE},
    });
    println!("{}", serde_json::to_string_pretty(&json!({"genesis": genesis, "blocks": blocks, "final": fin})).unwrap());
}
