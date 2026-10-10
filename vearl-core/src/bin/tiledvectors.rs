//! vearl-tiled-vectors: a COMPLETE chain of BLOCK-WISE matrix products (`tiled` jobs, docs/07 section 13) produced by the reference node,
//! in JSON: honest cycles (Accept, settlement by window), `Product` work blocks, PRODUCT fraud proof and SUM
//! fraud proof. Independent verifier: tools/tiled_ref.py.
//!   vearl-tiled-vectors                 -> JSON vectors on standard output
//!   vearl-tiled-vectors verify FILE     -> replays a chain (produced by ANOTHER implementation) with the Rust validator
//!   vearl-tiled-vectors diff FILE       -> differential fuzz: decisions of the Rust validator against those of the Python validator
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use ed25519_dalek::SigningKey;
use vearl_core::chain::*;
use vearl_core::crypto::*;
use vearl_core::state::*;
use vearl_core::tiled::*;
use vearl_core::types::*;
use vearl_core::work::*;
use serde_json::{json, Value};

fn hx(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}
fn b64(v: &[i8]) -> String {
    B64.encode(v.iter().map(|x| *x as u8).collect::<Vec<u8>>())
}
fn b64i32(v: &[i32]) -> String {
    B64.encode(v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
}
fn unhex_vec(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
}
fn unhex32(s: &str) -> Hash {
    unhex_vec(s).try_into().expect("32 bytes")
}
fn dec_i8(s: &str) -> Vec<i8> {
    B64.decode(s).unwrap().into_iter().map(|x| x as i8).collect()
}

struct Gen {
    chain: Chain,
    ts: u64,
    blocks: Vec<Value>,
    nonces: std::collections::HashMap<u8, u64>,
}

fn class_json(c: &WorkClass) -> Value {
    match c {
        WorkClass::Native => json!({"type": "Native"}),
        WorkClass::Product { job, i, j, k } => json!({"type": "Product", "job": hx(job), "i": i, "j": j, "k": k}),
        _ => unreachable!(),
    }
}

impl Gen {
    fn tx(&mut self, who: u8, kind: TxKind) -> SignedTx {
        let n = self.nonces.entry(who).or_insert(0);
        let t = SignedTx::new(Tx { from: pubkey(&key(who)), nonce: *n, fee: 0, kind }, &key(who));
        *n += 1;
        t
    }

    fn block(&mut self, miner: u8, class: WorkClass, cm: Option<&CommittedMatrices>, txs: Vec<SignedTx>, dt: u64) {
        self.ts += dt;
        let out = self.chain.mine(&key(miner), class, cm, txs, self.ts).expect("mine");
        let b = out.block.clone();
        self.chain.validate_and_apply(out.block).unwrap_or_else(|e| panic!("block refused: {e}"));
        let hd = &b.header;
        self.blocks.push(json!({
            "header": {"version": hd.version, "height": hd.height, "prev": hx(&hd.prev), "timestamp": hd.timestamp, "miner": hx(&hd.miner), "class": class_json(&hd.class),
                       "extra_nonce": hd.extra_nonce, "ti": hd.ti, "tj": hd.tj, "jackpot": hx(&hd.jackpot), "tx_root": hx(&hd.tx_root), "target": hd.target, "a_root": hx(&hd.a_root), "b_root": hx(&hd.b_root), "accounts_root": hx(&hd.accounts_root)},
            "hash": hx(&hd.hash()),
            "txs": b.txs.iter().map(|t| json!({"from": hx(&t.tx.from), "nonce": t.tx.nonce, "fee": t.tx.fee, "sig": hx(&t.sig), "id": hx(&t.id()),
                                               "tx_bincode": hx(&bincode::serialize(&t.tx).unwrap()), "kind": serde_json::to_value(&t.tx.kind).unwrap()})).collect::<Vec<_>>(),
        }));
    }

    fn native(&mut self, txs: Vec<SignedTx>) {
        self.block(4, WorkClass::Native, None, txs, 20);
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Tamper {
    None,
    Product,
    Sum,
}

#[derive(Clone, Copy, PartialEq)]
enum Finish {
    Accept,
    Settle,
    Fraud,
    /// The client demands a product entry and a sum entry on-chain, the worker serves them, then acceptance
    FetchServe,
    /// The client demands an entry, the worker does not serve: SlashUnserved by a third party
    Unserved,
}

fn genesis_cfg(g: &Value) -> (WorkParams, GenesisConfig) {
    let gp = |k: &str| g["params"][k].as_u64().unwrap() as usize;
    let p = WorkParams { n: gp("n"), tile: gp("tile"), rank: gp("rank"), fold_k: gp("fold_k") };
    let alloc: Vec<(PubKey, u64)> = g["alloc"].as_array().unwrap().iter().map(|a| (unhex32(a["key"].as_str().unwrap()), a["amount"].as_u64().unwrap())).collect();
    let timing = Timing { challenge_secs: g["timing"]["challenge_secs"].as_u64().unwrap(), serve_secs: g["timing"]["serve_secs"].as_u64().unwrap() };
    (p, GenesisConfig { params: p, alloc, ts: g["timestamp"].as_u64().unwrap(), target: g["target"].as_u64().unwrap(), timing, retarget: Retarget::Fixed })
}

fn parse_header(h: &Value) -> Header {
    let u = |f: &str| h[f].as_u64().unwrap();
    let class = match h["class"]["type"].as_str().unwrap() {
        "Native" => WorkClass::Native,
        "Product" => WorkClass::Product { job: unhex32(h["class"]["job"].as_str().unwrap()), i: h["class"]["i"].as_u64().unwrap() as u8, j: h["class"]["j"].as_u64().unwrap() as u8, k: h["class"]["k"].as_u64().unwrap() as u8 },
        t => panic!("class {t}"),
    };
    Header {
        version: u("version") as u32,
        height: u("height"),
        prev: unhex32(h["prev"].as_str().unwrap()),
        timestamp: u("timestamp"),
        miner: unhex32(h["miner"].as_str().unwrap()),
        class,
        extra_nonce: u("extra_nonce"),
        ti: u("ti") as u32,
        tj: u("tj") as u32,
        jackpot: unhex32(h["jackpot"].as_str().unwrap()),
        tx_root: unhex32(h["tx_root"].as_str().unwrap()),
        target: u("target"),
        a_root: unhex32(h["a_root"].as_str().unwrap()),
        b_root: unhex32(h["b_root"].as_str().unwrap()),
        accounts_root: unhex32(h["accounts_root"].as_str().unwrap()),
    }
}

fn parse_txs(b: &Value) -> Vec<SignedTx> {
    b["txs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| SignedTx { tx: Tx { from: unhex32(t["from"].as_str().unwrap()), nonce: t["nonce"].as_u64().unwrap(), fee: t["fee"].as_u64().unwrap(), kind: serde_json::from_value(t["kind"].clone()).expect("TxKind") }, sig: unhex_vec(t["sig"].as_str().unwrap()) })
        .collect()
}

fn status_name(s: &TiledStatus) -> String {
    format!("{s:?}").split(|c: char| c == ' ' || c == '{').next().unwrap().to_string()
}

/// `vearl-tiled-vectors verify FILE`: replays the chain (including blocks and transactions produced by ANOTHER implementation).
fn verify(path: &str) {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let (p, cfg) = genesis_cfg(&v["genesis"]);
    let n = p.n;
    let mut chain = Chain::from_config(&cfg);
    // data: (dims, roots of A, roots of B) -> (A, B)
    struct D {
        dims: (usize, usize, usize),
        a_roots: Vec<Hash>,
        b_roots: Vec<Hash>,
        a: Vec<i8>,
        b: Vec<i8>,
    }
    let mut data: Vec<D> = vec![];
    for d in v["tiled_data"].as_array().unwrap() {
        let dd = d["dims"].as_array().unwrap();
        let dims = (dd[0].as_u64().unwrap() as usize, dd[1].as_u64().unwrap() as usize, dd[2].as_u64().unwrap() as usize);
        let (a, b) = (dec_i8(d["a"].as_str().unwrap()), dec_i8(d["b"].as_str().unwrap()));
        data.push(D { dims, a_roots: a_block_roots(&a, dims.1 * n, dims.0, dims.1, n), b_roots: b_block_roots(&b, dims.2 * n, dims.1, dims.2, n), a, b });
    }
    let blocks = v["blocks"].as_array().unwrap();
    for (k, b) in blocks.iter().enumerate() {
        let header = parse_header(&b["header"]);
        let txs = parse_txs(b);
        let proof = match &header.class {
            WorkClass::Native => build_proof(chain.native_committed(header.height / EPOCH_LEN), header.ti as usize, header.tj as usize, &p),
            WorkClass::Product { job, i, j, k: kk } => {
                let tj = chain.state.tiled.get(job).unwrap_or_else(|| panic!("block {k}: unknown job"));
                let d = data.iter().find(|d| d.dims == tj.dims && d.a_roots == tj.a_roots && d.b_roots == tj.b_roots).unwrap_or_else(|| panic!("block {k}: data missing"));
                let (a, bm) = (get_block(&d.a, d.dims.1 * n, *i as usize, *kk as usize, n), get_block(&d.b, d.dims.2 * n, *kk as usize, *j as usize, n));
                build_proof(&CommittedMatrices::new(a, bm, &p), header.ti as usize, header.tj as usize, &p)
            }
            _ => unreachable!(),
        };
        if let Err(e) = chain.validate_and_apply(Block { header, proof, txs }) {
            println!("REFUSED at block {k}: {e}");
            std::process::exit(1);
        }
    }
    assert!(chain.check_supply_invariant());
    let st = &chain.state;
    let mut jobs: Vec<String> = st.tiled.iter().map(|(id, j)| format!("{}={}", hx(&id[..4]), status_name(&j.status))).collect();
    jobs.sort();
    println!("OK: {} blocks accepted by the Rust validator, height {}, emission {}, burned {}, block-wise jobs [{}]", blocks.len(), chain.height(), st.minted, st.burned, jobs.join(", "));
}

/// `vearl-tiled-vectors diff FILE`: DIFFERENTIAL FUZZ of the block-wise jobs (same rules as prog-vectors diff).
fn diff(path: &str) {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let (p, cfg) = genesis_cfg(&v["genesis"]);
    let mut chain = Chain::from_config(&cfg);
    let (mut agree, mut mismatches) = (0usize, vec![]);
    let attempts = v["attempts"].as_array().unwrap();
    for (k, at) in attempts.iter().enumerate() {
        let b = &at["block"];
        let header = parse_header(&b["header"]);
        let txs = parse_txs(b);
        let proof = build_proof(chain.native_committed(header.height / EPOCH_LEN), header.ti as usize, header.tj as usize, &p);
        let res = chain.validate_and_apply(Block { header, proof, txs });
        let py_ok = at["py_ok"].as_bool().unwrap();
        if res.is_ok() == py_ok {
            agree += 1;
        } else {
            mismatches.push(format!("attempt {k} [{}]: Python {} ({}); Rust {}", at["names"].as_str().unwrap(), if py_ok { "accepted" } else { "refused" }, at["why"].as_str().unwrap(), match &res { Ok(()) => "accepted".to_string(), Err(e) => format!("refused: {e}") }));
        }
    }
    assert!(chain.check_supply_invariant(), "money supply invariant broken on the Rust side");
    let f = &v["final"];
    let st = &chain.state;
    let mut diffs: Vec<String> = vec![];
    if st.minted != f["minted"].as_u64().unwrap() || st.burned != f["burned"].as_u64().unwrap() || chain.height() != f["height"].as_u64().unwrap() {
        diffs.push(format!("emission/burn/height: Rust ({}, {}, {}) against Python ({}, {}, {})", st.minted, st.burned, chain.height(), f["minted"], f["burned"], f["height"]));
    }
    for a in f["accounts"].as_array().unwrap() {
        let key = unhex32(a["key"].as_str().unwrap());
        if (st.balance(&key), st.nonce(&key)) != (a["balance"].as_u64().unwrap(), a["nonce"].as_u64().unwrap()) {
            diffs.push(format!("account {}: Rust ({}, {}) against Python ({}, {})", &a["key"].as_str().unwrap()[..8], st.balance(&key), st.nonce(&key), a["balance"], a["nonce"]));
        }
        let r = st.reputation(&key);
        let rp = &f["reputation"][a["key"].as_str().unwrap()];
        if (r.done as u64, r.slashed as u64, r.expired as u64) != (rp["done"].as_u64().unwrap(), rp["slashed"].as_u64().unwrap(), rp["expired"].as_u64().unwrap()) {
            diffs.push(format!("reputation {}", &a["key"].as_str().unwrap()[..8]));
        }
    }
    let pj = f["jobs"].as_object().unwrap();
    if pj.len() != st.tiled.len() {
        diffs.push(format!("number of jobs: Rust {} against Python {}", st.tiled.len(), pj.len()));
    }
    for (id, j) in &st.tiled {
        let name = status_name(&j.status);
        match pj.get(&hx(id)) {
            Some(pv) if pv["status"].as_str().unwrap() == name && pv["escrow_net"].as_u64().unwrap() == j.escrow_net && pv["rebate_left"].as_u64().unwrap() == j.rebate_left && pv["posted"].as_bool().unwrap() == !j.product_roots.is_empty() => {}
            other => diffs.push(format!("job {}: Rust ({name}, {}, {}, {}) against Python {other:?}", &hx(id)[..8], j.escrow_net, j.rebate_left, !j.product_roots.is_empty())),
        }
    }
    println!("[diff] final state (emission, burn, accounts, block-wise jobs, reputations): {}", if diffs.is_empty() { "IDENTICAL".to_string() } else { format!("{} DIFFERENCE(S)", diffs.len()) });
    for d in diffs.iter().take(8) {
        println!("  DIFFERENCE {d}");
    }
    println!("[diff] {agree}/{} identical decisions between the Rust validator and the Python validator; {} divergence(s)", attempts.len(), mismatches.len());
    for m in mismatches.iter().take(12) {
        println!("  DIVERGENCE {m}");
    }
    if !mismatches.is_empty() || !diffs.is_empty() {
        std::process::exit(1);
    }
}

fn main() {
    if let (Some("verify"), Some(path)) = (std::env::args().nth(1).as_deref(), std::env::args().nth(2)) {
        return verify(&path);
    }
    if let (Some("diff"), Some(path)) = (std::env::args().nth(1).as_deref(), std::env::args().nth(2)) {
        return diff(&path);
    }
    let p = WorkParams::TESTNET;
    let n = p.n;
    let (client, worker, challenger) = (1u8, 2u8, 3u8);
    let alloc: Vec<(PubKey, u64)> = (1u8..=4).map(|s| (pubkey(&key(s)), 10_000 * COIN)).collect();
    let timing = Timing { challenge_secs: 60, serve_secs: 30 };
    let cfg = GenesisConfig { params: p, alloc: alloc.clone(), ts: 1_700_000_000, target: u64::MAX / 16, timing, retarget: Retarget::Fixed };
    let mut g = Gen { chain: Chain::from_config(&cfg), ts: cfg.ts, blocks: vec![], nonces: Default::default() };
    for _ in 0..3 {
        g.native(vec![]);
    }
    let pk = |s: u8| pubkey(&key(s));
    let mut data: Vec<Value> = vec![];
    let mut job_ids: Vec<JobId> = vec![];

    // (name, dimensions in blocks (I, K, J), tampering, finish)
    let cases: Vec<(&str, (usize, usize, usize), Tamper, Finish)> = vec![
        ("2x2x2 honest, accepted", (2, 2, 2), Tamper::None, Finish::Accept),
        ("1x2x1 honest, settled by the window", (1, 2, 1), Tamper::None, Finish::Settle),
        ("2x1x3: wrong product", (2, 1, 3), Tamper::Product, Finish::Fraud),
        ("2x2x1: wrong sum", (2, 2, 1), Tamper::Sum, Finish::Fraud),
        ("1x2x2: entries demanded then served, accepted", (1, 2, 2), Tamper::None, Finish::FetchServe),
        ("1x1x2: entry demanded, not served: worker penalised", (1, 1, 2), Tamper::None, Finish::Unserved),
    ];
    for (ci, (_name, dims, tamper, fin)) in cases.iter().enumerate() {
        let (bi, bk, bj) = *dims;
        let (a, b) = demo_matrices(n, *dims, 70 + ci as u64);
        let (a_roots, b_roots) = (a_block_roots(&a, bk * n, bi, bk, n), b_block_roots(&b, bj * n, bk, bj, n));
        let dd = (bi as u8, bk as u8, bj as u8);
        let id = tiled_id(&pk(client), g.nonces.get(&client).copied().unwrap_or(0), dd, &a_roots, &b_roots);
        job_ids.push(id);
        let products_n = (bi * bk * bj) as u64;
        let t = g.tx(client, TxKind::Tiled(TiledTx::Submit { dims: dd, a_roots, b_roots, escrow: 30 * COIN * products_n, deadline: g.chain.height() + 500 }));
        g.native(vec![t]);
        let t = g.tx(worker, TxKind::Tiled(TiledTx::Claim { job_id: id }));
        g.native(vec![t]);
        // two useful work blocks on products of the job (the client's computation feeds the lottery)
        for (i, j, k) in [(0usize, 0usize, 0usize), (bi - 1, bj - 1, bk - 1)] {
            let cm = CommittedMatrices::new(get_block(&a, bk * n, i, k, n), get_block(&b, bj * n, k, j, n), &p);
            g.block(worker, WorkClass::Product { job: id, i: i as u8, j: j as u8, k: k as u8 }, Some(&cm), vec![], 20);
        }
        let (honest_products, honest_sums) = compute_all(&a, &b, *dims, n);
        let (mut products, mut sums) = (honest_products.clone(), honest_sums.clone());
        if *tamper == Tamper::Product {
            let l = products[1].len();
            products[1][l / 3] += 1;
            for ij in 0..bi * bj {
                let mut s = vec![0i32; n * n];
                for k in 0..bk {
                    for (x, v) in s.iter_mut().zip(&products[ij * bk + k]) {
                        *x += *v;
                    }
                }
                sums[ij] = s;
            }
        }
        if *tamper == Tamper::Sum {
            let l = sums[0].len();
            sums[0][l / 2] -= 1;
        }
        let post = g.tx(worker, TxKind::Tiled(TiledTx::Post { job_id: id, product_roots: products.iter().map(|x| c_tree(x, n).root()).collect(), c_roots: sums.iter().map(|x| c_tree(x, n).root()).collect() }));
        g.native(vec![post]);
        match fin {
            Finish::Accept => {
                let t = g.tx(client, TxKind::Tiled(TiledTx::Accept { job_id: id }));
                g.native(vec![t]);
            }
            Finish::Settle => {
                for _ in 0..12 {
                    g.native(vec![]);
                }
                let t = g.tx(4, TxKind::Tiled(TiledTx::Settle { job_id: id }));
                g.native(vec![t]);
            }
            Finish::FetchServe => {
                let (t1, t2) = (
                    g.tx(client, TxKind::Tiled(TiledTx::Fetch { job_id: id, which: 0, i: 0, j: 1, k: 1, idx: 7 })),
                    g.tx(client, TxKind::Tiled(TiledTx::Fetch { job_id: id, which: 1, i: 0, j: 1, k: 0, idx: 5 })),
                );
                g.native(vec![t1, t2]);
                let pi = (0 * bj + 1) * bk + 1;
                let s1 = g.tx(worker, TxKind::Tiled(TiledTx::Serve { job_id: id, which: 0, i: 0, j: 1, k: 1, idx: 7, row: row_of(&products[pi], n, 0), path: c_tree(&products[pi], n).proof(0) }));
                let s2 = g.tx(worker, TxKind::Tiled(TiledTx::Serve { job_id: id, which: 1, i: 0, j: 1, k: 0, idx: 5, row: row_of(&sums[1], n, 0), path: c_tree(&sums[1], n).proof(0) }));
                g.native(vec![s1, s2]);
                let t = g.tx(client, TxKind::Tiled(TiledTx::Accept { job_id: id }));
                g.native(vec![t]);
            }
            Finish::Unserved => {
                let t = g.tx(client, TxKind::Tiled(TiledTx::Fetch { job_id: id, which: 1, i: 0, j: 0, k: 0, idx: 3 }));
                g.native(vec![t]);
                g.native(vec![]);
                g.native(vec![]);
                let t = g.tx(challenger, TxKind::Tiled(TiledTx::SlashUnserved { job_id: id }));
                g.native(vec![t]);
            }
            Finish::Fraud => {
                let kind = if *tamper == Tamper::Product {
                    // first product for which an entry is not the dot product of the committed row / column
                    let k_flat = (0..products.len()).find(|&x| products[x] != honest_products[x]).unwrap();
                    let (i, j, k) = (k_flat / (bj * bk), (k_flat % (bj * bk)) / bk, k_flat % bk);
                    let idx = (0..n * n).find(|&e| products[k_flat][e] != honest_products[k_flat][e]).unwrap();
                    let (r, c) = (idx / n, idx % n);
                    let (ab, bb) = (get_block(&a, bk * n, i, k, n), get_block(&b, bj * n, k, j, n));
                    TiledFraudKind::Product {
                        i: i as u8, j: j as u8, k: k as u8, r: r as u32, c: c as u32,
                        a_row: ab[r * n..(r + 1) * n].to_vec(), a_path: rows_tree(&ab, n).proof(r),
                        b_col: (0..n).map(|t| bb[t * n + c]).collect(), b_path: cols_tree(&bb, n).proof(c),
                        z_row: row_of(&products[k_flat], n, idx / n), z_path: c_tree(&products[k_flat], n).proof(idx / n),
                    }
                } else {
                    let (ij, idx) = (0usize, (0..n * n).find(|&e| products[0..bk].iter().map(|x| x[e] as i64).sum::<i64>() != sums[0][e] as i64).unwrap());
                    let (i, j) = (ij / bj, ij % bj);
                    TiledFraudKind::Sum {
                        i: i as u8, j: j as u8, r: (idx / n) as u32, c: (idx % n) as u32,
                        z_rows: (0..bk).map(|k| row_of(&products[ij * bk + k], n, idx / n)).collect(), z_paths: (0..bk).map(|k| c_tree(&products[ij * bk + k], n).proof(idx / n)).collect(),
                        c_row: row_of(&sums[ij], n, idx / n), c_path: c_tree(&sums[ij], n).proof(idx / n),
                    }
                };
                let t = g.tx(challenger, TxKind::Tiled(TiledTx::Fraud { job_id: id, kind }));
                g.native(vec![t]);
            }
        }
        data.push(json!({"dims": [bi, bk, bj], "a": b64(&a), "b": b64(&b), "products": products.iter().map(|x| b64i32(x)).collect::<Vec<_>>(), "sums": sums.iter().map(|x| b64i32(x)).collect::<Vec<_>>()}));
    }

    assert!(g.chain.check_supply_invariant());
    let st = &g.chain.state;
    let jobs_json: Vec<Value> = job_ids.iter().map(|id| json!({"id": hx(id), "status": status_name(&st.tiled[id].status), "rebate_left": st.tiled[id].rebate_left, "escrow_net": st.tiled[id].escrow_net, "posted": !st.tiled[id].product_roots.is_empty()})).collect();
    let fin = json!({
        "height": g.chain.height(), "minted": st.minted, "burned": st.burned,
        "accounts": (1u8..=4).map(|s| json!({"key": hx(&pk(s)), "balance": st.balance(&pk(s)), "nonce": st.nonce(&pk(s))})).collect::<Vec<_>>(),
        "jobs": jobs_json,
        "reputation": (1u8..=4).map(|s| { let r = st.reputation(&pk(s)); json!({"key": hx(&pk(s)), "done": r.done, "slashed": r.slashed, "expired": r.expired}) }).collect::<Vec<_>>(),
    });
    let genesis = json!({
        "params": {"n": p.n, "tile": p.tile, "rank": p.rank, "fold_k": p.fold_k},
        "alloc": alloc.iter().map(|(k, a)| json!({"key": hx(k), "amount": a})).collect::<Vec<_>>(),
        "timestamp": cfg.ts, "target": cfg.target, "retarget": "Fixed",
        "timing": {"challenge_secs": timing.challenge_secs, "serve_secs": timing.serve_secs},
        "constants": {"block_time": BLOCK_TIME, "epoch_len": EPOCH_LEN, "r0": R0, "halving": HALVING, "tail": TAIL, "lwma_n": LWMA_N, "max_future": MAX_FUTURE,
                      "burn_bps": BURN_BPS, "bond_bps": BOND_BPS, "min_escrow": MIN_ESCROW, "challenge_window": CHALLENGE_WINDOW,
                      "fetch_fee": FETCH_FEE, "fetch_extend_secs": FETCH_EXTEND_SECS, "max_pending_fetches": MAX_PENDING_FETCHES},
    });
    println!("{}", serde_json::to_string(&json!({"genesis": genesis, "tiled_data": data, "blocks": g.blocks, "final": fin})).unwrap());
}
