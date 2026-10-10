//! vearl-market-vectors: a COMPLETE chain of the compute market (auction, bond, useful-work blocks with rebate,
//! data availability, fraud proof, expiry, settlement by dispute window, penalty for unserved data),
//! produced by the reference node, in JSON. A second implementer must re-validate it (docs/07 section 11) and find the
//! final state. Independent verifier: tools/market_ref.py.
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use ed25519_dalek::SigningKey;
use vearl_core::chain::*;
use vearl_core::crypto::*;
use vearl_core::state::*;
use vearl_core::types::*;
use vearl_core::work::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::{json, Value};

fn hx(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
fn hxs(v: &[Hash]) -> Vec<String> {
    v.iter().map(|h| hx(h)).collect()
}
fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}
fn i8s(v: &[i8]) -> Vec<i32> {
    v.iter().map(|x| *x as i32).collect()
}

fn txj(t: &SignedTx) -> Value {
    let kind = match &t.tx.kind {
        TxKind::Transfer { to, amount } => json!({"type": "Transfer", "to": hx(to), "amount": amount}),
        TxKind::SubmitJob { a_root, b_root, escrow, deadline } => json!({"type": "SubmitJob", "a_root": hx(a_root), "b_root": hx(b_root), "escrow": escrow, "deadline": deadline}),
        TxKind::SubmitJobAuction { a_root, b_root, escrow, deadline, auction_secs } => {
            json!({"type": "SubmitJobAuction", "a_root": hx(a_root), "b_root": hx(b_root), "escrow": escrow, "deadline": deadline, "auction_secs": auction_secs})
        }
        TxKind::Bid { job_id, price } => json!({"type": "Bid", "job_id": hx(job_id), "price": price}),
        TxKind::ClaimJob { job_id } => json!({"type": "ClaimJob", "job_id": hx(job_id)}),
        TxKind::PostResult { job_id, c_root } => json!({"type": "PostResult", "job_id": hx(job_id), "c_root": hx(c_root)}),
        TxKind::Accept { job_id } => json!({"type": "Accept", "job_id": hx(job_id)}),
        TxKind::Settle { job_id } => json!({"type": "Settle", "job_id": hx(job_id)}),
        TxKind::Fetch { job_id, i, j } => json!({"type": "Fetch", "job_id": hx(job_id), "i": i, "j": j}),
        TxKind::Serve { job_id, i, j, row, path } => json!({"type": "Serve", "job_id": hx(job_id), "i": i, "j": j, "row": row, "path": hxs(path)}),
        TxKind::SlashUnserved { job_id } => json!({"type": "SlashUnserved", "job_id": hx(job_id)}),
        TxKind::Fraud(f) => json!({"type": "Fraud", "job_id": hx(&f.job_id), "i": f.i, "j": f.j, "a_row": i8s(&f.a_row), "a_path": hxs(&f.a_path),
                                     "b_col": i8s(&f.b_col), "b_path": hxs(&f.b_path), "c_row": f.c_row, "c_path": hxs(&f.c_path)}),
        _ => unreachable!(),
    };
    json!({"from": hx(&t.tx.from), "nonce": t.tx.nonce, "fee": t.tx.fee, "sig": hx(&t.sig), "id": hx(&t.id()), "tx_bincode": hx(&bincode::serialize(&t.tx).unwrap()), "kind": kind})
}

struct Gen {
    chain: Chain,
    ts: u64,
    blocks: Vec<Value>,
    nonces: std::collections::HashMap<u8, u64>,
}

impl Gen {
    fn tx(&mut self, who: u8, kind: TxKind) -> SignedTx {
        let n = self.nonces.entry(who).or_insert(0);
        let t = SignedTx::new(Tx { from: pubkey(&key(who)), nonce: *n, fee: 0, kind }, &key(who));
        *n += 1;
        t
    }

    /// Mines and validates a block (Native by default, or useful-work class with the job's data).
    fn block(&mut self, miner: u8, class: WorkClass, cm: Option<&CommittedMatrices>, txs: Vec<SignedTx>, dt: u64) {
        self.ts += dt;
        let out = self.chain.mine(&key(miner), class, cm, txs, self.ts).expect("mine");
        let b = out.block.clone();
        self.chain.validate_and_apply(out.block).unwrap_or_else(|e| panic!("block refused: {e}"));
        let hd = &b.header;
        let class_json = match &hd.class {
            WorkClass::Native => json!({"type": "Native"}),
            WorkClass::Job(id) => json!({"type": "Job", "job": hx(id)}),
            _ => unreachable!(),
        };
        self.blocks.push(json!({
            "header": {"version": hd.version, "height": hd.height, "prev": hx(&hd.prev), "timestamp": hd.timestamp, "miner": hx(&hd.miner), "class": class_json,
                       "extra_nonce": hd.extra_nonce, "ti": hd.ti, "tj": hd.tj, "jackpot": hx(&hd.jackpot), "tx_root": hx(&hd.tx_root), "target": hd.target, "a_root": hx(&hd.a_root), "b_root": hx(&hd.b_root), "accounts_root": hx(&hd.accounts_root)},
            "hash": hx(&hd.hash()),
            "txs": b.txs.iter().map(txj).collect::<Vec<_>>(),
        }));
    }

    fn native(&mut self, txs: Vec<SignedTx>) {
        self.block(4, WorkClass::Native, None, txs, 20);
    }
    fn native_until(&mut self, ts: u64) {
        while self.ts < ts {
            self.native(vec![]);
        }
    }
    fn height(&self) -> u64 {
        self.chain.height()
    }
}

// ---------------- `verify` mode: replays a chain file with the Rust reference validator ----------------

fn unhex_vec(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
}
fn unhex32(s: &str) -> Hash {
    unhex_vec(s).try_into().expect("32 bytes")
}
fn hashes(v: &Value) -> Vec<Hash> {
    v.as_array().unwrap().iter().map(|x| unhex32(x.as_str().unwrap())).collect()
}
fn i8vec(v: &Value) -> Vec<i8> {
    v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i8).collect()
}
fn i32s(v: &Value) -> Vec<i32> {
    v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect()
}

fn parse_kind(k: &Value) -> TxKind {
    let h = |f: &str| unhex32(k[f].as_str().unwrap());
    let u = |f: &str| k[f].as_u64().unwrap();
    match k["type"].as_str().unwrap() {
        "Transfer" => TxKind::Transfer { to: h("to"), amount: u("amount") },
        "SubmitJob" => TxKind::SubmitJob { a_root: h("a_root"), b_root: h("b_root"), escrow: u("escrow"), deadline: u("deadline") },
        "SubmitJobAuction" => TxKind::SubmitJobAuction { a_root: h("a_root"), b_root: h("b_root"), escrow: u("escrow"), deadline: u("deadline"), auction_secs: u("auction_secs") },
        "Bid" => TxKind::Bid { job_id: h("job_id"), price: u("price") },
        "ClaimJob" => TxKind::ClaimJob { job_id: h("job_id") },
        "PostResult" => TxKind::PostResult { job_id: h("job_id"), c_root: h("c_root") },
        "Accept" => TxKind::Accept { job_id: h("job_id") },
        "Settle" => TxKind::Settle { job_id: h("job_id") },
        "Fetch" => TxKind::Fetch { job_id: h("job_id"), i: u("i") as u32, j: u("j") as u32 },
        "Serve" => TxKind::Serve { job_id: h("job_id"), i: u("i") as u32, j: u("j") as u32, row: i32s(&k["row"]), path: hashes(&k["path"]) },
        "SlashUnserved" => TxKind::SlashUnserved { job_id: h("job_id") },
        "Fraud" => TxKind::Fraud(FraudProof {
            job_id: h("job_id"),
            i: u("i") as u32,
            j: u("j") as u32,
            a_row: i8vec(&k["a_row"]),
            a_path: hashes(&k["a_path"]),
            b_col: i8vec(&k["b_col"]),
            b_path: hashes(&k["b_path"]),
            c_row: i32s(&k["c_row"]),
            c_path: hashes(&k["c_path"]),
        }),
        t => panic!("unknown transaction type: {t}"),
    }
}

/// `vearl-market-vectors verify FILE`: replays the chain (including blocks mined by ANOTHER implementation) with the
/// reference validator. Prints the final state of the jobs.
fn verify(path: &str) {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let g = &v["genesis"];
    let gp = |k: &str| g["params"][k].as_u64().unwrap() as usize;
    let p = WorkParams { n: gp("n"), tile: gp("tile"), rank: gp("rank"), fold_k: gp("fold_k") };
    let alloc: Vec<(PubKey, u64)> = g["alloc"].as_array().unwrap().iter().map(|a| (unhex32(a["key"].as_str().unwrap()), a["amount"].as_u64().unwrap())).collect();
    let timing = Timing { challenge_secs: g["timing"]["challenge_secs"].as_u64().unwrap(), serve_secs: g["timing"]["serve_secs"].as_u64().unwrap() };
    let cfg = GenesisConfig { params: p, alloc, ts: g["timestamp"].as_u64().unwrap(), target: g["target"].as_u64().unwrap(), timing, retarget: Retarget::Fixed };
    let mut chain = Chain::from_config(&cfg);
    let mut data: Vec<CommittedMatrices> = vec![];
    for d in v["jobs_data"].as_array().unwrap() {
        let dec = |f: &str| B64.decode(d[f].as_str().unwrap()).unwrap().into_iter().map(|x| x as i8).collect::<Vec<i8>>();
        data.push(CommittedMatrices::new(dec("a"), dec("b"), &p));
    }
    let blocks = v["blocks"].as_array().unwrap();
    for (k, b) in blocks.iter().enumerate() {
        let h = &b["header"];
        let u = |f: &str| h[f].as_u64().unwrap();
        let class = match h["class"]["type"].as_str().unwrap() {
            "Native" => WorkClass::Native,
            "Job" => WorkClass::Job(unhex32(h["class"]["job"].as_str().unwrap())),
            t => panic!("class {t}"),
        };
        let header = Header {
            version: u("version") as u32,
            height: u("height"),
            prev: unhex32(h["prev"].as_str().unwrap()),
            timestamp: u("timestamp"),
            miner: unhex32(h["miner"].as_str().unwrap()),
            class: class.clone(),
            extra_nonce: u("extra_nonce"),
            ti: u("ti") as u32,
            tj: u("tj") as u32,
            jackpot: unhex32(h["jackpot"].as_str().unwrap()),
            tx_root: unhex32(h["tx_root"].as_str().unwrap()),
            target: u("target"),
            a_root: unhex32(h["a_root"].as_str().unwrap()),
            b_root: unhex32(h["b_root"].as_str().unwrap()),
            accounts_root: unhex32(h["accounts_root"].as_str().unwrap()),
        };
        let txs: Vec<SignedTx> = b["txs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| SignedTx { tx: Tx { from: unhex32(t["from"].as_str().unwrap()), nonce: t["nonce"].as_u64().unwrap(), fee: t["fee"].as_u64().unwrap(), kind: parse_kind(&t["kind"]) }, sig: unhex_vec(t["sig"].as_str().unwrap()) })
            .collect();
        // matrices of the work: native, or those of the job whose roots are the ones committed on-chain
        let proof = match &class {
            WorkClass::Native => build_proof(chain.native_committed(header.height / EPOCH_LEN), header.ti as usize, header.tj as usize, &p),
            WorkClass::Job(id) => {
                let j = chain.state.jobs.get(id).unwrap_or_else(|| panic!("block {k}: unknown job"));
                let cm = data.iter().find(|d| d.a_root() == j.a_root && d.b_root() == j.b_root).unwrap_or_else(|| panic!("block {k}: job data missing"));
                build_proof(cm, header.ti as usize, header.tj as usize, &p)
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
    let mut jobs: Vec<String> = st.jobs.iter().map(|(id, j)| format!("{}={}", hx(&id[..4]), format!("{:?}", j.status).split(|c: char| c == ' ' || c == '{').next().unwrap())).collect();
    jobs.sort();
    println!("OK: {} blocks accepted by the Rust validator, height {}, emission {}, burned {}, jobs [{}]", blocks.len(), chain.height(), st.minted, st.burned, jobs.join(", "));
}

/// `vearl-market-vectors diff FILE`: DIFFERENTIAL FUZZ. Each attempt (random Native block carrying valid or faulty market transactions)
/// is presented to the Rust validator, whose decision (accept / refuse) must be that of the independent Python
/// validator. The Rust validator only advances on accepted blocks, like the Python one.
fn diff(path: &str) {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let g = &v["genesis"];
    let gp = |k: &str| g["params"][k].as_u64().unwrap() as usize;
    let p = WorkParams { n: gp("n"), tile: gp("tile"), rank: gp("rank"), fold_k: gp("fold_k") };
    let alloc: Vec<(PubKey, u64)> = g["alloc"].as_array().unwrap().iter().map(|a| (unhex32(a["key"].as_str().unwrap()), a["amount"].as_u64().unwrap())).collect();
    let timing = Timing { challenge_secs: g["timing"]["challenge_secs"].as_u64().unwrap(), serve_secs: g["timing"]["serve_secs"].as_u64().unwrap() };
    let cfg = GenesisConfig { params: p, alloc, ts: g["timestamp"].as_u64().unwrap(), target: g["target"].as_u64().unwrap(), timing, retarget: Retarget::Fixed };
    let mut chain = Chain::from_config(&cfg);
    let (mut agree, mut mismatches) = (0usize, vec![]);
    let attempts = v["attempts"].as_array().unwrap();
    for (k, at) in attempts.iter().enumerate() {
        let b = &at["block"];
        let h = &b["header"];
        let u = |f: &str| h[f].as_u64().unwrap();
        let header = Header {
            version: u("version") as u32,
            height: u("height"),
            prev: unhex32(h["prev"].as_str().unwrap()),
            timestamp: u("timestamp"),
            miner: unhex32(h["miner"].as_str().unwrap()),
            class: WorkClass::Native,
            extra_nonce: u("extra_nonce"),
            ti: u("ti") as u32,
            tj: u("tj") as u32,
            jackpot: unhex32(h["jackpot"].as_str().unwrap()),
            tx_root: unhex32(h["tx_root"].as_str().unwrap()),
            target: u("target"),
            a_root: unhex32(h["a_root"].as_str().unwrap()),
            b_root: unhex32(h["b_root"].as_str().unwrap()),
            accounts_root: unhex32(h["accounts_root"].as_str().unwrap()),
        };
        let txs: Vec<SignedTx> = b["txs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| SignedTx { tx: Tx { from: unhex32(t["from"].as_str().unwrap()), nonce: t["nonce"].as_u64().unwrap(), fee: t["fee"].as_u64().unwrap(), kind: parse_kind(&t["kind"]) }, sig: unhex_vec(t["sig"].as_str().unwrap()) })
            .collect();
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
    // IDENTICAL FINAL STATE: emission, burn, accounts, jobs (status, escrow, rebate), reputations
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
    if pj.len() != st.jobs.len() {
        diffs.push(format!("number of jobs: Rust {} against Python {}", st.jobs.len(), pj.len()));
    }
    for (id, j) in &st.jobs {
        let name = format!("{:?}", j.status).split(|c: char| c == ' ' || c == '{').next().unwrap().to_string();
        match pj.get(&hx(id)) {
            Some(pv) if pv["status"].as_str().unwrap() == name && pv["escrow_net"].as_u64().unwrap() == j.escrow_net && pv["rebate_left"].as_u64().unwrap() == j.rebate_left => {}
            other => diffs.push(format!("job {}: Rust ({name}, {}, {}) against Python {other:?}", &hx(id)[..8], j.escrow_net, j.rebate_left)),
        }
    }
    println!("[diff] final state (emission, burn, accounts, jobs, reputations): {}", if diffs.is_empty() { "IDENTICAL".to_string() } else { format!("{} DIFFERENCE(S)", diffs.len()) });
    for d in diffs.iter().take(8) {
        println!("  DIFFERENCE {d}");
    }
    if !diffs.is_empty() {
        std::process::exit(1);
    }
    println!("[diff] {agree}/{} identical decisions between the Rust validator and the Python validator; {} divergence(s)", attempts.len(), mismatches.len());
    for m in mismatches.iter().take(12) {
        println!("  DIVERGENCE {m}");
    }
    if !mismatches.is_empty() {
        std::process::exit(1);
    }
}

fn job_matrices(seed: u64, p: &WorkParams) -> CommittedMatrices {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut m = || (0..p.n * p.n).map(|_| rng.gen_range(-64..=64) as i8).collect::<Vec<i8>>();
    let (a, b) = (m(), m());
    CommittedMatrices::new(a, b, p)
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
    let (client, w1, w2) = (1u8, 2u8, 3u8);
    let alloc: Vec<(PubKey, u64)> = (1u8..=4).map(|s| (pubkey(&key(s)), 10_000 * COIN)).collect();
    let timing = Timing { challenge_secs: 60, serve_secs: 30 };
    let cfg = GenesisConfig { params: p, alloc: alloc.clone(), ts: 1_700_000_000, target: u64::MAX / 16, timing, retarget: Retarget::Fixed };
    let mut g = Gen { chain: Chain::from_config(&cfg), ts: cfg.ts, blocks: vec![], nonces: Default::default() };
    let mut data = vec![];
    for _ in 0..3 {
        g.native(vec![]);
    }
    let pk = |s: u8| pubkey(&key(s));

    // ---------- A: auction job, fetch/serve, accept ----------
    let cma = job_matrices(101, &p);
    data.push(&cma);
    let ida = job_id(&pk(client), g.nonces.get(&client).copied().unwrap_or(0), &cma.a_root(), &cma.b_root());
    let tx = g.tx(client, TxKind::SubmitJobAuction { a_root: cma.a_root(), b_root: cma.b_root(), escrow: 50 * COIN, deadline: g.height() + 500, auction_secs: 60 });
    g.native(vec![tx]);
    let end_a = g.ts + 60;
    let tx = g.tx(w2, TxKind::Bid { job_id: ida, price: 35 * COIN });
    g.native(vec![tx]);
    let tx = g.tx(w1, TxKind::Bid { job_id: ida, price: 20 * COIN });
    g.native(vec![tx]);
    g.native_until(end_a);
    let tx = g.tx(w1, TxKind::ClaimJob { job_id: ida });
    g.native(vec![tx]);
    g.block(w1, WorkClass::Job(ida), Some(&cma), vec![], 20); // 1st useful block: rebate
    g.block(w1, WorkClass::Job(ida), Some(&cma), vec![], 20); // 2nd: rebate exhausted
    let ca = plain_gemm(&cma.a, &cma.b, n);
    let ta = c_tree(&ca, n);
    let tx = g.tx(w1, TxKind::PostResult { job_id: ida, c_root: ta.root() });
    g.native(vec![tx]);
    let tx = g.tx(client, TxKind::Fetch { job_id: ida, i: 3, j: 5 });
    g.native(vec![tx]);
    let tx = g.tx(w1, TxKind::Serve { job_id: ida, i: 3, j: 5, row: row_of(&ca, n, 3), path: ta.proof(3) });
    g.native(vec![tx]);
    let tx = g.tx(client, TxKind::Accept { job_id: ida });
    g.native(vec![tx]);

    // ---------- B: wrong result, fraud proof ----------
    let cmb = job_matrices(102, &p);
    data.push(&cmb);
    let idb = job_id(&pk(client), g.nonces[&client], &cmb.a_root(), &cmb.b_root());
    let tx = g.tx(client, TxKind::SubmitJob { a_root: cmb.a_root(), b_root: cmb.b_root(), escrow: 30 * COIN, deadline: g.height() + 500 });
    g.native(vec![tx]);
    let tx = g.tx(w2, TxKind::ClaimJob { job_id: idb });
    g.native(vec![tx]);
    g.block(w2, WorkClass::Job(idb), Some(&cmb), vec![], 20);
    let mut cbad = plain_gemm(&cmb.a, &cmb.b, n);
    cbad[5 * n + 7] += 1;
    let tb = c_tree(&cbad, n);
    let tx = g.tx(w2, TxKind::PostResult { job_id: idb, c_root: tb.root() });
    g.native(vec![tx]);
    let (fi, fj) = (5usize, 7usize);
    let proof = FraudProof { job_id: idb, i: fi as u32, j: fj as u32, a_row: cmb.a[fi * n..(fi + 1) * n].to_vec(), a_path: cmb.a_tree.proof(fi), b_col: cmb.col(fj, n), b_path: cmb.b_tree.proof(fj), c_row: row_of(&cbad, n, fi), c_path: tb.proof(fi) };
    let tx = g.tx(client, TxKind::Fraud(proof));
    g.native(vec![tx]);

    // ---------- C: data requested and never served -> SlashUnserved ----------
    let cmc = job_matrices(103, &p);
    data.push(&cmc);
    let idc = job_id(&pk(client), g.nonces[&client], &cmc.a_root(), &cmc.b_root());
    let tx = g.tx(client, TxKind::SubmitJob { a_root: cmc.a_root(), b_root: cmc.b_root(), escrow: 20 * COIN, deadline: g.height() + 500 });
    g.native(vec![tx]);
    let tx = g.tx(w1, TxKind::ClaimJob { job_id: idc });
    g.native(vec![tx]);
    let cc = plain_gemm(&cmc.a, &cmc.b, n);
    let tc = c_tree(&cc, n);
    let tx = g.tx(w1, TxKind::PostResult { job_id: idc, c_root: tc.root() });
    g.native(vec![tx]);
    let tx = g.tx(client, TxKind::Fetch { job_id: idc, i: 1, j: 1 });
    g.native(vec![tx]);
    let due = g.ts + 30;
    g.native_until(due + 1);
    let tx = g.tx(4, TxKind::SlashUnserved { job_id: idc });
    g.native(vec![tx]);

    // ---------- D: job never claimed -> expiry ----------
    let cmd = job_matrices(104, &p);
    data.push(&cmd);
    let idd = job_id(&pk(client), g.nonces[&client], &cmd.a_root(), &cmd.b_root());
    let tx = g.tx(client, TxKind::SubmitJob { a_root: cmd.a_root(), b_root: cmd.b_root(), escrow: 10 * COIN, deadline: g.height() + 4 });
    g.native(vec![tx]);
    for _ in 0..6 {
        g.native(vec![]);
    }
    let tx = g.tx(client, TxKind::Settle { job_id: idd });
    g.native(vec![tx]);

    // ---------- E: settlement by dispute window (height AND seconds) ----------
    let cme = job_matrices(105, &p);
    data.push(&cme);
    let ide = job_id(&pk(client), g.nonces[&client], &cme.a_root(), &cme.b_root());
    let tx = g.tx(client, TxKind::SubmitJob { a_root: cme.a_root(), b_root: cme.b_root(), escrow: 20 * COIN, deadline: g.height() + 500 });
    g.native(vec![tx]);
    let tx = g.tx(w2, TxKind::ClaimJob { job_id: ide });
    g.native(vec![tx]);
    g.block(w2, WorkClass::Job(ide), Some(&cme), vec![], 20);
    let ce = plain_gemm(&cme.a, &cme.b, n);
    let tx = g.tx(w2, TxKind::PostResult { job_id: ide, c_root: c_tree(&ce, n).root() });
    g.native(vec![tx]);
    for _ in 0..12 {
        g.native(vec![]);
    }
    let tx = g.tx(4, TxKind::Settle { job_id: ide });
    g.native(vec![tx]);
    let tx = g.tx(client, TxKind::Transfer { to: pk(4), amount: 3 * COIN });
    g.native(vec![tx]);

    // ---------- final state ----------
    let st = &g.chain.state;
    let status = |id: &JobId| format!("{:?}", st.jobs[id].status).split(|c: char| c == ' ' || c == '{').next().unwrap().to_string();
    let jobs_json: Vec<Value> = [ida, idb, idc, idd, ide].iter().map(|id| json!({"id": hx(id), "status": status(id), "rebate_left": st.jobs[id].rebate_left, "escrow_net": st.jobs[id].escrow_net})).collect();
    let fin = json!({
        "height": g.chain.height(), "minted": st.minted, "burned": st.burned,
        "accounts": (1u8..=4).map(|s| json!({"key": hx(&pk(s)), "balance": st.balance(&pk(s)), "nonce": st.nonce(&pk(s))})).collect::<Vec<_>>(),
        "jobs": jobs_json,
        "reputation": (1u8..=4).map(|s| { let r = st.reputation(&pk(s)); json!({"key": hx(&pk(s)), "done": r.done, "slashed": r.slashed, "expired": r.expired}) }).collect::<Vec<_>>(),
    });
    assert!(g.chain.check_supply_invariant());
    let genesis = json!({
        "params": {"n": p.n, "tile": p.tile, "rank": p.rank, "fold_k": p.fold_k},
        "alloc": alloc.iter().map(|(k, a)| json!({"key": hx(k), "amount": a})).collect::<Vec<_>>(),
        "timestamp": cfg.ts, "target": cfg.target, "retarget": "Fixed",
        "timing": {"challenge_secs": timing.challenge_secs, "serve_secs": timing.serve_secs},
        "constants": {"block_time": BLOCK_TIME, "epoch_len": EPOCH_LEN, "r0": R0, "halving": HALVING, "tail": TAIL, "lwma_n": LWMA_N, "max_future": MAX_FUTURE,
                      "burn_bps": BURN_BPS, "bond_bps": BOND_BPS, "min_escrow": MIN_ESCROW, "challenge_window": CHALLENGE_WINDOW, "auction_grace_secs": AUCTION_GRACE_SECS,
                      "max_auction_secs": MAX_AUCTION_SECS, "fetch_fee": FETCH_FEE, "fetch_extend_secs": FETCH_EXTEND_SECS, "max_pending_fetches": MAX_PENDING_FETCHES},
    });
    let jobs_data: Vec<Value> = data.iter().map(|cm| json!({"a": B64.encode(cm.a.iter().map(|x| *x as u8).collect::<Vec<u8>>()), "b": B64.encode(cm.b.iter().map(|x| *x as u8).collect::<Vec<u8>>())})).collect();
    println!("{}", serde_json::to_string(&json!({"genesis": genesis, "jobs_data": jobs_data, "blocks": g.blocks, "final": fin})).unwrap());
}
