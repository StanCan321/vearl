//! vearl-prog-vectors: a COMPLETE chain of verifiable tensor programs (CNN with a residual block and transformer encoder
//! block: convolutions, pooling, residuals, GemmAT, integer softmax and layer normalisation) with honest cycles
//! (Accept, settlement by window) and the FOUR kinds of fraud proofs (Gemm, Act, Elem, Col), produced by the
//! reference node, in JSON. Independent verifier: tools/prog_ref.py (docs/07 section 12).
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use ed25519_dalek::SigningKey;
use vearl_core::chain::*;
use vearl_core::crypto::*;
use vearl_core::mlp::{cols_root_of, rows_root_of};
use vearl_core::prog::*;
use vearl_core::state::*;
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

    fn block(&mut self, miner: u8, class: WorkClass, cm: Option<&CommittedMatrices>, txs: Vec<SignedTx>, dt: u64) {
        self.ts += dt;
        let out = self.chain.mine(&key(miner), class, cm, txs, self.ts).expect("mine");
        let b = out.block.clone();
        self.chain.validate_and_apply(out.block).unwrap_or_else(|e| panic!("block refused: {e}"));
        let hd = &b.header;
        let class_json = match &hd.class {
            WorkClass::Native => json!({"type": "Native"}),
            WorkClass::Op { job, op } => json!({"type": "Op", "job": hx(job), "op": op}),
            _ => unreachable!(),
        };
        self.blocks.push(json!({
            "header": {"version": hd.version, "height": hd.height, "prev": hx(&hd.prev), "timestamp": hd.timestamp, "miner": hx(&hd.miner), "class": class_json,
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
    Gemm(usize),
    Act(usize),
    Elem(usize),
    Col(usize),
}

#[derive(Clone, Copy, PartialEq)]
enum Finish {
    Accept,
    Settle,
    /// Fraud proven as soon as possible: `early` = during `Working` (as soon as the wrong operation is published)
    Fraud { early: bool },
    /// The client demands a column on-chain, the worker serves it, then acceptance
    FetchServe,
    /// The client demands a column, the worker does not serve: SlashUnserved by a third party
    Unserved,
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

/// `vearl-prog-vectors verify FILE`: replays the chain (including blocks, programs and fraud proofs produced by
/// ANOTHER implementation) with the Rust reference validator.
fn verify(path: &str) {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let g = &v["genesis"];
    let gp = |k: &str| g["params"][k].as_u64().unwrap() as usize;
    let p = WorkParams { n: gp("n"), tile: gp("tile"), rank: gp("rank"), fold_k: gp("fold_k") };
    let n = p.n;
    let alloc: Vec<(PubKey, u64)> = g["alloc"].as_array().unwrap().iter().map(|a| (unhex32(a["key"].as_str().unwrap()), a["amount"].as_u64().unwrap())).collect();
    let timing = Timing { challenge_secs: g["timing"]["challenge_secs"].as_u64().unwrap(), serve_secs: g["timing"]["serve_secs"].as_u64().unwrap() };
    let cfg = GenesisConfig { params: p, alloc, ts: g["timestamp"].as_u64().unwrap(), target: g["target"].as_u64().unwrap(), timing, retarget: Retarget::Fixed };
    let mut chain = Chain::from_config(&cfg);
    // data: (root of X0, roots of the weights) -> (x0, weights, published tensors)
    type Data = (Hash, Vec<Hash>, Vec<Vec<i8>>);
    let mut data: Vec<(Data, Vec<i8>, Vec<Vec<i8>>)> = vec![];
    for d in v["prog_data"].as_array().unwrap() {
        let x0 = dec_i8(d["x0"].as_str().unwrap());
        let ws: Vec<Vec<i8>> = d["ws"].as_array().unwrap().iter().map(|w| dec_i8(w.as_str().unwrap())).collect();
        let tens: Vec<Vec<i8>> = d["tensors"].as_array().unwrap().iter().map(|t| dec_i8(t.as_str().unwrap())).collect();
        data.push(((cols_root_of(&x0, n), ws.iter().map(|w| rows_root_of(w, n)).collect(), vec![]), x0, [ws, tens].concat()));
    }
    let blocks = v["blocks"].as_array().unwrap();
    for (k, b) in blocks.iter().enumerate() {
        let h = &b["header"];
        let u = |f: &str| h[f].as_u64().unwrap();
        let class = match h["class"]["type"].as_str().unwrap() {
            "Native" => WorkClass::Native,
            "Op" => WorkClass::Op { job: unhex32(h["class"]["job"].as_str().unwrap()), op: h["class"]["op"].as_u64().unwrap() as u8 },
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
            .map(|t| SignedTx { tx: Tx { from: unhex32(t["from"].as_str().unwrap()), nonce: t["nonce"].as_u64().unwrap(), fee: t["fee"].as_u64().unwrap(), kind: serde_json::from_value(t["kind"].clone()).expect("TxKind") }, sig: unhex_vec(t["sig"].as_str().unwrap()) })
            .collect();
        let proof = match &class {
            WorkClass::Native => build_proof(chain.native_committed(header.height / EPOCH_LEN), header.ti as usize, header.tj as usize, &p),
            WorkClass::Op { job, op } => {
                let j = chain.state.progs.get(job).unwrap_or_else(|| panic!("block {k}: unknown program"));
                let ((_, _, _), _x0, blob) = data.iter().find(|((r0, wr, _), _, _)| *r0 == j.x0_root && *wr == j.w_roots).map(|(a, b, c)| (a.clone(), b.clone(), c.clone())).unwrap_or_else(|| panic!("block {k}: program data missing"));
                let nw = j.w_roots.len();
                let (ws, tens) = (&blob[..nw], &blob[nw..]);
                let (a, bm): (Vec<i8>, Vec<i8>) = match &j.ops[*op as usize] {
                    Op::Gemm { w, x, .. } => (ws[*w as usize].clone(), tens[*x as usize].clone()),
                    Op::GemmAT { a, x, .. } => {
                        let ta = &tens[*a as usize];
                        ((0..n * n).map(|idx| ta[(idx % n) * n + idx / n]).collect(), tens[*x as usize].clone())
                    }
                    _ => panic!("block {k}: the operation is not a Gemm"),
                };
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
    let mut jobs: Vec<String> = st.progs.iter().map(|(id, j)| format!("{}={}", hx(&id[..4]), format!("{:?}", j.status).split(|c: char| c == ' ' || c == '{').next().unwrap())).collect();
    jobs.sort();
    println!("OK: {} blocks accepted by the Rust validator, height {}, emission {}, burned {}, programs [{}]", blocks.len(), chain.height(), st.minted, st.burned, jobs.join(", "));
}

/// `vearl-prog-vectors diff FILE`: DIFFERENTIAL FUZZ of the tensor programs. Each attempt (Native block carrying valid or faulty
/// program transactions) is presented to the Rust validator, whose decision must be that of the independent Python validator;
/// the final state (accounts, programs, reputations, emission, burn) must be identical.
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
            .map(|t| SignedTx { tx: Tx { from: unhex32(t["from"].as_str().unwrap()), nonce: t["nonce"].as_u64().unwrap(), fee: t["fee"].as_u64().unwrap(), kind: serde_json::from_value(t["kind"].clone()).expect("TxKind") }, sig: unhex_vec(t["sig"].as_str().unwrap()) })
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
    if pj.len() != st.progs.len() {
        diffs.push(format!("number of programs: Rust {} against Python {}", st.progs.len(), pj.len()));
    }
    for (id, j) in &st.progs {
        let name = format!("{:?}", j.status).split(|c: char| c == ' ' || c == '{').next().unwrap().to_string();
        match pj.get(&hx(id)) {
            Some(pv) if pv["status"].as_str().unwrap() == name && pv["escrow_net"].as_u64().unwrap() == j.escrow_net && pv["rebate_left"].as_u64().unwrap() == j.rebate_left && pv["posted"].as_u64().unwrap() == j.posted.len() as u64 => {}
            other => diffs.push(format!("program {}: Rust ({name}, {}, {}, {}) against Python {other:?}", &hx(id)[..8], j.escrow_net, j.rebate_left, j.posted.len())),
        }
    }
    println!("[diff] final state (emission, burn, accounts, programs, reputations): {}", if diffs.is_empty() { "IDENTICAL".to_string() } else { format!("{} DIFFERENCE(S)", diffs.len()) });
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
    let mut data: Vec<Value> = vec![];
    let mut job_ids: Vec<JobId> = vec![];
    let pk = |s: u8| pubkey(&key(s));

    // (name, program, tamper, finish)
    let cases: Vec<(&str, &str, Tamper, Finish)> = vec![
        ("honest cnn, accepted", "cnn", Tamper::None, Finish::Accept),
        ("honest transformer, settled by the window", "transformer", Tamper::None, Finish::Settle),
        ("cnn: wrong GEMM", "cnn", Tamper::Gemm(4), Finish::Fraud { early: false }),
        ("cnn: wrong activation", "cnn", Tamper::Act(1), Finish::Fraud { early: true }),
        ("cnn: wrong residual addition", "cnn", Tamper::Elem(6), Finish::Fraud { early: false }),
        ("transformer: wrong softmax", "transformer", Tamper::Col(4), Finish::Fraud { early: false }),
        ("honest MLP with bias, accepted", "biasmlp", Tamper::None, Finish::Accept),
        ("MLP with bias: wrong bias", "biasmlp", Tamper::Elem(1), Finish::Fraud { early: false }),
        ("MLP with bias and table: wrong activation (table)", "biasmlp", Tamper::Elem(2), Finish::Fraud { early: false }),
        ("MLP with bias and table, published in ONE transaction (PostBatch), accepted", "biasmlp+batch", Tamper::None, Finish::Accept),
        ("general convolution network (stride 2, no padding), accepted", "convnet", Tamper::None, Finish::Accept),
        ("general convolution network: wrong convolution output", "convnet", Tamper::Elem(0), Finish::Fraud { early: false }),
        ("MLP with wide input (Shift), accepted", "widemlp", Tamper::None, Finish::Accept),
        ("MLP with wide input: wrong Shift output", "widemlp", Tamper::Elem(0), Finish::Fraud { early: false }),
        ("multi-head attention (HeadSpread, HeadGather, gamma by Mul), accepted", "heads", Tamper::None, Finish::Accept),
        ("multi-head attention: wrong head spreading", "heads", Tamper::Elem(3), Finish::Fraud { early: false }),
        ("multi-head attention: wrong head gathering", "heads", Tamper::Elem(8), Finish::Fraud { early: false }),
        ("multi-head attention: wrong gamma (Mul)", "heads", Tamper::Elem(10), Finish::Fraud { early: false }),
        ("fine attention (Softmax + SoftmaxLo), accepted", "fineattn", Tamper::None, Finish::Accept),
        ("fine attention: wrong fine part (SoftmaxLo)", "fineattn", Tamper::Col(6), Finish::Fraud { early: false }),
        ("causal attention (CausalSoftmax high + fine), accepted", "causalattn", Tamper::None, Finish::Accept),
        ("causal attention: wrong causal softmax", "causalattn", Tamper::Col(5), Finish::Fraud { early: false }),
        ("cnn: column demanded then served, accepted", "cnn", Tamper::None, Finish::FetchServe),
        ("cnn: column demanded, not served: worker penalised", "cnn", Tamper::None, Finish::Unserved),
    ];
    for (ci, (_name, prog, tamper, fin)) in cases.iter().enumerate() {
        let batch = prog.ends_with("+batch");
        let prog = prog.trim_end_matches("+batch");
        let (x0, ws, ops) = demo_program(prog, n, 40 + ci as u64).unwrap();
        let x0_root = cols_root_of(&x0, n);
        let w_roots: Vec<Hash> = ws.iter().map(|w| rows_root_of(w, n)).collect();
        let id = prog_id(&pk(client), g.nonces.get(&client).copied().unwrap_or(0), &x0_root, &w_roots, &ops);
        job_ids.push(id);
        let gemms = ops.iter().filter(|o| o.is_gemm()).count() as u64;
        let t = g.tx(client, TxKind::Prog(ProgTx::Submit { x0_root, w_roots, ops: ops.clone(), escrow: 30 * COIN * gemms, deadline: g.chain.height() + 500 }));
        g.native(vec![t]);
        let t = g.tx(worker, TxKind::Prog(ProgTx::Claim { job_id: id }));
        g.native(vec![t]);
        if batch {
            // publication of ALL the operations in one transaction, then useful blocks of the Gemms (during `Complete`), then acceptance
            let (mut tens, mut zs) = (vec![x0.clone()], Vec::<Option<Vec<i32>>>::new());
            for op in ops.iter() {
                let (z, out) = eval_op(op, &tens, &ws, n);
                zs.push(z);
                tens.push(out);
            }
            let posts: Vec<(Hash, Hash)> = zs.iter().zip(tens[1..].iter()).map(|(z, t)| (z.as_ref().map_or([0u8; 32], |zz| c_tree(zz, n).root()), cols_root_of(t, n))).collect();
            let post = g.tx(worker, TxKind::Prog(ProgTx::PostBatch { job_id: id, op: 0, posts }));
            g.native(vec![post]);
            for (k, op) in ops.iter().enumerate() {
                if let Op::Gemm { w, x, .. } = op {
                    let cm = CommittedMatrices::new(ws[*w as usize].clone(), tens[*x as usize].clone(), &p);
                    g.block(worker, WorkClass::Op { job: id, op: k as u8 }, Some(&cm), vec![], 20);
                }
            }
            let t = g.tx(client, TxKind::Prog(ProgTx::Accept { job_id: id }));
            g.native(vec![t]);
            data.push(json!({"x0": b64(&x0), "ws": ws.iter().map(|w| b64(w)).collect::<Vec<_>>(), "tensors": tens.iter().map(|t| b64(t)).collect::<Vec<_>>(),
                              "zs": zs.iter().map(|z| z.as_ref().map(|zz| B64.encode(zz.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()))).collect::<Vec<_>>()}));
            continue;
        }
        let mut tens = vec![x0.clone()];
        let mut zs: Vec<Option<Vec<i32>>> = vec![];
        let mut early_fraud: Option<(u8, ProgFraud)> = None;
        for (k, op) in ops.iter().enumerate() {
            let (mut z, mut out) = eval_op(op, &tens, &ws, n);
            if let Op::Gemm { w, x, shift, relu } = op {
                let cm = CommittedMatrices::new(ws[*w as usize].clone(), tens[*x as usize].clone(), &p);
                g.block(worker, WorkClass::Op { job: id, op: k as u8 }, Some(&cm), vec![], 20);
                if *tamper == Tamper::Gemm(k) {
                    let zz = z.as_mut().unwrap();
                    zz[3 * n + 5] += 1 << *shift;
                    out = zz.iter().map(|&v| act_fn(v, *shift, *relu)).collect();
                }
                if *tamper == Tamper::Act(k) {
                    let i = out.iter().position(|v| *v != 63 && *v != 64).unwrap();
                    out[i] = out[i].wrapping_add(1);
                }
            } else if let Op::GemmAT { .. } = op {
                // (not tampered in these vectors)
                let a_t: Vec<i8> = {
                    let Op::GemmAT { a, .. } = op else { unreachable!() };
                    let ta = &tens[*a as usize];
                    (0..n * n).map(|idx| ta[(idx % n) * n + idx / n]).collect()
                };
                let Op::GemmAT { x, .. } = op else { unreachable!() };
                let cm = CommittedMatrices::new(a_t, tens[*x as usize].clone(), &p);
                g.block(worker, WorkClass::Op { job: id, op: k as u8 }, Some(&cm), vec![], 20);
            } else if *tamper == Tamper::Elem(k) || *tamper == Tamper::Col(k) {
                // one entry too large by one; never beyond the range of the operands (|v| <= 64), which would make the later tile proofs unprovable
                let i = out.iter().position(|v| (*v as i32) < 64).unwrap_or(0);
                out[i] = out[i].wrapping_add(1);
            }
            let z_root = z.as_ref().map_or([0u8; 32], |zz| c_tree(zz, n).root());
            let post = g.tx(worker, TxKind::Prog(ProgTx::PostOp { job_id: id, op: k as u8, z_root, x_root: cols_root_of(&out, n) }));
            g.native(vec![post]);
            zs.push(z);
            tens.push(out);
            if let Finish::Fraud { early: true } = fin {
                if early_fraud.is_none() && matches!(tamper, Tamper::Act(kk) if *kk == k) {
                    // proof built as soon as the wrong operation is published (the job is still `Working`)
                    let partial = find_fraud_proof(&ops[..=k], &ws, &tens, &zs, n);
                    early_fraud = partial;
                }
            }
            if let Some((op, kind)) = early_fraud.take() {
                let t = g.tx(challenger, TxKind::Prog(ProgTx::Fraud { job_id: id, op, kind }));
                g.native(vec![t]);
                break;
            }
        }
        match fin {
            Finish::Accept => {
                let t = g.tx(client, TxKind::Prog(ProgTx::Accept { job_id: id }));
                g.native(vec![t]);
            }
            Finish::Settle => {
                for _ in 0..12 {
                    g.native(vec![]);
                }
                let t = g.tx(4, TxKind::Prog(ProgTx::Settle { job_id: id }));
                g.native(vec![t]);
            }
            Finish::Fraud { early: false } => {
                let (op, kind) = find_fraud_proof(&ops, &ws, &tens, &zs, n).expect("fraud to prove");
                let t = g.tx(challenger, TxKind::Prog(ProgTx::Fraud { job_id: id, op, kind }));
                g.native(vec![t]);
            }
            Finish::Fraud { early: true } => {}
            Finish::FetchServe => {
                let last = ops.len();
                let t = g.tx(client, TxKind::Prog(ProgTx::Fetch { job_id: id, tensor: last as u8, col: 3 }));
                g.native(vec![t]);
                let vals: Vec<i8> = (0..n).map(|i| tens[last][i * n + 3]).collect();
                let t = g.tx(worker, TxKind::Prog(ProgTx::Serve { job_id: id, tensor: last as u8, col: 3, vals, path: cols_tree(&tens[last], n).proof(3) }));
                g.native(vec![t]);
                let t = g.tx(client, TxKind::Prog(ProgTx::Accept { job_id: id }));
                g.native(vec![t]);
            }
            Finish::Unserved => {
                let t = g.tx(client, TxKind::Prog(ProgTx::Fetch { job_id: id, tensor: 1, col: 5 }));
                g.native(vec![t]);
                g.native(vec![]);
                g.native(vec![]);
                let t = g.tx(challenger, TxKind::Prog(ProgTx::SlashUnserved { job_id: id }));
                g.native(vec![t]);
            }
        }
        // Z of the Gemms: recomputed by the verifier; we only export the tensors published by the worker
        data.push(json!({"x0": b64(&x0), "ws": ws.iter().map(|w| b64(w)).collect::<Vec<_>>(), "tensors": tens.iter().map(|t| b64(t)).collect::<Vec<_>>(),
                          "zs": zs.iter().map(|z| z.as_ref().map(|zz| B64.encode(zz.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()))).collect::<Vec<_>>()}));
    }

    assert!(g.chain.check_supply_invariant());
    let st = &g.chain.state;
    let status = |id: &JobId| format!("{:?}", st.progs[id].status).split(|c: char| c == ' ' || c == '{').next().unwrap().to_string();
    let jobs_json: Vec<Value> = job_ids.iter().map(|id| json!({"id": hx(id), "status": status(id), "rebate_left": st.progs[id].rebate_left, "escrow_net": st.progs[id].escrow_net, "posted": st.progs[id].posted.len()})).collect();
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
    println!("{}", serde_json::to_string(&json!({"genesis": genesis, "prog_data": data, "blocks": g.blocks, "final": fin})).unwrap());
}
