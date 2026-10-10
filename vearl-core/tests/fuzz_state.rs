//! Fuzz of transaction application (market, inference, blocks, tensor programs): VALID transactions,
//! taken at various moments of real life cycles, are mutated byte by byte (lengths, indices, roots, paths,
//! identifiers), RE-signed with a valid key (otherwise the signature check discards them before any logic) then
//! applied to the state. Two properties:
//!   1. NO panic: a hostile peer must never be able to crash a node with a forged transaction;
//!   2. CONSERVATION: if a mutated transaction is accepted, balances + escrows + bonds + burned == genesis + emission.
use ed25519_dalek::SigningKey;
use vearl_core::chain::*;
use vearl_core::crypto::*;
use vearl_core::mlp::*;
use vearl_core::prog::*;
use vearl_core::state::*;
use vearl_core::tiled::*;
use vearl_core::types::*;
use vearl_core::work::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use bincode::Options;
use std::panic::{catch_unwind, AssertUnwindSafe};

const P: WorkParams = WorkParams::TESTNET;
const N: usize = 128;

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

fn new_chain() -> Chain {
    let alloc: Vec<(PubKey, u64)> = [10u8, 11, 12, 13].iter().map(|s| (pubkey(&key(*s)), 1000 * COIN)).collect();
    Chain::genesis(P, &alloc, 1_000, u64::MAX / 64)
}

/// Records the successive states of a life cycle and the transaction kinds encountered.
struct Rec {
    name: &'static str,
    snaps: Vec<(State, u64, u64)>,
    kinds: Vec<TxKind>,
    gs: u64,
}

impl Rec {
    fn new(name: &'static str, c: &Chain) -> Rec {
        let mut r = Rec { name, snaps: vec![], kinds: vec![], gs: c.genesis_supply };
        r.snap(c);
        r
    }
    fn snap(&mut self, c: &Chain) {
        self.snaps.push((c.state.clone(), c.height(), c.tip_ts()));
    }
    fn mine(&mut self, c: &mut Chain, who: u8, class: WorkClass, cm: Option<&CommittedMatrices>, txs: Vec<SignedTx>) -> Vec<i32> {
        for t in &txs {
            self.kinds.push(t.tx.kind.clone());
        }
        let ts = c.tip_ts() + BLOCK_TIME;
        let out = c.mine(&key(who), class, cm, txs, ts).expect("mine");
        let r = out.c.clone();
        c.validate_and_apply(out.block).expect("apply");
        self.snap(c);
        r
    }
}

fn tx(c: &Chain, who: u8, kind: TxKind) -> SignedTx {
    make_tx(c, &key(who), 0, kind)
}

fn rand_matrix(rows: usize, cols: usize, seed: u64) -> Vec<i8> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..rows * cols).map(|_| rng.gen_range(-64..=64) as i8).collect()
}

// ---------------- 4 families of life cycles ----------------

fn market() -> Rec {
    let mut c = new_chain();
    let mut r = Rec::new("market", &c);
    let mut rng = StdRng::seed_from_u64(1);
    let mut m = || (0..N * N).map(|_| rng.gen_range(-64..=64) as i8).collect::<Vec<i8>>();
    let (a, b) = (m(), m());
    let cm = CommittedMatrices::new(a, b, &P);
    let id = job_id(&pubkey(&key(10)), 0, &cm.a_root(), &cm.b_root());
    let t = tx(&c, 10, TxKind::SubmitJobAuction { a_root: cm.a_root(), b_root: cm.b_root(), escrow: 50 * COIN, deadline: 900, auction_secs: 100 });
    r.mine(&mut c, 12, WorkClass::Native, None, vec![t]);
    let t = tx(&c, 11, TxKind::Bid { job_id: id, price: 30 * COIN });
    r.mine(&mut c, 12, WorkClass::Native, None, vec![t]);
    for _ in 0..4 {
        r.mine(&mut c, 12, WorkClass::Native, None, vec![]);
    }
    let t = tx(&c, 11, TxKind::ClaimJob { job_id: id });
    r.mine(&mut c, 12, WorkClass::Native, None, vec![t]);
    let mut cbad = plain_gemm(&cm.a, &cm.b, N);
    cbad[5 * N + 7] += 1;
    let tb = c_tree(&cbad, N);
    let t = tx(&c, 11, TxKind::PostResult { job_id: id, c_root: tb.root() });
    r.mine(&mut c, 12, WorkClass::Native, None, vec![t]);
    let t = tx(&c, 10, TxKind::Fetch { job_id: id, i: 2, j: 3 });
    r.mine(&mut c, 12, WorkClass::Native, None, vec![t]);
    let t = tx(&c, 11, TxKind::Serve { job_id: id, i: 2, j: 3, row: row_of(&cbad, N, 2), path: tb.proof(2) });
    r.mine(&mut c, 12, WorkClass::Native, None, vec![t]);
    // additional tx seeds (never applied in the cycle): Fraud, Accept, Settle, SlashUnserved, Transfer
    let (i, j) = (5usize, 7usize);
    let fraud = FraudProof { job_id: id, i: i as u32, j: j as u32, a_row: cm.a[i * N..(i + 1) * N].to_vec(), a_path: cm.a_tree.proof(i), b_col: cm.col(j, N), b_path: cm.b_tree.proof(j), c_row: row_of(&cbad, N, i), c_path: tb.proof(i) };
    r.kinds.push(TxKind::Fraud(fraud));
    r.kinds.push(TxKind::Accept { job_id: id });
    r.kinds.push(TxKind::Settle { job_id: id });
    r.kinds.push(TxKind::SlashUnserved { job_id: id });
    r.kinds.push(TxKind::Transfer { to: pubkey(&key(13)), amount: 5 * COIN });
    r
}

fn mlp_family() -> Rec {
    let layers = 2usize;
    let mut c = new_chain();
    let mut r = Rec::new("inference", &c);
    let mut rng = StdRng::seed_from_u64(2);
    let x0: Vec<i8> = (0..N * N).map(|_| rng.gen_range(0..=64) as i8).collect();
    let ws: Vec<Vec<i8>> = (0..layers).map(|_| (0..N * N).map(|_| rng.gen_range(-8..=8) as i8).collect()).collect();
    let (x0_root, w_roots): (Hash, Vec<Hash>) = (cols_root_of(&x0, N), ws.iter().map(|w| rows_root_of(w, N)).collect());
    let id = mlp_id(&pubkey(&key(10)), 0, &x0_root, &w_roots);
    let t = tx(&c, 10, TxKind::Mlp(MlpTx::Submit { x0_root, w_roots, shifts: vec![9; layers], escrow: 60 * COIN, deadline: 500 }));
    r.mine(&mut c, 12, WorkClass::Native, None, vec![t]);
    let t = tx(&c, 11, TxKind::Mlp(MlpTx::Claim { job_id: id }));
    r.mine(&mut c, 12, WorkClass::Native, None, vec![t]);
    let mut xprev = x0;
    for l in 0..layers {
        let cm = CommittedMatrices::new(ws[l].clone(), xprev.clone(), &P);
        let z = plain_gemm(&ws[l], &xprev, N);
        let x = requantize(&z, 9, l + 1 < layers);
        let post = tx(&c, 11, TxKind::Mlp(MlpTx::PostLayer { job_id: id, layer: l as u8, z_root: c_tree(&z, N).root(), x_root: cols_root_of(&x, N) }));
        r.mine(&mut c, 11, WorkClass::Layer { job: id, layer: l as u8 }, Some(&cm), vec![post]);
        xprev = x;
    }
    r.kinds.push(TxKind::Mlp(MlpTx::Accept { job_id: id }));
    r.kinds.push(TxKind::Mlp(MlpTx::Settle { job_id: id }));
    r
}

fn tiled_family() -> Rec {
    let dims = (1usize, 2usize, 1usize);
    let (bi, bk, bj) = dims;
    let (a, b) = (rand_matrix(bi * N, bk * N, 3), rand_matrix(bk * N, bj * N, 4));
    let (ar, br) = (a_block_roots(&a, bk * N, bi, bk, N), b_block_roots(&b, bj * N, bk, bj, N));
    let mut c = new_chain();
    let mut r = Rec::new("blocks", &c);
    let d8 = (bi as u8, bk as u8, bj as u8);
    let id = tiled_id(&pubkey(&key(10)), 0, d8, &ar, &br);
    let t = tx(&c, 10, TxKind::Tiled(TiledTx::Submit { dims: d8, a_roots: ar, b_roots: br, escrow: 20 * COIN * 2, deadline: 900 }));
    r.mine(&mut c, 12, WorkClass::Native, None, vec![t]);
    let t = tx(&c, 11, TxKind::Tiled(TiledTx::Claim { job_id: id }));
    r.mine(&mut c, 12, WorkClass::Native, None, vec![t]);
    let (products, sums) = compute_all(&a, &b, dims, N);
    let t = tx(&c, 11, TxKind::Tiled(TiledTx::Post { job_id: id, product_roots: products.iter().map(|p| c_tree(p, N).root()).collect(), c_roots: sums.iter().map(|s| c_tree(s, N).root()).collect() }));
    r.mine(&mut c, 12, WorkClass::Native, None, vec![t]);
    r.kinds.push(TxKind::Tiled(TiledTx::Accept { job_id: id }));
    r.kinds.push(TxKind::Tiled(TiledTx::Settle { job_id: id }));
    r
}

/// Tensor program (CNN with a residual block); `tamper` = possible falsification of the output of an operation.
fn prog_family(name: &'static str, tamper: Option<usize>) -> Rec {
    let (x0, ws, ops) = demo_cnn(N, 5);
    let mut c = new_chain();
    let mut r = Rec::new(name, &c);
    let x0_root = cols_root_of(&x0, N);
    let w_roots: Vec<Hash> = ws.iter().map(|w| rows_root_of(w, N)).collect();
    let id = prog_id(&pubkey(&key(10)), 0, &x0_root, &w_roots, &ops);
    let gemms = ops.iter().filter(|o| o.is_gemm()).count() as u64;
    let t = tx(&c, 10, TxKind::Prog(ProgTx::Submit { x0_root, w_roots, ops: ops.clone(), escrow: 30 * COIN * gemms, deadline: 900 }));
    r.mine(&mut c, 12, WorkClass::Native, None, vec![t]);
    let t = tx(&c, 11, TxKind::Prog(ProgTx::Claim { job_id: id }));
    r.mine(&mut c, 12, WorkClass::Native, None, vec![t]);
    let mut tens = vec![x0];
    let mut zs: Vec<Option<Vec<i32>>> = vec![];
    for (k, op) in ops.iter().enumerate() {
        let (z, mut out) = eval_op(op, &tens, &ws, N);
        if let Op::Gemm { w, x, .. } = op {
            let cm = CommittedMatrices::new(ws[*w as usize].clone(), tens[*x as usize].clone(), &P);
            r.mine(&mut c, 11, WorkClass::Op { job: id, op: k as u8 }, Some(&cm), vec![]);
        }
        if tamper == Some(k) {
            out[0] = out[0].wrapping_add(1);
        }
        let z_root = z.as_ref().map_or([0u8; 32], |zz| c_tree(zz, N).root());
        let post = tx(&c, 11, TxKind::Prog(ProgTx::PostOp { job_id: id, op: k as u8, z_root, x_root: cols_root_of(&out, N) }));
        r.mine(&mut c, 12, WorkClass::Native, None, vec![post]);
        zs.push(z);
        tens.push(out);
    }
    if let Some((op, kind)) = find_fraud_proof(&ops, &ws, &tens, &zs, N) {
        r.kinds.push(TxKind::Prog(ProgTx::Fraud { job_id: id, op, kind }));
    }
    r.kinds.push(TxKind::Prog(ProgTx::Accept { job_id: id }));
    r.kinds.push(TxKind::Prog(ProgTx::Settle { job_id: id }));
    r
}

// ---------------- mutations ----------------

fn mutate(bytes: &mut Vec<u8>, rng: &mut StdRng) {
    if bytes.is_empty() {
        return;
    }
    for _ in 0..rng.gen_range(1..=3) {
        let len = bytes.len();
        if len == 0 {
            return;
        }
        match rng.gen_range(0..8) {
            0 => bytes[rng.gen_range(0..len)] ^= 1 << rng.gen_range(0..8),
            1 => bytes[rng.gen_range(0..len)] = rng.gen(),
            2 => {
                // 8 bytes at 0xFF: gigantic lengths / indices / amounts
                let p = rng.gen_range(0..len);
                for b in bytes.iter_mut().skip(p).take(8) {
                    *b = 0xFF;
                }
            }
            3 => {
                let p = rng.gen_range(0..len);
                for b in bytes.iter_mut().skip(p).take(8) {
                    *b = 0;
                }
            }
            4 => bytes.truncate(rng.gen_range(0..len)),
            5 => {
                let extra: Vec<u8> = (0..rng.gen_range(1..40)).map(|_| rng.gen()).collect();
                bytes.extend(extra);
            }
            6 if len > 70 => {
                // swap of two 32-byte windows (identifiers, roots)
                let (p, q) = (rng.gen_range(0..len - 32), rng.gen_range(0..len - 32));
                for k in 0..32 {
                    bytes.swap(p + k, q + k);
                }
            }
            _ => {
                let p = rng.gen_range(0..len);
                bytes[p] = bytes[p].wrapping_add(rng.gen_range(1..=3));
            }
        }
    }
}

struct Stats {
    tried: usize,
    decoded: usize,
    applied: usize,
    panics: Vec<String>,
    broken: Vec<String>,
}

fn fuzz(rec: &Rec, per_pair: usize, seed: u64, st: &mut Stats) {
    let mut rng = StdRng::seed_from_u64(seed);
    let codec = bincode::options().with_fixint_encoding().with_limit(4 << 20);
    let signers = [10u8, 11, 12, 13];
    for (state, height, ts) in &rec.snaps {
        for kind in &rec.kinds {
            let orig = bincode::serialize(kind).unwrap();
            for m in 0..per_pair {
                let mut bytes = orig.clone();
                if m > 0 {
                    mutate(&mut bytes, &mut rng);
                }
                st.tried += 1;
                let Ok(k) = codec.deserialize::<TxKind>(&bytes) else { continue };
                st.decoded += 1;
                let who = signers[rng.gen_range(0..signers.len())];
                let from = pubkey(&key(who));
                let t = SignedTx::new(Tx { from, nonce: state.nonce(&from), fee: if rng.gen_bool(0.9) { 0 } else { rng.gen_range(0..1000) }, kind: k.clone() }, &key(who));
                let mut s2 = state.clone();
                let miner = pubkey(&key(12));
                match catch_unwind(AssertUnwindSafe(|| s2.apply_tx(&t, height + 1, ts + BLOCK_TIME, &miner, &P))) {
                    Err(_) => st.panics.push(format!("[{}] PANIC on {:?}", rec.name, format!("{k:?}").chars().take(300).collect::<String>())),
                    Ok(Ok(())) => {
                        st.applied += 1;
                        if s2.total_value() + s2.burned != rec.gs + s2.minted {
                            st.broken.push(format!("[{}] CONSERVATION broken by {:?}", rec.name, format!("{k:?}").chars().take(300).collect::<String>()));
                        }
                    }
                    Ok(Err(_)) => {}
                }
            }
        }
    }
}

#[test]
fn mutated_valid_transactions_never_panic_and_never_break_conservation() {
    let fams = [market(), mlp_family(), tiled_family(), prog_family("program", None), prog_family("program (op 6 wrong)", Some(6)), prog_family("program (op 4 wrong)", Some(4))];
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| { if std::thread::current().name().map_or(false, |n| n.contains("mutated")) && std::env::var("FUZZ_VERBOSE").is_err() { return; } default_hook(info); }));
    let mut st = Stats { tried: 0, decoded: 0, applied: 0, panics: vec![], broken: vec![] };
    for (i, f) in fams.iter().enumerate() {
        let per_pair = std::env::var("FUZZ_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(60);
        fuzz(f, per_pair, 1000 + i as u64 + std::env::var("FUZZ_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(0), &mut st);
    }
    
    println!("fuzz: {} mutants, {} decoded, {} accepted (including the original), {} panics, {} conservation violations", st.tried, st.decoded, st.applied, st.panics.len(), st.broken.len());
    for p in st.panics.iter().take(5).chain(st.broken.iter().take(5)) {
        println!("{p}");
    }
    assert!(st.panics.is_empty(), "{} panics: {:?}", st.panics.len(), st.panics.iter().take(3).collect::<Vec<_>>());
    assert!(st.broken.is_empty(), "{:?}", st.broken.iter().take(3).collect::<Vec<_>>());
    assert!(st.decoded > st.tried / 10, "too few decodable mutants for the test to be significant");
    assert!(st.applied > 50, "too few accepted transactions: the fuzz does not reach the deep logic");
}
