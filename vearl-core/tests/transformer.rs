//! Per-column operations (INTEGER softmax and layer normalisation), GemmAT (attention scores) and a complete
//! transformer encoder block: semantics, life cycle, fraud proofs.
use ed25519_dalek::SigningKey;
use vearl_core::chain::*;
use vearl_core::crypto::*;
use vearl_core::mlp::{cols_root_of, rows_root_of};
use vearl_core::prog::*;
use vearl_core::state::*;
use vearl_core::types::*;
use vearl_core::work::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

const P: WorkParams = WorkParams::TESTNET;
const N: usize = 128;

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

// ------------------------------------------------------------------ semantics of the per-column operations

fn float_softmax(col: &[i8], len: usize) -> Vec<f64> {
    let m = col[..len].iter().map(|v| *v as f64).fold(f64::MIN, f64::max);
    let e: Vec<f64> = col[..len].iter().map(|v| 2f64.powf(-(m - *v as f64) / 8.0)).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

#[test]
fn integer_softmax_matches_the_real_softmax_and_its_invariances() {
    let mut rng = StdRng::seed_from_u64(1);
    for trial in 0..300 {
        let len = rng.gen_range(1..=64usize);
        let col: Vec<i8> = (0..N).map(|_| rng.gen_range(-64..=64)).collect();
        let out = softmax_col(&col, len);
        let f = float_softmax(&col, len);
        for j in 0..len {
            assert!((out[j] as f64 - 64.0 * f[j]).abs() <= 1.01, "trial {trial} j={j}: integer {} vs real {:.3}", out[j], 64.0 * f[j]);
            assert!((0..=64).contains(&(out[j] as i32)));
        }
        assert!(out[len..].iter().all(|v| *v == 0), "padding cells are zero");
        let sum: i32 = out.iter().map(|v| *v as i32).sum();
        assert!((sum - 64).abs() <= (len as i32 + 1) / 2 + 1, "sum {sum} ~ 64 (rounding)");
        // the order is preserved
        for a in 0..len {
            for b in 0..len {
                if col[a] > col[b] {
                    assert!(out[a] >= out[b]);
                }
            }
        }
    }
    // invariance under translation of the logits (only the differences count)
    let base: Vec<i8> = vec![-10, 0, 5, 20, 7, -30, 12, 0];
    let shifted: Vec<i8> = base.iter().map(|v| v + 15).collect();
    assert_eq!(softmax_col(&base, 8), softmax_col(&shifted, 8));
    // maximal gap: almost one-hot output; equality: uniform output
    let mut hot = vec![-64i8; 8];
    hot[3] = 64;
    assert_eq!(&softmax_col(&hot, 8)[..8], &[0, 0, 0, 64, 0, 0, 0, 0]);
    assert!(softmax_col(&[7, 7, 7, 7], 4).iter().take(4).all(|v| *v == 16));
    assert_eq!(softmax_col(&[5, 6], 0), vec![0, 0], "len = 0: everything is zero");
}

#[test]
fn integer_sqrt_is_exact() {
    for x in 0..70_000u64 {
        let r = isqrt(x);
        assert!(r * r <= x && (r + 1) * (r + 1) > x, "isqrt({x}) = {r}");
    }
    for x in [u32::MAX as u64, u32::MAX as u64 + 1, u64::MAX, (1u64 << 62) + 12345, 4_294_967_295u64 * 4_294_967_295u64] {
        let r = isqrt(x);
        assert!((r as u128) * (r as u128) <= x as u128 && ((r + 1) as u128) * ((r + 1) as u128) > x as u128, "isqrt({x}) = {r}");
    }
}

#[test]
fn integer_layernorm_matches_the_real_normalisation_and_its_invariances() {
    let mut rng = StdRng::seed_from_u64(2);
    for trial in 0..300 {
        let len = rng.gen_range(4..=96usize);
        // columns of reasonable dynamics: when the standard deviation is of the order of 1-2 units, the int8 QUANTISATION dominates
        // (documented limit of the integer semantics), so we only compare from sd ~ 7
        let spread = rng.gen_range(12..=40i32);
        let col: Vec<i8> = (0..N).map(|_| rng.gen_range(-spread..=spread) as i8).collect();
        let out = layernorm_col(&col, len);
        let mu = col[..len].iter().map(|v| *v as f64).sum::<f64>() / len as f64;
        let sd = (col[..len].iter().map(|v| (*v as f64 - mu).powi(2)).sum::<f64>() / len as f64 + 1.0).sqrt();
        for j in 0..len {
            let want = (((col[j] as f64 - mu) / sd) * 16.0).clamp(-64.0, 64.0);
            // errors due to the INTEGER SEMANTICS: rounded mean (<= 8/sd), rounded standard deviation (<= |output|/(2 sd)), + roundings
            assert!((out[j] as f64 - want).abs() <= (8.0 + want.abs() / 2.0) / sd + 2.0, "trial {trial} j={j}: integer {} vs real {want:.2} (sd {sd:.1})", out[j]);
        }
        assert!(out[len..].iter().all(|v| *v == 0));
        let sum: i32 = out.iter().map(|v| *v as i32).sum();
        assert!(sum.abs() as f64 <= len as f64 * (8.0 / sd + 0.6) + 2.0, "output of mean ~0: sum {sum} (len {len}, sd {sd:.1})");
    }
    assert!(layernorm_col(&[9, 9, 9, 9], 4).iter().all(|v| *v == 0), "constant column: zeros (no division by zero)");
    assert!(layernorm_col(&[-64, 64, -64, 64], 4).iter().all(|v| v.abs() <= 64), "saturation a [-64,64]");
    // invariance under a constant shift of the inputs
    let a: Vec<i8> = vec![3, -7, 12, 0, 5, -2];
    let b: Vec<i8> = a.iter().map(|v| v + 20).collect();
    assert_eq!(layernorm_col(&a, 6), layernorm_col(&b, 6));
}

// ------------------------------------------------------------------ attention: the program = an independent naive computation

#[test]
fn the_program_computes_exactly_the_naive_attention() {
    let (x0, ws, ops) = demo_transformer(N, 1);
    let (tens, _) = run_program(&x0, &ws, &ops, N);
    let (d, l) = (32usize, 16usize);
    let g = |t: usize, r: usize, c: usize| tens[t][r * N + c] as i32;
    // scores S[i][j] = (sum_k K[k][i] * Q[k][j]) >> 5   (T2 = K, T1 = Q, T4 = S)
    for i in 0..l {
        for j in 0..l {
            let z: i32 = (0..d).map(|k| g(2, k, i) * g(1, k, j)).sum();
            assert_eq!(g(4, i, j), (z >> 5).clamp(-64, 64), "S[{i}][{j}]");
        }
    }
    // per-column attention (softmax over the keys); the columns >= l are padding
    for j in 0..l {
        let col: Vec<i8> = (0..N).map(|i| tens[4][i * N + j]).collect();
        let a = softmax_col(&col, l);
        for i in 0..N {
            assert_eq!(tens[5][i * N + j], a[i], "A[{i}][{j}]");
        }
    }
    // output O[k][j] = (sum_i V[k][i] * A[i][j]) >> 6   (T3 = V, T5 = A, T7 = O)
    for k in 0..d {
        for j in 0..l {
            let z: i32 = (0..l).map(|i| g(3, k, i) * g(5, i, j)).sum();
            assert_eq!(g(7, k, j), (z >> 6).clamp(-64, 64), "O[{k}][{j}]");
        }
    }
    // normalised output column: mean ~ 0 over the 32 features
    for j in 0..l {
        let s: i32 = (0..d).map(|r| g(14, r, j)).sum();
        assert!(s.abs() <= d as i32 + 2, "final LayerNorm, token {j}: sum {s}");
    }
    // the signal goes through the whole block
    let distinct: std::collections::BTreeSet<i8> = (0..d).flat_map(|r| (0..l).map(move |c| (r, c))).map(|(r, c)| tens[14][r * N + c]).collect();
    assert!(distinct.len() >= 8, "varied output ({} values)", distinct.len());
}

// ------------------------------------------------------------------ life cycle on the chain

fn new_chain() -> Chain {
    let alloc: Vec<(PubKey, u64)> = [10u8, 11, 13].iter().map(|s| (pubkey(&key(*s)), 1000 * COIN)).collect();
    Chain::genesis(P, &alloc, 1_000, u64::MAX / 64)
}

fn mine_at(c: &mut Chain, sk: &SigningKey, class: WorkClass, data: Option<&CommittedMatrices>, txs: Vec<SignedTx>) -> Result<Vec<i32>, String> {
    let t = c.tip_ts() + BLOCK_TIME;
    let out = c.mine(sk, class, data, txs, t)?;
    let r = out.c.clone();
    c.validate_and_apply(out.block)?;
    Ok(r)
}

fn ptx(c: &Chain, who: u8, m: ProgTx) -> SignedTx {
    make_tx(c, &key(who), 0, TxKind::Prog(m))
}

/// Matrices (A, B) mined for a GEMM operation: A = W_w or T_a^T, B = T_x.
fn gemm_matrices(op: &Op, tens: &[Vec<i8>], ws: &[Vec<i8>]) -> (Vec<i8>, Vec<i8>) {
    match op {
        Op::Gemm { w, x, .. } => (ws[*w as usize].clone(), tens[*x as usize].clone()),
        Op::GemmAT { a, x, .. } => {
            let ta = &tens[*a as usize];
            ((0..N * N).map(|idx| ta[(idx % N) * N + idx / N]).collect(), tens[*x as usize].clone())
        }
        _ => unreachable!(),
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Tamper {
    None,
    Gemm(usize),
    Col(usize),
}

struct Run {
    c: Chain,
    id: JobId,
    ws: Vec<Vec<i8>>,
    ops: Vec<Op>,
    tens: Vec<Vec<i8>>,
    zs: Vec<Option<Vec<i32>>>,
}

fn run(seed: u64, tamper: Tamper) -> Run {
    let (x0, ws, ops) = demo_transformer(N, seed);
    let mut c = new_chain();
    let x0_root = cols_root_of(&x0, N);
    let w_roots: Vec<Hash> = ws.iter().map(|w| rows_root_of(w, N)).collect();
    let nonce = c.state.nonce(&pubkey(&key(10)));
    let id = prog_id(&pubkey(&key(10)), nonce, &x0_root, &w_roots, &ops);
    let gemms = ops.iter().filter(|o| o.is_gemm()).count() as u64;
    assert_eq!(gemms, 8);
    let submit = ptx(&c, 10, ProgTx::Submit { x0_root, w_roots, ops: ops.clone(), escrow: 30 * COIN * gemms, deadline: 900 });
    mine_at(&mut c, &key(12), WorkClass::Native, None, vec![submit]).unwrap();
    let claim = ptx(&c, 11, ProgTx::Claim { job_id: id });
    mine_at(&mut c, &key(12), WorkClass::Native, None, vec![claim]).unwrap();

    let mut tens = vec![x0];
    let mut zs: Vec<Option<Vec<i32>>> = vec![];
    for (k, op) in ops.iter().enumerate() {
        let (mut z, mut out) = eval_op(op, &tens, &ws, N);
        if let Op::Gemm { shift, relu, .. } | Op::GemmAT { shift, relu, .. } = op {
            // the mined block computes the true GEMM on the committed matrices (W or T_a^T, and T_x)
            let (am, bm) = gemm_matrices(op, &tens, &ws);
            let cm = CommittedMatrices::new(am, bm, &P);
            let zm = mine_at(&mut c, &key(11), WorkClass::Op { job: id, op: k as u8 }, Some(&cm), vec![]).unwrap();
            assert_eq!(&zm, z.as_ref().unwrap(), "the Op block {k} produces the true GEMM");
            if tamper == Tamper::Gemm(k) {
                let zz = z.as_mut().unwrap();
                zz[2 * N + 3] += 1 << *shift;
                out = zz.iter().map(|&v| act_fn(v, *shift, *relu)).collect();
            }
        } else if tamper == Tamper::Col(k) {
            out[0] = out[0].wrapping_add(1);
        }
        let z_root = z.as_ref().map_or([0u8; 32], |zz| c_tree(zz, N).root());
        let post = ptx(&c, 11, ProgTx::PostOp { job_id: id, op: k as u8, z_root, x_root: cols_root_of(&out, N) });
        mine_at(&mut c, &key(12), WorkClass::Native, None, vec![post]).unwrap();
        zs.push(z);
        tens.push(out);
    }
    Run { c, id, ws, ops, tens, zs }
}

fn submit_fraud(r: &mut Run, op: u8, kind: ProgFraud) -> Result<Vec<i32>, String> {
    let f = ptx(&r.c, 10, ProgTx::Fraud { job_id: r.id, op, kind });
    mine_at(&mut r.c, &key(12), WorkClass::Native, None, vec![f])
}

#[test]
fn honest_transformer_block_is_verified_and_paid() {
    let mut r = run(1, Tamper::None);
    assert!(matches!(r.c.state.progs[&r.id].status, ProgStatus::Complete { .. }));
    assert!(find_fraud_proof(&r.ops, &r.ws, &r.tens, &r.zs, N).is_none(), "nothing to contest");
    let before = r.c.state.balance(&pubkey(&key(11)));
    let accept = ptx(&r.c, 10, ProgTx::Accept { job_id: r.id });
    mine_at(&mut r.c, &key(12), WorkClass::Native, None, vec![accept]).unwrap();
    assert_eq!(r.c.state.progs[&r.id].status, ProgStatus::Settled);
    let net = 240 * COIN - bps(240 * COIN, BURN_BPS);
    assert_eq!(r.c.state.balance(&pubkey(&key(11))), before + net + bps(net, BOND_BPS));
    assert!(r.c.check_supply_invariant());
}

#[test]
fn every_kind_of_wrong_transformer_operation_is_proven_and_slashed() {
    // (falsification, expected proof kind)
    let cases = [
        (Tamper::Gemm(3), "gemm"), // attention scores K^T Q (GemmAT)
        (Tamper::Gemm(6), "gemm"), // output V.A (GemmAT)
        (Tamper::Gemm(0), "gemm"), // projection Q (Gemm)
        (Tamper::Col(4), "col"),   // softmax
        (Tamper::Col(9), "col"),   // first normalisation
        (Tamper::Col(13), "col"),  // final normalisation
    ];
    for (i, (t, want)) in cases.into_iter().enumerate() {
        let mut r = run(30 + i as u64, t);
        let (op, kind) = find_fraud_proof(&r.ops, &r.ws, &r.tens, &r.zs, N).unwrap_or_else(|| panic!("case {i}: no fraud found"));
        let got = match &kind {
            ProgFraud::Gemm { .. } => "gemm",
            ProgFraud::Act { .. } => "act",
            ProgFraud::Elem { .. } => "elem",
            ProgFraud::Col { .. } => "col",
            ProgFraud::Bias { .. } => "bias",
            ProgFraud::Lut { .. } => "lut",
        };
        assert_eq!(got, want, "cas {i}");
        let alice0 = r.c.state.balance(&pubkey(&key(10)));
        submit_fraud(&mut r, op, kind).unwrap_or_else(|e| panic!("case {i}: {e}"));
        assert_eq!(r.c.state.progs[&r.id].status, ProgStatus::Slashed, "cas {i}");
        let net = 240 * COIN - bps(240 * COIN, BURN_BPS);
        assert_eq!(r.c.state.balance(&pubkey(&key(10))), alice0 + net + bps(net, BOND_BPS) / 2, "cas {i}");
        assert!(r.c.check_supply_invariant());
    }
}

#[test]
fn false_accusations_on_attention_operations_are_rejected() {
    let mut r = run(2, Tamper::None);
    // correct softmax column
    let j = 5usize;
    let tx = &r.tens[4];
    let (in_col, out_col): (Vec<i8>, Vec<i8>) = ((0..N).map(|i| tx[i * N + j]).collect(), (0..N).map(|i| r.tens[5][i * N + j]).collect());
    let col = ProgFraud::Col { j: j as u32, in_col: in_col.clone(), in_path: cols_tree(&r.tens[4], N).proof(j), out_col: out_col.clone(), out_path: cols_tree(&r.tens[5], N).proof(j) };
    assert!(submit_fraud(&mut r, 4, col.clone()).is_err(), "correct column");
    assert!(submit_fraud(&mut r, 3, col.clone()).is_err(), "proof kind unsuitable for a GemmAT");
    assert!(submit_fraud(&mut r, 9, col).is_err(), "wrong operation (other input tensor)");
    // GemmAT: correct entry of the scores
    let (z, ta, tx) = (r.zs[3].clone().unwrap(), r.tens[2].clone(), r.tens[1].clone());
    let (i, jj) = (4usize, 7usize);
    let gemm = ProgFraud::Gemm {
        i: i as u32, j: jj as u32,
        w_row: (0..N).map(|k| ta[k * N + i]).collect(), // row i of T_a^T = column i of T_a
        w_path: cols_tree(&ta, N).proof(i),
        x_col: (0..N).map(|k| tx[k * N + jj]).collect(), x_path: cols_tree(&tx, N).proof(jj),
        z_row: row_of(&z, N, i), z_path: c_tree(&z, N).proof(i),
    };
    assert!(submit_fraud(&mut r, 3, gemm).is_err(), "correct entry of the GemmAT");
    assert!(matches!(r.c.state.progs[&r.id].status, ProgStatus::Complete { .. }));
}

#[test]
fn validation_of_attention_and_row_operations() {
    let (x0, ws, ops) = demo_transformer(N, 3);
    let x0_root = cols_root_of(&x0, N);
    let w_roots: Vec<Hash> = ws.iter().map(|w| rows_root_of(w, N)).collect();
    let mut c = new_chain();
    let mut try_submit = |ops: Vec<Op>| {
        let t = ptx(&c, 10, ProgTx::Submit { x0_root, w_roots: w_roots.clone(), ops, escrow: 300 * COIN, deadline: 900 });
        mine_at(&mut c, &key(12), WorkClass::Native, None, vec![t]).is_ok()
    };
    assert!(try_submit(ops.clone()), "reference");
    let base = |extra: Op| vec![Op::Gemm { w: 0, x: 0, shift: 3, relu: false }, extra];
    assert!(try_submit(base(Op::Softmax { x: 1, len: 16 })));
    assert!(!try_submit(base(Op::Softmax { x: 0, len: 16 })), "per-column operations read a COMPUTED tensor (not the raw input)");
    assert!(!try_submit(base(Op::Softmax { x: 1, len: 0 })), "len = 0");
    assert!(!try_submit(base(Op::LayerNorm { x: 1, len: 129 })), "len > n");
    assert!(!try_submit(base(Op::Softmax { x: 5, len: 16 })), "tensor not yet computed");
    assert!(!try_submit(base(Op::GemmAT { a: 5, x: 1, shift: 3, relu: false })), "GemmAT reads a future tensor");
    assert!(!try_submit(base(Op::GemmAT { a: 1, x: 1, shift: 30, relu: false })), "shift out of bounds");
    assert!(try_submit(base(Op::GemmAT { a: 1, x: 1, shift: 3, relu: false })));
    // a program whose only GEMM is a GemmAT counts for the price floor
    assert!(!try_submit(vec![Op::Softmax { x: 0, len: 4 }]), "no GEMM to mine");
    assert!(c.check_supply_invariant());
}
