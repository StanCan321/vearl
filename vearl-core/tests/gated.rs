//! The operations added for the Qwen3 models: grouped RMS normalisation (`RmsNorm`), element-wise product of two tensors (`MulT`, the gate
//! of a SwiGLU network), element-wise product with a weight table (`MulW`, rotary position embedding) and the causal softmax with a coarser
//! exponent step (`CausalSoftmaxS`, attention logits wider than 11 nats): semantics, life cycle, fraud proofs.
use ed25519_dalek::SigningKey;
use rand::{rngs::StdRng, Rng, SeedableRng};
use vearl_core::chain::*;
use vearl_core::crypto::*;
use vearl_core::mlp::{cols_root_of, rows_root_of};
use vearl_core::prog::*;
use vearl_core::state::*;
use vearl_core::types::*;
use vearl_core::work::*;

const P: WorkParams = WorkParams::TESTNET;
const N: usize = 128;

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

// ------------------------------------------------------------------ semantics

#[test]
fn rms_normalisation_matches_the_real_one_and_has_no_mean_subtraction() {
    let mut rng = StdRng::seed_from_u64(5);
    for trial in 0..300 {
        let group = [8usize, 16, 32][rng.gen_range(0..3)];
        let groups = rng.gen_range(1..=4usize);
        let len = group * groups;
        let spread = rng.gen_range(12..=40i32);
        let col: Vec<i8> = (0..N).map(|_| rng.gen_range(-spread..=spread) as i8).collect();
        let out = rmsnorm_col(&col, len, group);
        for g in 0..groups {
            let cells = &col[g * group..(g + 1) * group];
            let rms = (cells.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / group as f64 + 1.0).sqrt();
            for (k, v) in cells.iter().enumerate() {
                let want = (*v as f64 / rms * 16.0).clamp(-64.0, 64.0);
                assert!((out[g * group + k] as f64 - want).abs() <= (1.0 + want.abs() / 2.0) / rms + 1.5, "trial {trial} group {g}: integer {} vs real {want:.2} (rms {rms:.1})", out[g * group + k]);
            }
        }
        assert!(out[len..].iter().all(|v| *v == 0), "cells beyond len are zero");
        // each group is normalised on its own: changing another group does not change this one
        let mut other = col.clone();
        other[(groups - 1) * group] = other[(groups - 1) * group].wrapping_add(7).clamp(-64, 64);
        if groups > 1 {
            assert_eq!(rmsnorm_col(&other, len, group)[..group], out[..group], "the first group does not see the last one");
        }
    }
    // no centring (this is what separates it from LayerNorm): a constant column is NOT sent to zero
    let c = rmsnorm_col(&[9, 9, 9, 9], 4, 4);
    assert!(c.iter().take(4).all(|v| *v == 16), "constant column: 9 / rms 9 * 16 = 16, got {c:?}");
    assert!(layernorm_col(&[9, 9, 9, 9], 4).iter().all(|v| *v == 0), "(LayerNorm sends it to zero)");
    // sign symmetry, saturation, zero column (no division by zero)
    let a: Vec<i8> = vec![3, -7, 12, 0, 5, -2, 9, -11];
    let neg: Vec<i8> = a.iter().map(|v| -v).collect();
    let (ra, rn) = (rmsnorm_col(&a, 8, 8), rmsnorm_col(&neg, 8, 8));
    for k in 0..8 {
        assert!((ra[k] as i32 + rn[k] as i32).abs() <= 1, "odd function up to the rounding: {} {}", ra[k], rn[k]);
    }
    assert!(rmsnorm_col(&[0, 0, 0, 0], 4, 4).iter().all(|v| *v == 0));
    assert!(rmsnorm_col(&[64, -64, 64, 1], 4, 4).iter().all(|v| v.abs() <= 64));
    // group = len is the normalisation of the whole column; group not dividing len leaves the last cells at zero (such programs are refused)
    assert_eq!(rmsnorm_col(&a, 8, 8), rmsnorm_col(&a, 8, 100));
    assert_eq!(rmsnorm_col(&a, 8, 0), vec![0; 8]);
}

#[test]
fn the_rms_output_shift_widens_the_range_and_coarsens_the_unit() {
    // a vector with one outlier at 8 sigma: shift 0 clips it at 4 sigma (64), shift 1 keeps it (8 sigma = 64), shift 2 keeps it with room
    let mut col = vec![1i8; 64];
    col[3] = 36; // rms = sqrt((63 + 1296) / 64) = 4.6 -> the outlier is 7.8 sigma
    let (o0, o1, o2) = (rmsnorm_col_s(&col, 64, 64, 0), rmsnorm_col_s(&col, 64, 64, 1), rmsnorm_col_s(&col, 64, 64, 2));
    assert_eq!(o0, rmsnorm_col(&col, 64, 64));
    assert_eq!(o0[3], 64, "saturated at 4 sigma");
    assert!(o1[3] > 60 && o1[3] <= 64, "7.8 sigma / 8 sigma * 64 = {}", o1[3]);
    assert!((o2[3] as i32 - 31).abs() <= 1, "unit 1/4 of a sigma: 7.8 * 4 = 31, got {}", o2[3]);
    // the unit halves with each shift: out(shift s) = out(0) / 2^s up to the rounding, as long as nothing saturates
    let mut rng = StdRng::seed_from_u64(31);
    for _ in 0..100 {
        let c: Vec<i8> = (0..64).map(|_| rng.gen_range(-12..=12) as i8).collect();
        let (a, b) = (rmsnorm_col_s(&c, 64, 64, 0), rmsnorm_col_s(&c, 64, 64, 2));
        for k in 0..64 {
            assert!((a[k] as i32 - 4 * b[k] as i32).abs() <= 2, "{} vs {}", a[k], b[k]);
        }
    }
}

#[test]
fn element_wise_products_use_an_arithmetic_shift_and_saturate() {
    assert_eq!(mul_apply(10, 20, 4), 12); // 200 >> 4
    assert_eq!(mul_apply(-10, 20, 4), -13, "floor, not truncation toward zero"); // -200 >> 4 = -13
    assert_eq!(mul_apply(64, 64, 0), 64, "saturation at +64");
    assert_eq!(mul_apply(-64, 64, 0), -64);
    let op = Op::MulT { a: 1, b: 2, shift: 5 };
    assert_eq!(op_inputs(&op, 3, 4, N), vec![(1, 3, 4), (2, 3, 4)]);
    assert_eq!(op_eval(&op, &[37, -41]), mul_apply(37, -41, 5));
    let w = Op::MulW { x: 1, w: 0, shift: 5 };
    assert!(op_inputs(&w, 3, 4, N).is_empty(), "MulW reads a weight entry: its own proof shape");
    let mut rng = StdRng::seed_from_u64(9);
    let mk = |rng: &mut StdRng| (0..N * N).map(|_| rng.gen_range(-64..=64) as i8).collect::<Vec<i8>>();
    let (ta, tb, tw) = (mk(&mut rng), mk(&mut rng), mk(&mut rng));
    let (_, out) = eval_op(&Op::MulT { a: 0, b: 1, shift: 6 }, &[ta.clone(), tb.clone()], &[], N);
    assert!((0..N * N).all(|e| out[e] == mul_apply(ta[e], tb[e], 6)));
    let (_, out) = eval_op(&Op::MulW { x: 0, w: 0, shift: 6 }, &[ta.clone()], &[tw.clone()], N);
    assert!((0..N * N).all(|e| out[e] == mul_apply(ta[e], tw[e], 6)));
}

#[test]
fn the_gated_program_computes_what_its_operations_say() {
    let (x0, ws, ops) = demo_gated(N, 1);
    let (tens, _) = run_program(&x0, &ws, &ops, N);
    let g = |t: usize, r: usize, c: usize| tens[t][r * N + c];
    let (d, l, ff) = (32usize, 16usize, 64usize);
    for c in 0..l {
        // T2: two groups of 16 rows normalised separately
        let col: Vec<i8> = (0..N).map(|r| g(1, r, c)).collect();
        assert_eq!((0..N).map(|r| g(2, r, c)).collect::<Vec<i8>>(), rmsnorm_col_s(&col, d, 16, 1));
        for r in 0..ff {
            // T5 = silu table, T6 = silu(gate) * up, T7 = T6 * table
            assert_eq!(g(5, r, c), lut_apply(&ws[3][..N], g(3, r, c)));
            assert_eq!(g(6, r, c), mul_apply(g(5, r, c), g(4, r, c), 6));
            assert_eq!(g(7, r, c), mul_apply(g(6, r, c), ws[4][r * N + c], 6));
        }
        let fin: Vec<i8> = (0..N).map(|r| g(9, r, c)).collect();
        assert_eq!((0..N).map(|r| g(10, r, c)).collect::<Vec<i8>>(), rmsnorm_col(&fin, d, d));
    }
    let distinct: std::collections::BTreeSet<i8> = (0..d).flat_map(|r| (0..l).map(move |c| (r, c))).map(|(r, c)| g(10, r, c)).collect();
    assert!(distinct.len() >= 8, "the signal goes through the block ({} values)", distinct.len());
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

#[derive(Clone, Copy, PartialEq)]
enum Tamper {
    None,
    Gemm(usize),
    Cell(usize), // one wrong cell in the output of a non-Gemm operation
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
    run_demo(demo_gated(N, seed), 4, tamper)
}

fn run_demo((x0, ws, ops): (Vec<i8>, Vec<Vec<i8>>, Vec<Op>), expect_gemms: u64, tamper: Tamper) -> Run {
    let mut c = new_chain();
    let x0_root = cols_root_of(&x0, N);
    let w_roots: Vec<Hash> = ws.iter().map(|w| rows_root_of(w, N)).collect();
    let nonce = c.state.nonce(&pubkey(&key(10)));
    let id = prog_id(&pubkey(&key(10)), nonce, &x0_root, &w_roots, &ops);
    let gemms = ops.iter().filter(|o| o.is_gemm()).count() as u64;
    assert_eq!(gemms, expect_gemms);
    let submit = ptx(&c, 10, ProgTx::Submit { x0_root, w_roots, ops: ops.clone(), escrow: 30 * COIN * gemms, deadline: 900 });
    mine_at(&mut c, &key(12), WorkClass::Native, None, vec![submit]).unwrap();
    let claim = ptx(&c, 11, ProgTx::Claim { job_id: id });
    mine_at(&mut c, &key(12), WorkClass::Native, None, vec![claim]).unwrap();
    let mut tens = vec![x0];
    let mut zs: Vec<Option<Vec<i32>>> = vec![];
    for (k, op) in ops.iter().enumerate() {
        let (mut z, mut out) = eval_op(op, &tens, &ws, N);
        if let Op::Gemm { w, x, shift, relu } = op {
            let cm = CommittedMatrices::new(ws[*w as usize].clone(), tens[*x as usize].clone(), &P);
            let zm = mine_at(&mut c, &key(11), WorkClass::Op { job: id, op: k as u8 }, Some(&cm), vec![]).unwrap();
            assert_eq!(&zm, z.as_ref().unwrap());
            if tamper == Tamper::Gemm(k) {
                let zz = z.as_mut().unwrap();
                zz[2 * N + 3] += 1 << *shift;
                out = zz.iter().map(|&v| act_fn(v, *shift, *relu)).collect();
            }
        } else if tamper == Tamper::Cell(k) {
            let i = out.iter().position(|v| (*v as i32) < 60 && *v != 0).expect("a non-zero cell");
            out[i] = out[i].wrapping_add(2);
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
fn an_honest_gated_block_is_verified_and_paid() {
    let mut r = run(1, Tamper::None);
    assert!(matches!(r.c.state.progs[&r.id].status, ProgStatus::Complete { .. }));
    assert!(find_fraud_proof(&r.ops, &r.ws, &r.tens, &r.zs, N).is_none(), "nothing to contest");
    let before = r.c.state.balance(&pubkey(&key(11)));
    let accept = ptx(&r.c, 10, ProgTx::Accept { job_id: r.id });
    mine_at(&mut r.c, &key(12), WorkClass::Native, None, vec![accept]).unwrap();
    assert_eq!(r.c.state.progs[&r.id].status, ProgStatus::Settled);
    let net = 120 * COIN - bps(120 * COIN, BURN_BPS);
    assert_eq!(r.c.state.balance(&pubkey(&key(11))), before + net + bps(net, BOND_BPS));
    assert!(r.c.check_supply_invariant());
}

#[test]
fn every_new_kind_of_wrong_operation_is_proven_and_slashed() {
    // (falsification, expected kind of proof)
    let cases = [
        (Tamper::Cell(1), "col"),   // grouped RMS normalisation
        (Tamper::Cell(9), "col"),   // final RMS normalisation
        (Tamper::Cell(5), "elem"),  // MulT
        (Tamper::Cell(6), "mulw"),  // MulW
        (Tamper::Cell(4), "lut"),   // table (existing operation, in the same program)
        (Tamper::Gemm(7), "gemm"),  // down projection
    ];
    for (i, (t, want)) in cases.into_iter().enumerate() {
        let mut r = run(40 + i as u64, t);
        let (op, kind) = find_fraud_proof(&r.ops, &r.ws, &r.tens, &r.zs, N).unwrap_or_else(|| panic!("case {i}: no fraud found"));
        let got = match &kind {
            ProgFraud::Gemm { .. } => "gemm",
            ProgFraud::Act { .. } => "act",
            ProgFraud::Elem { .. } => "elem",
            ProgFraud::Col { .. } => "col",
            ProgFraud::Bias { .. } => "bias",
            ProgFraud::Lut { .. } => "lut",
            ProgFraud::MulW { .. } => "mulw",
        };
        assert_eq!(got, want, "case {i}");
        let alice0 = r.c.state.balance(&pubkey(&key(10)));
        submit_fraud(&mut r, op, kind).unwrap_or_else(|e| panic!("case {i}: {e}"));
        assert_eq!(r.c.state.progs[&r.id].status, ProgStatus::Slashed, "case {i}");
        let net = 120 * COIN - bps(120 * COIN, BURN_BPS);
        assert_eq!(r.c.state.balance(&pubkey(&key(10))), alice0 + net + bps(net, BOND_BPS) / 2, "case {i}");
        assert!(r.c.check_supply_invariant());
    }
}

#[test]
fn false_accusations_and_forged_proofs_on_the_new_operations_are_rejected() {
    let mut r = run(3, Tamper::None);
    // MulW: a correct entry
    let (x, w, shift) = (6usize, 4usize, 6u8);
    let (tx, out, tw) = (&r.tens[x], &r.tens[7], &r.ws[w]);
    let (i, j) = (3usize, 5usize);
    let good = ProgFraud::MulW {
        i: i as u32,
        j: j as u32,
        in_col: (0..N).map(|q| tx[q * N + j]).collect(),
        in_path: cols_tree(tx, N).proof(j),
        w_row: tw[i * N..(i + 1) * N].to_vec(),
        w_path: rows_tree(tw, N).proof(i),
        out_col: (0..N).map(|q| out[q * N + j]).collect(),
        out_path: cols_tree(out, N).proof(j),
    };
    assert_eq!(out[i * N + j], mul_apply(tx[i * N + j], tw[i * N + j], shift));
    assert!(submit_fraud(&mut r, 6, good.clone()).is_err(), "the entry is correct");
    assert!(submit_fraud(&mut r, 5, good.clone()).is_err(), "a MulW proof does not fit a MulT");
    assert!(submit_fraud(&mut r, 1, good.clone()).is_err(), "nor a normalisation");
    // MulW: a forged output column (the openings no longer match the published root)
    let forged = match good {
        ProgFraud::MulW { i, j, in_col, in_path, w_row, w_path, mut out_col, out_path } => {
            out_col[i as usize] = out_col[i as usize].wrapping_add(1);
            ProgFraud::MulW { i, j, in_col, in_path, w_row, w_path, out_col, out_path }
        }
        _ => unreachable!(),
    };
    assert!(submit_fraud(&mut r, 6, forged).is_err(), "a forged output opening is refused");
    // RMS normalisation: a correct column
    let j = 4usize;
    let (tin, tout) = (&r.tens[1], &r.tens[2]);
    let col = ProgFraud::Col {
        j: j as u32,
        in_col: (0..N).map(|q| tin[q * N + j]).collect(),
        in_path: cols_tree(tin, N).proof(j),
        out_col: (0..N).map(|q| tout[q * N + j]).collect(),
        out_path: cols_tree(tout, N).proof(j),
    };
    assert!(submit_fraud(&mut r, 1, col.clone()).is_err(), "correct column");
    assert!(submit_fraud(&mut r, 9, col).is_err(), "wrong operation");
    // MulT: a correct entry (Elem proof with the two source columns)
    let (ta, tb, tout) = (&r.tens[5], &r.tens[4], &r.tens[6]);
    let ins = vec![
        OpenedCol { tensor: 5, col: j as u32, vals: (0..N).map(|q| ta[q * N + j]).collect(), path: cols_tree(ta, N).proof(j) },
        OpenedCol { tensor: 4, col: j as u32, vals: (0..N).map(|q| tb[q * N + j]).collect(), path: cols_tree(tb, N).proof(j) },
    ];
    let elem = ProgFraud::Elem { i: 7, j: j as u32, ins, out_col: (0..N).map(|q| tout[q * N + j]).collect(), out_path: cols_tree(tout, N).proof(j) };
    assert!(submit_fraud(&mut r, 5, elem).is_err(), "correct product");
    assert!(matches!(r.c.state.progs[&r.id].status, ProgStatus::Complete { .. }), "nothing was slashed");
    assert!(r.c.check_supply_invariant());
}

#[test]
fn the_scaled_causal_softmax_steps_the_exponent() {
    // scale 0 is CausalSoftmax
    let mut rng = StdRng::seed_from_u64(77);
    for _ in 0..200 {
        let col: Vec<i8> = (0..N).map(|_| rng.gen_range(-64..=64) as i8).collect();
        for fine in [false, true] {
            let a = col_op(&Op::CausalSoftmax { x: 1, len: 40, l: 16, fine }, 21, &col);
            let b = col_op(&Op::CausalSoftmaxS { x: 1, len: 40, l: 16, fine, scale: 0 }, 21, &col);
            assert_eq!(a, b);
        }
    }
    // one unit of the score is 2^scale units of the unscaled op: (0, -8) at scale 3 is (0, -64) at scale 0
    assert_eq!(softmax_col_s(&[0, -8, 0, 0], 2, 3), softmax_col(&[0, -64, 0, 0], 2));
    assert_eq!(softmax_lo_col_s(&[5, -3, 0, 0], 2, 2), softmax_lo_col(&[5, -27, 0, 0], 2));
    // beyond the table (shifted difference > 128) the weight is the last entry, whatever the scale
    assert_eq!(softmax_col_s(&[0, -40, 0, 0], 2, 4), softmax_col(&[0, -128, 0, 0], 2));
    // real attention: scores 0, -6 nats, -12 nats, with a score unit of ln2/8 * 4: the probabilities follow exp(-nats)
    let unit = 4.0 * 2f64.ln() / 8.0;
    let sc: Vec<i8> = [0.0f64, -6.0, -12.0, -3.0].iter().map(|n| (n / unit).round() as i8).collect();
    let p = softmax_lo_col_s(&sc, 4, 2);
    let hi = softmax_col_s(&sc, 4, 2);
    let probs: Vec<f64> = (0..4).map(|k| (hi[k] as f64 * 64.0 + p[k] as f64) / 4096.0).collect();
    let z: f64 = [0.0f64, -6.0, -12.0, -3.0].iter().map(|n: &f64| n.exp()).sum();
    for (k, n) in [0.0f64, -6.0, -12.0, -3.0].iter().enumerate() {
        assert!((probs[k] - n.exp() / z).abs() < 0.02, "key {k}: {} vs {}", probs[k], n.exp() / z);
    }
    // causal mask and padding rows are those of CausalSoftmax
    let col = vec![3i8; N];
    let out = col_op(&Op::CausalSoftmaxS { x: 1, len: 100, l: 16, fine: false, scale: 3 }, 2, &col);
    assert!(out[..3].iter().all(|v| *v > 0) && out[3..].iter().all(|v| *v == 0), "query 2 sees keys 0..=2 only");
    // the program (the demo attention with scale 2) computes what its operations say
    let (x0, ws, ops) = demo_causal_attention_scaled(N, 2);
    assert!(ops.iter().any(|o| matches!(o, Op::CausalSoftmaxS { scale: 2, .. })));
    let (tens, _) = run_program(&x0, &ws, &ops, N);
    for j in 0..32 {
        let col: Vec<i8> = (0..N).map(|i| tens[5][i * N + j]).collect();
        assert_eq!((0..N).map(|i| tens[6][i * N + j]).collect::<Vec<i8>>(), col_op(&ops[5], j, &col));
    }
}

#[test]
fn a_scaled_causal_attention_is_verified_and_a_wrong_softmax_is_slashed() {
    let honest = run_demo(demo_causal_attention_scaled(N, 6), 6, Tamper::None);
    assert!(matches!(honest.c.state.progs[&honest.id].status, ProgStatus::Complete { .. }));
    assert!(find_fraud_proof(&honest.ops, &honest.ws, &honest.tens, &honest.zs, N).is_none(), "nothing to contest");
    for (i, k) in [5usize, 6].into_iter().enumerate() {
        let mut r = run_demo(demo_causal_attention_scaled(N, 7 + i as u64), 6, Tamper::Cell(k));
        let (op, kind) = find_fraud_proof(&r.ops, &r.ws, &r.tens, &r.zs, N).expect("fraud found");
        assert!(matches!(kind, ProgFraud::Col { .. }), "a column proof");
        assert_eq!(op as usize, k);
        submit_fraud(&mut r, op, kind).unwrap();
        assert_eq!(r.c.state.progs[&r.id].status, ProgStatus::Slashed);
        assert!(r.c.check_supply_invariant());
    }
}

#[test]
fn validation_of_the_new_operations() {
    let (x0, ws, ops) = demo_gated(N, 3);
    let x0_root = cols_root_of(&x0, N);
    let w_roots: Vec<Hash> = ws.iter().map(|w| rows_root_of(w, N)).collect();
    let mut c = new_chain();
    let mut try_submit = |ops: Vec<Op>| {
        let t = ptx(&c, 10, ProgTx::Submit { x0_root, w_roots: w_roots.clone(), ops, escrow: 130 * COIN, deadline: 900 });
        mine_at(&mut c, &key(12), WorkClass::Native, None, vec![t]).is_ok()
    };
    assert!(try_submit(ops.clone()), "reference");
    let base = |extra: Op| vec![Op::Gemm { w: 0, x: 0, shift: 3, relu: false }, extra];
    assert!(try_submit(base(Op::RmsNorm { x: 1, len: 32, group: 16, shift: 0 })));
    assert!(try_submit(base(Op::RmsNorm { x: 1, len: 32, group: 32, shift: 4 })));
    assert!(!try_submit(base(Op::RmsNorm { x: 1, len: 32, group: 32, shift: 5 })), "output shift out of bounds");
    assert!(!try_submit(base(Op::RmsNorm { x: 1, len: 32, group: 12, shift: 0 })), "group does not divide len");
    assert!(!try_submit(base(Op::RmsNorm { x: 1, len: 32, group: 0, shift: 0 })), "group = 0");
    assert!(!try_submit(base(Op::RmsNorm { x: 1, len: 16, group: 32, shift: 0 })), "group > len");
    assert!(!try_submit(base(Op::RmsNorm { x: 0, len: 32, group: 16, shift: 0 })), "per-column operations read a computed tensor");
    assert!(!try_submit(base(Op::RmsNorm { x: 1, len: 129, group: 1, shift: 0 })), "len > n");
    assert!(try_submit(base(Op::MulT { a: 1, b: 1, shift: 6 })));
    assert!(!try_submit(base(Op::MulT { a: 1, b: 5, shift: 6 })), "reads a tensor that does not exist yet");
    assert!(!try_submit(base(Op::MulT { a: 1, b: 1, shift: 30 })), "shift out of bounds");
    assert!(try_submit(base(Op::MulW { x: 1, w: 1, shift: 6 })));
    assert!(!try_submit(base(Op::MulW { x: 1, w: 9, shift: 6 })), "weight block that does not exist");
    assert!(!try_submit(base(Op::MulW { x: 2, w: 1, shift: 6 })), "tensor not computed yet");
    assert!(try_submit(base(Op::CausalSoftmaxS { x: 1, len: 12, l: 16, fine: false, scale: 4 })));
    assert!(try_submit(base(Op::CausalSoftmaxS { x: 1, len: 12, l: 16, fine: true, scale: 0 })));
    assert!(!try_submit(base(Op::CausalSoftmaxS { x: 1, len: 12, l: 16, fine: false, scale: 5 })), "exponent step out of bounds");
    assert!(!try_submit(base(Op::CausalSoftmaxS { x: 0, len: 12, l: 16, fine: false, scale: 1 })), "reads the raw input");
    assert!(!try_submit(base(Op::CausalSoftmaxS { x: 1, len: 0, l: 16, fine: false, scale: 1 })), "len = 0");
    assert!(!try_submit(base(Op::CausalSoftmaxS { x: 1, len: 12, l: 129, fine: false, scale: 1 })), "l > n");
    assert!(c.check_supply_invariant());
}
