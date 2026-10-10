//! Operation `Bias` (bias vector = one row of a weight block): exact integer semantics, complete life cycle of a small MLP
//! WITH biases, `Bias` fraud proof (openings of the input column, of the bias row and of the output column).
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

fn mine_at(c: &mut Chain, sk: &SigningKey, class: WorkClass, data: Option<&CommittedMatrices>, txs: Vec<SignedTx>) -> Result<(), String> {
    let t = c.tip_ts() + BLOCK_TIME;
    let out = c.mine(sk, class, data, txs, t)?;
    c.validate_and_apply(out.block)
}

fn ptx(c: &Chain, who: u8, m: ProgTx) -> SignedTx {
    make_tx(c, &key(who), 0, TxKind::Prog(m))
}

/// MLP 16 -> 24 -> 10 with biases: Gemm, Bias(relu), Gemm, Bias. Weights: W0, W1 then the BIAS TABLE (row 0: layer 1, row 1: layer 2).
fn program(seed: u64) -> (Vec<i8>, Vec<Vec<i8>>, Vec<Op>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x0 = vec![0i8; N * N];
    for r in 0..16 {
        for c in 0..32 {
            x0[r * N + c] = rng.gen_range(0..=64) as i8;
        }
    }
    let mut mk = |rows: usize, cols: usize| {
        let mut w = vec![0i8; N * N];
        for r in 0..rows {
            for c in 0..cols {
                w[r * N + c] = rng.gen_range(-6..=6) as i8;
            }
        }
        w
    };
    let (w0, w1) = (mk(24, 16), mk(10, 24));
    let mut bias = vec![0i8; N * N];
    for i in 0..24 {
        bias[i] = rng.gen_range(-20..=20) as i8;
    }
    for i in 0..10 {
        bias[N + i] = rng.gen_range(-20..=20) as i8;
    }
    let ops = vec![
        Op::Gemm { w: 0, x: 0, shift: 6, relu: false },
        Op::Bias { x: 1, w: 2, row: 0, relu: true },
        Op::Gemm { w: 1, x: 2, shift: 6, relu: false },
        Op::Bias { x: 3, w: 2, row: 1, relu: false },
    ];
    (x0, vec![w0, w1, bias], ops)
}

#[test]
fn bias_semantics_are_exact_saturating_and_per_row() {
    let (x0, ws, _) = program(1);
    let op = Op::Bias { x: 0, w: 2, row: 0, relu: false };
    let (z, out) = eval_op(&op, &[x0.clone()], &ws, N);
    assert!(z.is_none());
    for i in 0..N {
        for j in 0..N {
            let want = (x0[i * N + j] as i32 + ws[2][i] as i32).clamp(-64, 64) as i8;
            assert_eq!(out[i * N + j], want, "({i},{j})");
        }
    }
    // with ReLU the lower bound is 0; the bias of row 1 applies to output row i through ws[2][N + i]
    let (_, out_r) = eval_op(&Op::Bias { x: 0, w: 2, row: 1, relu: true }, &[x0.clone()], &ws, N);
    for i in 0..N {
        assert_eq!(out_r[i * N + 3], (x0[i * N + 3] as i32 + ws[2][N + i] as i32).clamp(0, 64) as i8);
    }
    // validation: row outside the block, nonexistent weight, future tensor
    assert!(validate_ops(&[Op::Gemm { w: 0, x: 0, shift: 3, relu: false }, Op::Bias { x: 1, w: 1, row: 0, relu: false }], 2, N).is_ok());
    assert!(validate_ops(&[Op::Bias { x: 0, w: 0, row: N as u16, relu: false }], 1, N).is_err(), "row >= n");
    assert!(validate_ops(&[Op::Bias { x: 0, w: 3, row: 0, relu: false }], 3, N).is_err(), "nonexistent weight");
    assert!(validate_ops(&[Op::Bias { x: 1, w: 0, row: 0, relu: false }], 1, N).is_err(), "tensor not yet computed");
}

struct Run {
    c: Chain,
    id: JobId,
    ws: Vec<Vec<i8>>,
    ops: Vec<Op>,
    tens: Vec<Vec<i8>>,
    zs: Vec<Option<Vec<i32>>>,
}

/// Submission, claim, useful blocks of the Gemms, publication; the Bias `tamper_op` (if any) is falsified at cell (5, 3).
fn run(seed: u64, tamper_op: Option<usize>) -> Run {
    let (x0, ws, ops) = program(seed);
    let alloc: Vec<(PubKey, u64)> = [10u8, 11, 13].iter().map(|s| (pubkey(&key(*s)), 1000 * COIN)).collect();
    let mut c = Chain::genesis(P, &alloc, 1_000, u64::MAX / 64);
    let x0_root = cols_root_of(&x0, N);
    let w_roots: Vec<Hash> = ws.iter().map(|w| rows_root_of(w, N)).collect();
    let id = prog_id(&pubkey(&key(10)), c.state.nonce(&pubkey(&key(10))), &x0_root, &w_roots, &ops);
    let submit = ptx(&c, 10, ProgTx::Submit { x0_root, w_roots, ops: ops.clone(), escrow: 60 * COIN, deadline: 900 });
    mine_at(&mut c, &key(12), WorkClass::Native, None, vec![submit]).unwrap();
    let claim = ptx(&c, 11, ProgTx::Claim { job_id: id });
    mine_at(&mut c, &key(12), WorkClass::Native, None, vec![claim]).unwrap();
    let (mut tens, mut zs) = (vec![x0], vec![]);
    for (k, op) in ops.iter().enumerate() {
        let (z, mut out) = eval_op(op, &tens, &ws, N);
        if let Op::Gemm { w, x, .. } = op {
            let cm = CommittedMatrices::new(ws[*w as usize].clone(), tens[*x as usize].clone(), &P);
            mine_at(&mut c, &key(11), WorkClass::Op { job: id, op: k as u8 }, Some(&cm), vec![]).unwrap();
        }
        if tamper_op == Some(k) {
            out[5 * N + 3] = out[5 * N + 3].wrapping_add(1);
        }
        let z_root = z.as_ref().map_or([0u8; 32], |zz| c_tree(zz, N).root());
        let post = ptx(&c, 11, ProgTx::PostOp { job_id: id, op: k as u8, z_root, x_root: cols_root_of(&out, N) });
        mine_at(&mut c, &key(12), WorkClass::Native, None, vec![post]).unwrap();
        zs.push(z);
        tens.push(out);
    }
    Run { c, id, ws, ops, tens, zs }
}

#[test]
fn honest_mlp_with_biases_is_verified_and_paid() {
    let mut r = run(3, None);
    assert!(matches!(r.c.state.progs[&r.id].status, ProgStatus::Complete { .. }));
    assert!(find_fraud_proof(&r.ops, &r.ws, &r.tens, &r.zs, N).is_none(), "the client finds nothing to contest");
    let accept = ptx(&r.c, 10, ProgTx::Accept { job_id: r.id });
    mine_at(&mut r.c, &key(12), WorkClass::Native, None, vec![accept]).unwrap();
    assert_eq!(r.c.state.progs[&r.id].status, ProgStatus::Settled);
    assert!(r.c.check_supply_invariant());
    // the biases do change the result (without them, the output differs)
    let mut no_bias = r.ws.clone();
    no_bias[2] = vec![0; N * N];
    let (t2, _) = run_program(&program(3).0, &no_bias, &r.ops, N);
    assert_ne!(t2.last(), r.tens.last(), "the bias vector influences the output");
}

#[test]
fn a_wrong_bias_column_is_proven_and_slashed_and_bad_proofs_are_rejected() {
    for bad_op in [1usize, 3] {
        let mut r = run(4 + bad_op as u64, Some(bad_op));
        let (op, kind) = find_fraud_proof(&r.ops, &r.ws, &r.tens, &r.zs, N).expect("fraud found");
        assert_eq!(op as usize, bad_op);
        assert!(matches!(kind, ProgFraud::Bias { j: 3, .. }), "the wrong column is 3");
        // falsified bias opening, altered bias value, unsuitable proof type: refused
        let ProgFraud::Bias { j, in_col, in_path, b_row, b_path, out_col, out_path } = kind.clone() else { unreachable!() };
        let mut b2 = b_row.clone();
        b2[5] = b2[5].wrapping_add(1);
        let forged = ProgFraud::Bias { j, in_col: in_col.clone(), in_path: in_path.clone(), b_row: b2, b_path: b_path.clone(), out_col: out_col.clone(), out_path: out_path.clone() };
        let f = ptx(&r.c, 13, ProgTx::Fraud { job_id: r.id, op, kind: forged });
        assert!(mine_at(&mut r.c, &key(12), WorkClass::Native, None, vec![f]).is_err(), "bias that is not the committed one");
        let wrong_kind = ProgFraud::Col { j, in_col: in_col.clone(), in_path: in_path.clone(), out_col: out_col.clone(), out_path: out_path.clone() };
        let f = ptx(&r.c, 13, ProgTx::Fraud { job_id: r.id, op, kind: wrong_kind });
        assert!(mine_at(&mut r.c, &key(12), WorkClass::Native, None, vec![f]).is_err(), "unsuitable proof type");
        // a CORRECT column is not a fraud
        let good = (0..N).find(|c| *c != 3).unwrap();
        let tx_in = &r.tens[if bad_op == 1 { 1 } else { 3 }];
        let out_t = &r.tens[bad_op + 1];
        let Op::Bias { w, row, .. } = &r.ops[bad_op] else { unreachable!() };
        let honest = ProgFraud::Bias {
            j: good as u32,
            in_col: (0..N).map(|i| tx_in[i * N + good]).collect(),
            in_path: cols_tree(tx_in, N).proof(good),
            b_row: r.ws[*w as usize][*row as usize * N..(*row as usize + 1) * N].to_vec(),
            b_path: rows_tree(&r.ws[*w as usize], N).proof(*row as usize),
            out_col: (0..N).map(|i| out_t[i * N + good]).collect(),
            out_path: cols_tree(out_t, N).proof(good),
        };
        let f = ptx(&r.c, 13, ProgTx::Fraud { job_id: r.id, op, kind: honest });
        assert!(mine_at(&mut r.c, &key(12), WorkClass::Native, None, vec![f]).is_err(), "correct column: not a fraud");
        // the true proof is accepted: the worker is penalised
        let alice0 = r.c.state.balance(&pubkey(&key(10)));
        let f = ptx(&r.c, 10, ProgTx::Fraud { job_id: r.id, op, kind });
        mine_at(&mut r.c, &key(12), WorkClass::Native, None, vec![f]).unwrap();
        assert_eq!(r.c.state.progs[&r.id].status, ProgStatus::Slashed);
        assert!(r.c.state.balance(&pubkey(&key(10))) > alice0, "the client is refunded");
        assert!(r.c.check_supply_invariant());
    }
}

// ------------------------------------------------------------------ Lut: activation table

fn lut_program(seed: u64) -> (Vec<i8>, Vec<Vec<i8>>, Vec<Op>) {
    demo_program("biasmlp", N, seed).unwrap()
}

#[test]
fn lut_semantics_index_is_the_clamped_value_plus_64() {
    let (x0, ws, ops) = lut_program(9);
    assert_eq!(ops.iter().filter(|o| matches!(o, Op::Lut { .. })).count(), 1);
    // table of row 2: quantised tanh; input -64 -> table[0], 0 -> table[64], 63 -> table[127], 64 (saturated) -> table[127]
    let t = &ws[2][2 * N..3 * N];
    assert_eq!((t[0], t[64], t[127]), (-62, 0, 62));
    assert_eq!((lut_apply(t, -64), lut_apply(t, 0), lut_apply(t, 63), lut_apply(t, 64)), (t[0], t[64], t[127], t[127]));
    let (tens, _) = run_program(&x0, &ws, &ops, N);
    let (tin, tout) = (&tens[2], &tens[3]);
    for idx in 0..N * N {
        assert_eq!(tout[idx], t[(tin[idx].clamp(-64, 63) as i32 + 64) as usize]);
    }
    // validation: row outside the block, nonexistent weight
    assert!(validate_ops(&[Op::Lut { x: 0, w: 0, row: N as u16 }], 1, N).is_err());
    assert!(validate_ops(&[Op::Lut { x: 0, w: 1, row: 0 }], 1, N).is_err());
    assert!(validate_ops(&[Op::Lut { x: 0, w: 0, row: 3 }], 1, N).is_ok());
}

#[test]
fn a_wrong_table_activation_is_proven_and_slashed() {
    // reuses the small MLP with bias and table; we falsify the output of the table (operation 2)
    let (x0, ws, ops) = lut_program(10);
    let alloc: Vec<(PubKey, u64)> = [10u8, 11, 13].iter().map(|s| (pubkey(&key(*s)), 1000 * COIN)).collect();
    let mut c = Chain::genesis(P, &alloc, 1_000, u64::MAX / 64);
    let x0_root = cols_root_of(&x0, N);
    let w_roots: Vec<Hash> = ws.iter().map(|w| rows_root_of(w, N)).collect();
    let id = prog_id(&pubkey(&key(10)), c.state.nonce(&pubkey(&key(10))), &x0_root, &w_roots, &ops);
    let submit = ptx(&c, 10, ProgTx::Submit { x0_root, w_roots, ops: ops.clone(), escrow: 60 * COIN, deadline: 900 });
    mine_at(&mut c, &key(12), WorkClass::Native, None, vec![submit]).unwrap();
    { let t = ptx(&c, 11, ProgTx::Claim { job_id: id }); mine_at(&mut c, &key(12), WorkClass::Native, None, vec![t]).unwrap(); }
    let (mut tens, mut zs) = (vec![x0], vec![]);
    for (k, op) in ops.iter().enumerate() {
        let (z, mut out) = eval_op(op, &tens, &ws, N);
        if let Op::Gemm { w, x, .. } = op {
            let cm = CommittedMatrices::new(ws[*w as usize].clone(), tens[*x as usize].clone(), &P);
            mine_at(&mut c, &key(11), WorkClass::Op { job: id, op: k as u8 }, Some(&cm), vec![]).unwrap();
        }
        if k == 2 {
            out[7 * N + 4] = out[7 * N + 4].wrapping_add(1); // column 4 of the table output is wrong
        }
        let z_root = z.as_ref().map_or([0u8; 32], |zz| c_tree(zz, N).root());
        { let t = ptx(&c, 11, ProgTx::PostOp { job_id: id, op: k as u8, z_root, x_root: cols_root_of(&out, N) }); mine_at(&mut c, &key(12), WorkClass::Native, None, vec![t]).unwrap(); }
        zs.push(z);
        tens.push(out);
    }
    let (op, kind) = find_fraud_proof(&ops, &ws, &tens, &zs, N).expect("fraud found");
    assert_eq!(op, 2);
    let ProgFraud::Lut { j, in_col, in_path, t_row, t_path, out_col, out_path } = kind.clone() else { panic!("expected a Lut proof") };
    assert_eq!(j, 4);
    // table altered in the proof: invalid opening; proof of another kind: refused
    let mut t2 = t_row.clone();
    t2[10] = t2[10].wrapping_add(1);
    let forged = ProgFraud::Lut { j, in_col: in_col.clone(), in_path: in_path.clone(), t_row: t2, t_path: t_path.clone(), out_col: out_col.clone(), out_path: out_path.clone() };
    { let t = ptx(&c, 13, ProgTx::Fraud { job_id: id, op, kind: forged }); assert!(mine_at(&mut c, &key(12), WorkClass::Native, None, vec![t]).is_err()); }
    let as_bias = ProgFraud::Bias { j, in_col, in_path, b_row: t_row, b_path: t_path, out_col, out_path };
    { let t = ptx(&c, 13, ProgTx::Fraud { job_id: id, op, kind: as_bias }); assert!(mine_at(&mut c, &key(12), WorkClass::Native, None, vec![t]).is_err(), "unsuitable proof type"); }
    { let t = ptx(&c, 10, ProgTx::Fraud { job_id: id, op, kind }); mine_at(&mut c, &key(12), WorkClass::Native, None, vec![t]).unwrap(); }
    assert_eq!(c.state.progs[&id].status, ProgStatus::Slashed);
    assert!(c.check_supply_invariant());
}

// ------------------------------------------------------------------ Conv: general convolution (stride, padding)

#[test]
fn general_conv_with_stride_and_valid_padding_is_exactly_a_naive_convolution() {
    let mut rng = StdRng::seed_from_u64(31);
    let (ci, h, w, b, kh, kw) = (2usize, 7usize, 6usize, 2usize, 3usize, 3usize);
    for (pad, stride) in [(0usize, 2usize), (1, 2), (0, 1), (2, 3)] {
        let (oh, ow) = conv_out_dims(h, w, kh, kw, pad, stride).unwrap();
        let mut x0 = vec![0i8; N * N];
        for c in 0..ci {
            for q in 0..h * w * b {
                x0[c * N + q] = rng.gen_range(-30..=30);
            }
        }
        let co = 3;
        let mut w0 = vec![0i8; N * N];
        for r in 0..co {
            for c in 0..ci * kh * kw {
                w0[r * N + c] = rng.gen_range(-4..=4);
            }
        }
        let ops = vec![
            Op::Map { x: 0, map: MapSpec::Conv { c_in: ci as u16, h: h as u16, w: w as u16, batch: b as u16, kh: kh as u8, kw: kw as u8, pad: pad as u8, stride: stride as u8 } },
            Op::Gemm { w: 0, x: 1, shift: 0, relu: false },
        ];
        validate_ops(&ops, 1, N).unwrap();
        let (_, zs) = run_program(&x0, &[w0.clone()], &ops, N);
        let z = zs[1].as_ref().unwrap();
        for o in 0..co {
            for oy in 0..oh {
                for ox in 0..ow {
                    for bb in 0..b {
                        let mut acc = 0i32;
                        for c in 0..ci {
                            for dy in 0..kh {
                                for dx in 0..kw {
                                    let (sy, sx) = ((oy * stride + dy) as i64 - pad as i64, (ox * stride + dx) as i64 - pad as i64);
                                    if sy < 0 || sx < 0 || sy >= h as i64 || sx >= w as i64 {
                                        continue;
                                    }
                                    acc += w0[o * N + c * kh * kw + dy * kw + dx] as i32 * x0[c * N + (sy as usize * w + sx as usize) * b + bb] as i32;
                                }
                            }
                        }
                        assert_eq!(z[o * N + (oy * ow + ox) * b + bb], acc, "pad {pad} stride {stride} ({o},{oy},{ox},{bb})");
                    }
                }
            }
        }
    }
    // window that does not fit, zero stride: refused
    let bad = |pad: u8, stride: u8, kh: u8| validate_ops(&[Op::Map { x: 0, map: MapSpec::Conv { c_in: 1, h: 4, w: 4, batch: 1, kh, kw: 3, pad, stride } }], 1, N);
    assert!(bad(0, 0, 3).is_err() && bad(0, 1, 9).is_err() && bad(0, 1, 3).is_ok());
}

// ---------------------------------------------------------------- multi-head attention: HeadSpread, HeadGather, Mul (gamma)

#[test]
fn head_spread_gather_and_mul_follow_their_definitions() {
    use vearl_core::prog::*;
    let n = 128usize;
    let (x0, ws, ops) = demo_heads(n, 7);
    let mut tens = vec![x0.clone()];
    for op in &ops {
        let (_, out) = eval_op(op, &tens, &ws, n);
        tens.push(out);
    }
    let (dh, l) = (8usize, 16usize);
    // HeadSpread (T4 from Q = T1): row i goes into the column block of its head, zero elsewhere
    for i in 0..n {
        for j in 0..n {
            let h = i / dh;
            let want = if i < 4 * dh && j >= h * l && j < h * l + l { tens[1][i * n + (j - h * l)] } else { 0 };
            assert_eq!(tens[4][i * n + j], want, "HeadSpread ({i},{j})");
        }
    }
    // HeadGather (T9 from T8): inverse
    for i in 0..n {
        for j in 0..n {
            let want = if i < 4 * dh && j < l { tens[8][i * n + (i / dh) * l + j] } else { 0 };
            assert_eq!(tens[9][i * n + j], want, "HeadGather ({i},{j})");
        }
    }
    // Mul (T11 from T10, gamma = row 0 of block 3, shift 6)
    for i in 0..n {
        for j in 0..n {
            let want = (((tens[10][i * n + j] as i32) * (ws[3][i] as i32)) >> 6).clamp(-64, 64) as i8;
            assert_eq!(tens[11][i * n + j], want);
        }
    }
    // honest: nothing to contest; one wrong cell in each of the three new operations: proof found on the RIGHT operation
    let zs: Vec<Option<Vec<i32>>> = ops.iter().zip(0..).map(|(op, k)| eval_op(op, &tens[..=k], &ws, n).0).collect();
    assert!(find_fraud_proof(&ops, &ws, &tens, &zs, n).is_none());
    for (k, expect) in [(3usize, "Elem"), (8, "Elem"), (10, "Bias")] {
        let mut bad = tens.clone();
        bad[k + 1][0] = bad[k + 1][0].wrapping_add(1);
        let (op, kind) = find_fraud_proof(&ops, &ws, &bad, &zs, n).expect("fraud found");
        assert_eq!(op as usize, k);
        assert!(format!("{kind:?}").starts_with(expect), "{kind:?}");
    }
}

#[test]
fn head_ops_are_validated() {
    use vearl_core::prog::*;
    let n = 128;
    let ok = |map: MapSpec| validate_ops(&[Op::Gemm { w: 0, x: 0, shift: 0, relu: false }, Op::Map { x: 1, map }], 1, n).is_ok();
    assert!(ok(MapSpec::HeadSpread { dh: 8, heads: 4, l: 32 }));
    assert!(!ok(MapSpec::HeadSpread { dh: 8, heads: 4, l: 33 }), "4 x 33 columns > n");
    assert!(!ok(MapSpec::HeadGather { dh: 0, heads: 4, l: 16 }));
    assert!(!ok(MapSpec::HeadGather { dh: 64, heads: 3, l: 16 }), "3 x 64 rows > n");
    assert!(validate_ops(&[Op::Gemm { w: 0, x: 0, shift: 0, relu: false }, Op::Mul { x: 1, w: 0, row: 3, shift: 24 }], 1, n).is_ok());
    assert!(validate_ops(&[Op::Gemm { w: 0, x: 0, shift: 0, relu: false }, Op::Mul { x: 1, w: 1, row: 3, shift: 6 }], 1, n).is_err(), "nonexistent weight");
    assert!(validate_ops(&[Op::Gemm { w: 0, x: 0, shift: 0, relu: false }, Op::Mul { x: 1, w: 0, row: 3, shift: 25 }], 1, n).is_err(), "shift too large");
}

#[test]
fn softmax_lo_refines_the_integer_softmax_to_1_over_4096() {
    use vearl_core::prog::*;
    // column of 197 keys (attention of an image): most probabilities < 1/128 are lost by Softmax alone
    let n = 256usize;
    let mut rng = StdRng::seed_from_u64(11);
    for trial in 0..200 {
        let len = [1usize, 2, 7, 64, 197, 256][trial % 6];
        let col: Vec<i8> = (0..n).map(|_| rng.gen_range(-64i32..=64) as i8).collect();
        let (hi, lo) = (softmax_col(&col, len), softmax_lo_col(&col, len));
        let m = col[..len].iter().map(|v| *v as i32).max().unwrap();
        let e: Vec<u64> = col[..len].iter().map(|v| EXP2_TABLE[(m - *v as i32).clamp(0, 128) as usize] as u64).collect();
        let s: u64 = e.iter().sum();
        for i in 0..n {
            if i >= len {
                assert_eq!((hi[i], lo[i]), (0, 0), "cells >= len: 0");
                continue;
            }
            let fine = ((e[i] * 4096 + s / 2) / s) as i64;
            assert_eq!(64 * hi[i] as i64 + lo[i] as i64, fine, "high*64 + fine = p to within 1/4096");
            assert!((-32..=32).contains(&(lo[i] as i64)), "fine part in [-32, 32]: {}", lo[i]);
        }
    }
    // program operation: per-column evaluation, column fraud proof on the RIGHT operation
    let ws = vec![(0..n * n).map(|i| ((i * 7 % 13) as i8) - 6).collect::<Vec<i8>>()];
    let x0: Vec<i8> = (0..n * n).map(|i| ((i * 31 % 17) as i8) - 8).collect();
    let ops = vec![Op::Gemm { w: 0, x: 0, shift: 4, relu: false }, Op::Softmax { x: 1, len: 197 }, Op::SoftmaxLo { x: 1, len: 197 }, Op::Add { a: 2, b: 3, relu: false }];
    validate_ops(&ops, 1, n).unwrap();
    let mut tens = vec![x0];
    let mut zs = vec![];
    for op in &ops {
        let (z, out) = eval_op(op, &tens, &ws, n);
        zs.push(z);
        tens.push(out);
    }
    for j in 0..n {
        let col: Vec<i8> = (0..n).map(|i| tens[1][i * n + j]).collect();
        let lo = softmax_lo_col(&col, 197);
        for i in 0..n {
            assert_eq!(tens[3][i * n + j], lo[i]);
        }
    }
    assert!(find_fraud_proof(&ops, &ws, &tens, &zs, n).is_none());
    let mut bad = tens.clone();
    bad[3][5 * n + 9] = bad[3][5 * n + 9].wrapping_add(1);
    let (op, kind) = find_fraud_proof(&ops, &ws, &bad, &zs, n).expect("fraud found");
    assert_eq!(op, 2);
    assert!(format!("{kind:?}").starts_with("Col"), "{kind:?}");
    // validation: same rules as Softmax
    assert!(validate_ops(&[Op::Gemm { w: 0, x: 0, shift: 0, relu: false }, Op::SoftmaxLo { x: 0, len: 4 }], 1, n).is_err(), "reads the raw input");
    assert!(validate_ops(&[Op::Gemm { w: 0, x: 0, shift: 0, relu: false }, Op::SoftmaxLo { x: 1, len: 0 }], 1, n).is_err(), "len = 0");
    assert!(validate_ops(&[Op::Gemm { w: 0, x: 0, shift: 0, relu: false }, Op::SoftmaxLo { x: 1, len: 257 }], 1, n).is_err(), "len > n");
}

#[test]
fn causal_softmax_lets_each_query_see_only_the_previous_keys() {
    use vearl_core::prog::*;
    // scores (keys x (head, query)): 4 heads of l = 64 columns, 50 real tokens; the query q (column j, q = j mod 64) only sees
    // the keys 0..=q: Softmax / SoftmaxLo on that prefix, 0 beyond
    let (n, l, len) = (256usize, 64u16, 50u16);
    let ws = vec![(0..n * n).map(|i| ((i * 7 % 13) as i8) - 6).collect::<Vec<i8>>()];
    let x0: Vec<i8> = (0..n * n).map(|i| ((i * 31 % 17) as i8) - 8).collect();
    let ops = vec![
        Op::Gemm { w: 0, x: 0, shift: 4, relu: false },
        Op::CausalSoftmax { x: 1, len, l, fine: false },
        Op::CausalSoftmax { x: 1, len, l, fine: true },
    ];
    validate_ops(&ops, 1, n).unwrap();
    let mut tens = vec![x0];
    let mut zs = vec![];
    for op in &ops {
        let (z, out) = eval_op(op, &tens, &ws, n);
        zs.push(z);
        tens.push(out);
    }
    for j in 0..n {
        let q = j % l as usize;
        let eff = (len as usize).min(q + 1);
        let col: Vec<i8> = (0..n).map(|i| tens[1][i * n + j]).collect();
        let (hi, lo) = (softmax_col(&col, eff), softmax_lo_col(&col, eff));
        for i in 0..n {
            assert_eq!(tens[2][i * n + j], hi[i], "high, column {j} row {i}");
            assert_eq!(tens[3][i * n + j], lo[i], "fine, column {j} row {i}");
            if i > q {
                assert_eq!((tens[2][i * n + j], tens[3][i * n + j]), (0, 0), "future key: 0");
            }
        }
        let s: i32 = (0..n).map(|i| tens[2][i * n + j] as i32).sum();
        assert!((s - 64).abs() <= eff as i32, "sum of the probabilities ~ 64: {s}");
    }
    assert_eq!(tens[2][0], 64, "the first query only sees itself");
    assert!(find_fraud_proof(&ops, &ws, &tens, &zs, n).is_none());
    // a worker that lets a future key be seen (NON-causal softmax) is caught: column proof on operation 1
    let mut bad = tens.clone();
    let noncausal = eval_op(&Op::Softmax { x: 1, len }, &tens, &ws, n).1;
    bad[2] = noncausal;
    let (op, kind) = find_fraud_proof(&ops, &ws, &bad, &zs, n).expect("fraud found");
    assert_eq!(op, 1);
    assert!(format!("{kind:?}").starts_with("Col"), "{kind:?}");
    // validation: same rules as Softmax, plus 1 <= l <= n
    let g = Op::Gemm { w: 0, x: 0, shift: 0, relu: false };
    assert!(validate_ops(&[g.clone(), Op::CausalSoftmax { x: 0, len: 4, l: 4, fine: false }], 1, n).is_err(), "reads the raw input");
    assert!(validate_ops(&[g.clone(), Op::CausalSoftmax { x: 1, len: 0, l: 4, fine: false }], 1, n).is_err(), "len = 0");
    assert!(validate_ops(&[g.clone(), Op::CausalSoftmax { x: 1, len: 4, l: 0, fine: true }], 1, n).is_err(), "l = 0");
    assert!(validate_ops(&[g.clone(), Op::CausalSoftmax { x: 1, len: 4, l: 257, fine: true }], 1, n).is_err(), "l > n");
    assert!(validate_ops(&[g, Op::CausalSoftmax { x: 1, len: 256, l: 256, fine: true }], 1, n).is_ok());
}
