//! Verifiable tensor programs: semantics of the operations, life cycle, fraud proofs (Gemm, Act, Elem),
//! access controls. The demonstration program is a small CNN with a residual block.
use ed25519_dalek::SigningKey;
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
    /// wrong GEMM at operation k: Z falsified, output consistent with the falsified Z
    Gemm(usize),
    /// Z correct but wrong activation
    Act(usize),
    /// non-Gemm operation with one wrong output cell
    Elem(usize),
}

struct Run {
    c: Chain,
    id: JobId,
    x0: Vec<i8>,
    ws: Vec<Vec<i8>>,
    ops: Vec<Op>,
    tens: Vec<Vec<i8>>, // T_0..T_L published (possibly falsified)
    zs: Vec<Option<Vec<i32>>>,
}

fn setup(seed: u64) -> (Chain, JobId, Vec<i8>, Vec<Vec<i8>>, Vec<Op>) {
    let (x0, ws, ops) = demo_cnn(N, seed);
    let mut c = new_chain();
    let x0_root = cols_root_of(&x0, N);
    let w_roots: Vec<Hash> = ws.iter().map(|w| rows_root_of(w, N)).collect();
    let nonce = c.state.nonce(&pubkey(&key(10)));
    let id = prog_id(&pubkey(&key(10)), nonce, &x0_root, &w_roots, &ops);
    let gemms = ops.iter().filter(|o| matches!(o, Op::Gemm { .. })).count() as u64;
    let submit = ptx(&c, 10, ProgTx::Submit { x0_root, w_roots, ops: ops.clone(), escrow: 30 * COIN * gemms, deadline: 900 });
    mine_at(&mut c, &key(12), WorkClass::Native, None, vec![submit]).unwrap();
    let claim = ptx(&c, 11, ProgTx::Claim { job_id: id });
    mine_at(&mut c, &key(12), WorkClass::Native, None, vec![claim]).unwrap();
    (c, id, x0, ws, ops)
}

/// The worker executes the program operation by operation (with possible falsification), mines a block for each Gemm
/// (whose input is already published), and publishes each operation.
fn run(seed: u64, tamper: Tamper) -> Run {
    let (mut c, id, x0, ws, ops) = setup(seed);
    let mut tens = vec![x0.clone()];
    let mut zs: Vec<Option<Vec<i32>>> = vec![];
    for (k, op) in ops.iter().enumerate() {
        let (mut z, mut out) = eval_op(op, &tens, &ws, N);
        if let Op::Gemm { w, x, shift, relu } = op {
            // the mined block computes the true product W_w * T_x on the committed matrices
            let cm = CommittedMatrices::new(ws[*w as usize].clone(), tens[*x as usize].clone(), &P);
            let zm = mine_at(&mut c, &key(11), WorkClass::Op { job: id, op: k as u8 }, Some(&cm), vec![]).unwrap();
            assert_eq!(&zm, z.as_ref().unwrap(), "the Op block produces the true GEMM of operation {k}");
            if tamper == Tamper::Gemm(k) {
                let zz = z.as_mut().unwrap();
                zz[3 * N + 5] += 1 << *shift;
                out = zz.iter().map(|&v| act_fn(v, *shift, *relu)).collect();
            }
            if tamper == Tamper::Act(k) {
                let i = out.iter().position(|v| *v != 63 && *v != 64).unwrap();
                out[i] = out[i].wrapping_add(1);
            }
        } else if tamper == Tamper::Elem(k) {
            out[0] = out[0].wrapping_add(1); // any cell: the expected value is defined by the operation
        }
        let z_root = z.as_ref().map_or([0u8; 32], |zz| c_tree(zz, N).root());
        let post = ptx(&c, 11, ProgTx::PostOp { job_id: id, op: k as u8, z_root, x_root: cols_root_of(&out, N) });
        mine_at(&mut c, &key(12), WorkClass::Native, None, vec![post]).unwrap();
        zs.push(z);
        tens.push(out);
    }
    Run { c, id, x0, ws, ops, tens, zs }
}

/// Builds the fraud proof of operation k if it is wrong (as the client would).
fn find_fraud(r: &Run) -> Option<(u8, ProgFraud)> {
    for (k, op) in r.ops.iter().enumerate() {
        let out = &r.tens[k + 1];
        let out_tree = cols_tree(out, N);
        if let Op::Gemm { w, x, shift, relu } = op {
            let (z, tx) = (r.zs[k].as_ref().unwrap(), &r.tens[*x as usize]);
            let wm = &r.ws[*w as usize];
            if let Some((i, j)) = find_bad_entry(wm, tx, z, N, &[3u8; 32]) {
                let zt = c_tree(z, N);
                return Some((k as u8, ProgFraud::Gemm {
                    i: i as u32, j: j as u32, w_row: wm[i * N..(i + 1) * N].to_vec(), w_path: rows_tree(wm, N).proof(i),
                    x_col: (0..N).map(|t| tx[t * N + j]).collect(), x_path: cols_tree(tx, N).proof(j),
                    z_row: row_of(&z, N, i), z_path: zt.proof(i),
                }));
            }
            if let Some(idx) = (0..N * N).find(|&e| out[e] != act_fn(z[e], *shift, *relu)) {
                let (i, j) = (idx / N, idx % N);
                return Some((k as u8, ProgFraud::Act {
                    i: i as u32, j: j as u32, z_row: row_of(z, N, i), z_path: c_tree(z, N).proof(i),
                    out_col: (0..N).map(|t| out[t * N + j]).collect(), out_path: out_tree.proof(j),
                }));
            }
        } else {
            for idx in 0..N * N {
                let (i, j) = (idx / N, idx % N);
                let req = op_inputs(op, i, j, N);
                let vals: Vec<i8> = req.iter().map(|&(t, si, sj)| r.tens[t][si * N + sj]).collect();
                if out[idx] != op_eval(op, &vals) {
                    let ins = req.iter().map(|&(t, _, sj)| OpenedCol { tensor: t as u8, col: sj as u32, vals: (0..N).map(|q| r.tens[t][q * N + sj]).collect(), path: cols_tree(&r.tens[t], N).proof(sj) }).collect();
                    return Some((k as u8, ProgFraud::Elem { i: i as u32, j: j as u32, ins, out_col: (0..N).map(|q| out[q * N + j]).collect(), out_path: out_tree.proof(j) }));
                }
            }
        }
    }
    None
}

fn submit_fraud(r: &mut Run, op: u8, kind: ProgFraud) -> Result<Vec<i32>, String> {
    let f = ptx(&r.c, 10, ProgTx::Fraud { job_id: r.id, op, kind });
    mine_at(&mut r.c, &key(12), WorkClass::Native, None, vec![f])
}

// ------------------------------------------------------------------ semantics

#[test]
fn im2col_gemm_is_exactly_a_naive_convolution() {
    let (x0, ws, ops) = demo_cnn(N, 1);
    let (tens, zs) = run_program(&x0, &ws, &ops, N);
    // op 1 = Gemm(W0, im2col(x0)): 3x3 conv, stride 1, padding 1, 4 -> 8 channels, 8x8 images, batch 2
    let z = zs[1].as_ref().unwrap();
    let (h, w, b) = (8usize, 8usize, 2usize);
    for co in 0..8 {
        for y in 0..h {
            for x in 0..w {
                for bb in 0..b {
                    let mut acc = 0i32;
                    for c in 0..4 {
                        for dy in 0..3i64 {
                            for dx in 0..3i64 {
                                let (sy, sx) = (y as i64 + dy - 1, x as i64 + dx - 1);
                                if sy < 0 || sx < 0 || sy >= h as i64 || sx >= w as i64 {
                                    continue; // zero padding
                                }
                                acc += ws[0][co * N + c * 9 + (dy * 3 + dx) as usize] as i32 * x0[c * N + (sy as usize * w + sx as usize) * b + bb] as i32;
                            }
                        }
                    }
                    assert_eq!(z[co * N + (y * w + x) * b + bb], acc, "conv({co},{y},{x},{bb})");
                }
            }
        }
    }
    assert_eq!(tens.len(), ops.len() + 1);
}

#[test]
fn pooling_add_and_activation_semantics_are_exactly_defined() {
    // 2x2 mean rounded to the nearest integer upward, negatives included (div_euclid)
    let pool = Op::Pool { x: 0, pool: PoolSpec::Avg { c: 1, h: 2, w: 2, batch: 1, k: 2 } };
    assert_eq!(op_eval(&pool, &[1, 1, 1, 2]), 1); // 5/4 = 1.25 -> 1
    assert_eq!(op_eval(&pool, &[1, 1, 2, 2]), 2); // 6/4 = 1.5 -> 2
    assert_eq!(op_eval(&pool, &[-1, -1, -1, -2]), -1); // -5/4 = -1.25 -> -1
    assert_eq!(op_eval(&pool, &[-1, -1, -2, -2]), -1); // -6/4 = -1.5 -> -1 (rounded toward +inf)
    // residual: saturation to [-64,64]; with ReLU the lower bound is 0
    let add = Op::Add { a: 0, b: 0, relu: false };
    assert_eq!(op_eval(&add, &[60, 60]), 64);
    assert_eq!(op_eval(&add, &[-60, -60]), -64);
    assert_eq!(op_eval(&Op::Add { a: 0, b: 0, relu: true }, &[-30, 5]), 0);
    // activation: arithmetic shift (floor) then saturation
    assert_eq!(act_fn(-1, 3, false), -1, "-1 >> 3 = -1 (arithmetic)");
    assert_eq!(act_fn(-1, 3, true), 0);
    assert_eq!(act_fn(1 << 20, 4, true), 64);
    // im2col: outside the image => no source (zero)
    let conv = Op::Map { x: 0, map: MapSpec::Im2col { c_in: 1, h: 4, w: 4, batch: 1, kh: 3, kw: 3, pad: 1 } };
    assert!(op_inputs(&conv, 0, 0, N).is_empty(), "top-left corner of the patch (dy=0,dx=0) at pixel (0,0): outside the image");
    assert_eq!(op_inputs(&conv, 4, 5, N), vec![(0, 0, 5)], "central patch at pixel (1,1) of channel 0: source column 1*4+1 = 5");
    assert!(op_inputs(&conv, 9, 0, N).is_empty(), "row beyond c_in*kh*kw: zero");
}

// ------------------------------------------------------------------ life cycle and frauds

#[test]
fn honest_cnn_with_residual_block_is_verified_and_paid() {
    let mut r = run(1, Tamper::None);
    assert!(matches!(r.c.state.progs[&r.id].status, ProgStatus::Complete { .. }));
    assert!(find_fraud(&r).is_none(), "the client finds nothing to contest");
    // the client verifies op by op: Freivalds for each Gemm, element-by-element recomputation for the rest
    for (k, op) in r.ops.iter().enumerate() {
        if let Op::Gemm { w, x, .. } = op {
            assert!(freivalds(&r.ws[*w as usize], &r.tens[*x as usize], r.zs[k].as_ref().unwrap(), N, 24, &[7u8; 32]), "Freivalds op {k}");
        }
    }
    let (t_ref, _) = run_program(&r.x0, &r.ws, &r.ops, N);
    assert_eq!(t_ref, r.tens, "the worker's result is exactly the program's");
    let before = r.c.state.balance(&pubkey(&key(11)));
    let accept = ptx(&r.c, 10, ProgTx::Accept { job_id: r.id });
    mine_at(&mut r.c, &key(12), WorkClass::Native, None, vec![accept]).unwrap();
    assert_eq!(r.c.state.progs[&r.id].status, ProgStatus::Settled);
    let net = 120 * COIN - bps(120 * COIN, BURN_BPS);
    assert_eq!(r.c.state.balance(&pubkey(&key(11))), before + net + bps(net, BOND_BPS));
    assert!(r.c.state.progs[&r.id].rebate_left <= bps(120 * COIN, BURN_BPS));
    assert!(r.c.check_supply_invariant());
    // the final scores (10 classes x 32 columns) are not trivial
    let scores = r.tens.last().unwrap();
    let distinct: std::collections::BTreeSet<i8> = scores.iter().copied().collect();
    assert!(distinct.len() >= 5, "the signal does not die out in the network: {} distinct values in the scores", distinct.len());
    assert!((0..10).any(|r| (0..32).any(|c| scores[r * N + c] != 0)));
}

#[test]
fn each_kind_of_wrong_operation_is_proven_and_slashed() {
    // (falsification, expected proof kind)
    let cases = [
        (Tamper::Gemm(4), "gemm"),   // conv 2: wrong product
        (Tamper::Act(1), "act"),     // conv 1: wrong activation
        (Tamper::Elem(0), "elem"),   // im2col
        (Tamper::Elem(2), "elem"),   // pooling
        (Tamper::Elem(6), "elem"),   // residual addition
        (Tamper::Gemm(7), "gemm"),   // classifier
    ];
    for (i, (t, want)) in cases.into_iter().enumerate() {
        let mut r = run(20 + i as u64, t);
        let (op, kind) = find_fraud(&r).unwrap_or_else(|| panic!("case {i}: no fraud found"));
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
        let net = 120 * COIN - bps(120 * COIN, BURN_BPS);
        assert_eq!(r.c.state.balance(&pubkey(&key(10))), alice0 + net + bps(net, BOND_BPS) / 2, "cas {i}");
        assert!(r.c.check_supply_invariant());
    }
}

#[test]
fn false_accusations_are_rejected() {
    let mut r = run(2, Tamper::None);
    // Gemm: correct entry
    let (z, x, w) = (r.zs[1].clone().unwrap(), r.tens[1].clone(), r.ws[0].clone());
    let (i, j) = (2usize, 6usize);
    let gemm = ProgFraud::Gemm {
        i: i as u32, j: j as u32, w_row: w[i * N..(i + 1) * N].to_vec(), w_path: rows_tree(&w, N).proof(i),
        x_col: (0..N).map(|t| x[t * N + j]).collect(), x_path: cols_tree(&x, N).proof(j), z_row: row_of(&z, N, i), z_path: c_tree(&z, N).proof(i),
    };
    assert!(submit_fraud(&mut r, 1, gemm.clone()).is_err());
    // a proof kind that does not match the operation (Gemm on a pooling)
    assert!(submit_fraud(&mut r, 2, gemm).is_err());
    // Elem: correct entry of a residual addition
    let op = &r.ops[6];
    let (i, j) = (3usize, 4usize);
    let req = op_inputs(op, i, j, N);
    let ins = req.iter().map(|&(t, _, sj)| OpenedCol { tensor: t as u8, col: sj as u32, vals: (0..N).map(|q| r.tens[t][q * N + sj]).collect(), path: cols_tree(&r.tens[t], N).proof(sj) }).collect();
    let elem = ProgFraud::Elem { i: i as u32, j: j as u32, ins, out_col: (0..N).map(|q| r.tens[7][q * N + j]).collect(), out_path: cols_tree(&r.tens[7], N).proof(j) };
    assert!(submit_fraud(&mut r, 6, elem.clone()).is_err());
    // same openings but wrong operation
    assert!(submit_fraud(&mut r, 0, elem).is_err());
    assert!(matches!(r.c.state.progs[&r.id].status, ProgStatus::Complete { .. }));
}

#[test]
fn mining_access_control_and_unlocking() {
    let (mut c, id, x0, ws, ops) = setup(3);
    let cm_op1 = CommittedMatrices::new(ws[0].clone(), eval_op(&ops[0], &[x0.clone()], &ws, N).1, &P);
    // op 1 (Gemm): its input T1 is not yet published => locked
    assert!(mine_at(&mut c, &key(11), WorkClass::Op { job: id, op: 1 }, Some(&cm_op1), vec![]).is_err(), "input not published");
    // op 0 is a Map: never mineable
    assert!(mine_at(&mut c, &key(11), WorkClass::Op { job: id, op: 0 }, Some(&cm_op1), vec![]).is_err(), "only Gemms are mined");
    // publication of op 0 => op 1 unlocks; another miner stays refused
    let t1 = eval_op(&ops[0], &[x0.clone()], &ws, N).1;
    let post = ptx(&c, 11, ProgTx::PostOp { job_id: id, op: 0, z_root: [0; 32], x_root: cols_root_of(&t1, N) });
    mine_at(&mut c, &key(12), WorkClass::Native, None, vec![post]).unwrap();
    assert!(mine_at(&mut c, &key(13), WorkClass::Op { job: id, op: 1 }, Some(&cm_op1), vec![]).is_err(), "not the worker");
    // data of another operation
    let wrong = CommittedMatrices::new(ws[1].clone(), t1.clone(), &P);
    assert!(mine_at(&mut c, &key(11), WorkClass::Op { job: id, op: 1 }, Some(&wrong), vec![]).is_err());
    assert!(mine_at(&mut c, &key(11), WorkClass::Op { job: id, op: 1 }, Some(&cm_op1), vec![]).is_ok());
    // out-of-order publication, z_root on a non-Gemm, by a third party
    let skip = ptx(&c, 11, ProgTx::PostOp { job_id: id, op: 3, z_root: [0; 32], x_root: [1; 32] });
    assert!(mine_at(&mut c, &key(12), WorkClass::Native, None, vec![skip]).is_err(), "out of order");
    let post1 = ptx(&c, 11, ProgTx::PostOp { job_id: id, op: 1, z_root: [9; 32], x_root: [1; 32] });
    mine_at(&mut c, &key(12), WorkClass::Native, None, vec![post1]).unwrap();
    let z_on_pool = ptx(&c, 11, ProgTx::PostOp { job_id: id, op: 2, z_root: [7; 32], x_root: [1; 32] });
    assert!(mine_at(&mut c, &key(12), WorkClass::Native, None, vec![z_on_pool]).is_err(), "z_root forbidden outside Gemm");
    let thief = ptx(&c, 13, ProgTx::PostOp { job_id: id, op: 2, z_root: [0; 32], x_root: [1; 32] });
    assert!(mine_at(&mut c, &key(12), WorkClass::Native, None, vec![thief]).is_err(), "third party");
}

#[test]
fn program_validation_and_price_floor() {
    let (x0, ws, ops) = demo_cnn(N, 4);
    let x0_root = cols_root_of(&x0, N);
    let w_roots: Vec<Hash> = ws.iter().map(|w| rows_root_of(w, N)).collect();
    let mut c = new_chain();
    let mut try_submit = |ops: Vec<Op>, roots: Vec<Hash>, escrow: u64| {
        let t = ptx(&c, 10, ProgTx::Submit { x0_root, w_roots: roots, ops, escrow, deadline: 900 });
        mine_at(&mut c, &key(12), WorkClass::Native, None, vec![t]).is_ok()
    };
    // price floor PER GEMM: 4 Gemms require at least 4 x MIN_ESCROW
    assert!(!try_submit(ops.clone(), w_roots.clone(), 3 * COIN), "below the floor");
    // reference: the valid program passes with a sufficient escrow
    assert!(try_submit(ops.clone(), w_roots.clone(), 40 * COIN));
    // operation that reads a tensor not yet computed
    let mut fwd = ops.clone();
    fwd[0] = Op::Map { x: 5, map: MapSpec::Transpose };
    assert!(!try_submit(fwd, w_roots.clone(), 40 * COIN));
    // nonexistent weight
    let mut nw = ops.clone();
    nw[1] = Op::Gemm { w: 9, x: 1, shift: 6, relu: true };
    assert!(!try_submit(nw, w_roots.clone(), 40 * COIN));
    // program without any Gemm (nothing to mine)
    assert!(!try_submit(vec![Op::Map { x: 0, map: MapSpec::Transpose }], w_roots.clone(), 40 * COIN));
    // im2col that overflows the n x n tensor (36*... rows)
    let big = vec![Op::Map { x: 0, map: MapSpec::Im2col { c_in: 64, h: 8, w: 8, batch: 2, kh: 3, kw: 3, pad: 1 } }, Op::Gemm { w: 0, x: 1, shift: 6, relu: true }];
    assert!(!try_submit(big, w_roots.clone(), 40 * COIN));
    // pooling whose k does not divide the size
    let bad_pool = vec![Op::Pool { x: 0, pool: PoolSpec::Avg { c: 4, h: 8, w: 8, batch: 2, k: 3 } }, Op::Gemm { w: 0, x: 1, shift: 6, relu: true }];
    assert!(!try_submit(bad_pool, w_roots.clone(), 40 * COIN));
    // too many operations
    let many: Vec<Op> = (0..MAX_OPS + 1).map(|k| if k == 0 { Op::Gemm { w: 0, x: 0, shift: 6, relu: true } } else { Op::Add { a: 0, b: k as u8 - 1, relu: false } }).collect();
    assert!(!try_submit(many, w_roots.clone(), 500 * COIN));
    // shift out of bounds
    let sh = vec![Op::Gemm { w: 0, x: 0, shift: 30, relu: true }];
    assert!(!try_submit(sh, w_roots, 40 * COIN));
    assert!(c.check_supply_invariant());
}
