//! Wider networks: the same protocol (tensor programs, fraud proofs) on a chain whose parameter
//! n is 256 or 512. The kernel, the Merkle trees and the proofs are generic in n; only the cost per attempt
//! (proportional to n) and the size of the proofs (O(n)) change.
use ed25519_dalek::SigningKey;
use vearl_core::chain::*;
use vearl_core::crypto::*;
use vearl_core::mlp::{cols_root_of, rows_root_of};
use vearl_core::prog::*;
use vearl_core::state::*;
use vearl_core::types::*;
use vearl_core::work::*;

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
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

/// Runs the demonstration CNN on a chain of dimension `n`, with possible falsification of GEMM `cheat`.
fn run(n: usize, cheat: Option<usize>) -> (Chain, JobId, Vec<Vec<i8>>, Vec<Option<Vec<i32>>>, Vec<Vec<i8>>, Vec<Op>) {
    let p = WorkParams { n, tile: 16, rank: 63, fold_k: 64 };
    p.security_check().unwrap();
    let alloc: Vec<(PubKey, u64)> = [10u8, 11, 13].iter().map(|s| (pubkey(&key(*s)), 1000 * COIN)).collect();
    let mut c = Chain::genesis(p, &alloc, 1_000, u64::MAX / 64);
    let (x0, ws, ops) = demo_cnn(n, 5);
    let x0_root = cols_root_of(&x0, n);
    let w_roots: Vec<Hash> = ws.iter().map(|w| rows_root_of(w, n)).collect();
    let id = prog_id(&pubkey(&key(10)), c.state.nonce(&pubkey(&key(10))), &x0_root, &w_roots, &ops);
    let gemms = ops.iter().filter(|o| o.is_gemm()).count() as u64;
    let submit = ptx(&c, 10, ProgTx::Submit { x0_root, w_roots, ops: ops.clone(), escrow: 30 * COIN * gemms, deadline: 900 });
    mine_at(&mut c, &key(12), WorkClass::Native, None, vec![submit]).unwrap();
    let claim = ptx(&c, 11, ProgTx::Claim { job_id: id });
    mine_at(&mut c, &key(12), WorkClass::Native, None, vec![claim]).unwrap();
    let mut tens = vec![x0];
    let mut zs = vec![];
    for (k, op) in ops.iter().enumerate() {
        let (mut z, mut out) = eval_op(op, &tens, &ws, n);
        if let Op::Gemm { w, x, shift, relu } = op {
            let cm = CommittedMatrices::new(ws[*w as usize].clone(), tens[*x as usize].clone(), &p);
            let zm = mine_at(&mut c, &key(11), WorkClass::Op { job: id, op: k as u8 }, Some(&cm), vec![]).unwrap();
            assert_eq!(&zm, z.as_ref().unwrap(), "n={n}: the Op block produces the true GEMM {k}");
            if cheat == Some(k) {
                let zz = z.as_mut().unwrap();
                zz[3 * n + 5] += 1 << *shift;
                out = zz.iter().map(|&v| act_fn(v, *shift, *relu)).collect();
            }
        }
        let z_root = z.as_ref().map_or([0u8; 32], |zz| c_tree(zz, n).root());
        let post = ptx(&c, 11, ProgTx::PostOp { job_id: id, op: k as u8, z_root, x_root: cols_root_of(&out, n) });
        mine_at(&mut c, &key(12), WorkClass::Native, None, vec![post]).unwrap();
        zs.push(z);
        tens.push(out);
    }
    (c, id, tens, zs, ws, ops)
}

#[test]
fn wider_chains_run_honest_programs_and_prove_fraud() {
    for n in [256usize, 512, 1024] {
        // honest: paid, and identical to the program engine
        let (mut c, id, tens, zs, ws, ops) = run(n, None);
        assert!(matches!(c.state.progs[&id].status, ProgStatus::Complete { .. }), "n={n}");
        assert!(find_fraud_proof(&ops, &ws, &tens, &zs, n).is_none());
        let (x0, ..) = demo_cnn(n, 5);
        assert_eq!(run_program(&x0, &ws, &ops, n).0, tens, "n={n}");
        let accept = ptx(&c, 10, ProgTx::Accept { job_id: id });
        mine_at(&mut c, &key(12), WorkClass::Native, None, vec![accept]).unwrap();
        assert_eq!(c.state.progs[&id].status, ProgStatus::Settled);
        assert!(c.check_supply_invariant());
        // fraud: falsified GEMM, proven with openings of size O(n)
        let (mut c, id, tens, zs, ws, ops) = run(n, Some(4));
        let (op, kind) = find_fraud_proof(&ops, &ws, &tens, &zs, n).expect("fraud found");
        assert!(matches!(kind, ProgFraud::Gemm { .. }) && op == 4);
        let f = ptx(&c, 10, ProgTx::Fraud { job_id: id, op, kind });
        mine_at(&mut c, &key(12), WorkClass::Native, None, vec![f]).unwrap();
        assert_eq!(c.state.progs[&id].status, ProgStatus::Slashed, "n={n}");
        assert!(c.check_supply_invariant());
    }
}

#[test]
fn unsupported_widths_are_refused_at_genesis_rules() {
    for (n, ok) in [(128usize, true), (256, true), (512, true), (1024, true), (192, false), (100, false)] {
        assert_eq!(WorkParams { n, tile: 16, rank: 63, fold_k: 64 }.security_check().is_ok(), ok, "n={n}");
    }
}
