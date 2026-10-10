//! Attacks by structured inputs against the consensus work (see src/attack.rs and docs/03).
use vearl_core::attack::*;
use vearl_core::work::*;

fn all_tiles(p: &WorkParams) -> Vec<(usize, usize)> {
    (0..p.tiles()).flat_map(|i| (0..p.tiles()).map(move |j| (i, j))).collect()
}

#[test]
fn low_rank_inputs_yield_valid_jackpots_without_the_dense_gemm() {
    // deliberately NON-secure parameters (fold_k > rank) so that the attack is visible
    let p = WorkParams { n: 128, tile: 16, rank: 16, fold_k: 64 };
    for (a0, b0, seed) in [(0usize, 0usize, 1u64), (4, 4, 2), (0, 8, 3), (16, 16, 4)] {
        let lr = LowRank::random(p.n, a0, b0, seed);
        let (a, b) = lr.dense(p.n);
        assert!(a.iter().chain(b.iter()).all(|v| (*v as i32).abs() <= VAL_MAX), "inputs valid for the protocol");
        let sigma = [seed as u8 + 40; 32];
        let (honest, _) = run_noisy_gemm(&a, &b, &sigma, &p);
        let atk = attack_tiles(&lr, &sigma, &p, &all_tiles(&p));
        for ((ti, tj), jp) in &atk.jackpots {
            assert_eq!(*jp, honest.jackpots[ti * p.tiles() + tj], "a0={a0} b0={b0} tile ({ti},{tj})");
        }
    }
}

#[test]
fn cost_model_matches_the_real_attack_operation_count() {
    let p = WorkParams { n: 128, tile: 16, rank: 16, fold_k: 32 };
    let lr = LowRank::random(p.n, 4, 4, 9);
    let atk = attack_tiles(&lr, &[5u8; 32], &p, &all_tiles(&p));
    let per_tile = atk.macs as f64 / (p.tiles() * p.tiles()) as f64;
    let model = attack_macs_per_tile(&p, 4, 4);
    assert!((per_tile - model).abs() / model < 1e-9, "real count {per_tile} vs model {model}");
}

#[test]
fn old_parameters_were_broken_and_new_ones_close_the_gap() {
    // the parameters of the previous GPU bench: rank 16, folding every 256 steps, n = 4096
    let old = WorkParams { n: 4096, tile: 16, rank: 16, fold_k: 256 };
    let r_old = attack_cost_ratio(&old, 0, 0);
    assert!(r_old < 0.1, "with rank=16, fold_k=256 an attempt costs {r_old:.3} times the honest computation");
    assert!(old.security_check().is_err(), "these parameters must be refused");

    let new = WorkParams { n: 4096, tile: 16, rank: 63, fold_k: 64 };
    assert!(new.security_check().is_ok());
    for (a0, b0) in [(0, 0), (8, 8), (32, 32), (0, 63)] {
        let r = attack_cost_ratio(&new, a0, b0);
        assert!(r > 0.95, "rank=63, fold_k=64, inputs of rank ({a0},{b0}): ratio {r:.3}");
    }
    assert!(WorkParams::TESTNET.security_check().is_ok());
}

#[test]
fn security_check_rejects_non_int8_safe_and_malformed_parameters() {
    let bad = |p: WorkParams| p.security_check().is_err();
    assert!(bad(WorkParams { n: 128, tile: 16, rank: 64, fold_k: 64 }), "rank 64: A+E can overflow int8");
    assert!(bad(WorkParams { n: 128, tile: 8, rank: 32, fold_k: 32 }), "tile != 16");
    assert!(bad(WorkParams { n: 100, tile: 16, rank: 32, fold_k: 32 }));
    assert!(bad(WorkParams { n: 128, tile: 16, rank: 32, fold_k: 48 }), "fold_k does not divide n");
}
