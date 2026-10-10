//! RESEARCH TOOL (not consensus): attack by STRUCTURED inputs against NoisyGEMM-T.
//!
//! Tested hypothesis: an accomplice client chooses low-rank A and B (the extreme case: A = B = 0). The miner
//! (or the attacker itself) can then obtain the tile jackpots WITHOUT doing the dense GEMM:
//!   A' = A + EL*ER = [UA | EL] * [VA ; ER]   (rank a = a0 + r)
//!   B' = B + FL*FR = [WB | FL] * [ZB ; FR]   (rank b = b0 + r)
//!   acc_s(tile) = UA'_rows * G_s * ZB'_columns,   G_s = sum_{k < K_s} VA'[:,k] * WB'[k,:]   (a x b)
//! G_s is common to ALL the tiles; per tile and per fold step there remain ~ T^2 * b MACs instead of
//! T^2 * fold_k. Attacker gain ~ fold_k / b with b >= rank. This module implements the real attack
//! (the jackpots obtained are IDENTICAL to those of the honest computation) and its cost model, to choose the parameters.
use crate::crypto::Hash;
use crate::work::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;

pub struct LowRank {
    pub a0: usize,
    pub b0: usize,
    pub ua: Vec<i32>, // n x a0
    pub va: Vec<i32>, // a0 x n
    pub wb: Vec<i32>, // n x b0
    pub zb: Vec<i32>, // b0 x n
}

impl LowRank {
    /// Factors with entries in {-1,0,1}: |A| <= a0 <= 64, so A stays in [-64,64] (valid for the protocol).
    pub fn random(n: usize, a0: usize, b0: usize, seed: u64) -> Self {
        assert!(a0 <= 64 && b0 <= 64);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut g = |len: usize| -> Vec<i32> { (0..len).map(|_| rng.gen_range(-1..=1)).collect() };
        let (ua, va, wb, zb) = (g(n * a0), g(a0 * n), g(n * b0), g(b0 * n));
        Self { a0, b0, ua, va, wb, zb }
    }

    /// The DENSE matrices A, B (what the protocol sees / what the honest computation handles).
    pub fn dense(&self, n: usize) -> (Vec<i8>, Vec<i8>) {
        let mut a = vec![0i8; n * n];
        let mut b = vec![0i8; n * n];
        for i in 0..n {
            for k in 0..n {
                a[i * n + k] = (0..self.a0).map(|t| self.ua[i * self.a0 + t] * self.va[t * n + k]).sum::<i32>() as i8;
                b[i * n + k] = (0..self.b0).map(|t| self.wb[i * self.b0 + t] * self.zb[t * n + k]).sum::<i32>() as i8;
            }
        }
        (a, b)
    }
}

pub struct AttackResult {
    pub jackpots: Vec<((usize, usize), Hash)>,
    /// MACs actually executed (common part G_s + X per (tile row, step) + products per tile).
    pub macs: u64,
}

/// Computes the jackpots of the requested tiles WITHOUT dense GEMM.
pub fn attack_tiles(lr: &LowRank, sigma: &Hash, p: &WorkParams, tiles: &[(usize, usize)]) -> AttackResult {
    let (n, t, r, fk) = (p.n, p.tile, p.rank, p.fold_k);
    let noise = derive_noise(sigma, p);
    let (a, b) = (lr.a0 + r, lr.b0 + r);
    let folds = n / fk;
    let mut macs = 0u64;

    // augmented factors: A' = UA' * VA', B' = WB' * ZB'
    let ua_p = |i: usize, u: usize| if u < lr.a0 { lr.ua[i * lr.a0 + u] } else { noise.el[i * r + (u - lr.a0)] };
    let va_p = |u: usize, k: usize| if u < lr.a0 { lr.va[u * n + k] } else { noise.er[(u - lr.a0) * n + k] };
    let wb_p = |k: usize, v: usize| if v < lr.b0 { lr.wb[k * lr.b0 + v] } else { noise.fl[k * r + (v - lr.b0)] };
    let zb_p = |v: usize, j: usize| if v < lr.b0 { lr.zb[v * n + j] } else { noise.fr[(v - lr.b0) * n + j] };

    // cumulative G_s (part common to all the tiles)
    let mut gs: Vec<Vec<i32>> = Vec::with_capacity(folds);
    let mut g = vec![0i32; a * b];
    for s in 0..folds {
        for k in s * fk..(s + 1) * fk {
            for u in 0..a {
                let x = va_p(u, k);
                if x == 0 {
                    macs += b as u64; // counted anyway: a dense implementation does not skip zeros
                    continue;
                }
                for v in 0..b {
                    g[u * b + v] = g[u * b + v].wrapping_add(x.wrapping_mul(wb_p(k, v)));
                }
                macs += b as u64;
            }
        }
        gs.push(g.clone());
    }

    let mut xcache: HashMap<(usize, usize), Vec<i32>> = HashMap::new(); // (ti, s) -> X (t x b)
    let mut out = Vec::new();
    for &(ti, tj) in tiles {
        let mut fold = [0u32; FOLD_WORDS];
        for s in 0..folds {
            let x = xcache.entry((ti, s)).or_insert_with(|| {
                let mut x = vec![0i32; t * b];
                for r0 in 0..t {
                    for u in 0..a {
                        let w = ua_p(ti * t + r0, u);
                        for v in 0..b {
                            x[r0 * b + v] = x[r0 * b + v].wrapping_add(w.wrapping_mul(gs[s][u * b + v]));
                        }
                    }
                }
                macs += (t * a * b) as u64;
                x
            });
            let mut acc = vec![0i32; t * t];
            for r0 in 0..t {
                for c in 0..t {
                    let mut sum = 0i32;
                    for v in 0..b {
                        sum = sum.wrapping_add(x[r0 * b + v].wrapping_mul(zb_p(v, tj * t + c)));
                    }
                    acc[r0 * t + c] = sum;
                }
            }
            macs += (t * t * b) as u64;
            fold_step(&mut fold, &acc, s);
        }
        out.push(((ti, tj), jackpot_from_fold(sigma, ti, tj, &fold)));
    }
    AttackResult { jackpots: out, macs }
}

/// Honest cost of a tile (MACs): T^2 * n.
pub fn honest_macs_per_tile(p: &WorkParams) -> f64 {
    (p.tile * p.tile * p.n) as f64
}

/// AVERAGE cost of the attack per tile when ALL the tiles of a sigma are mined (the common cost is amortised).
pub fn attack_macs_per_tile(p: &WorkParams, a0: usize, b0: usize) -> f64 {
    let (n, t, r, fk) = (p.n as f64, p.tile as f64, p.rank as f64, p.fold_k as f64);
    let (a, b) = (a0 as f64 + r, b0 as f64 + r);
    let folds = n / fk;
    let tiles = n / t;
    let shared = n * a * b; // G_s
    let per_row = tiles * folds * t * a * b; // X for each (tile row, step)
    let per_tile = tiles * tiles * folds * t * t * b;
    (shared + per_row + per_tile) / (tiles * tiles)
}

/// Attacker cost / honest cost ratio per attempt. < 1: the structured input makes mining cheaper.
pub fn attack_cost_ratio(p: &WorkParams, a0: usize, b0: usize) -> f64 {
    attack_macs_per_tile(p, a0, b0) / honest_macs_per_tile(p)
}
