//! Report on the attack by structured inputs: attacker cost / honest cost per attempt.
//! attackreport            -> analytical table (n=4096) + real CPU time measurement (n=1024)
use vearl_core::attack::*;
use vearl_core::work::*;
use std::time::Instant;

fn main() {
    println!("Attacker cost / honest cost report (n=4096, MACs per attempt). < 1.00 = mining cheaper for the attacker.\n");
    let ranks = [16usize, 32, 48, 63];
    let folds = [16usize, 32, 64, 128, 256, 512];
    for (a0, b0) in [(0usize, 0usize), (8, 8), (32, 32)] {
        println!("rank inputs (a0={a0}, b0={b0}){}", if a0 == 0 { "  [A = B = 0]" } else { "" });
        print!("| rang \\ fold_k |");
        for f in folds {
            print!(" {f} |");
        }
        println!("\n|---|{}", "---|".repeat(folds.len()));
        for r in ranks {
            print!("| {r} |");
            for f in folds {
                let p = WorkParams { n: 4096, tile: 16, rank: r, fold_k: f };
                let ratio = attack_cost_ratio(&p, a0, b0);
                let ok = p.security_check().is_ok();
                print!(" {ratio:.2}{} |", if ok { " ok" } else { "" });
            }
            println!();
        }
        println!();
    }
    println!("('ok' = parameters accepted by WorkParams::security_check)\n");

    // Real measurement (CPU, naive loops on both sides) over a COMPLETE sigma (all the tiles): n = 512.
    // The noise A', B' is computed once per sigma on the honest side (as the GPU kernel does).
    println!("Real CPU times, n=512, ALL the tiles of a sigma (A = B = 0): honest computation vs attack");
    for (r, f) in [(16usize, 256usize), (63, 64)] {
        let p = WorkParams { n: 512, tile: 16, rank: r, fold_k: f };
        let lr = LowRank::random(p.n, 0, 0, 1);
        let (a, b) = lr.dense(p.n);
        let sigma = [3u8; 32];
        let t0 = Instant::now();
        let (honest, _) = run_noisy_gemm(&a, &b, &sigma, &p);
        let honest_t = t0.elapsed();
        let tiles: Vec<(usize, usize)> = (0..p.tiles()).flat_map(|i| (0..p.tiles()).map(move |j| (i, j))).collect();
        let t1 = Instant::now();
        let atk = attack_tiles(&lr, &sigma, &p, &tiles);
        let attack_t = t1.elapsed();
        let same = atk.jackpots.iter().all(|((i, j), x)| *x == honest.jackpots[i * p.tiles() + j]);
        println!(
            "  rank={r:>2} fold_k={f:>3}: honest {:>8.1} ms | attack {:>8.1} ms | identical jackpots: {same} | time ratio {:.2} | MACs ratio (model) {:.2}",
            honest_t.as_secs_f64() * 1e3,
            attack_t.as_secs_f64() * 1e3,
            attack_t.as_secs_f64() / honest_t.as_secs_f64(),
            attack_cost_ratio(&p, 0, 0)
        );
    }
    println!("
(CPU times mix arithmetic cost and loop efficiency; the MACs ratio is the reference measure.)");
}
