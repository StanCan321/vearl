//! Benchmark of the VEARL GPU kernel: tiles/s rate and folding/hash overhead depending on fold_k.
//! Usage: gpubench [n=4096] [rank=16]
use vearl_core::gpu::GpuJob;
use vearl_core::work::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

fn s_fold_probe(n: usize) -> bool {
    n >= 1024
}

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let n = *args.first().unwrap_or(&4096);
    let rank = *args.get(1).unwrap_or(&16);
    let only_batch = args.get(2).copied(); // 3rd argument: only measure batches of this size (for ncu profiling)
    let mut rng = StdRng::seed_from_u64(7);
    let mut m = || (0..n * n).map(|_| rng.gen_range(-64..=64) as i8).collect::<Vec<i8>>();
    let (a, b) = (m(), m());
    println!("n={n} rank={rank} | tiles={} per sigma | useful ops per sigma = {:.2} Gop (int8 MAC x2)", (n / 16) * (n / 16), 2.0 * (n as f64).powi(3) / 1e9);
    println!("| fold_k | tile kernel (ms) | noise (ms) | TOPS int8 (tiles) | attempts/s | wall (ms/sigma) |");
    println!("|---|---|---|---|---|---|");
    for fold_k in [32usize, 64, 128, 256, 512, 1024, 2048, 4096, 8192] {
        if only_batch.is_some() || fold_k > n || n % fold_k != 0 {
            continue;
        }
        let p = WorkParams { n, tile: 16, rank, fold_k };
        let job = GpuJob::new(&a, &b, &p).expect("gpu init");
        job.run(&[0u8; 32], false).unwrap(); // warm-up
        let reps = 5;
        let (mut tiles_ms, mut noise_ms) = (0f32, 0f32);
        let t0 = Instant::now();
        for i in 0..reps {
            let r = job.run(&[i as u8 + 1; 32], false).unwrap();
            tiles_ms += r.ms_tiles;
            noise_ms += r.ms_noise;
        }
        let wall = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let (tm, nm) = (tiles_ms / reps as f32, noise_ms / reps as f32);
        let tops = 2.0 * (n as f64).powi(3) / (tm as f64 * 1e-3) / 1e12;
        let attempts = ((n / 16) * (n / 16)) as f64 / (wall / 1e3);
        println!("| {fold_k} | {tm:.2} | {nm:.2} | {tops:.1} | {attempts:.0} | {wall:.1} |");
    }
    // Real mining: batches of sigmas (small n = one sigma does not fill the GPU). fold_k = 64 (security rule, rank 63).
    if n % 64 == 0 {
        let p = WorkParams { n, tile: 16, rank, fold_k: 64 };
        let job = GpuJob::new(&a, &b, &p).expect("gpu init");
        println!("
batches of sigmas (fold_k=64): recommended batch size = {}", job.batch_size());
        println!("| batch | attempts/s | wall (ms/batch) | GPU detail |");
        println!("|---|---|---|---|");
        for s in only_batch.map_or(vec![1usize, 8, 32, 128, 512], |b| vec![b]) {
            if s > 1 && s * n * n > (1 << 30) {
                continue;
            }
            let sig = |k: usize| -> Vec<[u8; 32]> { (0..s).map(|i| { let mut h = [0u8; 32]; h[..8].copy_from_slice(&((k * s + i) as u64).to_le_bytes()); h }).collect() };
            job.run_batch(&sig(1000)).unwrap();
            let reps = 5;
            let t0 = Instant::now();
            let mut ms = [0f32; 2];
            for k in 0..reps {
                let (_, m) = job.run_batch_timed(&sig(k)).unwrap();
                ms[0] += m[0] / reps as f32;
                ms[1] += m[1] / reps as f32;
            }
            let wall = t0.elapsed().as_secs_f64() / reps as f64;
            println!("| {s} | {:.0} | {:.2} | noise {:.3} ms, tiles {:.3} ms ({:.1} TOPS) |", (s * (n / 16) * (n / 16)) as f64 / wall, wall * 1e3, ms[0], ms[1], 2.0 * (n as f64).powi(3) * s as f64 / (ms[1] as f64 * 1e-3) / 1e12);
        }
        // Time of each kernel of a batch (the average of 20 runs after a warm-up).
        for s in [8usize, 32] {
            if s * n * n > (1 << 30) {
                continue;
            }
            let sg: Vec<[u8; 32]> = (0..s).map(|i| { let mut h = [0u8; 32]; h[..8].copy_from_slice(&(i as u64 + 5).to_le_bytes()); h }).collect();
            job.profile_batch(&sg).unwrap();
            let mut acc = [0f32; 8];
            for _ in 0..20 {
                let m = job.profile_batch(&sg).unwrap();
                for i in 0..8 { acc[i] += m[i] / 20.0; }
            }
            let tot: f32 = acc.iter().sum();
            println!("\nper kernel, batch {s} (ms): noise_gen {:.3} | pack16 {:.3} {:.3} {:.3} {:.3} | noise_tc A {:.3} B {:.3} | tile_kernel {:.3} | total {:.3} (noise side {:.0} %)", acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7], tot, 100.0 * (tot - acc[7]) / tot);
        }
        // Where the tile kernel's time goes: the same batch with fewer foldings (fold_k up to n: one folding at the very end).
        if s_fold_probe(n) {
            for fk in [64usize, 128, 256, 1024] {
                if n % fk != 0 { continue; }
                let pp = WorkParams { n, tile: 16, rank, fold_k: fk };
                let jj = GpuJob::new(&a, &b, &pp).expect("gpu init");
                let sg: Vec<[u8; 32]> = (0..32usize).map(|i| { let mut h = [0u8; 32]; h[..8].copy_from_slice(&(i as u64 + 9).to_le_bytes()); h }).collect();
                jj.profile_batch(&sg).unwrap();
                let mut t = 0f32;
                for _ in 0..20 { t += jj.profile_batch(&sg).unwrap()[7] / 20.0; }
                println!("tile_kernel, batch 32, fold_k {fk}: {t:.3} ms ({:.1} TOPS)", 2.0 * (n as f64).powi(3) * 32.0 / (t as f64 * 1e-3) / 1e12);
            }
        }
        // What the miners now call: the GPU filters the jackpots against the target and only the winners come back (target 2^30: no winner in practice).
        println!("
batches of sigmas, GPU-side filter (`run_batch_hits`, what the miners use):");
        println!("| batch | attempts/s | wall (ms/batch) |");
        println!("|---|---|---|");
        for s in only_batch.map_or(vec![8usize, 16, 32, 64, 128, 512], |b| vec![b]) {
            if s > 1 && s * n * n > (1 << 30) {
                continue;
            }
            let sig = |k: usize| -> Vec<[u8; 32]> { (0..s).map(|i| { let mut h = [0u8; 32]; h[..8].copy_from_slice(&((k * s + i) as u64).to_le_bytes()); h }).collect() };
            job.run_batch_hits(&sig(1000), 1 << 30).unwrap();
            let reps = 40;
            let t0 = Instant::now();
            for k in 0..reps {
                job.run_batch_hits(&sig(k), 1 << 30).unwrap();
            }
            let wall = t0.elapsed().as_secs_f64() / reps as f64;
            println!("| {s} | {:.0} | {:.3} |", (s * (n / 16) * (n / 16)) as f64 / wall, wall * 1e3);
        }
    }
}
