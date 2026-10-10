//! Benchmark of the VEARL GPU kernel: tiles/s rate and folding/hash overhead depending on fold_k.
//! Usage: gpubench [n=4096] [rank=16]
use vearl_core::gpu::GpuJob;
use vearl_core::work::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

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
    }
}
