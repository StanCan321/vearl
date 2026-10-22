//! The GPU kernel (int8 tensor cores + folding + blake3 on GPU) must be BIT-FOR-BIT
//! identical to the CPU reference: this is what makes the blocks verifiable by a node without a GPU.
use vearl_core::gpu::GpuJob;
use vearl_core::work::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn rand_mat(rng: &mut StdRng, n: usize) -> Vec<i8> {
    (0..n * n).map(|_| rng.gen_range(-64..=64) as i8).collect()
}

fn check(p: WorkParams, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let (a, b) = (rand_mat(&mut rng, p.n), rand_mat(&mut rng, p.n));
    let job = GpuJob::new(&a, &b, &p).expect("gpu init");
    for k in 0..3u8 {
        let sigma = [k + 1; 32];
        let (cpu, noise) = run_noisy_gemm(&a, &b, &sigma, &p);
        let gpu = job.run(&sigma, true).expect("gpu run");
        assert_eq!(gpu.c_noised.as_ref().unwrap(), &cpu.c_noised, "identical C' (n={})", p.n);
        assert_eq!(gpu.jackpots, cpu.jackpots, "identical jackpots (n={}, fold_k={})", p.n, p.fold_k);
        // and the decoding gives back the true product
        let c = decode(gpu.c_noised.as_ref().unwrap(), &a, &b, &noise, &p);
        assert_eq!(c, plain_gemm(&a, &b, p.n));
    }
}

/// The exact GEMM on tensor cores returns the SAME integer as the CPU computation (this is the useful computation of a job), including at the extremes.
#[test]
fn gpu_exact_gemm_equals_cpu() {
    let mut rng = StdRng::seed_from_u64(77);
    for n in [128usize, 256, 384, 1024] {
        let (a, b) = (rand_mat(&mut rng, n), rand_mat(&mut rng, n));
        assert_eq!(vearl_core::gpu::exact_gemm(&a, &b, n).unwrap(), plain_gemm(&a, &b, n), "n = {n}");
    }
    let n = 512;
    let (hi, lo) = (vec![64i8; n * n], vec![-64i8; n * n]);
    assert_eq!(vearl_core::gpu::exact_gemm(&hi, &lo, n).unwrap(), plain_gemm(&hi, &lo, n), "extremes");
    assert!(vearl_core::gpu::exact_gemm(&hi[..100 * 100], &lo[..100 * 100], 100).is_none(), "size not handled");
}

#[test]
fn gpu_matches_cpu_testnet_params() {
    check(WorkParams::TESTNET, 1);
}

#[test]
fn gpu_matches_cpu_larger_and_other_fold() {
    check(WorkParams { n: 256, tile: 16, rank: 32, fold_k: 64 }, 2);
    check(WorkParams { n: 256, tile: 16, rank: 16, fold_k: 32 }, 3);
}

/// The REAL testnet size (n = 1024, rank 63, fold_k 64): the GPU agrees with the CPU reference on C' and the jackpots, and what the batch + GPU-side
/// filter path of the miners returns is exactly what the CPU reference says (checked on the first sigmas of a real-size batch).
#[test]
fn gpu_matches_cpu_at_the_real_testnet_size() {
    let p = WorkParams { n: 1024, tile: 16, rank: 63, fold_k: 64 };
    check(p, 5);
    let mut rng = StdRng::seed_from_u64(9);
    let (a, b) = (rand_mat(&mut rng, p.n), rand_mat(&mut rng, p.n));
    let job = GpuJob::new(&a, &b, &p).unwrap();
    let sigmas: Vec<[u8; 32]> = (0..job.batch_size() as u8).map(|i| [i ^ 0x77; 32]).collect();
    let target = u64::MAX / 600; // about 7 winners per sigma
    let got = job.run_batch_hits(&sigmas, target).unwrap();
    let mut want = Vec::new();
    for (d, sg) in sigmas.iter().take(3).enumerate() {
        let (cpu, _) = run_noisy_gemm(&a, &b, sg, &p);
        for (i, jp) in cpu.jackpots.iter().enumerate() {
            if meets_target(jp, target) {
                want.push((d, i, *jp));
            }
        }
    }
    assert!(want.len() > 5, "the test needs winners ({})", want.len());
    let got3: Vec<_> = got.iter().filter(|h| h.0 < 3).cloned().collect();
    assert_eq!(got3, want);
    assert!(got.iter().any(|h| h.0 >= 3), "the other sigmas of the batch have winners too");
}

use ed25519_dalek::SigningKey;
use vearl_core::chain::*;
use vearl_core::crypto::*;
use vearl_core::state::*;
use vearl_core::types::*;

#[test]
fn gpu_mined_blocks_are_accepted_by_the_cpu_validator_including_job_blocks() {
    let p = WorkParams::TESTNET;
    let (alice, bob) = (SigningKey::from_bytes(&[10; 32]), SigningKey::from_bytes(&[11; 32]));
    let alloc = [(pubkey(&alice), 1000 * COIN), (pubkey(&bob), 100 * COIN)];
    let mut chain = Chain::genesis(p, &alloc, 1_000, u64::MAX / 64);

    // Native block mined on GPU, validated by the CPU
    let ts = chain.tip_ts() + BLOCK_TIME;
    let out = chain.mine_gpu(&bob, WorkClass::Native, None, vec![], ts).unwrap();
    chain.validate_and_apply(out.block).unwrap();

    // job: alice commits A,B; bob claims it then mines WHILE computing alice's product, on GPU
    let mut rng = StdRng::seed_from_u64(99);
    let (a, b) = (rand_mat(&mut rng, p.n), rand_mat(&mut rng, p.n));
    let cm = CommittedMatrices::new(a.clone(), b.clone(), &p);
    let id = job_id(&pubkey(&alice), 0, &cm.a_root(), &cm.b_root());
    let submit = make_tx(&chain, &alice, 0, TxKind::SubmitJob { a_root: cm.a_root(), b_root: cm.b_root(), escrow: 10 * COIN, deadline: 100 });
    let ts = chain.tip_ts() + BLOCK_TIME;
    let out = chain.mine_gpu(&bob, WorkClass::Native, None, vec![submit], ts).unwrap();
    chain.validate_and_apply(out.block).unwrap();
    let claim = make_tx(&chain, &bob, 0, TxKind::ClaimJob { job_id: id });
    let ts = chain.tip_ts() + BLOCK_TIME;
    let out = chain.mine_gpu(&bob, WorkClass::Native, None, vec![claim], ts).unwrap();
    chain.validate_and_apply(out.block).unwrap();

    let ts = chain.tip_ts() + BLOCK_TIME;
    let out = chain.mine_gpu(&bob, WorkClass::Job(id), Some(&cm), vec![], ts).unwrap();
    assert_eq!(out.c, plain_gemm(&a, &b, p.n), "the GPU does produce the client's true A*B");
    chain.validate_and_apply(out.block).unwrap();
    assert!(chain.check_supply_invariant());
}

#[test]
fn gpu_matches_cpu_at_wider_secure_params() {
    // parameters accepted by security_check (rank 63, fold_k 64) on wider chains
    check(WorkParams { n: 256, tile: 16, rank: 63, fold_k: 64 }, 4);
    check(WorkParams { n: 512, tile: 16, rank: 63, fold_k: 64 }, 5);
}

/// Several GPUs share the search for a block (one thread per GPU, each taking its own batches of sigmas): the blocks they produce are
/// accepted by the CPU validator, native and job blocks alike. With one physical GPU the test puts three workers on device 0, which
/// exercises exactly the same code (threads, batches handed out in turn, the winner chosen among them).
#[test]
fn several_gpus_share_the_search_and_the_blocks_are_valid() {
    use vearl_core::gpu::GpuPool;
    let p = WorkParams::TESTNET;
    let (alice, bob) = (SigningKey::from_bytes(&[12; 32]), SigningKey::from_bytes(&[13; 32]));
    let alloc = [(pubkey(&alice), 1000 * COIN), (pubkey(&bob), 100 * COIN)];
    // a harder target than the other tests (about 64 sigmas per block) so that the search takes several batches
    let mut chain = Chain::genesis(p, &alloc, 1_000, u64::MAX >> 18);
    let pool_for = |a: &[i8], b: &[i8]| GpuPool { jobs: (0..3).map(|_| GpuJob::new_on(0, a, b, &p).expect("gpu init")).collect() };
    let mut cache: GpuCache = None;
    let mut mine = |chain: &mut Chain, class: WorkClass, data: Option<&CommittedMatrices>, txs: Vec<SignedTx>| {
        let ts = chain.tip_ts() + BLOCK_TIME;
        let t = chain.template(class, data).unwrap();
        if !matches!(&cache, Some((ra, rb, _)) if (*ra, *rb) == (t.cm.a_root(), t.cm.b_root())) {
            cache = Some((t.cm.a_root(), t.cm.b_root(), pool_for(&t.cm.a, &t.cm.b)));
        }
        let before = SIGMA_DONE.load(std::sync::atomic::Ordering::Relaxed);
        let out = mine_template(&t, &bob, txs, ts, None, true, &mut cache).unwrap();
        assert!(SIGMA_DONE.load(std::sync::atomic::Ordering::Relaxed) > before, "the GPUs count their work");
        out
    };
    for _ in 0..3 {
        let out = mine(&mut chain, WorkClass::Native, None, vec![]);
        chain.validate_and_apply(out.block).unwrap();
    }
    let mut rng = StdRng::seed_from_u64(98);
    let (a, b) = (rand_mat(&mut rng, p.n), rand_mat(&mut rng, p.n));
    let cm = CommittedMatrices::new(a.clone(), b.clone(), &p);
    let id = job_id(&pubkey(&alice), 0, &cm.a_root(), &cm.b_root());
    let submit = make_tx(&chain, &alice, 0, TxKind::SubmitJob { a_root: cm.a_root(), b_root: cm.b_root(), escrow: 10 * COIN, deadline: 100 });
    let out = mine(&mut chain, WorkClass::Native, None, vec![submit]);
    chain.validate_and_apply(out.block).unwrap();
    let claim = make_tx(&chain, &bob, 0, TxKind::ClaimJob { job_id: id });
    let out = mine(&mut chain, WorkClass::Native, None, vec![claim]);
    chain.validate_and_apply(out.block).unwrap();
    let out = mine(&mut chain, WorkClass::Job(id), Some(&cm), vec![]);
    assert_eq!(out.c, plain_gemm(&a, &b, p.n), "several GPUs still produce the client's true A*B");
    chain.validate_and_apply(out.block).unwrap();
    assert!(chain.check_supply_invariant());
}

/// The cancellation of a search shared by several GPUs is honoured (a block from the network arrived).
#[test]
fn a_search_on_several_gpus_stops_when_cancelled() {
    use vearl_core::gpu::GpuPool;
    use std::sync::atomic::AtomicBool;
    let p = WorkParams::TESTNET;
    let bob = SigningKey::from_bytes(&[14; 32]);
    let mut chain = Chain::genesis(p, &[(pubkey(&bob), 100 * COIN)], 1_000, 1); // target 1: never found
    let t = chain.template(WorkClass::Native, None).unwrap();
    let mut cache: GpuCache = Some((t.cm.a_root(), t.cm.b_root(), GpuPool { jobs: (0..2).map(|_| GpuJob::new_on(0, &t.cm.a, &t.cm.b, &p).unwrap()).collect() }));
    let cancel = AtomicBool::new(true);
    let r = mine_template(&t, &bob, vec![], chain.tip_ts() + BLOCK_TIME, Some(&cancel), true, &mut cache);
    assert_eq!(r.err().as_deref(), Some("cancelled"));
}

#[test]
fn gpu_batch_equals_sigma_by_sigma() {
    // the batch (noise derived on several threads, concatenated kernels) must give exactly the same jackpots
    for p in [WorkParams::TESTNET, WorkParams { n: 256, tile: 16, rank: 63, fold_k: 64 }] {
        let mut rng = StdRng::seed_from_u64(11);
        let (a, b) = (rand_mat(&mut rng, p.n), rand_mat(&mut rng, p.n));
        let job = GpuJob::new(&a, &b, &p).unwrap();
        for s in [1usize, 5, job.batch_size() + 3] {
            let sigmas: Vec<[u8; 32]> = (0..s).map(|i| [i as u8 ^ 0x5a; 32]).collect();
            let lot = job.run_batch(&sigmas).unwrap();
            assert_eq!(lot.len(), s);
            for (i, sg) in sigmas.iter().enumerate() {
                assert_eq!(lot[i], job.run(sg, false).unwrap().jackpots, "n={} sigma {i}/{s}", p.n);
                assert_eq!(lot[i], job.run_batch_host_noise(&sigmas).unwrap()[i], "GPU noise = CPU noise (n={})", p.n);
                let (cpu, _) = run_noisy_gemm(&a, &b, sg, &p);
                assert_eq!(lot[i], cpu.jackpots, "lot = reference CPU (n={}, sigma {i})", p.n);
            }
        }
    }
}

/// The GPU-side filter (`run_batch_hits`, what the miners call) returns EXACTLY the attempts that scanning the full batch with `meets_target`
/// finds, in the order (sigma, tile), whatever the target: none, a few, many, and so many that the GPU's short list overflows (full copy).
#[test]
fn gpu_hits_equal_scanning_the_full_batch() {
    for p in [WorkParams::TESTNET, WorkParams { n: 256, tile: 16, rank: 63, fold_k: 64 }] {
        let mut rng = StdRng::seed_from_u64(23);
        let (a, b) = (rand_mat(&mut rng, p.n), rand_mat(&mut rng, p.n));
        let job = GpuJob::new(&a, &b, &p).unwrap();
        let tiles = p.tiles();
        let sigmas: Vec<[u8; 32]> = (0..12u8).map(|i| [i.wrapping_mul(29) ^ 0x3c; 32]).collect();
        let lot = job.run_batch(&sigmas).unwrap();
        let mut tops: Vec<u64> = lot.iter().flatten().map(|jp| u64::from_be_bytes(jp[..8].try_into().unwrap())).collect();
        tops.sort_unstable();
        let q = |f: f64| tops[((tops.len() - 1) as f64 * f) as usize];
        // 0 = nothing can win; q(0) = the smallest jackpot itself is NOT below it; q(0)+1 = exactly that one wins; then few, some, all
        for target in [0u64, tops[0], tops[0].wrapping_add(1), q(0.001), q(0.01), q(0.2), q(0.5), u64::MAX] {
            let mut want = Vec::new();
            for (d, row) in lot.iter().enumerate() {
                for (i, jp) in row.iter().enumerate() {
                    if meets_target(jp, target) {
                        want.push((d, i, *jp));
                    }
                }
            }
            let got = job.run_batch_hits(&sigmas, target).unwrap();
            assert_eq!(got, want, "n={} target={target:#x}: {} expected, {} returned", p.n, want.len(), got.len());
            assert!(got.iter().all(|h| h.1 < tiles * tiles));
        }
        // a sigma alone, and the real batch size
        let one = job.run_batch_hits(&sigmas[..1], u64::MAX).unwrap();
        assert_eq!(one.len(), tiles * tiles);
        let many: Vec<[u8; 32]> = (0..job.batch_size() as u32).map(|i| { let mut h = [0u8; 32]; h[..4].copy_from_slice(&i.to_le_bytes()); h }).collect();
        let full = job.run_batch(&many).unwrap();
        let t = u64::MAX / 3000;
        let want: Vec<_> = full.iter().enumerate().flat_map(|(d, r)| r.iter().enumerate().filter(|(_, jp)| meets_target(jp, t)).map(move |(i, jp)| (d, i, *jp))).collect();
        assert_eq!(job.run_batch_hits(&many, t).unwrap(), want, "real batch size, n={}", p.n);
    }
}

#[test]
fn gpu_noise_generation_matches_blake3_xof_for_many_shapes() {
    // the blake3-XOF derive_key stream generated on GPU must give back exactly derive_noise(): several n and ranks
    for (n, rank) in [(128usize, 63usize), (128, 16), (256, 63), (256, 1)] {
        let p = WorkParams { n, tile: 16, rank, fold_k: if rank >= 32 { 64 } else { 32 } };
        let mut rng = StdRng::seed_from_u64(n as u64 + rank as u64);
        let (a, b) = (rand_mat(&mut rng, n), rand_mat(&mut rng, n));
        let job = GpuJob::new(&a, &b, &p).unwrap();
        let sigmas: Vec<[u8; 32]> = (0..6u8).map(|i| [i.wrapping_mul(37) ^ 0xa5; 32]).collect();
        let dev = job.run_batch(&sigmas).unwrap();
        for (i, sg) in sigmas.iter().enumerate() {
            let (cpu, _) = run_noisy_gemm(&a, &b, sg, &p);
            assert_eq!(dev[i], cpu.jackpots, "n={n} rank={rank} sigma {i}");
        }
    }
}

/// Render2 tiles computed on the GPU are bit-identical to the CPU reference (this is what lets a node without a GPU judge a fraud proof).
#[test]
fn gpu_render2_tiles_equal_the_cpu_reference() {
    use vearl_core::render2::*;
    let scenes = [scene_orb(64, 48, 2, 2, 25.0, 1.2, [3; 32]), scene_orb(48, 32, 1, 0, 140.0, 0.9, [4; 32]), scene_prism(48, 32, 3, 2, [5; 32]), scene_prism(32, 32, 1, 1, [6; 32]), scene_watch(64, 48, 2, 2, 30.0, 2.4, [7; 32])];
    for p in scenes {
        render2_validate(&p).unwrap();
        let n = (p.width as u32 / 16) * (p.height as u32 / 16);
        let gpu = vearl_core::gpu::render2_tiles(&p, 0, n).expect("gpu render2");
        assert_eq!(gpu.len(), n as usize);
        for c in 0..n {
            assert_eq!(gpu[c as usize], render2_chunk(&p, c), "tile {c} of a {}x{} scene (spp {}, bounces {})", p.width, p.height, p.spp, p.bounces);
        }
        // a sub-range gives the same tiles
        let part = vearl_core::gpu::render2_tiles(&p, n / 2, n - n / 2).expect("gpu render2 range");
        assert_eq!(part[0], gpu[(n / 2) as usize]);
    }
}

/// Render3 chunks computed on the GPU are bit-identical to the CPU reference, for the test scene, the watch and the cocktail (thin leaves once made a GPU thread loop forever).
#[test]
fn gpu_render3_chunks_equal_the_cpu_reference() {
    use vearl_core::render3::*;
    let scenes = [scene3_mini(48, 32, 2, 2, 6, [31; 32]), scene3_mini(32, 32, 1, 1, 3, [32; 32]), scene3_watch(64, 48, 1, 1, [33; 32]), scene3_cocktail(48, 32, 1, 1, [34; 32]), scene3_textures(64, 48, 2, 1, 5, [36; 32]), scene3_orb(64, 32, 1, 1, [37; 32]), scene3_gallery(64, 32, 1, 1, [38; 32])];
    for p in scenes {
        render3_validate(&p).unwrap();
        let n = render3_chunks(&p);
        let gpu = vearl_core::gpu::render3_chunks_gpu(&p, 0, n).expect("gpu render3");
        assert_eq!(gpu.len(), n as usize);
        for c in 0..n {
            assert_eq!(gpu[c as usize], render3_chunk(&p, c), "chunk {c} of a {}x{} scene (spp {}, passes {}, bounces {})", p.width, p.height, p.spp, p.passes, p.bounces);
        }
    }
}

/// An ellipsoid thinner than 1/16 makes the i64 distance computation overflow: the CPU wraps but a GPU thread could loop forever, so it is refused.
#[test]
fn render3_refuses_ellipsoids_that_could_overflow() {
    use vearl_core::render3::*;
    let mut p = scene3_cocktail(16, 16, 1, 1, [35; 32]);
    render3_validate(&p).unwrap();
    for pt in p.parts.iter_mut() {
        if let Geo3::Ellipsoid { r } = &mut pt.geo {
            r[1] = MIN_ELLIPSOID_R - 1;
        }
    }
    assert!(render3_validate(&p).is_err());
}
