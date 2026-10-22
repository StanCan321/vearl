//! Rust <-> CUDA kernel bridge (feature `gpu`). The noise is derived on the CPU side (blake3-XOF);
//! the GPU does the int8 GEMM on tensor cores, the folding and the hashing of the tiles.
use crate::crypto::Hash;
use crate::work::*;
use std::ffi::c_void;

/// (name, major, minor) of CUDA device `dev`, or None if it does not exist.
pub fn device_n(dev: usize) -> Option<(String, i32, i32)> {
    let mut name = [0u8; 256];
    let (mut major, mut minor) = (0i32, 0i32);
    let rc = unsafe { vearl_gpu_device_n(dev as i32, name.as_mut_ptr(), name.len() as i32, &mut major, &mut minor) };
    if rc != 0 {
        return None;
    }
    let end = name.iter().position(|b| *b == 0).unwrap_or(name.len());
    Some((String::from_utf8_lossy(&name[..end]).into_owned(), major, minor))
}

/// (name, major, minor) of GPU 0, or None if there is no usable CUDA GPU.
pub fn device() -> Option<(String, i32, i32)> {
    device_n(0)
}

/// Number of CUDA devices seen by this process (`CUDA_VISIBLE_DEVICES` applies).
pub fn device_count() -> usize {
    unsafe { vearl_gpu_count() }.max(0) as usize
}

/// Every CUDA device with its description, usable or not: (index, name, major, minor).
pub fn devices() -> Vec<(usize, String, i32, i32)> {
    (0..device_count()).filter_map(|d| device_n(d).map(|(n, a, b)| (d, n, a, b))).collect()
}

/// Mining threads per GPU, each with its own `GpuJob` (own buffers and CUDA stream): while one of them is busy on the CPU side of its batch
/// (copying the result, preparing the next sigmas, talking to the pool) the kernels of the other keep the card busy, so it is never idle between
/// two batches. `VEARL_GPU_WORKERS` = 1..4 (default 2; 1 = the old behaviour).
pub fn workers_per_gpu() -> usize {
    std::env::var("VEARL_GPU_WORKERS").ok().and_then(|v| v.trim().parse::<usize>().ok()).map_or(2, |v| v.clamp(1, 4))
}

/// The CUDA devices the miner will use: all the GPUs of capability >= 8.0, or those listed in `VEARL_GPUS` (comma-separated CUDA
/// indices, for example `0,2`; an index of a missing or unsupported GPU is ignored, with a message).
pub fn mining_devices() -> Vec<usize> {
    let all = devices();
    let ok = |d: usize| all.iter().any(|(i, _, major, _)| *i == d && *major >= 8);
    match std::env::var("VEARL_GPUS") {
        Ok(list) if !list.trim().is_empty() => {
            let mut out = Vec::new();
            for tok in list.split(',').map(str::trim).filter(|t| !t.is_empty()) {
                match tok.parse::<usize>() {
                    Ok(d) if ok(d) => out.push(d),
                    _ => eprintln!("[gpu] VEARL_GPUS: \"{tok}\" is not a usable GPU (an index of an Ampere or newer GPU is required), ignored"),
                }
            }
            out
        }
        _ => all.iter().filter(|(_, _, major, _)| *major >= 8).map(|(d, ..)| *d).collect(),
    }
}

/// The miner requires a GPU of capability >= 8.0 (Ampere and newer: cp.async, ldmatrix, mma.m16n8k32 int8). The text lists
/// every GPU the miner will use.
pub fn check_supported() -> Result<String, String> {
    let all = devices();
    if all.is_empty() {
        return Err("no usable CUDA GPU (NVIDIA driver and CUDA required)".into());
    }
    let used = mining_devices();
    if used.is_empty() {
        let (_, name, major, minor) = &all[0];
        return Err(format!("{name} (compute capability {major}.{minor}) is not supported: an Ampere or newer GPU is required (compute capability >= 8.0)"));
    }
    let describe = |d: &usize| {
        let (_, name, major, minor) = all.iter().find(|(i, ..)| i == d).unwrap();
        format!("{name} (compute capability {major}.{minor})")
    };
    if used.len() == 1 {
        return Ok(describe(&used[0]));
    }
    Ok(format!("{} GPUs: {}", used.len(), used.iter().map(|d| format!("#{d} {}", describe(d))).collect::<Vec<_>>().join(", ")))
}

extern "C" {
    fn vearl_gpu_count() -> i32;
    fn vearl_gpu_device_n(dev: i32, name: *mut u8, name_len: i32, major: *mut i32, minor: *mut i32) -> i32;
    fn vearl_gpu_new_on(dev: i32, a: *const i8, b: *const i8, n: i32, rank: i32) -> *mut c_void;
    fn vearl_gpu_free(p: *mut c_void);
    fn vearl_gpu_run(
        p: *mut c_void,
        hdr48: *const u8,
        el: *const i8,
        er: *const i8,
        fl: *const i8,
        fr: *const i8,
        fold_k: i32,
        jackpots_out: *mut u8,
        c_out: *mut i32,
        ms_out: *mut f32,
    ) -> i32;
    fn vearl_gpu_exact_gemm(a: *const i8, b: *const i8, n: i32, c_out: *mut i32) -> i32;
    fn vearl_gpu_run_batch_gen(p: *mut c_void, s: i32, hdr48: *const u8, fold_k: i32, jackpots_out: *mut u8, ms_out: *mut f32) -> i32;
    fn vearl_gpu_profile(p: *mut c_void, s: i32, hdr48: *const u8, fold_k: i32, ms_out: *mut f32) -> i32;
    fn vearl_gpu_run_batch_hits(
        p: *mut c_void,
        s: i32,
        hdr48: *const u8,
        fold_k: i32,
        target: u64,
        hit_count_out: *mut u32,
        hit_idx_out: *mut u32,
        hit_hash_out: *mut u8,
        max_out: i32,
        ms_out: *mut f32,
    ) -> i32;
    fn vearl_gpu_run_batch(
        p: *mut c_void,
        s: i32,
        hdr48: *const u8,
        el: *const i8,
        er: *const i8,
        fl: *const i8,
        fr: *const i8,
        fold_k: i32,
        jackpots_out: *mut u8,
        ms_out: *mut f32,
    ) -> i32;
}

pub struct GpuJob {
    ctx: *mut c_void,
    p: WorkParams,
}

pub struct GpuRun {
    pub jackpots: Vec<Hash>,
    pub c_noised: Option<Vec<i32>>,
    pub ms_noise: f32,
    pub ms_tiles: f32,
}

impl GpuJob {
    /// Loads A and B (resident in VRAM for the whole duration of the job) on GPU 0.
    pub fn new(a: &[i8], b: &[i8], p: &WorkParams) -> Option<Self> {
        Self::new_on(0, a, b, p)
    }

    /// Same, on CUDA device `dev`. Every call on the returned job runs on that device, from whichever thread owns the job.
    pub fn new_on(dev: usize, a: &[i8], b: &[i8], p: &WorkParams) -> Option<Self> {
        assert_eq!(p.tile, 16, "the GPU kernel is specialised for 16x16 tiles");
        let ctx = unsafe { vearl_gpu_new_on(dev as i32, a.as_ptr(), b.as_ptr(), p.n as i32, p.rank as i32) };
        (!ctx.is_null()).then_some(Self { ctx, p: *p })
    }

    /// Number of sigmas processed per launch: enough tiles to occupy the whole GPU even at small n
    /// (at n=128 a sigma is only 64 attempts = a single CUDA block); ~32,768 attempts per batch, ~131,072 at the real size
    /// (n >= 1024): a batch is then ~1 ms, the fixed cost of a launch (noise, 8 kernels, a synchronisation) is spread over 4 times
    /// more attempts and a new job or a cancellation is still noticed within a millisecond or two.
    pub fn batch_size(&self) -> usize {
        let t = self.p.tiles();
        let attempts = if t >= 64 { 131072 } else { 32768 };
        (attempts / (t * t)).max(1)
    }

    /// Jackpots (a vector of tiles^2 hashes per sigma) of a batch of sigmas; result identical to `run` sigma by sigma.
    /// The noise is generated ON THE GPU (blake3-XOF): only the 48-byte headers per sigma are sent.
    pub fn run_batch_timed(&self, sigmas: &[Hash]) -> Result<(Vec<Vec<Hash>>, [f32; 2]), i32> {
        let p = &self.p;
        let s = sigmas.len();
        let t2 = p.tiles() * p.tiles();
        let mut hdr = vec![0u8; 48 * s];
        for (i, sigma) in sigmas.iter().enumerate() {
            hdr[i * 48..i * 48 + 16].copy_from_slice(JACKPOT_DOMAIN);
            hdr[i * 48 + 16..(i + 1) * 48].copy_from_slice(sigma);
        }
        let mut jb = vec![0u8; s * t2 * 32];
        let mut ms = [0f32; 2];
        let rc = unsafe { vearl_gpu_run_batch_gen(self.ctx, s as i32, hdr.as_ptr(), p.fold_k as i32, jb.as_mut_ptr(), ms.as_mut_ptr()) };
        if rc != 0 {
            return Err(rc);
        }
        Ok((jb.chunks_exact(t2 * 32).map(|per_sigma| per_sigma.chunks_exact(32).map(|ch| <Hash>::try_from(ch).unwrap()).collect()).collect(), ms))
    }

    /// Development aid (gpubench): the time in ms of each kernel of a batch: noise_gen, 4 x pack16, 2 x noise_tc (A', B'^T), tile_kernel.
    pub fn profile_batch(&self, sigmas: &[Hash]) -> Result<[f32; 8], i32> {
        let mut hdr = vec![0u8; 48 * sigmas.len()];
        for (i, sigma) in sigmas.iter().enumerate() {
            hdr[i * 48..i * 48 + 16].copy_from_slice(JACKPOT_DOMAIN);
            hdr[i * 48 + 16..(i + 1) * 48].copy_from_slice(sigma);
        }
        let mut ms = [0f32; 8];
        let rc = unsafe { vearl_gpu_profile(self.ctx, sigmas.len() as i32, hdr.as_ptr(), self.p.fold_k as i32, ms.as_mut_ptr()) };
        if rc != 0 { Err(rc) } else { Ok(ms) }
    }

    pub fn run_batch(&self, sigmas: &[Hash]) -> Result<Vec<Vec<Hash>>, i32> {
        self.run_batch_timed(sigmas).map(|(j, _)| j)
    }

    /// The attempts of a batch whose jackpot meets `target` (`work::meets_target`), as (index of the sigma in the batch, tile index
    /// `ti * tiles + tj`, jackpot), in the order (sigma, tile): exactly what scanning `run_batch` would find, but the GPU does the
    /// filtering and only the few winners are copied back (a batch of 32 sigmas is 4 MiB of jackpots, copied and scanned for nothing
    /// almost every time). When the target is so easy that the GPU's short list overflows, the full copy is used instead.
    pub fn run_batch_hits(&self, sigmas: &[Hash], target: u64) -> Result<Vec<(usize, usize, Hash)>, i32> {
        const MAX_OUT: usize = 512; // the capacity of the GPU's list (MAX_HITS in vearl_gpu.cu)
        let p = &self.p;
        let s = sigmas.len();
        let t2 = p.tiles() * p.tiles();
        let mut hdr = vec![0u8; 48 * s];
        for (i, sigma) in sigmas.iter().enumerate() {
            hdr[i * 48..i * 48 + 16].copy_from_slice(JACKPOT_DOMAIN);
            hdr[i * 48 + 16..(i + 1) * 48].copy_from_slice(sigma);
        }
        let (mut count, mut idx, mut hashes, mut ms) = (0u32, [0u32; MAX_OUT], [0u8; MAX_OUT * 32], [0f32; 2]);
        let rc = unsafe {
            vearl_gpu_run_batch_hits(self.ctx, s as i32, hdr.as_ptr(), p.fold_k as i32, target, &mut count, idx.as_mut_ptr(), hashes.as_mut_ptr(), MAX_OUT as i32, ms.as_mut_ptr())
        };
        if rc != 0 {
            return Err(rc);
        }
        if count as usize > MAX_OUT {
            let mut v = Vec::new();
            for (d, row) in self.run_batch(sigmas)?.iter().enumerate() {
                for (i, jp) in row.iter().enumerate() {
                    if crate::work::meets_target(jp, target) {
                        v.push((d, i, *jp));
                    }
                }
            }
            return Ok(v);
        }
        let mut v: Vec<(usize, usize, Hash)> = (0..count as usize).map(|k| (idx[k] as usize / t2, idx[k] as usize % t2, <Hash>::try_from(&hashes[k * 32..k * 32 + 32]).unwrap())).collect();
        v.sort_by_key(|h| (h.0, h.1));
        Ok(v)
    }


    /// Variant with the noise derived on the CPU (several threads): serves as a cross-check and a reference measurement.
    pub fn run_batch_host_noise(&self, sigmas: &[Hash]) -> Result<Vec<Vec<Hash>>, i32> {
        let p = &self.p;
        let s = sigmas.len();
        let (n, r, t2) = (p.n, p.rank, p.tiles() * p.tiles());
        let nr = n * r;
        let mut bufs = vec![0i8; 4 * nr * s];
        let mut hdr = vec![0u8; 48 * s];
        let threads = std::thread::available_parallelism().map_or(1, |x| x.get()).min(s).max(1);
        let per = (s + threads - 1) / threads;
        std::thread::scope(|sc| {
            // slice of each sigma in the 4 buffers: [el | er | fl | fr] of size nr * s each
            let (el_all, rest) = bufs.split_at_mut(nr * s);
            let (er_all, rest) = rest.split_at_mut(nr * s);
            let (fl_all, fr_all) = rest.split_at_mut(nr * s);
            let chunks = el_all.chunks_mut(nr * per).zip(er_all.chunks_mut(nr * per)).zip(fl_all.chunks_mut(nr * per)).zip(fr_all.chunks_mut(nr * per)).zip(hdr.chunks_mut(48 * per)).zip(sigmas.chunks(per));
            for (((((el, er), fl), fr), h), sg) in chunks {
                sc.spawn(move || {
                    for (i, sigma) in sg.iter().enumerate() {
                        let noise = derive_noise(sigma, p);
                        for (dst, src) in [(&mut el[i * nr..(i + 1) * nr], &noise.el), (&mut er[i * nr..(i + 1) * nr], &noise.er), (&mut fl[i * nr..(i + 1) * nr], &noise.fl), (&mut fr[i * nr..(i + 1) * nr], &noise.fr)] {
                            for (d, x) in dst.iter_mut().zip(src.iter()) {
                                *d = *x as i8;
                            }
                        }
                        h[i * 48..i * 48 + 16].copy_from_slice(JACKPOT_DOMAIN);
                        h[i * 48 + 16..(i + 1) * 48].copy_from_slice(sigma);
                    }
                });
            }
        });
        let mut jb = vec![0u8; s * t2 * 32];
        let mut ms = [0f32; 2];
        let rc = unsafe {
            vearl_gpu_run_batch(
                self.ctx,
                s as i32,
                hdr.as_ptr(),
                bufs.as_ptr(),
                bufs[nr * s..].as_ptr(),
                bufs[2 * nr * s..].as_ptr(),
                bufs[3 * nr * s..].as_ptr(),
                p.fold_k as i32,
                jb.as_mut_ptr(),
                ms.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(rc);
        }
        Ok(jb.chunks_exact(t2 * 32).map(|per_sigma| per_sigma.chunks_exact(32).map(|ch| <Hash>::try_from(ch).unwrap()).collect()).collect())
    }

    pub fn run(&self, sigma: &Hash, want_c: bool) -> Result<GpuRun, i32> {
        let p = &self.p;
        let noise = derive_noise(sigma, p);
        let to8 = |v: &Vec<i32>| v.iter().map(|x| *x as i8).collect::<Vec<i8>>();
        let (el, er, fl, fr) = (to8(&noise.el), to8(&noise.er), to8(&noise.fl), to8(&noise.fr));
        let mut hdr = [0u8; 48];
        hdr[..16].copy_from_slice(JACKPOT_DOMAIN);
        hdr[16..].copy_from_slice(sigma);
        let tiles = p.tiles();
        let mut jb = vec![0u8; tiles * tiles * 32];
        let mut c = if want_c { Some(vec![0i32; p.n * p.n]) } else { None };
        let mut ms = [0f32; 2];
        let rc = unsafe {
            vearl_gpu_run(
                self.ctx,
                hdr.as_ptr(),
                el.as_ptr(),
                er.as_ptr(),
                fl.as_ptr(),
                fr.as_ptr(),
                p.fold_k as i32,
                jb.as_mut_ptr(),
                c.as_mut().map_or(std::ptr::null_mut(), |v| v.as_mut_ptr()),
                ms.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(rc);
        }
        let jackpots = jb.chunks_exact(32).map(|ch| <Hash>::try_from(ch).unwrap()).collect();
        Ok(GpuRun { jackpots, c_noised: c, ms_noise: ms[0], ms_tiles: ms[1] })
    }
}

impl Drop for GpuJob {
    fn drop(&mut self) {
        unsafe { vearl_gpu_free(self.ctx) }
    }
}

// SAFETY: GpuJob only holds DEVICE memory pointers (primary CUDA context of the process, valid
// from any thread). It is never shared between threads without mutual exclusion: it lives in
// `Chain`, itself protected by the node's Mutex; `mine_template` receives it by &mut/& from a single thread.
// It is therefore NOT `Sync`, only `Send`.
unsafe impl Send for GpuJob {}

/// The same job (A and B resident) on every GPU the miner uses: one `GpuJob` per device. The search for a winning sigma is shared
/// between them (`chain::mine_template`), each GPU taking its own batches.
pub struct GpuPool {
    pub jobs: Vec<GpuJob>,
}

impl GpuPool {
    /// Loads A and B on each device of `mining_devices()`. A device that fails to initialise is skipped with a message; None if no
    /// device works.
    pub fn new(a: &[i8], b: &[i8], p: &WorkParams) -> Option<Self> {
        let mut jobs = Vec::new();
        for d in mining_devices() {
            // `workers_per_gpu()` jobs on each device: the search runs one thread per job, each with its own CUDA stream
            for w in 0..workers_per_gpu() {
                match GpuJob::new_on(d, a, b, p) {
                    Some(j) => jobs.push(j),
                    None if w == 0 => {
                        eprintln!("[gpu] GPU #{d} could not be initialised (out of memory or driver error): skipped");
                        break;
                    }
                    None => break, // no room for one more: the device mines with the jobs it has
                }
            }
        }
        (!jobs.is_empty()).then_some(Self { jobs })
    }
}

/// Only one exact GEMM at a time on the GPU (the buffers are allocated per call).
static EXACT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Exact C = A*B (int32) on tensor cores; `None` if the size is not handled (n multiple of 128) or on a CUDA error.
pub fn exact_gemm(a: &[i8], b: &[i8], n: usize) -> Option<Vec<i32>> {
    if n < 128 || n % 128 != 0 || a.len() != n * n || b.len() != n * n {
        return None;
    }
    let _g = EXACT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut c = vec![0i32; n * n];
    let rc = unsafe { vearl_gpu_exact_gemm(a.as_ptr(), b.as_ptr(), n as i32, c.as_mut_ptr()) };
    (rc == 0).then_some(c)
}


// ---------------------------------------------------------------- Render2 (level 2) on the GPU

#[repr(C)]
#[derive(Clone, Copy)]
struct R2Prim {
    kind: i32,
    c: [i32; 3],
    h: [i32; 3],
    rnd: i32,
    col: [u8; 4],
    col2: [u8; 4],
    refl: u8,
    emit: u8,
    checker: u8,
    pad: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct R2Params {
    width: i32,
    height: i32,
    spp: i32,
    bounces: i32,
    cam: [i32; 3],
    target: [i32; 3],
    fov: i32,
    light: [i32; 3],
    light_col: [u8; 4],
    sky_top: [u8; 4],
    sky_hor: [u8; 4],
    fog: i32,
    smooth_k: i32,
    exposure: i32,
    nprims: i32,
    prims: [R2Prim; 32],
}

extern "C" {
    fn vearl_gpu_render2(params: *const R2Params, first_chunk: u32, chunks: u32, jitter: *const u8, out_host: *mut u8) -> i32;
}

fn pack_render2(p: &crate::render2::RenderParams2) -> R2Params {
    use crate::render2::Shape;
    let blank = R2Prim { kind: 0, c: [0; 3], h: [0; 3], rnd: 0, col: [0; 4], col2: [0; 4], refl: 0, emit: 0, checker: 0, pad: 0 };
    let mut prims = [blank; 32];
    for (i, pr) in p.prims.iter().enumerate() {
        let m = &pr.mat;
        let mut q = R2Prim { col: [m.col[0], m.col[1], m.col[2], 0], col2: [m.col2[0], m.col2[1], m.col2[2], 0], refl: m.refl, emit: m.emit, checker: m.checker as u8, ..blank };
        match &pr.shape {
            Shape::Sphere { c, r } => {
                q.kind = 0;
                q.c = *c;
                q.h[0] = *r;
            }
            Shape::Box { c, h, rnd } => {
                q.kind = 1;
                q.c = *c;
                q.h = *h;
                q.rnd = *rnd;
            }
            Shape::Torus { c, big, small } => {
                q.kind = 2;
                q.c = *c;
                q.h[0] = *big;
                q.h[1] = *small;
            }
            Shape::Plane { y } => {
                q.kind = 3;
                q.c[1] = *y;
            }
            Shape::Cylinder { c, r, h, rnd } => {
                q.kind = 4;
                q.c = *c;
                q.h[0] = *r;
                q.h[1] = *h;
                q.rnd = *rnd;
            }
            Shape::Capsule { a, b, r } => {
                q.kind = 5;
                q.c = *a;
                q.h = *b;
                q.rnd = *r;
            }
        }
        prims[i] = q;
    }
    R2Params {
        width: p.width as i32,
        height: p.height as i32,
        spp: p.spp as i32,
        bounces: p.bounces as i32,
        cam: p.cam,
        target: p.target,
        fov: p.fov,
        light: p.light,
        light_col: [p.light_col[0], p.light_col[1], p.light_col[2], 0],
        sky_top: [p.sky_top[0], p.sky_top[1], p.sky_top[2], 0],
        sky_hor: [p.sky_hor[0], p.sky_hor[1], p.sky_hor[2], 0],
        fog: p.fog,
        smooth_k: p.smooth_k,
        exposure: p.exposure,
        nprims: p.prims.len() as i32,
        prims,
    }
}

/// Renders the tiles `first .. first + count` of a Render2 scene on the GPU: one 768-byte output per tile, bit-identical to
/// `render2::render2_chunk` (tests/gpu.rs). Launches are split so that the sampling-offset table stays small.
pub fn render2_tiles(p: &crate::render2::RenderParams2, first: u32, count: u32) -> Result<Vec<Vec<u8>>, i32> {
    let packed = pack_render2(p);
    let spp = p.spp as usize;
    let mut outs: Vec<Vec<u8>> = Vec::with_capacity(count as usize);
    let batch = 2048u32;
    let mut done = 0u32;
    while done < count {
        let n = batch.min(count - done);
        let start = first + done;
        let jitter: Vec<u8> = if spp > 1 {
            let mut j = Vec::with_capacity(n as usize * 256 * spp * 4);
            for c in start..start + n {
                j.extend_from_slice(&crate::render2::jitter_bytes(&p.seed, c, spp));
            }
            j
        } else {
            vec![]
        };
        let mut buf = vec![0u8; n as usize * 768];
        let rc = unsafe { vearl_gpu_render2(&packed, start, n, jitter.as_ptr(), buf.as_mut_ptr()) };
        if rc != 0 {
            return Err(rc);
        }
        for t in buf.chunks(768) {
            outs.push(t.to_vec());
        }
        done += n;
    }
    Ok(outs)
}


// ---------------------------------------------------------------- Render3 (level 2 path tracer) on the GPU

#[repr(C)]
#[derive(Clone, Copy)]
struct R3Part {
    kind: i32,
    g: [i32; 4],
    c: [i32; 3],
    rot: [i32; 9],
    op: i32,
    k: i32,
    mat: i32,
    polar: i32,
    mirror: i32,
    off: [i32; 3],
    group: i32,
    shell: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct R3Mat {
    alb: [i32; 3],
    alb2: [i32; 3],
    rough: i32,
    metal: i32,
    spec: i32,
    emit: [i32; 3],
    trans: i32,
    glass: i32,
    ior: i32,
    absorb: [i32; 3],
    coat: i32,
    tex: i32,
    tex_scale: i32,
    tex_p: i32,
    tex_q: i32,
    tex_r: i32,
    bump: i32,
    bump_freq: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct R3Box {
    dir: [i32; 3],
    hs: [i32; 2],
    soft: i32,
    col: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct R3Params {
    width: i32,
    height: i32,
    spp: i32,
    passes: i32,
    bounces: i32,
    fog0: i32,
    fog1: i32,
    seed0: u32,
    seed1: u32,
    cam_pos: [i32; 3],
    cam_look: [i32; 3],
    focal: i32,
    aperture: i32,
    focus: i32,
    top: [u32; 3],
    hor: [u32; 3],
    gnd: [u32; 3],
    nboxes: i32,
    boxes: [R3Box; 4],
    bd_top: [u32; 3],
    bd_bot: [u32; 3],
    nparts: i32,
    nmats: i32,
    mats: [R3Mat; 24],
    parts: [R3Part; 48],
}

extern "C" {
    fn vearl_gpu_render3(params: *const R3Params, first_chunk: u32, chunks: u32, out_host: *mut u8) -> i32;
}

fn pack_render3(p: &crate::render3::RenderParams3) -> Box<R3Params> {
    use crate::render3::{Geo3, Op3};
    let zp = R3Part { kind: 0, g: [0; 4], c: [0; 3], rot: [0; 9], op: 0, k: 0, mat: 0, polar: 0, mirror: 0, off: [0; 3], group: 0, shell: 0 };
    let zm = R3Mat { alb: [0; 3], alb2: [0; 3], rough: 0, metal: 0, spec: 0, emit: [0; 3], trans: 0, glass: 0, ior: 1536, absorb: [0; 3], coat: 0, tex: 0, tex_scale: 0, tex_p: 0, tex_q: 0, tex_r: 0, bump: 0, bump_freq: 0 };
    let zb = R3Box { dir: [0; 3], hs: [0; 2], soft: 0, col: [0; 3] };
    let mut parts = [zp; 48];
    for (i, pt) in p.parts.iter().enumerate() {
        let mut q = R3Part { c: pt.c, rot: pt.rot, mat: pt.mat as i32, polar: pt.polar as i32, mirror: pt.mirror as i32, off: pt.off, group: pt.group as i32, shell: pt.shell, ..zp };
        match &pt.geo {
            Geo3::Sphere { r } => {
                q.kind = 0;
                q.g[0] = *r;
            }
            Geo3::Ellipsoid { r } => {
                q.kind = 1;
                q.g[..3].copy_from_slice(r);
            }
            Geo3::Box { h, rnd } => {
                q.kind = 2;
                q.g[..3].copy_from_slice(h);
                q.g[3] = *rnd;
            }
            Geo3::Cylinder { r, h, rnd } => {
                q.kind = 3;
                q.g = [*r, *h, *rnd, 0];
            }
            Geo3::Capsule { h, r } => {
                q.kind = 4;
                q.g = [*h, *r, 0, 0];
            }
            Geo3::Torus { big, small } => {
                q.kind = 5;
                q.g = [*big, *small, 0, 0];
            }
            Geo3::Cone { h, r1, r2 } => {
                q.kind = 6;
                q.g = [*h, *r1, *r2, 0];
            }
            Geo3::Plane => q.kind = 7,
        }
        match &pt.op {
            Op3::Union => q.op = 0,
            Op3::Smooth(k) => {
                q.op = 1;
                q.k = *k;
            }
            Op3::Sub => q.op = 2,
            Op3::Inter => q.op = 3,
        }
        parts[i] = q;
    }
    let mut mats = [zm; 24];
    for (i, m) in p.mats.iter().enumerate() {
        mats[i] = R3Mat {
            alb: [m.alb[0] as i32, m.alb[1] as i32, m.alb[2] as i32],
            alb2: [m.alb2[0] as i32, m.alb2[1] as i32, m.alb2[2] as i32],
            rough: m.rough as i32,
            metal: m.metal as i32,
            spec: m.spec as i32,
            emit: [m.emit[0] as i32, m.emit[1] as i32, m.emit[2] as i32],
            trans: m.trans as i32,
            glass: m.glass as i32,
            ior: m.ior as i32,
            absorb: [m.absorb[0] as i32, m.absorb[1] as i32, m.absorb[2] as i32],
            coat: m.coat as i32,
            tex: m.tex as i32,
            tex_scale: m.tex_scale,
            tex_p: m.tex_p,
            tex_q: m.tex_q,
            tex_r: m.tex_r,
            bump: m.bump as i32,
            bump_freq: m.bump_freq,
        };
    }
    let mut boxes = [zb; 4];
    for (i, b) in p.env.boxes.iter().enumerate() {
        boxes[i] = R3Box { dir: b.dir, hs: b.hs, soft: b.soft, col: b.col };
    }
    let w0 = u32::from_le_bytes([p.seed[0], p.seed[1], p.seed[2], p.seed[3]]);
    let w1 = u32::from_le_bytes([p.seed[4], p.seed[5], p.seed[6], p.seed[7]]);
    Box::new(R3Params {
        width: p.width as i32,
        height: p.height as i32,
        spp: p.spp as i32,
        passes: p.passes as i32,
        bounces: p.bounces as i32,
        fog0: p.fog0,
        fog1: p.fog1,
        seed0: w0,
        seed1: w1,
        cam_pos: p.cam.pos,
        cam_look: p.cam.look,
        focal: p.cam.focal,
        aperture: p.cam.aperture,
        focus: p.cam.focus,
        top: p.env.top,
        hor: p.env.hor,
        gnd: p.env.gnd,
        nboxes: p.env.boxes.len() as i32,
        boxes,
        bd_top: p.env.bd_top,
        bd_bot: p.env.bd_bot,
        nparts: p.parts.len() as i32,
        nmats: p.mats.len() as i32,
        mats,
        parts,
    })
}

/// Renders the chunks `first .. first + count` of a Render3 job on the GPU (1,536 bytes each), bit-identical to `render3::render3_chunk`.
pub fn render3_chunks_gpu(p: &crate::render3::RenderParams3, first: u32, count: u32) -> Result<Vec<Vec<u8>>, i32> {
    let packed = pack_render3(p);
    let mut outs: Vec<Vec<u8>> = Vec::with_capacity(count as usize);
    // one launch must stay well under the 2 s watchdog of a display GPU (Windows TDR): about 0.6 s of work per launch (measured on an RTX 4070 SUPER)
    let batch = (40_000_000_000u64 / crate::render3::render3_cost(p).max(1)).clamp(1, 1024) as u32;
    let mut done = 0u32;
    while done < count {
        let n = batch.min(count - done);
        let mut buf = vec![0u8; n as usize * 1536];
        let rc = unsafe { vearl_gpu_render3(&*packed, first + done, n, buf.as_mut_ptr()) };
        if rc != 0 {
            return Err(rc);
        }
        for t in buf.chunks(1536) {
            outs.push(t.to_vec());
        }
        done += n;
    }
    Ok(outs)
}
