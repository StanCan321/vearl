//! LEVEL 2: computations that do NOT reduce to matrix products verifiable by Freivalds: Monte Carlo simulation
//! (integrand = small fixed-point program) and ray-traced RENDERING (signed distance fields). Docs: docs/10.
//!
//! This is NOT the consensus work: it is a SERVICE paid at full cost (escrow, bond, burn), with a guarantee
//! WEAKER than level 1:
//!   - the computation is cut into independent and DETERMINISTIC `chunks` (integers, no floating point);
//!   - the worker commits to the Merkle root of the outputs of all the chunks;
//!   - anyone can prove fraud on ONE chunk: the chain recomputes that single chunk (cost bounded by the kernel);
//!   - the client checks by SAMPLING: a worker that falsifies a fraction phi of the chunks is only detected, with m verified
//!     chunks, with probability 1-(1-phi)^m. This is therefore NOT a proof of exactness of the complete result.
//! The kernels are fixed by the protocol (no arbitrary client code): no sandbox is needed.
use crate::crypto::*;
use crate::prog::isqrt;
use crate::state::*;
use crate::types::*;
use serde::{Deserialize, Serialize};

pub const MAX_CHUNKS: u32 = 1024;
pub const MAX_CHUNK_COST: u64 = 8_000_000;
pub const MAX_OUT: usize = 4096;
pub const MAX_PROGRAM: usize = 64;
pub const MAX_PRIMS: usize = 8;
pub const MAX_PENDING_CHUNKS: usize = 8;
pub const MC_ONE: i64 = 1 << 32;

// ---------------------------------------------------------------- Monte Carlo

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Instr {
    pub op: u8,
    pub a: u8,
    pub b: u8,
    pub c: u8,
    pub imm: i64,
}

/// Integrand: program with 16 i64 registers in Q32 fixed point; output of a chunk = (sum, sum of squares) of the `samples`
/// values of register `out`. Operations (r = registers, any overflow is SATURATED in i64):
///  0 LDC a,imm : r[a]=imm      1 INP a,b : r[a]=input b (u32 -> [0,1) in Q32)   2 ADD a,b,c   3 SUB a,b,c
///  4 MUL a,b,c : (r[b]*r[c])>>32 (i128, arithmetic shift)   5 MIN   6 MAX   7 ABS a,b   8 NEG a,b
///  9 SQRT a,b : isqrt(max(r[b],0)<<32)   10 LT a,b,c : r[a]= (r[b]<r[c]) ? 1.0 : 0   11 SEL a,b,c,imm : r[a] = r[b]>0 ? r[c] : r[imm&15]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct McParams {
    pub seed: [u8; 32],
    pub dim: u8,
    pub samples: u32,
    pub out: u8,
    pub program: Vec<Instr>,
}

fn sat(x: i128) -> i64 {
    x.clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

pub fn isqrt128(x: u128) -> u128 {
    if x < 2 {
        return x;
    }
    let mut r = 1u128 << ((128 - x.leading_zeros() + 1) / 2);
    loop {
        let n = (r + x / r) / 2;
        if n >= r {
            return r;
        }
        r = n;
    }
}

fn mc_validate(p: &McParams) -> Result<(), String> {
    if p.dim == 0 || p.dim > 8 || p.samples == 0 || p.samples > 4096 || p.out >= 16 {
        return Err("bad Monte Carlo parameters".into());
    }
    if p.program.is_empty() || p.program.len() > MAX_PROGRAM {
        return Err("bad program length".into());
    }
    for i in &p.program {
        let ok = match i.op {
            0 => i.a < 16,
            1 => i.a < 16 && i.b < p.dim,
            2..=6 | 10 => i.a < 16 && i.b < 16 && i.c < 16,
            7 | 8 | 9 => i.a < 16 && i.b < 16,
            11 => i.a < 16 && i.b < 16 && i.c < 16,
            _ => false,
        };
        if !ok {
            return Err("bad instruction".into());
        }
    }
    Ok(())
}

/// Inputs of the chunk: blake3 XOF (derive_key mode) of (seed, chunk), `samples*dim` little-endian u32 words.
pub fn mc_inputs(p: &McParams, chunk: u32) -> Vec<u32> {
    let mut hs = blake3::Hasher::new_derive_key("vearl/l2/mc/v1");
    hs.update(&p.seed);
    hs.update(&chunk.to_le_bytes());
    let mut buf = vec![0u8; p.samples as usize * p.dim as usize * 4];
    hs.finalize_xof().fill(&mut buf);
    buf.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect()
}

pub fn mc_chunk(p: &McParams, chunk: u32) -> Vec<u8> {
    let (d, s) = (p.dim as usize, p.samples as usize);
    let inp = mc_inputs(p, chunk);
    let (mut sum, mut sq) = (0i128, 0i128);
    for k in 0..s {
        let mut r = [0i64; 16];
        for i in &p.program {
            let (a, b, c) = (i.a as usize, i.b as usize, i.c as usize);
            match i.op {
                0 => r[a] = i.imm,
                1 => r[a] = inp[k * d + b] as i64,
                2 => r[a] = sat(r[b] as i128 + r[c] as i128),
                3 => r[a] = sat(r[b] as i128 - r[c] as i128),
                4 => r[a] = sat((r[b] as i128 * r[c] as i128) >> 32),
                5 => r[a] = r[b].min(r[c]),
                6 => r[a] = r[b].max(r[c]),
                7 => r[a] = sat((r[b] as i128).abs()),
                8 => r[a] = sat(-(r[b] as i128)),
                9 => r[a] = sat(isqrt128((r[b].max(0) as u128) << 32) as i128),
                10 => r[a] = if r[b] < r[c] { MC_ONE } else { 0 },
                11 => r[a] = if r[b] > 0 { r[c] } else { r[(i.imm & 15) as usize] },
                _ => unreachable!(),
            }
        }
        let v = r[p.out as usize] as i128;
        sum += v;
        sq += (v * v) >> 32;
    }
    let mut out = sum.to_le_bytes().to_vec();
    out.extend_from_slice(&sq.to_le_bytes());
    out
}

/// Demonstration program: indicator of the quarter disc (E = pi/4).
pub fn mc_pi_program() -> Vec<Instr> {
    let i = |op, a, b, c, imm| Instr { op, a, b, c, imm };
    vec![i(1, 0, 0, 0, 0), i(1, 1, 1, 0, 0), i(4, 2, 0, 0, 0), i(4, 3, 1, 1, 0), i(2, 4, 2, 3, 0), i(0, 5, 0, 0, MC_ONE), i(10, 6, 4, 5, 0)]
}

// ---------------------------------------------------------------- Rendering

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Prim {
    Sphere { c: [i32; 3], r: i32, col: [u8; 3] },
    /// Plane y = `y`, two-colour checkerboard (cells of 1.0).
    Plane { y: i32, col: [u8; 3], col2: [u8; 3] },
    Box { c: [i32; 3], h: [i32; 3], col: [u8; 3] },
}

/// Scene and camera, in Q16.16 fixed point (i32). Camera at (0, 0, -cam_z) looking +z, 90 degree field of view.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RenderParams {
    pub seed: [u8; 32],
    pub width: u16,
    pub height: u16,
    pub spp: u8,
    pub cam_z: i32,
    pub light: [i32; 3],
    pub prims: Vec<Prim>,
}

const Q: i64 = 1 << 16;
const MARCH_STEPS: u64 = 48;
const SHADOW_STEPS: u64 = 24;
const HIT_EPS: i64 = 64;
const T_MAX: i64 = 20 * Q;
pub const TILE: usize = 16;

fn mul16(a: i64, b: i64) -> i64 {
    (a * b) >> 16
}

fn len3(v: [i64; 3]) -> i64 {
    isqrt((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) as u64) as i64
}

/// Normalisation in Q16; division truncated toward zero; null vector -> (0, 1, 0).
fn norm3(v: [i64; 3]) -> [i64; 3] {
    let l = len3(v);
    if l == 0 {
        return [0, Q, 0];
    }
    [(v[0] << 16) / l, (v[1] << 16) / l, (v[2] << 16) / l]
}

fn prim_dist(pr: &Prim, p: [i64; 3]) -> i64 {
    match pr {
        Prim::Sphere { c, r, .. } => len3([p[0] - c[0] as i64, p[1] - c[1] as i64, p[2] - c[2] as i64]) - *r as i64,
        Prim::Plane { y, .. } => p[1] - *y as i64,
        Prim::Box { c, h, .. } => {
            let q = [(p[0] - c[0] as i64).abs() - h[0] as i64, (p[1] - c[1] as i64).abs() - h[1] as i64, (p[2] - c[2] as i64).abs() - h[2] as i64];
            let outside = len3([q[0].max(0), q[1].max(0), q[2].max(0)]);
            outside + q[0].max(q[1]).max(q[2]).min(0)
        }
    }
}

/// (minimum distance, index of the nearest primitive; tie -> smallest index)
fn sdf(prims: &[Prim], p: [i64; 3]) -> (i64, usize) {
    let mut best = (prim_dist(&prims[0], p), 0usize);
    for (k, pr) in prims.iter().enumerate().skip(1) {
        let d = prim_dist(pr, p);
        if d < best.0 {
            best = (d, k);
        }
    }
    best
}

fn at(o: [i64; 3], d: [i64; 3], t: i64) -> [i64; 3] {
    [o[0] + mul16(d[0], t), o[1] + mul16(d[1], t), o[2] + mul16(d[2], t)]
}

fn render_validate(p: &RenderParams) -> Result<(), String> {
    let (w, h) = (p.width as usize, p.height as usize);
    if w % TILE != 0 || h % TILE != 0 || !(TILE..=512).contains(&w) || !(TILE..=512).contains(&h) || p.spp == 0 || p.spp > 4 {
        return Err("bad image size or samples per pixel".into());
    }
    if p.prims.is_empty() || p.prims.len() > MAX_PRIMS {
        return Err("bad number of primitives".into());
    }
    let lim = 1i64 << 24;
    let ok = |v: i32| (v as i64).abs() <= lim;
    let all_ok = p.prims.iter().all(|pr| match pr {
        Prim::Sphere { c, r, .. } => c.iter().all(|v| ok(*v)) && ok(*r) && *r > 0,
        Prim::Plane { y, .. } => ok(*y),
        Prim::Box { c, h, .. } => c.iter().all(|v| ok(*v)) && h.iter().all(|v| ok(*v) && *v > 0),
    });
    if !all_ok || !ok(p.cam_z) || !p.light.iter().all(|v| ok(*v)) {
        return Err("scene coordinates out of range".into());
    }
    Ok(())
}

/// Output of a chunk = one 16x16 tile, 3 RGB bytes per pixel, row by row. Chunk i -> tile (i % (W/16), i / (W/16)).
pub fn render_chunk(p: &RenderParams, chunk: u32) -> Vec<u8> {
    let tiles_x = p.width as usize / TILE;
    let (tx, ty) = (chunk as usize % tiles_x, chunk as usize / tiles_x);
    let spp = p.spp as usize;
    let jitter: Vec<u8> = if spp > 1 {
        let mut hs = blake3::Hasher::new_derive_key("vearl/l2/render/v1");
        hs.update(&p.seed);
        hs.update(&chunk.to_le_bytes());
        let mut buf = vec![0u8; TILE * TILE * spp * 4];
        hs.finalize_xof().fill(&mut buf);
        buf
    } else {
        vec![]
    };
    let light = norm3([p.light[0] as i64, p.light[1] as i64, p.light[2] as i64]);
    let o = [0i64, 0, -(p.cam_z as i64)];
    let (w, h) = (p.width as i64, p.height as i64);
    let mut out = Vec::with_capacity(TILE * TILE * 3);
    for y in 0..TILE {
        for x in 0..TILE {
            let mut acc = [0i64; 3];
            for s in 0..spp {
                let (jx, jy) = if spp > 1 {
                    let k = ((y * TILE + x) * spp + s) * 4;
                    (u16::from_le_bytes([jitter[k], jitter[k + 1]]) as i64, u16::from_le_bytes([jitter[k + 2], jitter[k + 3]]) as i64)
                } else {
                    (32768, 32768)
                };
                let (sx, sy) = (((tx * TILE + x) as i64) * Q + jx, ((ty * TILE + y) as i64) * Q + jy);
                let u = (2 * sx - w * Q) / w;
                let v = -((2 * sy - h * Q) / w);
                let d = norm3([u, v, Q]);
                let c = shade(p, o, d, light);
                for k in 0..3 {
                    acc[k] += c[k];
                }
            }
            for k in 0..3 {
                out.push((acc[k] / spp as i64) as u8);
            }
        }
    }
    out
}

fn shade(p: &RenderParams, o: [i64; 3], d: [i64; 3], light: [i64; 3]) -> [i64; 3] {
    let mut t = 0i64;
    let mut hit = None;
    for _ in 0..MARCH_STEPS {
        let pos = at(o, d, t);
        let (dist, k) = sdf(&p.prims, pos);
        if dist < HIT_EPS {
            hit = Some((pos, k));
            break;
        }
        t += dist;
        if t > T_MAX {
            break;
        }
    }
    let Some((pos, k)) = hit else {
        let sky = [90i64, 120, 180];
        return [20 + ((sky[0] * (d[1] + Q)) >> 17), 20 + ((sky[1] * (d[1] + Q)) >> 17), 20 + ((sky[2] * (d[1] + Q)) >> 17)];
    };
    const H: i64 = 256;
    let n = norm3([
        sdf(&p.prims, [pos[0] + H, pos[1], pos[2]]).0 - sdf(&p.prims, [pos[0] - H, pos[1], pos[2]]).0,
        sdf(&p.prims, [pos[0], pos[1] + H, pos[2]]).0 - sdf(&p.prims, [pos[0], pos[1] - H, pos[2]]).0,
        sdf(&p.prims, [pos[0], pos[1], pos[2] + H]).0 - sdf(&p.prims, [pos[0], pos[1], pos[2] - H]).0,
    ]);
    let diffuse = ((n[0] * light[0] + n[1] * light[1] + n[2] * light[2]) >> 16).max(0);
    // cast shadow: march from a point slightly lifted off the surface toward the light
    let ps = [pos[0] + mul16(n[0], 1310), pos[1] + mul16(n[1], 1310), pos[2] + mul16(n[2], 1310)];
    let (mut ts, mut lit) = (0i64, true);
    for _ in 0..SHADOW_STEPS {
        let (dist, _) = sdf(&p.prims, at(ps, light, ts));
        if dist < HIT_EPS {
            lit = false;
            break;
        }
        ts += dist;
        if ts > T_MAX / 2 {
            break;
        }
    }
    let shade = if lit { Q } else { Q / 4 };
    let inten = (13107 + mul16(52429, mul16(diffuse, shade))).min(Q);
    let base = match &p.prims[k] {
        Prim::Sphere { col, .. } | Prim::Box { col, .. } => *col,
        Prim::Plane { col, col2, .. } => {
            if ((pos[0] >> 16) + (pos[2] >> 16)) & 1 == 0 {
                *col
            } else {
                *col2
            }
        }
    };
    [(base[0] as i64 * inten) >> 16, (base[1] as i64 * inten) >> 16, (base[2] as i64 * inten) >> 16]
}

/// Demonstration scene: two spheres, a box, a checkerboard floor, a light.
pub fn demo_scene(width: u16, height: u16, spp: u8, seed: [u8; 32]) -> RenderParams {
    let f = |x: f64| (x * 65536.0).round() as i32;
    RenderParams {
        seed,
        width,
        height,
        spp,
        cam_z: f(4.0),
        light: [f(-0.6), f(1.0), f(-0.5)],
        prims: vec![
            Prim::Sphere { c: [f(-0.9), f(0.0), f(1.0)], r: f(0.9), col: [220, 60, 50] },
            Prim::Sphere { c: [f(1.1), f(-0.3), f(0.4)], r: f(0.6), col: [60, 120, 230] },
            Prim::Box { c: [f(0.2), f(-0.55), f(-0.9)], h: [f(0.35), f(0.35), f(0.35)], col: [80, 200, 90] },
            Prim::Plane { y: f(-0.9), col: [230, 230, 230], col2: [70, 70, 80] },
        ],
    }
}

// ---------------------------------------------------------------- kernels (common interface)

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Kernel {
    MonteCarlo(McParams),
    Render(RenderParams),
    /// Render2: reflections, soft shadows, occlusion, glow, free camera (docs/10 section 4 bis). Appended last: tag 2.
    Render2(crate::render2::RenderParams2),
    /// Render3: Monte Carlo path tracer (docs/10 section 4 ter). One chunk = one pass of one tile. Tag 3.
    Render3(crate::render3::RenderParams3),
}

impl Kernel {
    /// Number of chunks imposed by the kernel, if it imposes one (rendering: one tile per chunk).
    pub fn forced_chunks(&self) -> Option<u32> {
        match self {
            Kernel::Render(p) => Some((p.width as u32 / TILE as u32) * (p.height as u32 / TILE as u32)),
            Kernel::Render2(p) => Some((p.width as u32 / TILE as u32) * (p.height as u32 / TILE as u32)),
            Kernel::Render3(p) => Some(crate::render3::render3_chunks(p)),
            Kernel::MonteCarlo(_) => None,
        }
    }

    /// Cost per chunk (evaluation units): bounds what a node must recompute to judge ONE fraud proof.
    pub fn cost(&self) -> u64 {
        match self {
            Kernel::MonteCarlo(p) => p.samples as u64 * (p.program.len() as u64 + p.dim as u64),
            Kernel::Render(p) => (TILE * TILE) as u64 * p.spp as u64 * (MARCH_STEPS + 6 + SHADOW_STEPS + 2) * p.prims.len() as u64,
            Kernel::Render2(p) => crate::render2::render2_cost(p),
            Kernel::Render3(p) => crate::render3::render3_cost(p),
        }
    }

    pub fn validate(&self, chunks: u32) -> Result<(), String> {
        match self {
            Kernel::MonteCarlo(p) => mc_validate(p)?,
            Kernel::Render(p) => render_validate(p)?,
            Kernel::Render2(p) => crate::render2::render2_validate(p)?,
            Kernel::Render3(p) => crate::render3::render3_validate(p)?,
        }
        let max_chunks = match self {
            Kernel::Render2(_) => crate::render2::MAX_RENDER_CHUNKS,
            Kernel::Render3(_) => crate::render3::MAX_CHUNKS3,
            _ => MAX_CHUNKS,
        };
        if chunks == 0 || chunks > max_chunks {
            return Err("bad chunk count".into());
        }
        if let Some(f) = self.forced_chunks() {
            if f != chunks {
                return Err("chunk count does not match the image tiles".into());
            }
        }
        let bound = if matches!(self, Kernel::Render3(_)) { crate::render3::MAX_CHUNK_COST3 } else { MAX_CHUNK_COST };
        if self.cost() > bound {
            return Err("chunk cost above the protocol bound".into());
        }
        Ok(())
    }

    pub fn chunk(&self, i: u32) -> Vec<u8> {
        match self {
            Kernel::MonteCarlo(p) => mc_chunk(p, i),
            Kernel::Render(p) => render_chunk(p, i),
            Kernel::Render2(p) => crate::render2::render2_chunk(p, i),
            Kernel::Render3(p) => crate::render3::render3_chunk(p, i),
        }
    }
}

/// Computes all the chunks of a kernel on all the cores (the chunks are independent: the result does not depend on the parallelism).
pub fn compute_all(kernel: &Kernel, chunks: u32) -> Vec<Vec<u8>> {
    // Render2 on a `--gpu` worker: the whole image on the GPU, bit-identical to the CPU path (tests/gpu.rs)
    #[cfg(feature = "gpu")]
    if let Kernel::Render2(p) = kernel {
        if crate::work::gpu_exact_on() {
            if let Ok(v) = crate::gpu::render2_tiles(p, 0, chunks) {
                return v;
            }
        }
    }
    #[cfg(feature = "gpu")]
    if let Kernel::Render3(p) = kernel {
        if crate::work::gpu_exact_on() {
            if let Ok(v) = crate::gpu::render3_chunks_gpu(p, 0, chunks) {
                return v;
            }
        }
    }
    compute_all_cpu(kernel, chunks)
}

/// Same result on every core of the CPU (chunks are independent: the output does not depend on the parallelism).
pub fn compute_all_cpu(kernel: &Kernel, chunks: u32) -> Vec<Vec<u8>> {
    let threads = std::thread::available_parallelism().map_or(1, |x| x.get()).min(chunks.max(1) as usize);
    let mut outs: Vec<Vec<u8>> = vec![Vec::new(); chunks as usize];
    let per = (chunks as usize + threads - 1) / threads.max(1);
    std::thread::scope(|sc| {
        for (t, slice) in outs.chunks_mut(per.max(1)).enumerate() {
            sc.spawn(move || {
                for (k, o) in slice.iter_mut().enumerate() {
                    *o = kernel.chunk((t * per + k) as u32);
                }
            });
        }
    });
    outs
}

pub fn chunk_leaf(i: u32, out: &[u8]) -> Hash {
    leaf_hash(i as u64, out)
}

/// Merkle tree of the outputs (one leaf per chunk).
pub fn chunk_tree(outs: &[Vec<u8>]) -> MerkleTree {
    MerkleTree::new(outs.iter().enumerate().map(|(i, o)| chunk_leaf(i as u32, o)).collect())
}

// ---------------------------------------------------------------- transactions and state

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum L2Tx {
    Submit { kernel: Kernel, chunks: u32, escrow: u64, deadline: u64 },
    Claim { job_id: JobId },
    /// The worker commits to the Merkle root of the outputs of ALL the chunks.
    Post { job_id: JobId, root: Hash },
    Accept { job_id: JobId },
    /// The client demands the output of a chunk ON-CHAIN (data availability); with no answer: SlashUnserved.
    Fetch { job_id: JobId, chunk: u32 },
    Serve { job_id: JobId, chunk: u32, out: Vec<u8>, path: Vec<Hash> },
    SlashUnserved { job_id: JobId },
    /// Fraud proof: the committed output of a chunk is not the kernel's (recomputed by the chain).
    Fraud { job_id: JobId, chunk: u32, out: Vec<u8>, path: Vec<Hash> },
    Settle { job_id: JobId },
}

pub fn l2_id(client: &PubKey, nonce: u64, kernel: &Kernel, chunks: u32) -> JobId {
    h("vearl/l2/v1", &[client, &nonce.to_le_bytes(), &bincode::serialize(kernel).unwrap(), &chunks.to_le_bytes()])
}

#[derive(Clone, Debug, PartialEq)]
pub enum L2Status {
    Open,
    Claimed { worker: PubKey, bond: u64 },
    Posted { worker: PubKey, bond: u64, root: Hash, at: u64, at_ts: u64 },
    Settled,
    Slashed,
    Expired,
}

#[derive(Clone, Debug)]
pub struct L2Job {
    pub client: PubKey,
    pub kernel: Kernel,
    pub chunks: u32,
    pub escrow_net: u64,
    pub deadline: u64,
    pub status: L2Status,
    /// (chunk, deadline in timestamp seconds)
    pub fetches: Vec<(u32, u64)>,
    pub fetch_held: u64,
}

fn bad<T>(m: &str) -> Result<T, String> {
    Err(m.to_string())
}

impl State {
    pub(crate) fn l2_held(&self) -> u64 {
        self.l2
            .values()
            .map(|j| {
                j.fetch_held
                    + match &j.status {
                        L2Status::Open => j.escrow_net,
                        L2Status::Claimed { bond, .. } | L2Status::Posted { bond, .. } => j.escrow_net + bond,
                        _ => 0,
                    }
            })
            .sum()
    }

    fn l2_sanction(&mut self, beneficiary: &PubKey, id: &JobId) {
        let j = self.l2.get(id).unwrap().clone();
        let L2Status::Posted { worker, bond, .. } = j.status else { return };
        let reward = bond / 2;
        self.credit(beneficiary, reward);
        self.burned += bond - reward;
        self.credit(&j.client, j.escrow_net + j.fetch_held);
        self.rep.entry(worker).or_default().slashed += 1;
        let jm = self.l2.get_mut(id).unwrap();
        jm.fetch_held = 0;
        jm.fetches.clear();
        jm.status = L2Status::Slashed;
    }

    pub(crate) fn apply_l2(&mut self, from: &PubKey, nonce: u64, m: &L2Tx, height: u64, ts: u64, _p: &crate::work::WorkParams) -> Result<(), String> {
        match m {
            L2Tx::Submit { kernel, chunks, escrow, deadline } => {
                kernel.validate(*chunks)?;
                if *escrow < MIN_ESCROW {
                    return bad("escrow below floor price");
                }
                if *deadline <= height {
                    return bad("deadline in the past");
                }
                let id = l2_id(from, nonce, kernel, *chunks);
                if self.l2.contains_key(&id) {
                    return bad("duplicate job");
                }
                self.debit(from, *escrow)?;
                let burn = bps(*escrow, BURN_BPS);
                self.burned += burn;
                self.l2.insert(id, L2Job { client: *from, kernel: kernel.clone(), chunks: *chunks, escrow_net: escrow - burn, deadline: *deadline, status: L2Status::Open, fetches: vec![], fetch_held: 0 });
            }
            L2Tx::Claim { job_id } => {
                let (net, status, deadline) = {
                    let j = self.l2.get(job_id).ok_or("unknown job")?;
                    (j.escrow_net, j.status.clone(), j.deadline)
                };
                if status != L2Status::Open || height > deadline {
                    return bad("job not claimable");
                }
                let bond = bps(net, BOND_BPS);
                self.debit(from, bond)?;
                self.l2.get_mut(job_id).unwrap().status = L2Status::Claimed { worker: *from, bond };
            }
            L2Tx::Post { job_id, root } => {
                let j = self.l2.get_mut(job_id).ok_or("unknown job")?;
                let L2Status::Claimed { worker, bond } = j.status.clone() else { return bad("job is not in progress") };
                if worker != *from || height > j.deadline {
                    return bad("cannot post");
                }
                j.status = L2Status::Posted { worker, bond, root: *root, at: height, at_ts: ts };
            }
            L2Tx::Accept { job_id } => {
                let (client, status, net) = {
                    let j = self.l2.get(job_id).ok_or("unknown job")?;
                    (j.client, j.status.clone(), j.escrow_net)
                };
                match status {
                    L2Status::Posted { worker, bond, .. } if client == *from => {
                        self.credit(&worker, net + bond);
                        self.rep.entry(worker).or_default().done += 1;
                        let j = self.l2.get_mut(job_id).unwrap();
                        let held = std::mem::take(&mut j.fetch_held);
                        j.fetches.clear();
                        j.status = L2Status::Settled;
                        self.credit(&client, held);
                    }
                    _ => return bad("cannot accept"),
                }
            }
            L2Tx::Fetch { job_id, chunk } => {
                let j = self.l2.get(job_id).ok_or("unknown job")?;
                let L2Status::Posted { at, at_ts, .. } = j.status.clone() else { return bad("no posted result") };
                if j.client != *from {
                    return bad("only the client can request data");
                }
                if *chunk >= j.chunks {
                    return bad("chunk out of range");
                }
                if !self.timing.challenge_open(height, ts, at, at_ts) {
                    return bad("challenge window closed");
                }
                if j.fetches.len() >= MAX_PENDING_CHUNKS || j.fetches.iter().any(|f| f.0 == *chunk) {
                    return bad("too many or duplicate pending requests");
                }
                self.debit(from, FETCH_FEE)?;
                let serve = self.timing.serve_secs;
                let jm = self.l2.get_mut(job_id).unwrap();
                jm.fetch_held += FETCH_FEE;
                jm.fetches.push((*chunk, ts + serve));
            }
            L2Tx::Serve { job_id, chunk, out, path } => {
                let j = self.l2.get(job_id).ok_or("unknown job")?;
                let L2Status::Posted { worker, bond, root, at, at_ts } = j.status.clone() else { return bad("no posted result") };
                if worker != *from {
                    return bad("only the worker can serve");
                }
                let Some(pos) = j.fetches.iter().position(|f| f.0 == *chunk) else { return bad("no such pending request") };
                if out.len() > MAX_OUT || !verify_proof(&root, chunk_leaf(*chunk, out), *chunk as usize, path, depth_for(j.chunks as usize)) {
                    return bad("bad Merkle proof for the served chunk");
                }
                let chal = self.timing.challenge_secs;
                let jm = self.l2.get_mut(job_id).unwrap();
                jm.fetches.remove(pos);
                jm.fetch_held -= FETCH_FEE;
                let new_at_ts = (ts + FETCH_EXTEND_SECS).saturating_sub(chal).max(at_ts);
                let new_at = (height + 2).saturating_sub(CHALLENGE_WINDOW).max(at);
                jm.status = L2Status::Posted { worker, bond, root, at: new_at, at_ts: new_at_ts };
                self.credit(&worker, FETCH_FEE);
            }
            L2Tx::SlashUnserved { job_id } => {
                let j = self.l2.get(job_id).ok_or("unknown job")?;
                if !matches!(j.status, L2Status::Posted { .. }) {
                    return bad("no posted result");
                }
                if !j.fetches.iter().any(|f| ts > f.1) {
                    return bad("no expired data request");
                }
                self.l2_sanction(from, job_id);
            }
            L2Tx::Fraud { job_id, chunk, out, path } => {
                let j = self.l2.get(job_id).ok_or("unknown job")?;
                let L2Status::Posted { root, at, at_ts, .. } = j.status.clone() else { return bad("no posted result") };
                if !self.timing.challenge_open(height, ts, at, at_ts) {
                    return bad("challenge window closed");
                }
                if *chunk >= j.chunks || out.len() > MAX_OUT {
                    return bad("malformed fraud proof");
                }
                if !verify_proof(&root, chunk_leaf(*chunk, out), *chunk as usize, path, depth_for(j.chunks as usize)) {
                    return bad("bad chunk opening");
                }
                // the chain recomputes THIS chunk (cost bounded by MAX_CHUNK_COST at submission)
                if j.kernel.chunk(*chunk) == *out {
                    return bad("chunk is correct: not a fraud");
                }
                self.l2_sanction(from, job_id);
            }
            L2Tx::Settle { job_id } => {
                let (client, status, net, deadline, pending) = {
                    let j = self.l2.get(job_id).ok_or("unknown job")?;
                    (j.client, j.status.clone(), j.escrow_net, j.deadline, !j.fetches.is_empty())
                };
                match status {
                    L2Status::Posted { .. } if pending => return bad("data requests pending: serve or slash first"),
                    L2Status::Posted { worker, bond, at, at_ts, .. } if !self.timing.challenge_open(height, ts, at, at_ts) => {
                        self.credit(&worker, net + bond);
                        self.rep.entry(worker).or_default().done += 1;
                        self.l2.get_mut(job_id).unwrap().status = L2Status::Settled;
                    }
                    L2Status::Open if height > deadline => {
                        self.credit(&client, net);
                        self.l2.get_mut(job_id).unwrap().status = L2Status::Expired;
                    }
                    L2Status::Claimed { worker, bond } if height > deadline => {
                        self.rep.entry(worker).or_default().expired += 1;
                        self.burned += bond;
                        self.credit(&client, net);
                        self.l2.get_mut(job_id).unwrap().status = L2Status::Expired;
                    }
                    _ => return bad("cannot settle yet"),
                }
            }
        }
        Ok(())
    }
}
