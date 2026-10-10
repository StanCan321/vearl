//! Level-2 kernel "Render2": a path-free, integer-only 3D renderer (ray marching of signed distance fields) with smooth blending of
//! shapes, soft shadows, ambient occlusion, up to two mirror bounces with a Fresnel term, emissive glow, fog, a procedural sky with a
//! sun, a free camera, ACES tone mapping and a vignette. Every operation is an integer operation: the output of a 16x16 tile is
//! bit-identical on any CPU and on the GPU (`gpu/vearl_gpu.cu::render2_tiles`), which is what lets a node judge a one-tile fraud proof.
//! Specification: docs/10-level-2.md section 4 bis. Independent implementation: tools/l2_ref.py.
use serde::{Deserialize, Serialize};

pub const TILE: usize = 16;
pub const MAX_PRIMS2: usize = 32;
pub const MAX_SIDE: u16 = 2048;
/// An image of 1920 x 1088 has 120 x 68 = 8160 tiles.
pub const MAX_RENDER_CHUNKS: u32 = 8192;

const Q: i64 = 1 << 16;
pub const MARCH: u64 = 64;
pub const SOFT_STEPS: u64 = 32;
pub const AO_N: u64 = 5;
const HIT_EPS: i64 = 64;
const T_MAX: i64 = 40 * Q;
const FAR_HIT: i64 = 8192;
const SOFT_T0: i64 = 2621;
const SOFT_T_MAX: i64 = 14 * Q;
const NORMAL_H: i64 = 256;
const BIAS: i64 = 2621;
const GLOW_R: i64 = 3 * Q / 5;
/// Reach of the light that emissive shapes cast on nearby surfaces.
const EMIT_LIGHT_R: i64 = 5 * Q / 2;
const EMIT_LIGHT_GAIN: i64 = 3;
const AO_W: [i64; 5] = [65536, 62259, 59146, 56188, 53379];
const AO_STEP: i64 = 5243;
const DIRECT_GAIN: i64 = 2;
const SPEC_GAIN: i64 = 2;
const AMB_GAIN: i64 = 2;
const VIGNETTE: i64 = 22000;
// ACES approximation (Narkowicz) in Q16
const ACES_A: i64 = 164495;
const ACES_B: i64 = 1966;
const ACES_C: i64 = 159252;
const ACES_D: i64 = 38666;
const ACES_E: i64 = 9175;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Material {
    pub col: [u8; 3],
    /// Second colour of a checkerboard (used when `checker` is true).
    pub col2: [u8; 3],
    /// Mirror reflectivity, 0..=255 (Fresnel raises it at grazing angles).
    pub refl: u8,
    /// Emission, 0..=255 (128 = the base colour at full brightness); an emissive shape also glows.
    pub emit: u8,
    pub checker: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Shape {
    Sphere { c: [i32; 3], r: i32 },
    /// Rounded box: half sizes `h`, corner radius `rnd` (0 = sharp).
    Box { c: [i32; 3], h: [i32; 3], rnd: i32 },
    /// Torus in the horizontal plane through `c`: major radius `big`, tube radius `small`.
    Torus { c: [i32; 3], big: i32, small: i32 },
    Plane { y: i32 },
    /// Rounded cylinder with a vertical axis through `c`: radius `r`, half height `h`, edge rounding `rnd` (both sizes grow by `rnd`).
    Cylinder { c: [i32; 3], r: i32, h: i32, rnd: i32 },
    /// Segment from `a` to `b` thickened by radius `r` (any orientation: hands, markers, lugs, crowns).
    Capsule { a: [i32; 3], b: [i32; 3], r: i32 },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Prim2 {
    pub shape: Shape,
    pub mat: Material,
}

/// All coordinates are Q16.16 fixed point (`i32`, 65536 = 1.0).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RenderParams2 {
    pub seed: [u8; 32],
    pub width: u16,
    pub height: u16,
    pub spp: u8,
    pub bounces: u8,
    pub cam: [i32; 3],
    pub target: [i32; 3],
    /// Focal length: 65536 = 90 degrees of horizontal field of view; larger = narrower.
    pub fov: i32,
    /// Direction TOWARD the light (does not need to be normalised).
    pub light: [i32; 3],
    pub light_col: [u8; 3],
    pub sky_top: [u8; 3],
    pub sky_hor: [u8; 3],
    /// Fog density per unit of distance (Q16); 0 = none.
    pub fog: i32,
    /// Smooth-union radius between the non-plane shapes (Q16); 0 = hard union.
    pub smooth_k: i32,
    /// Exposure multiplier (Q16; 65536 = 1.0).
    pub exposure: i32,
    pub prims: Vec<Prim2>,
}

type V3 = [i64; 3];

fn isqrt(x: i64) -> i64 {
    if x < 2 {
        return x;
    }
    let mut r = (x as f64).sqrt() as i64;
    while r * r > x {
        r -= 1;
    }
    while (r + 1) * (r + 1) <= x {
        r += 1;
    }
    r
}

fn mul16(a: i64, b: i64) -> i64 {
    (a * b) >> 16
}

fn len3(v: V3) -> i64 {
    isqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
}

fn len2(a: i64, b: i64) -> i64 {
    isqrt(a * a + b * b)
}

/// Normalisation in Q16; division truncated toward zero; the null vector gives (0, 1, 0).
fn norm3(v: V3) -> V3 {
    let l = len3(v);
    if l == 0 {
        return [0, Q, 0];
    }
    [(v[0] << 16) / l, (v[1] << 16) / l, (v[2] << 16) / l]
}

fn dot16(a: V3, b: V3) -> i64 {
    (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) >> 16
}

fn cross16(a: V3, b: V3) -> V3 {
    [(a[1] * b[2] - a[2] * b[1]) >> 16, (a[2] * b[0] - a[0] * b[2]) >> 16, (a[0] * b[1] - a[1] * b[0]) >> 16]
}

fn at(o: V3, d: V3, t: i64) -> V3 {
    [o[0] + mul16(d[0], t), o[1] + mul16(d[1], t), o[2] + mul16(d[2], t)]
}

fn v3(c: [i32; 3]) -> V3 {
    [c[0] as i64, c[1] as i64, c[2] as i64]
}

/// Colour channel 0..=255 to linear Q16 (squared: the output stage takes the square root back).
fn lin(c: u8) -> i64 {
    ((c as i64 * c as i64) << 16) / 65025
}

fn lin3(c: [u8; 3]) -> V3 {
    [lin(c[0]), lin(c[1]), lin(c[2])]
}

fn pow_sq(mut x: i64, squarings: u32) -> i64 {
    for _ in 0..squarings {
        x = mul16(x, x);
    }
    x
}

fn shape_dist(s: &Shape, p: V3) -> i64 {
    match s {
        Shape::Sphere { c, r } => len3([p[0] - c[0] as i64, p[1] - c[1] as i64, p[2] - c[2] as i64]) - *r as i64,
        Shape::Plane { y } => p[1] - *y as i64,
        Shape::Box { c, h, rnd } => {
            let q = [(p[0] - c[0] as i64).abs() - h[0] as i64, (p[1] - c[1] as i64).abs() - h[1] as i64, (p[2] - c[2] as i64).abs() - h[2] as i64];
            let outside = len3([q[0].max(0), q[1].max(0), q[2].max(0)]);
            outside + q[0].max(q[1]).max(q[2]).min(0) - *rnd as i64
        }
        Shape::Torus { c, big, small } => {
            let q = [p[0] - c[0] as i64, p[1] - c[1] as i64, p[2] - c[2] as i64];
            let l = len2(q[0], q[2]) - *big as i64;
            len2(l, q[1]) - *small as i64
        }
        Shape::Cylinder { c, r, h, rnd } => {
            let q = [p[0] - c[0] as i64, p[1] - c[1] as i64, p[2] - c[2] as i64];
            let dx = len2(q[0], q[2]) - *r as i64;
            let dy = q[1].abs() - *h as i64;
            len2(dx.max(0), dy.max(0)) + dx.max(dy).min(0) - *rnd as i64
        }
        Shape::Capsule { a, b, r } => {
            let pa = [p[0] - a[0] as i64, p[1] - a[1] as i64, p[2] - a[2] as i64];
            let ba = [b[0] as i64 - a[0] as i64, b[1] as i64 - a[1] as i64, b[2] as i64 - a[2] as i64];
            let tq = ((dot16(pa, ba) << 16) / dot16(ba, ba)).clamp(0, Q);
            let q = [pa[0] - mul16(ba[0], tq), pa[1] - mul16(ba[1], tq), pa[2] - mul16(ba[2], tq)];
            len3(q) - *r as i64
        }
    }
}

fn smin(a: i64, b: i64, k: i64) -> i64 {
    if k <= 0 {
        return a.min(b);
    }
    let h = (k - (a - b).abs()).max(0);
    let hq = (h << 16) / k;
    a.min(b) - ((mul16(hq, hq) * k) >> 18)
}

struct Eval {
    d: i64,
    idx: usize,
    /// nearest emissive shape (any kind): used for the neon light cast on surfaces
    emin: i64,
    eidx: usize,
    /// nearest emissive shape that is NOT a sphere: its halo is sampled along the march (emissive spheres get an exact halo)
    gmin: i64,
    gidx: usize,
}

/// Distance to the scene (smooth union of the non-plane shapes, hard union with the planes), nearest shape index (raw distances,
/// ties to the lowest index), and the distance to / index of the nearest emissive shape.
fn eval(prims: &[Prim2], k: i64, p: V3) -> Eval {
    let (mut blend, mut plane) = (i64::MAX, i64::MAX);
    let (mut best, mut idx) = (i64::MAX, 0usize);
    let (mut emin, mut eidx) = (i64::MAX, 0usize);
    let (mut gmin, mut gidx) = (i64::MAX, 0usize);
    let mut first = true;
    for (i, pr) in prims.iter().enumerate() {
        let d = shape_dist(&pr.shape, p);
        if d < best {
            best = d;
            idx = i;
        }
        if pr.mat.emit > 0 {
            if d < emin {
                emin = d;
                eidx = i;
            }
            if !matches!(pr.shape, Shape::Sphere { .. }) && d < gmin {
                gmin = d;
                gidx = i;
            }
        }
        if let Shape::Plane { .. } = pr.shape {
            plane = plane.min(d);
        } else if first {
            blend = d;
            first = false;
        } else {
            blend = smin(blend, d, k);
        }
    }
    Eval { d: blend.min(plane), idx, emin, eidx, gmin, gidx }
}

pub(crate) struct Ctx<'a> {
    p: &'a RenderParams2,
    k: i64,
    light: V3,
    light_lin: V3,
    top_lin: V3,
    hor_lin: V3,
}

fn sky(c: &Ctx, d: V3) -> V3 {
    let e = (5 * d[1]).clamp(0, Q);
    let m = mul16(e, 2 * Q - e);
    let sun = {
        let s = dot16(d, c.light).max(0);
        let s64 = pow_sq(s, 6);
        let s256 = pow_sq(s64, 2);
        (s64 >> 2) + (s256 << 2)
    };
    // stars: a hash of the direction quantised to ~0.05 degree cells; about one cell in 8192 is a star, fading toward the horizon
    let star = if d[1] > 0 {
        let h = star_hash((d[0] >> 6) as u32, (d[1] >> 6) as u32, (d[2] >> 6) as u32);
        if h & 0x1FFF == 0 {
            mul16((((h >> 13) & 0xFF) as i64) << 8, Q - m) * 2
        } else {
            0
        }
    } else {
        0
    };
    let mut out = [0i64; 3];
    for i in 0..3 {
        out[i] = c.hor_lin[i] + mul16(c.top_lin[i] - c.hor_lin[i], m) + mul16(c.light_lin[i], sun) + star;
    }
    out
}

fn star_hash(x: u32, y: u32, z: u32) -> u32 {
    let mut h = x.wrapping_mul(0x9E3779B1) ^ y.wrapping_mul(0x85EBCA77) ^ z.wrapping_mul(0xC2B2AE3D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B3C6D);
    h ^ (h >> 12)
}

fn normal(c: &Ctx, pos: V3) -> V3 {
    let e = |dx: i64, dy: i64, dz: i64| eval(&c.p.prims, c.k, [pos[0] + dx, pos[1] + dy, pos[2] + dz]).d;
    let h = NORMAL_H;
    norm3([e(h, 0, 0) - e(-h, 0, 0), e(0, h, 0) - e(0, -h, 0), e(0, 0, h) - e(0, 0, -h)])
}

fn soft_shadow(c: &Ctx, o: V3) -> i64 {
    let (mut res, mut t) = (Q, SOFT_T0);
    for _ in 0..SOFT_STEPS {
        let d = eval(&c.p.prims, c.k, at(o, c.light, t)).d;
        if d < HIT_EPS {
            res = 0;
            break;
        }
        let w = ((8 * d) << 16) / t;
        if w < res {
            res = w;
        }
        t += d.clamp(1311, 32768);
        if t > SOFT_T_MAX {
            break;
        }
    }
    let r = res.clamp(0, Q);
    mul16(r, mul16(r, 3 * Q - 2 * r))
}

fn ambient_occlusion(c: &Ctx, pos: V3, n: V3) -> i64 {
    let mut occ = 0i64;
    for i in 0..AO_N as usize {
        let hr = (i as i64 + 1) * AO_STEP;
        let q = [pos[0] + mul16(n[0], hr), pos[1] + mul16(n[1], hr), pos[2] + mul16(n[2], hr)];
        let dd = eval(&c.p.prims, c.k, q).d;
        occ += ((hr - dd).max(0) * AO_W[i]) >> 16;
    }
    (Q - (occ + (occ >> 1))).clamp(0, Q)
}

/// Linear HDR colour of a surface point (no reflection), including emission and fog.
fn surface(c: &Ctx, pos: V3, idx: usize, d: V3, t: i64) -> (V3, V3) {
    let pr = &c.p.prims[idx];
    let n = normal(c, pos);
    let base = if pr.mat.checker && matches!(pr.shape, Shape::Plane { .. }) {
        if ((pos[0] >> 16) + (pos[2] >> 16)) & 1 == 0 {
            lin3(pr.mat.col)
        } else {
            lin3(pr.mat.col2)
        }
    } else {
        lin3(pr.mat.col)
    };
    let ndl = dot16(n, c.light).max(0);
    let o = [pos[0] + mul16(n[0], BIAS), pos[1] + mul16(n[1], BIAS), pos[2] + mul16(n[2], BIAS)];
    let shadow = soft_shadow(c, o);
    let ao = ambient_occlusion(c, pos, n);
    let hemi = ((n[1] + Q) >> 1).clamp(0, Q);
    let half = norm3([c.light[0] - d[0], c.light[1] - d[1], c.light[2] - d[2]]);
    let spec = pow_sq(dot16(n, half).max(0), 5);
    let emit = pr.mat.emit as i64 * 512;
    let fogf = {
        let f = mul16(t, c.p.fog as i64).clamp(0, Q);
        mul16(f, f)
    };
    let skyc = if fogf > 0 { sky(c, d) } else { [0; 3] };
    let near = eval(&c.p.prims, c.k, pos);
    let (ew, ecol) = if near.emin < EMIT_LIGHT_R {
        let w = ((EMIT_LIGHT_R - near.emin.max(0)) << 16) / EMIT_LIGHT_R;
        (mul16(w, w), lin3(c.p.prims[near.eidx].mat.col))
    } else {
        (0, [0; 3])
    };
    let mut out = [0i64; 3];
    for i in 0..3 {
        let amb = mul16(c.hor_lin[i] + mul16(c.top_lin[i] - c.hor_lin[i], hemi), ao) * AMB_GAIN;
        let direct = mul16(c.light_lin[i], mul16(ndl, shadow)) * DIRECT_GAIN;
        let spec_c = mul16(c.light_lin[i], mul16(spec, shadow)) * SPEC_GAIN;
        let col = mul16(base[i], direct + amb + mul16(ecol[i], ew) * EMIT_LIGHT_GAIN) + spec_c + mul16(base[i], emit);
        out[i] = col + mul16(skyc[i] - col, fogf);
    }
    (out, n)
}

fn trace(c: &Ctx, o0: V3, d0: V3) -> V3 {
    let (mut o, mut d) = (o0, d0);
    let mut acc = [0i64; 3];
    let mut thr = Q;
    for bounce in 0..=c.p.bounces {
        let (mut t, mut hit) = (0i64, None);
        let (mut emin, mut eidx) = (i64::MAX, 0usize);
        let mut last = None;
        for _ in 0..MARCH {
            let pos = at(o, d, t);
            let e = eval(&c.p.prims, c.k, pos);
            if bounce == 0 && e.gmin < emin {
                emin = e.gmin;
                eidx = e.gidx;
            }
            if e.d < HIT_EPS {
                hit = Some((pos, e.idx));
                break;
            }
            last = Some((pos, e.idx, e.d, t));
            t += e.d;
            if t > T_MAX {
                last = None;
                break;
            }
        }
        // 64 steps exhausted close to a surface (a floor seen at a grazing angle): count it as a hit, the fog hides the approximation
        if hit.is_none() {
            if let Some((pos, idx, dd, tt)) = last {
                if dd < FAR_HIT {
                    hit = Some((pos, idx));
                    t = tt;
                }
            }
        }
        if bounce == 0 {
            if emin < GLOW_R {
                let g = ((GLOW_R - emin.max(0)) << 16) / GLOW_R;
                let g = mul16(g, g);
                let ec = lin3(c.p.prims[eidx].mat.col);
                for i in 0..3 {
                    acc[i] += mul16(ec[i], g) >> 1;
                }
            }
            // exact halo of every emissive sphere: distance from its centre to the part of the ray that was marched
            let t_end = if hit.is_some() { t } else { T_MAX };
            for pr in c.p.prims.iter() {
                if let (Shape::Sphere { c: sc, r }, true) = (&pr.shape, pr.mat.emit > 0) {
                    let oc = [sc[0] as i64 - o[0], sc[1] as i64 - o[1], sc[2] as i64 - o[2]];
                    let tc = dot16(oc, d).clamp(0, t_end);
                    let q = at(o, d, tc);
                    let dist = len3([q[0] - sc[0] as i64, q[1] - sc[1] as i64, q[2] - sc[2] as i64]) - *r as i64;
                    if dist < GLOW_R {
                        let g = ((GLOW_R - dist.max(0)) << 16) / GLOW_R;
                        let g = mul16(g, g);
                        let ec = lin3(pr.mat.col);
                        for i in 0..3 {
                            acc[i] += mul16(ec[i], g) >> 1;
                        }
                    }
                }
            }
        }
        let Some((pos, idx)) = hit else {
            let s = sky(c, d);
            for i in 0..3 {
                acc[i] += mul16(thr, s[i]);
            }
            break;
        };
        let (col, n) = surface(c, pos, idx, d, t);
        let refl = c.p.prims[idx].mat.refl as i64 * 257;
        if bounce == c.p.bounces || refl == 0 {
            for i in 0..3 {
                acc[i] += mul16(thr, col[i]);
            }
            break;
        }
        let cosv = dot16([-d[0], -d[1], -d[2]], n).clamp(0, Q);
        let f = (refl + mul16(Q - refl, pow5(Q - cosv))).min(Q);
        for i in 0..3 {
            acc[i] += mul16(thr, mul16(col[i], Q - f));
        }
        thr = mul16(thr, f);
        let dn = dot16(d, n);
        let r = norm3([d[0] - 2 * mul16(dn, n[0]), d[1] - 2 * mul16(dn, n[1]), d[2] - 2 * mul16(dn, n[2])]);
        o = [pos[0] + mul16(n[0], BIAS), pos[1] + mul16(n[1], BIAS), pos[2] + mul16(n[2], BIAS)];
        d = r;
    }
    acc
}

fn pow5(x: i64) -> i64 {
    let x2 = mul16(x, x);
    mul16(mul16(x2, x2), x)
}

fn aces(c: i64) -> i64 {
    let c = c.max(0);
    let num = mul16(c, mul16(ACES_A, c) + ACES_B);
    let den = mul16(c, mul16(ACES_C, c) + ACES_D) + ACES_E;
    ((num << 16) / den).clamp(0, Q)
}

pub fn render2_validate(p: &RenderParams2) -> Result<(), String> {
    let (w, h) = (p.width as usize, p.height as usize);
    if w % TILE != 0 || h % TILE != 0 || w < TILE || h < TILE || p.width > MAX_SIDE || p.height > MAX_SIDE || p.spp == 0 || p.spp > 4 || p.bounces > 2 {
        return Err("bad image size, samples per pixel or bounces".into());
    }
    if p.prims.is_empty() || p.prims.len() > MAX_PRIMS2 {
        return Err("bad number of shapes".into());
    }
    let lim = 1i64 << 23;
    let ok = |v: i32| (v as i64).abs() <= lim;
    let all_ok = p.prims.iter().all(|pr| match &pr.shape {
        Shape::Sphere { c, r } => c.iter().all(|v| ok(*v)) && ok(*r) && *r > 0,
        Shape::Plane { y } => ok(*y),
        Shape::Box { c, h, rnd } => c.iter().all(|v| ok(*v)) && h.iter().all(|v| ok(*v) && *v > 0) && ok(*rnd) && *rnd >= 0,
        Shape::Torus { c, big, small } => c.iter().all(|v| ok(*v)) && ok(*big) && ok(*small) && *small > 0 && *big > *small,
        Shape::Cylinder { c, r, h, rnd } => c.iter().all(|v| ok(*v)) && ok(*r) && ok(*h) && *r > 0 && *h > 0 && ok(*rnd) && *rnd >= 0,
        Shape::Capsule { a, b, r } => {
            let ba = [b[0] as i64 - a[0] as i64, b[1] as i64 - a[1] as i64, b[2] as i64 - a[2] as i64];
            a.iter().all(|v| ok(*v)) && b.iter().all(|v| ok(*v)) && ok(*r) && *r > 0 && ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2] >= 1 << 24
        }
    });
    let in_range = |v: &[i32]| v.iter().all(|x| ok(*x));
    if !all_ok || !in_range(&p.cam) || !in_range(&p.target) || !in_range(&p.light) {
        return Err("scene coordinates out of range".into());
    }
    if p.fov < (Q as i32) / 4 || p.fov > 8 * Q as i32 || p.fog < 0 || p.fog > 8 * Q as i32 || p.smooth_k < 0 || p.smooth_k > 4 * Q as i32 || p.exposure < 0 || p.exposure > 16 * Q as i32 {
        return Err("camera, fog, blending or exposure out of range".into());
    }
    Ok(())
}

/// Cost bound of one tile, in evaluation units: pixels x samples x rays x (march + normal + soft shadow + occlusion) x shapes.
pub fn render2_cost(p: &RenderParams2) -> u64 {
    (TILE * TILE) as u64 * p.spp as u64 * (1 + p.bounces as u64) * (MARCH + 7 + SOFT_STEPS + AO_N) * p.prims.len() as u64
}

pub fn jitter_bytes(seed: &[u8; 32], chunk: u32, spp: usize) -> Vec<u8> {
    let mut hs = blake3::Hasher::new_derive_key("vearl/l2/render/v1");
    hs.update(seed);
    hs.update(&chunk.to_le_bytes());
    let mut buf = vec![0u8; TILE * TILE * spp * 4];
    hs.finalize_xof().fill(&mut buf);
    buf
}

pub(crate) fn make_ctx(p: &RenderParams2) -> (Ctx<'_>, V3, V3, V3, V3) {
    let c = Ctx { p, k: p.smooth_k as i64, light: norm3(v3(p.light)), light_lin: lin3(p.light_col), top_lin: lin3(p.sky_top), hor_lin: lin3(p.sky_hor) };
    let cam = v3(p.cam);
    let fwd = norm3([p.target[0] as i64 - cam[0], p.target[1] as i64 - cam[1], p.target[2] as i64 - cam[2]]);
    let right = norm3(cross16([0, Q, 0], fwd));
    let up = norm3(cross16(fwd, right));
    (c, cam, fwd, right, up)
}

/// Output of one tile: 16 x 16 pixels, RGB bytes, row by row. Chunk i = tile (i % (W/16), i / (W/16)).
pub fn render2_chunk(p: &RenderParams2, chunk: u32) -> Vec<u8> {
    let tiles_x = p.width as usize / TILE;
    let (tx, ty) = (chunk as usize % tiles_x, chunk as usize / tiles_x);
    let spp = p.spp as usize;
    let jitter = if spp > 1 { jitter_bytes(&p.seed, chunk, spp) } else { vec![] };
    let (ctx, cam, fwd, right, up) = make_ctx(p);
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
                let f = p.fov as i64;
                let draw = [mul16(fwd[0], f) + mul16(right[0], u) + mul16(up[0], v), mul16(fwd[1], f) + mul16(right[1], u) + mul16(up[1], v), mul16(fwd[2], f) + mul16(right[2], u) + mul16(up[2], v)];
                let c = trace(&ctx, cam, norm3(draw));
                for k in 0..3 {
                    acc[k] += c[k];
                }
            }
            let (cx, cy) = (((tx * TILE + x) as i64) * Q + Q / 2, ((ty * TILE + y) as i64) * Q + Q / 2);
            let nx = (2 * cx - w * Q) / w;
            let ny = (2 * cy - h * Q) / w;
            let vig = (Q - mul16(mul16(nx, nx) + mul16(ny, ny), VIGNETTE)).max(16384);
            for k in 0..3 {
                let mean = acc[k] / spp as i64;
                let c = mul16(mul16(mean, p.exposure as i64), vig);
                let g = isqrt((aces(c) * 65025) >> 16);
                out.push(g.min(255) as u8);
            }
        }
    }
    out
}

// ---------------------------------------------------------------- scenes (client side: floating point is fine here, it only builds the integer parameters)

fn f(x: f64) -> i32 {
    (x * 65536.0).round() as i32
}

fn mat(col: [u8; 3], refl: u8, emit: u8) -> Material {
    Material { col, col2: col, refl, emit, checker: false }
}

/// "Vearl Orb": a chrome blob of three merged spheres inside a glowing neon ring, five floating light orbs, two polished pillars and a
/// mirror checkerboard floor, under a dusk sky. `angle` (degrees) orbits the camera around the scene; `rise` is the camera height.
pub fn scene_orb(width: u16, height: u16, spp: u8, bounces: u8, angle: f64, rise: f64, seed: [u8; 32]) -> RenderParams2 {
    scene_orb_anim(width, height, spp, bounces, angle, rise, 6.4, 0.0, seed)
}

/// The same scene as a frame of an animation: `dist` is the camera distance and `phase` (radians) moves the light orbs around the
/// ring and makes the three blob spheres drift, so that consecutive frames form a film. Everything here is client-side floating
/// point that only builds the integer parameters; each frame is an ordinary Render2 job.
pub fn scene_orb_anim(width: u16, height: u16, spp: u8, bounces: u8, angle: f64, rise: f64, dist: f64, phase: f64, seed: [u8; 32]) -> RenderParams2 {
    let a = angle.to_radians();
    let cam = [f(dist * a.sin()), f(rise), f(-dist * a.cos())];
    let mut prims = vec![Prim2 { shape: Shape::Plane { y: f(-1.0) }, mat: Material { col: [26, 14, 52], col2: [96, 52, 160], refl: 95, emit: 0, checker: true } }];
    prims.push(Prim2 { shape: Shape::Sphere { c: [f(0.0), f(0.1), f(0.0)], r: f(0.82) }, mat: mat([214, 218, 240], 120, 0) });
    prims.push(Prim2 { shape: Shape::Sphere { c: [f(0.72 * phase.cos() + 0.0 * phase.sin()), f(-0.08 + 0.12 * (phase * 2.0).sin()), f(0.34 + 0.72 * phase.sin() * 0.55)], r: f(0.44) }, mat: mat([255, 86, 64], 70, 0) });
    prims.push(Prim2 { shape: Shape::Sphere { c: [f(-0.66 * phase.cos()), f(-0.22 - 0.1 * (phase * 2.0).sin()), f(-0.42 - 0.5 * phase.sin())], r: f(0.4) }, mat: mat([58, 200, 255], 70, 0) });
    prims.push(Prim2 { shape: Shape::Torus { c: [f(0.0), f(0.1), f(0.0)], big: f(1.55), small: f(0.075) }, mat: mat([255, 70, 210], 40, 210) });
    for i in 0..5 {
        let t = (i as f64) * std::f64::consts::TAU / 5.0 + 0.4 + phase;
        prims.push(Prim2 { shape: Shape::Sphere { c: [f(2.5 * t.cos()), f(-0.45 + 0.35 * (i as f64 * 1.7).sin()), f(2.5 * t.sin())], r: f(0.13) }, mat: mat([90, 225, 255], 0, 230) });
    }
    prims.push(Prim2 { shape: Shape::Box { c: [f(-2.7), f(-0.35), f(1.9)], h: [f(0.26), f(0.65), f(0.26)], rnd: f(0.07) }, mat: mat([46, 52, 72], 130, 0) });
    prims.push(Prim2 { shape: Shape::Box { c: [f(2.9), f(-0.5), f(-1.4)], h: [f(0.3), f(0.5), f(0.3)], rnd: f(0.07) }, mat: mat([52, 46, 72], 130, 0) });
    RenderParams2 {
        seed,
        width,
        height,
        spp,
        bounces,
        cam,
        target: [f(0.0), f(0.05), f(0.0)],
        fov: f(1.3),
        light: [f(0.3), f(0.3), f(0.9)],
        light_col: [255, 176, 118],
        sky_top: [4, 6, 44],
        sky_hor: [150, 44, 140],
        fog: f(0.055),
        smooth_k: f(0.45),
        exposure: f(1.0),
        prims,
    }
}

/// A second, different scene used by the test vectors: no floor, no emissive shape, hard unions, a torus, a rounded box, fog, an off-axis
/// camera and an exposure above 1: it exercises the branches that `scene_orb` does not (sky-only rays, `smooth_k = 0`, no glow).
pub fn scene_prism(width: u16, height: u16, spp: u8, bounces: u8, seed: [u8; 32]) -> RenderParams2 {
    let prims = vec![
        Prim2 { shape: Shape::Sphere { c: [f(-0.8), f(0.2), f(0.5)], r: f(0.7) }, mat: mat([240, 90, 60], 160, 0) },
        Prim2 { shape: Shape::Box { c: [f(0.9), f(-0.1), f(0.2)], h: [f(0.5), f(0.5), f(0.5)], rnd: f(0.12) }, mat: mat([70, 190, 120], 90, 0) },
        Prim2 { shape: Shape::Torus { c: [f(0.0), f(-0.55), f(1.4)], big: f(1.0), small: f(0.3) }, mat: mat([200, 200, 230], 220, 0) },
        Prim2 { shape: Shape::Box { c: [f(0.0), f(1.4), f(0.8)], h: [f(2.0), f(0.1), f(0.6)], rnd: 0 }, mat: mat([90, 110, 200], 40, 0) },
    ];
    RenderParams2 {
        seed,
        width,
        height,
        spp,
        bounces,
        cam: [f(-1.6), f(0.7), f(-4.2)],
        target: [f(0.1), f(0.1), f(0.4)],
        fov: f(1.1),
        light: [f(-0.4), f(0.8), f(-0.5)],
        light_col: [255, 244, 224],
        sky_top: [40, 90, 200],
        sky_hor: [200, 220, 255],
        fog: f(0.06),
        smooth_k: 0,
        exposure: f(1.35),
        prims,
    }
}

/// A wristwatch lying on a dark mirror table: rounded steel case, gold bezel, navy dial with twelve glowing markers, three hands set to
/// 10:10, a crown, four lugs and a leather strap. `angle` (degrees) orbits the camera; `rise` is its height.
pub fn scene_watch(width: u16, height: u16, spp: u8, bounces: u8, angle: f64, rise: f64, seed: [u8; 32]) -> RenderParams2 {
    let a = angle.to_radians();
    let dist = 3.9;
    let cam = [f(dist * a.sin()), f(rise), f(-dist * a.cos())];
    let steel = mat([200, 206, 216], 150, 0);
    let gold = mat([232, 184, 92], 170, 0);
    let lume = mat([244, 214, 130], 120, 0);
    let mut prims: Vec<Prim2> = vec![];
    prims.push(Prim2 { shape: Shape::Plane { y: f(0.0) }, mat: Material { col: [14, 14, 22], col2: [30, 30, 44], refl: 80, emit: 0, checker: true } });
    // leather strap (two flat segments) and the four lugs
    let leather = mat([58, 34, 22], 25, 0);
    prims.push(Prim2 { shape: Shape::Box { c: [f(0.0), f(0.05), f(2.3)], h: [f(0.42), f(0.035), f(1.05)], rnd: f(0.03) }, mat: leather.clone() });
    prims.push(Prim2 { shape: Shape::Box { c: [f(0.0), f(0.05), f(-2.3)], h: [f(0.42), f(0.035), f(1.05)], rnd: f(0.03) }, mat: leather });
    for (sx, sz) in [(-0.5, 1.0), (0.5, 1.0), (-0.5, -1.0), (0.5, -1.0)] {
        prims.push(Prim2 { shape: Shape::Capsule { a: [f(sx * 0.9), f(0.27), f(sz * 0.8)], b: [f(sx * 0.9), f(0.15), f(sz * 1.3)], r: f(0.085) }, mat: steel.clone() });
    }
    // case, bezel, dial
    prims.push(Prim2 { shape: Shape::Cylinder { c: [f(0.0), f(0.25), f(0.0)], r: f(0.92), h: f(0.2), rnd: f(0.06) }, mat: steel });
    prims.push(Prim2 { shape: Shape::Torus { c: [f(0.0), f(0.47), f(0.0)], big: f(0.88), small: f(0.075) }, mat: gold.clone() });
    prims.push(Prim2 { shape: Shape::Cylinder { c: [f(0.0), f(0.43), f(0.0)], r: f(0.8), h: f(0.03), rnd: 0 }, mat: Material { col: [8, 20, 56], col2: [8, 20, 56], refl: 70, emit: 0, checker: false } });
    // twelve markers, 12 o'clock toward +z, 3 o'clock toward +x
    for i in 0..12 {
        let t = (i as f64) * std::f64::consts::TAU / 12.0;
        let (r0, r1) = if i % 3 == 0 { (0.56, 0.76) } else { (0.64, 0.74) };
        let m = if i % 3 == 0 { lume.clone() } else { mat([225, 225, 235], 60, 0) };
        prims.push(Prim2 { shape: Shape::Capsule { a: [f(r0 * t.sin()), f(0.475), f(r0 * t.cos())], b: [f(r1 * t.sin()), f(0.475), f(r1 * t.cos())], r: f(if i % 3 == 0 { 0.032 } else { 0.022 }) }, mat: m });
    }
    // hands at 10:10 (hour hand 305 degrees, minute hand 60 degrees, seconds hand 200 degrees)
    let hand = |deg: f64, len: f64, rad: f64, y: f64, m: Material| {
        let t = deg.to_radians();
        Prim2 { shape: Shape::Capsule { a: [f(-0.08 * t.sin()), f(y), f(-0.08 * t.cos())], b: [f(len * t.sin()), f(y), f(len * t.cos())], r: f(rad) }, mat: m }
    };
    prims.push(hand(305.0, 0.44, 0.045, 0.495, gold.clone()));
    prims.push(hand(60.0, 0.68, 0.032, 0.515, gold.clone()));
    prims.push(hand(200.0, 0.74, 0.012, 0.535, mat([230, 40, 40], 90, 0)));
    prims.push(Prim2 { shape: Shape::Sphere { c: [f(0.0), f(0.52), f(0.0)], r: f(0.06) }, mat: gold.clone() });
    prims.push(Prim2 { shape: Shape::Capsule { a: [f(0.95), f(0.27), f(0.0)], b: [f(1.2), f(0.27), f(0.0)], r: f(0.085) }, mat: gold });
    RenderParams2 {
        seed,
        width,
        height,
        spp,
        bounces,
        cam,
        target: [f(0.1), f(0.2), f(0.0)],
        fov: f(1.4),
        light: [f(-0.7), f(0.55), f(-0.3)],
        light_col: [255, 226, 190],
        sky_top: [6, 10, 38],
        sky_hor: [150, 96, 80],
        fog: f(0.02),
        smooth_k: 0,
        exposure: f(1.15),
        prims,
    }
}
