//! Level-2 kernel "Render3": a Monte Carlo path tracer over signed distance fields, in integer arithmetic only, so that every
//! tile is bit-identical on any CPU and on the GPU (`gpu/vearl_render3.cu`). It is the production renderer of the protocol (Render2
//! is the instant preview): global illumination by random walks with up to 14 bounces, glass with refraction, Fresnel and
//! Beer absorption, rough metals (GGX), clear coat, thin translucency, emission, a studio environment of soft boxes, a thin-lens
//! camera (depth of field, round bokeh), distance fog, procedural textures with bump mapping, and a shape set (ellipsoids, rounded
//! boxes and cylinders, capsules, tori, cones, planes) placed with arbitrary rotations and combined by union, smooth union,
//! subtraction and intersection, with polar repetition and mirroring.
//!
//! One chunk = one 16x16 tile for one PASS of `spp` samples per pixel; the image of a job with `passes` passes is the per-tile average
//! of its passes (done off-chain by `assemble`, followed by exposure, ACES, vignette and gamma). A chunk holds the MEAN radiance of its
//! samples per pixel as three little-endian `u16` in Q11 (2048 = 1.0), so a fraud proof only ever recomputes one pass of one tile.
//! Specification: docs/10-level-2.md section 4 ter. Independent implementation: tools/l2_ref.py (`render3_chunk`).
use serde::{Deserialize, Serialize};

pub const TILE: usize = 16;
pub const MAX_PARTS: usize = 48;
pub const MAX_MATS: usize = 24;
pub const MAX_BOXES: usize = 4;
pub const MAX_SIDE3: u16 = 2048;
pub const MAX_CHUNKS3: u32 = 262_144;
pub const MAX_CHUNK_COST3: u64 = 1_200_000_000;
pub const MAX_BOUNCES3: u8 = 14;
pub const MAX_SPP_PASS: u8 = 32;
pub const MARCH_STEPS3: u64 = 420;
/// Smallest ellipsoid radius (Q16): 1/16 of a unit. Below it the distance computation can overflow i64 (see `render3_validate`).
pub const MIN_ELLIPSOID_R: i32 = 4096;

const Q: i64 = 1 << 16;
const PI_Q: i64 = 205887;
const TAU_Q: i64 = 411775;
const MAXT: i64 = 60 * Q;
const EPS_Q: i64 = 26; // .0004
const T_START: i64 = 66; // .001
const NORMAL_E: i64 = 33; // .0005
const BIAS_G: i64 = 131; // .002
const BIAS_T: i64 = 262; // .004
const CLAMP_RAD: i64 = 24 * Q;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Geo3 {
    Sphere { r: i32 },
    Ellipsoid { r: [i32; 3] },
    /// Rounded box: half sizes `h` (the rounding `rnd` is part of them, as in `sdRoundBox`).
    Box { h: [i32; 3], rnd: i32 },
    /// Rounded cylinder along the local y axis: radius `r`, half height `h` (rounding `rnd` included).
    Cylinder { r: i32, h: i32, rnd: i32 },
    /// Capsule along the local y axis from -h to +h, radius `r`.
    Capsule { h: i32, r: i32 },
    Torus { big: i32, small: i32 },
    /// Capped cone along y: radius `r1` at y = -h, `r2` at y = +h.
    Cone { h: i32, r1: i32, r2: i32 },
    /// The half space y < 0 of the local frame.
    Plane,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Op3 {
    Union,
    /// Smooth union with blending radius `k` (Q16).
    Smooth(i32),
    Sub,
    Inter,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Part3 {
    pub geo: Geo3,
    /// Centre in world space (Q16).
    pub c: [i32; 3],
    /// World-to-local rotation, row-major, Q16: `local = rot · (p − c)`.
    pub rot: [i32; 9],
    pub op: Op3,
    pub mat: u8,
    /// Polar repetition count around the local y axis (0 or 1 = none).
    pub polar: u8,
    /// Mirror the local x and z coordinates (`|x|`, `|z|`).
    pub mirror: bool,
    /// Offset of the shape inside the local frame, applied after mirroring and polar repetition (Q16).
    pub off: [i32; 3],
    /// Group id (non-decreasing along the list): the operations of a part (smooth union, subtraction, intersection) combine it with the
    /// parts of ITS group only; the groups are then merged by a hard union.
    pub group: u8,
    /// Hollow shell: the distance becomes `|d| − shell/2`-style thickness `shell` (0 = solid).
    pub shell: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Mat3 {
    pub alb: [u8; 3],
    pub alb2: [u8; 3],
    pub rough: u8,
    pub metal: bool,
    /// Specular strength of a dielectric (128 = 0.5).
    pub spec: u8,
    /// Emission, Q10 (1024 = 1.0) per channel.
    pub emit: [u16; 3],
    /// Probability (0..255) of a thin-translucency transmission instead of a diffuse bounce.
    pub trans: u8,
    pub glass: bool,
    /// Index of refraction, Q10 (1536 = 1.5).
    pub ior: u16,
    /// Beer absorption per unit length, Q10 per channel.
    pub absorb: [u16; 3],
    /// Clear coat strength (0..255).
    pub coat: u8,
    /// Procedural texture 0..=7 (docs/10 section 4 ter) and its parameters.
    pub tex: u8,
    pub tex_scale: i32,
    pub tex_p: i32,
    pub tex_q: i32,
    pub tex_r: i32,
    /// Bump mapping strength (Q8) and frequency (Q16); 0 = smooth.
    pub bump: u8,
    pub bump_freq: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SoftBox {
    pub dir: [i32; 3],
    /// Half extents as tangents of the half angles (Q16).
    pub hs: [i32; 2],
    /// Edge softness (Q16, 0..65536).
    pub soft: i32,
    /// Radiance, Q10 per channel.
    pub col: [u32; 3],
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Env3 {
    pub top: [u32; 3],
    pub hor: [u32; 3],
    pub gnd: [u32; 3],
    pub boxes: Vec<SoftBox>,
    /// Backdrop seen directly by the camera.
    pub bd_top: [u32; 3],
    pub bd_bot: [u32; 3],
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Cam3 {
    pub pos: [i32; 3],
    pub look: [i32; 3],
    /// Focal length: 1 / tan(fov / 2), Q16.
    pub focal: i32,
    /// Lens radius (Q16); 0 = pinhole.
    pub aperture: i32,
    /// Distance of the plane in focus (Q16).
    pub focus: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RenderParams3 {
    pub seed: [u8; 32],
    pub width: u16,
    pub height: u16,
    /// Samples per pixel in ONE chunk.
    pub spp: u8,
    /// Number of passes (chunks per tile).
    pub passes: u16,
    pub bounces: u8,
    pub fog0: i32,
    pub fog1: i32,
    pub cam: Cam3,
    pub env: Env3,
    pub mats: Vec<Mat3>,
    pub parts: Vec<Part3>,
}

type V3 = [i64; 3];

// ------------------------------------------------------------------------------------------------ integer helpers

pub(crate) fn isqrt(x: i64) -> i64 {
    if x < 2 {
        return x.max(0);
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

fn add3(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale3(a: V3, s: i64) -> V3 {
    [mul16(a[0], s), mul16(a[1], s), mul16(a[2], s)]
}

fn neg3(a: V3) -> V3 {
    [-a[0], -a[1], -a[2]]
}

fn v3(c: [i32; 3]) -> V3 {
    [c[0] as i64, c[1] as i64, c[2] as i64]
}

fn lin8(c: u8) -> i64 {
    ((c as i64 * c as i64) << 16) / 65025
}

fn lin8x3(c: [u8; 3]) -> V3 {
    [lin8(c[0]), lin8(c[1]), lin8(c[2])]
}

/// Q10 -> Q16
fn q10(v: u32) -> i64 {
    (v as i64) << 6
}

fn q10x3(c: [u32; 3]) -> V3 {
    [q10(c[0]), q10(c[1]), q10(c[2])]
}

fn smoothstep(e0: i64, e1: i64, x: i64) -> i64 {
    if e1 == e0 {
        return if x >= e1 { Q } else { 0 };
    }
    let t = (((x - e0) << 16) / (e1 - e0)).clamp(0, Q);
    mul16(mul16(t, t), 3 * Q - 2 * t)
}

fn mix(a: i64, b: i64, t: i64) -> i64 {
    a + mul16(b - a, t)
}

// ---- CORDIC: sine, cosine and arctangent in Q16 radians with 24 iterations and Q24 internals
const CORDIC_ATAN: [i64; 24] = [
    13176795, 7778716, 4110060, 2086331, 1047214, 524117, 262123, 131069, 65536, 32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2,
];
const CORDIC_K: i64 = 10188014; // 0.607252935 * 2^24

/// (sin, cos) of an angle in Q16 radians, in Q16.
fn sincos(a: i64) -> (i64, i64) {
    let mut a = a.rem_euclid(TAU_Q);
    if a > PI_Q {
        a -= TAU_Q;
    }
    let mut flip = false;
    if a > PI_Q / 2 {
        a -= PI_Q;
        flip = true;
    } else if a < -PI_Q / 2 {
        a += PI_Q;
        flip = true;
    }
    let (mut x, mut y, mut z) = (CORDIC_K, 0i64, a << 8);
    for i in 0..24 {
        let (dx, dy) = (y >> i, x >> i);
        if z >= 0 {
            x -= dx;
            y += dy;
            z -= CORDIC_ATAN[i];
        } else {
            x += dx;
            y -= dy;
            z += CORDIC_ATAN[i];
        }
    }
    let (mut s, mut c) = ((y + 128) >> 8, (x + 128) >> 8);
    if flip {
        s = -s;
        c = -c;
    }
    (s, c)
}

/// atan2(y, x) in Q16 radians, in (-pi, pi].
fn atan2q(y: i64, x: i64) -> i64 {
    if x == 0 && y == 0 {
        return 0;
    }
    let (mut xx, mut yy) = (x << 8, y << 8);
    let mut base = 0i64;
    if xx < 0 {
        // rotate by +-pi so that the vector points into the right half plane
        if yy >= 0 {
            base = PI_Q;
        } else {
            base = -PI_Q;
        }
        xx = -xx;
        yy = -yy;
    }
    let mut z = 0i64;
    for i in 0..24 {
        let (dx, dy) = (yy >> i, xx >> i);
        if yy > 0 {
            xx += dx;
            yy -= dy;
            z += CORDIC_ATAN[i];
        } else {
            xx -= dx;
            yy += dy;
            z -= CORDIC_ATAN[i];
        }
    }
    base + ((z + 128) >> 8)
}

// ---- hashing, value noise
fn pcg(v: u32) -> u32 {
    let s = v.wrapping_mul(747796405).wrapping_add(2891336453);
    let w = ((s >> ((s >> 28) + 4)) ^ s).wrapping_mul(277803737);
    (w >> 22) ^ w
}

fn hash3(x: i64, y: i64, z: i64) -> i64 {
    let h = pcg((x as u32).wrapping_mul(73856093) ^ (y as u32).wrapping_mul(19349663) ^ (z as u32).wrapping_mul(83492791));
    (h >> 16) as i64
}

/// Value noise in [0, Q) at a point in Q16 coordinates.
fn vnoise(p: V3) -> i64 {
    let (ix, iy, iz) = (p[0] >> 16, p[1] >> 16, p[2] >> 16);
    let f = |v: i64| {
        let t = v & 0xFFFF;
        mul16(mul16(t, t), 3 * Q - 2 * t)
    };
    let (fx, fy, fz) = (f(p[0]), f(p[1]), f(p[2]));
    let h = |dx: i64, dy: i64, dz: i64| hash3(ix + dx, iy + dy, iz + dz);
    let x00 = mix(h(0, 0, 0), h(1, 0, 0), fx);
    let x10 = mix(h(0, 1, 0), h(1, 1, 0), fx);
    let x01 = mix(h(0, 0, 1), h(1, 0, 1), fx);
    let x11 = mix(h(0, 1, 1), h(1, 1, 1), fx);
    mix(mix(x00, x10, fy), mix(x01, x11, fy), fz)
}

fn fbm_n(mut p: V3, octaves: usize) -> i64 {
    let (mut a, mut s) = (Q / 2, 0i64);
    for _ in 0..octaves {
        s += mul16(a, vnoise(p));
        p = [p[0] * 2 + (p[0] >> 5) + 111411, p[1] * 2 + (p[1] >> 5) + 602931, p[2] * 2 + (p[2] >> 5) + 203162];
        a >>= 1;
    }
    s
}

/// Independent hash of a lattice cell, `k` selects one of several streams (used by the cell textures).
fn cell_hash(ix: i64, iy: i64, iz: i64, k: i64) -> i64 {
    hash3(ix + 17 * k, iy + 31 * k, iz + 47 * k)
}

/// Worley noise: distances (Q16) to the nearest and to the second nearest feature point, and a hash of the nearest cell.
fn voronoi(p: V3) -> (i64, i64, i64) {
    let (ix, iy, iz) = (p[0] >> 16, p[1] >> 16, p[2] >> 16);
    let (mut d1, mut d2, mut id) = (i64::MAX, i64::MAX, 0i64);
    for dz in -1..=1i64 {
        for dy in -1..=1i64 {
            for dx in -1..=1i64 {
                let (cx, cy, cz) = (ix + dx, iy + dy, iz + dz);
                let (ex, ey, ez) = ((cx << 16) + cell_hash(cx, cy, cz, 1) - p[0], (cy << 16) + cell_hash(cx, cy, cz, 2) - p[1], (cz << 16) + cell_hash(cx, cy, cz, 3) - p[2]);
                let d = ex * ex + ey * ey + ez * ez;
                if d < d1 {
                    d2 = d1;
                    d1 = d;
                    id = cell_hash(cx, cy, cz, 0);
                } else if d < d2 {
                    d2 = d;
                }
            }
        }
    }
    (isqrt(d1), isqrt(d2), id)
}

// ------------------------------------------------------------------------------------------------ geometry

fn rotate(rot: &[i32; 9], p: V3) -> V3 {
    [
        (rot[0] as i64 * p[0] + rot[1] as i64 * p[1] + rot[2] as i64 * p[2]) >> 16,
        (rot[3] as i64 * p[0] + rot[4] as i64 * p[1] + rot[5] as i64 * p[2]) >> 16,
        (rot[6] as i64 * p[0] + rot[7] as i64 * p[1] + rot[8] as i64 * p[2]) >> 16,
    ]
}

fn geo_dist(g: &Geo3, q: V3) -> i64 {
    match g {
        Geo3::Sphere { r } => len3(q) - *r as i64,
        Geo3::Ellipsoid { r } => {
            let r = v3(*r);
            let (a, b) = ([(q[0] << 16) / r[0], (q[1] << 16) / r[1], (q[2] << 16) / r[2]], [(((q[0] << 16) / r[0]) << 16) / r[0], (((q[1] << 16) / r[1]) << 16) / r[1], (((q[2] << 16) / r[2]) << 16) / r[2]]);
            let k0 = len3(a);
            let k1 = len3(b).max(1);
            (mul16(k0, k0 - Q) << 16) / k1
        }
        Geo3::Box { h, rnd } => {
            let r = *rnd as i64;
            let w = [q[0].abs() - h[0] as i64 + r, q[1].abs() - h[1] as i64 + r, q[2].abs() - h[2] as i64 + r];
            len3([w[0].max(0), w[1].max(0), w[2].max(0)]) + w[0].max(w[1].max(w[2])).min(0) - r
        }
        Geo3::Cylinder { r, h, rnd } => {
            let rr = *rnd as i64;
            let dx = len2(q[0], q[2]) - *r as i64 + rr;
            let dy = q[1].abs() - *h as i64 + rr;
            len2(dx.max(0), dy.max(0)) + dx.max(dy).min(0) - rr
        }
        Geo3::Capsule { h, r } => {
            let y = q[1].clamp(-(*h as i64), *h as i64);
            len3([q[0], q[1] - y, q[2]]) - *r as i64
        }
        Geo3::Torus { big, small } => len2(len2(q[0], q[2]) - *big as i64, q[1]) - *small as i64,
        Geo3::Cone { h, r1, r2 } => {
            let (h, r1, r2) = (*h as i64, *r1 as i64, *r2 as i64);
            let (qx, qy) = (len2(q[0], q[2]), q[1]);
            let (k1x, k1y) = (r2, h);
            let (k2x, k2y) = (r2 - r1, 2 * h);
            let cax = qx - qx.min(if qy < 0 { r1 } else { r2 });
            let cay = qy.abs() - h;
            let dd = mul16(k2x, k2x) + mul16(k2y, k2y);
            let tt = (if dd == 0 { 0 } else { ((mul16(k1x - qx, k2x) + mul16(k1y - qy, k2y)) << 16) / dd }).clamp(0, Q);
            let (cbx, cby) = (qx - k1x + mul16(k2x, tt), qy - k1y + mul16(k2y, tt));
            let s = if cbx < 0 && cay < 0 { -1 } else { 1 };
            s * isqrt((cax * cax + cay * cay).min(cbx * cbx + cby * cby))
        }
        Geo3::Plane => q[1],
    }
}

fn smin3(a: i64, b: i64, k: i64) -> i64 {
    if k <= 0 {
        return a.min(b);
    }
    let h = (Q / 2 + (((b - a) << 15) / k)).clamp(0, Q);
    b + mul16(a - b, h) - mul16(mul16(k, h), Q - h)
}

const BIG: i64 = 1 << 60;

struct Hit {
    d: i64,
    mat: usize,
}

fn part_dist(pt: &Part3, p: V3) -> i64 {
    let mut q = rotate(&pt.rot, sub3(p, v3(pt.c)));
    if pt.mirror {
        q[0] = q[0].abs();
        q[2] = q[2].abs();
    }
    if pt.polar > 1 {
        let n = pt.polar as i64;
        let sec = TAU_Q / n;
        let ang = atan2q(q[2], q[0]);
        let k = (ang + sec / 2).div_euclid(sec);
        let (s, c) = sincos(k * sec);
        q = [mul16(q[0], c) + mul16(q[2], s), q[1], mul16(-q[0], s) + mul16(q[2], c)];
    }
    let d = geo_dist(&pt.geo, sub3(q, v3(pt.off)));
    if pt.shell > 0 {
        d.abs() - pt.shell as i64
    } else {
        d
    }
}

fn scene(p: &RenderParams3, pos: V3) -> Hit {
    let (mut all, mut all_mat) = (BIG, 0usize);
    let (mut acc, mut mat) = (BIG, 0usize);
    let mut cur = p.parts[0].group;
    for pt in p.parts.iter() {
        if pt.group != cur {
            if acc < all {
                all = acc;
                all_mat = mat;
            }
            acc = BIG;
            mat = 0;
            cur = pt.group;
        }
        let d = part_dist(pt, pos);
        match pt.op {
            Op3::Union => {
                if d < acc {
                    acc = d;
                    mat = pt.mat as usize;
                }
            }
            Op3::Smooth(k) => {
                if acc == BIG {
                    acc = d;
                    mat = pt.mat as usize;
                } else {
                    let k = k as i64;
                    let m = if d < acc { pt.mat as usize } else { mat };
                    acc = smin3(acc, d, k);
                    mat = m;
                }
            }
            Op3::Sub => {
                if -d > acc {
                    acc = -d;
                    mat = pt.mat as usize;
                }
            }
            Op3::Inter => {
                if d > acc {
                    acc = d;
                    mat = pt.mat as usize;
                }
            }
        }
    }
    if acc < all {
        all = acc;
        all_mat = mat;
    }
    Hit { d: all, mat: all_mat }
}

fn normal_at(p: &RenderParams3, pos: V3) -> V3 {
    let e = NORMAL_E;
    let f = |dx: i64, dy: i64, dz: i64| scene(p, [pos[0] + dx, pos[1] + dy, pos[2] + dz]).d;
    let (a, b, c, d) = (f(e, -e, -e), f(-e, -e, e), f(-e, e, -e), f(e, e, e));
    norm3([a - b - c + d, -a - b + c + d, -a + b - c + d])
}

// ------------------------------------------------------------------------------------------------ materials

struct Surf {
    alb: V3,
    rough: i64,
    nrm: Option<V3>,
}

fn bump_normal(pos: V3, n: V3, freq: i64, amp: i64) -> V3 {
    let e = 98i64;
    let sp = |p: V3| fbm_n(scale3(p, freq), 3);
    let b = sp(pos);
    let g = [(sp([pos[0] + e, pos[1], pos[2]]) - b) * Q / e, (sp([pos[0], pos[1] + e, pos[2]]) - b) * Q / e, (sp([pos[0], pos[1], pos[2] + e]) - b) * Q / e];
    let gn = dot16(g, n);
    let g = [g[0] - mul16(n[0], gn), g[1] - mul16(n[1], gn), g[2] - mul16(n[2], gn)];
    // n - amp * g / freq  (amp in Q8 -> Q16)
    let k = ((amp << 8) << 16) / freq.max(1);
    norm3([n[0] - mul16(g[0], k), n[1] - mul16(g[1], k), n[2] - mul16(g[2], k)])
}

fn surface_mat(m: &Mat3, pos: V3, ng: V3) -> Surf {
    let mut alb = lin8x3(m.alb);
    let alb2 = lin8x3(m.alb2);
    let mut rough = (m.rough as i64) << 8;
    let sc = m.tex_scale as i64;
    let mixc = |t: i64, a: V3, b: V3| [mix(a[0], b[0], t), mix(a[1], b[1], t), mix(a[2], b[2], t)];
    match m.tex {
        1 => {
            let cell = (mul16(pos[0], sc) >> 16) + (mul16(pos[1], sc) >> 16) + (mul16(pos[2], sc) >> 16);
            if cell & 1 != 0 {
                alb = alb2;
            }
        }
        2 => {
            let axis = (m.tex_p as usize).min(2);
            let cell = mul16(pos[axis], sc) >> 16;
            if cell & 1 != 0 {
                alb = alb2;
            }
        }
        3 => {
            let t = fbm_n(scale3(pos, sc), 5);
            alb = mixc(t.clamp(0, Q), alb, alb2);
        }
        4 => {
            let r = len2(pos[0], pos[2]);
            if (mul16(r, sc) & 0xFFFF) < Q / 2 {
                alb = alb2;
            }
        }
        5 => {
            let u = mul16(pos[2], sc) + 2 * fbm_n(scale3(pos, 3 * Q), 3);
            let fr = u & 0xFFFF;
            let t = smoothstep(0, Q / 2, fr) - smoothstep(Q / 2, Q, fr);
            alb = mixc(t.clamp(0, Q), alb, alb2);
        }
        6 => {
            let ang = atan2q(pos[2], pos[0]);
            let (s, _) = sincos(ang * m.tex_p as i64);
            let t = (Q + s) / 2;
            alb = mixc(t.clamp(0, Q), alb, alb2);
            rough = (rough + mul16(t, Q / 4)).min(Q);
        }
        7 => {
            let r = len2(pos[0], pos[2]);
            let ang = atan2q(pos[2], pos[0]);
            let (_, c) = sincos(ang * m.tex_p as i64);
            if c.abs() > 64553 && r >= m.tex_q as i64 && r <= m.tex_r as i64 && ng[1] > 39322 {
                alb = alb2;
            }
        }
        // 8: veined marble. A cloudy stone (alb toward alb2 by at most a quarter) crossed by veins of alb2: `tex_p` warps the veins with noise
        // (Q16), `tex_q` is their half width (Q16 of one stripe period).
        8 => {
            let w = fbm_n(scale3(pos, sc), 4);
            let u = mul16(pos[0] + mul16(pos[1], 26214) + mul16(pos[2], 39322), sc) + mul16(w, m.tex_p as i64);
            let fr = u & 0xFFFF;
            let d = fr.min(Q - fr);
            let vein = Q - smoothstep(0, (m.tex_q as i64).max(1), d);
            let base = mixc(mul16(w, 16384), alb, alb2);
            alb = mixc(vein, base, alb2);
        }
        // 9: cracked cells (hammered metal, ceramic, dried mud): each cell has its own mix of alb and alb2 (`tex_q` = spread, Q16), the cell
        // borders are dark cracks of half width `tex_p` (Q16) that are also rougher.
        9 => {
            let (f1, f2, id) = voronoi(scale3(pos, sc));
            let t = mul16(id, m.tex_q as i64);
            let base = mixc(t, alb, alb2);
            let crack = Q - smoothstep(0, (m.tex_p as i64).max(1), f2 - f1);
            alb = mixc(mul16(crack, 55705), base, [0, 0, 0]);
            rough = (rough + mul16(crack, Q / 4)).min(Q);
        }
        // 10: swirl. Two colours mixed by a domain-warped fractal noise (`tex_p` = warp strength, Q16), with fine contour lines of the field
        // (`tex_q` = number of lines per unit of the field).
        10 => {
            let p = scale3(pos, sc);
            let wx = fbm_n(p, 3);
            let wy = fbm_n([p[0] + 5 * Q + 21299, p[1] + Q + 19661, p[2] + 2 * Q + 45875], 3);
            let wz = fbm_n([p[0] + 3 * Q + 13107, p[1] + 4 * Q + 52428, p[2] + Q + 6554], 3);
            let wp = m.tex_p as i64;
            let q = [p[0] + mul16(wx - Q / 4, wp), p[1] + mul16(wy - Q / 4, wp), p[2] + mul16(wz - Q / 4, wp)];
            let t = fbm_n(q, 4);
            let s = smoothstep(Q / 5, 4 * Q / 5, t);
            let c = mixc(s, alb, alb2);
            let u = mul16(t, m.tex_q as i64);
            let fr = u & 0xFFFF;
            let line = Q - smoothstep(0, Q / 20, fr.min(Q - fr));
            alb = mixc(mul16(line, Q / 4), c, [Q, Q, Q]);
        }
        // 11: floor tiles on the xz plane: `tex_scale` tiles per unit, a tone per tile (`tex_q` = spread, Q16), dark rough grout of half
        // width `tex_p` (Q16 of a tile).
        11 => {
            let (u, v) = (mul16(pos[0], sc), mul16(pos[2], sc));
            let (fu, fv) = (u & 0xFFFF, v & 0xFFFF);
            let e = fu.min(Q - fu).min(fv).min(Q - fv);
            let g = (m.tex_p as i64).max(2);
            let grout = Q - smoothstep(g / 2, g, e);
            let tone = mul16(hash3(u >> 16, 5, v >> 16), m.tex_q as i64);
            let base = mixc(tone, alb, alb2);
            alb = mixc(grout, base, scale3(base, 8000));
            rough = mix(rough, 58982, grout);
        }
        _ => {}
    }
    let nrm = if m.bump > 0 { Some(bump_normal(pos, ng, m.bump_freq as i64, m.bump as i64)) } else { None };
    Surf { alb, rough, nrm }
}

// ------------------------------------------------------------------------------------------------ randomness and sampling

struct Rng(u32);

impl Rng {
    fn next(&mut self) -> i64 {
        self.0 = pcg(self.0);
        (self.0 >> 16) as i64
    }
}

fn disc(r: &mut Rng) -> (i64, i64) {
    for _ in 0..8 {
        let x = 2 * r.next() - Q;
        let y = 2 * r.next() - Q;
        if x * x + y * y <= Q * Q {
            return (x, y);
        }
    }
    (0, 0)
}

fn basis_t(n: V3) -> V3 {
    if n[0].abs() > Q / 2 {
        norm3(cross16(n, [0, Q, 0]))
    } else {
        norm3(cross16(n, [Q, 0, 0]))
    }
}

fn cos_hemi(n: V3, r: &mut Rng) -> V3 {
    let (x, y) = disc(r);
    let z = isqrt((Q * Q - x * x - y * y).max(0));
    let t = basis_t(n);
    let b = cross16(n, t);
    norm3([mul16(t[0], x) + mul16(b[0], y) + mul16(n[0], z), mul16(t[1], x) + mul16(b[1], y) + mul16(n[1], z), mul16(t[2], x) + mul16(b[2], y) + mul16(n[2], z)])
}

/// GGX half vector; `a` is the alpha parameter of the original shader (squared once more inside).
fn ggx_h(n: V3, a: i64, r: &mut Rng) -> V3 {
    let (x, y) = disc(r);
    let v = r.next();
    let a2 = mul16(a, a);
    let denom = Q + mul16(a2 - Q, v);
    let ratio = (((Q - v) << 16) / denom.max(1)).clamp(0, Q);
    let ct = isqrt(ratio << 16);
    let st = isqrt((Q * Q - ct * ct).max(0));
    let l = len2(x, y);
    let (cx, cy) = if l == 0 { (Q, 0) } else { ((x << 16) / l, (y << 16) / l) };
    let t = basis_t(n);
    let b = cross16(n, t);
    let (sx, sy) = (mul16(cx, st), mul16(cy, st));
    norm3([mul16(t[0], sx) + mul16(b[0], sy) + mul16(n[0], ct), mul16(t[1], sx) + mul16(b[1], sy) + mul16(n[1], ct), mul16(t[2], sx) + mul16(b[2], sy) + mul16(n[2], ct)])
}

fn reflect(i: V3, n: V3) -> V3 {
    let d = dot16(i, n);
    [i[0] - 2 * mul16(d, n[0]), i[1] - 2 * mul16(d, n[1]), i[2] - 2 * mul16(d, n[2])]
}

fn refract(i: V3, n: V3, eta: i64) -> Option<V3> {
    let d = dot16(n, i);
    let k = Q - mul16(mul16(eta, eta), Q - mul16(d, d));
    if k < 0 {
        return None;
    }
    let sk = isqrt(k << 16);
    let s = mul16(eta, d) + sk;
    Some(norm3([mul16(eta, i[0]) - mul16(s, n[0]), mul16(eta, i[1]) - mul16(s, n[1]), mul16(eta, i[2]) - mul16(s, n[2])]))
}

fn pow5(x: i64) -> i64 {
    let x2 = mul16(x, x);
    mul16(mul16(x2, x2), x)
}

// ------------------------------------------------------------------------------------------------ environment

fn softbox(d: V3, b: &SoftBox) -> i64 {
    let dir = norm3(v3(b.dir));
    let c = dot16(d, dir);
    if c <= 0 {
        return 0;
    }
    let up0 = if dir[1].abs() > 62259 { [Q, 0, 0] } else { [0, Q, 0] };
    let r = norm3(cross16(dir, up0));
    let u = cross16(r, dir);
    let qx = ((dot16(d, r) << 16) / c).abs();
    let qy = ((dot16(d, u) << 16) / c).abs();
    let (hx, hy, s) = (b.hs[0] as i64, b.hs[1] as i64, b.soft as i64);
    let wx = Q - smoothstep(mul16(hx, Q - s), hx, qx);
    let wy = Q - smoothstep(mul16(hy, Q - s), hy, qy);
    mul16(wx, wy)
}

fn env(p: &RenderParams3, d: V3) -> V3 {
    let e = &p.env;
    let (top, hor, gnd) = (q10x3(e.top), q10x3(e.hor), q10x3(e.gnd));
    let mut out = [0i64; 3];
    for i in 0..3 {
        out[i] = if d[1] >= 0 { mix(hor[i], top[i], mul16(d[1].min(Q), d[1].min(Q))) } else { mix(hor[i], gnd[i], (-d[1]).min(Q)) };
    }
    for b in e.boxes.iter() {
        let w = softbox(d, b);
        if w > 0 {
            let c = q10x3(b.col);
            for i in 0..3 {
                out[i] += mul16(c[i], w);
            }
        }
    }
    out
}

fn backdrop(p: &RenderParams3, d: V3) -> V3 {
    let (t, b) = (q10x3(p.env.bd_top), q10x3(p.env.bd_bot));
    let k = ((d[1] + Q) >> 1).clamp(0, Q);
    [mix(b[0], t[0], k), mix(b[1], t[1], k), mix(b[2], t[2], k)]
}

// ------------------------------------------------------------------------------------------------ path tracing

fn march(p: &RenderParams3, ro: V3, rd: V3) -> Option<(i64, usize)> {
    let mut t = T_START;
    for _ in 0..MARCH_STEPS3 {
        let pos = add3(ro, scale3(rd, t));
        let h = scene(p, pos);
        let d = h.d.abs();
        if d < ((EPS_Q * t.max(19661)) >> 16).max(1) {
            return Some((t, h.mat));
        }
        t += (d * 9) / 10;
        if t > MAXT {
            break;
        }
    }
    None
}

fn trace(p: &RenderParams3, rng: &mut Rng, ro0: V3, rd0: V3) -> V3 {
    let (mut ro, mut rd) = (ro0, rd0);
    let (mut col, mut thr) = ([0i64; 3], [Q; 3]);
    let mut t0 = -1i64;
    let mut inside = false;
    let mut sig = [0i64; 3];
    for b in 0..=p.bounces as usize {
        let Some((t, mi)) = march(p, ro, rd) else {
            let e = if b == 0 { backdrop(p, rd) } else { env(p, rd) };
            for i in 0..3 {
                col[i] += mul16(thr[i], e[i]);
            }
            break;
        };
        if b == 0 {
            t0 = t;
        }
        if inside {
            // thr *= exp(-sig * t): exp of a Q16 argument via a squaring ladder (exp(-x) = (exp(-x/2^k))^(2^k))
            for i in 0..3 {
                thr[i] = mul16(thr[i], exp_neg(mul16(sig[i], t)));
            }
        }
        let pos = add3(ro, scale3(rd, t));
        let ng = normal_at(p, pos);
        let m = &p.mats[mi];
        let sf = surface_mat(m, pos, ng);
        let n = sf.nrm.unwrap_or(ng);
        let emit = q10x3([m.emit[0] as u32, m.emit[1] as u32, m.emit[2] as u32]);
        for i in 0..3 {
            col[i] += mul16(thr[i], emit[i]);
        }
        if (emit[0] + emit[1] + emit[2]) > 0 && (m.alb[0] as u32 + m.alb[1] as u32 + m.alb[2] as u32) == 0 {
            break;
        }
        if m.glass {
            let ent = dot16(rd, ng) < 0;
            let gn = if ent { ng } else { neg3(ng) };
            let mut nn = if ent { n } else { neg3(n) };
            if sf.rough > 0 {
                let r2 = mul16(sf.rough, sf.rough);
                nn = ggx_h(nn, r2, rng);
            }
            let ior = (m.ior as i64) << 6;
            let eta = if ent { (Q << 16) / ior } else { ior };
            let ci = (-dot16(rd, nn)).clamp(0, Q);
            let f0 = {
                let a = ((ior - Q) << 16) / (ior + Q);
                mul16(a, a)
            };
            let fr = f0 + mul16(Q - f0, pow5(Q - ci));
            let tr = refract(rd, nn, eta);
            let refl_choice = match tr {
                None => true,
                Some(_) => rng.next() < fr,
            };
            if refl_choice {
                let mut r = reflect(rd, nn);
                if dot16(r, gn) < 0 {
                    r = reflect(r, gn);
                }
                rd = r;
                ro = add3(pos, scale3(gn, 164));
            } else {
                rd = tr.unwrap();
                ro = sub3(pos, scale3(gn, 164));
                inside = ent;
                sig = q10x3([m.absorb[0] as u32, m.absorb[1] as u32, m.absorb[2] as u32]);
                if ent {
                    let a = lin8x3(m.alb);
                    for i in 0..3 {
                        thr[i] = mul16(thr[i], a[i]);
                    }
                }
            }
            continue;
        }
        let ci = dot16(neg3(rd), n).clamp(0, Q);
        if m.coat > 0 {
            let fc = 2621 + mul16(62915, pow5(Q - ci));
            if rng.next() < mul16(fc, (m.coat as i64) << 8) {
                let r2 = reflect(rd, ggx_h(n, 983, rng));
                rd = if dot16(r2, ng) > 0 { r2 } else { reflect(rd, ng) };
                ro = add3(pos, scale3(ng, 131));
                continue;
            }
        }
        let spec = (m.spec as i64) << 8;
        let f0: V3 = if m.metal { sf.alb } else { [mul16(5243, spec); 3] };
        let pw = pow5(Q - ci);
        let fre = [f0[0] + mul16(Q - f0[0], pw), f0[1] + mul16(Q - f0[1], pw), f0[2] + mul16(Q - f0[2], pw)];
        let mut ps = (mul16((fre[0] + fre[1] + fre[2]) / 3, Q - mul16(sf.rough, Q / 2))).clamp(2621, 62259);
        if m.metal {
            ps = Q;
        }
        if rng.next() < ps {
            let a = mul16(sf.rough, sf.rough);
            let mut rf = reflect(rd, ggx_h(n, a.max(26), rng));
            if dot16(rf, ng) <= 0 {
                rf = reflect(rd, ng);
            }
            rd = rf;
            for i in 0..3 {
                thr[i] = if m.metal { mul16(thr[i], sf.alb[i]) } else { ((thr[i] * fre[i]) >> 16) * Q / ps };
            }
        } else {
            if m.trans > 0 && rng.next() < ((m.trans as i64) << 8) {
                rd = cos_hemi(neg3(n), rng);
                ro = sub3(pos, scale3(ng, BIAS_T));
                for i in 0..3 {
                    thr[i] = mul16(thr[i], sf.alb[i]) * Q / (Q - ps);
                }
                continue;
            }
            rd = cos_hemi(n, rng);
            if dot16(rd, ng) < 0 {
                rd = reflect(rd, ng);
            }
            for i in 0..3 {
                thr[i] = mul16(mul16(thr[i], sf.alb[i]), Q - fre[i]) * Q / (Q - ps);
            }
        }
        ro = add3(pos, scale3(ng, BIAS_G));
        if b > 2 {
            let mx = thr[0].max(thr[1]).max(thr[2]);
            let q = (Q - mx).max(3277);
            if rng.next() < q {
                break;
            }
            for i in 0..3 {
                thr[i] = thr[i] * Q / (Q - q);
            }
        }
    }
    let mut out = [col[0].min(CLAMP_RAD), col[1].min(CLAMP_RAD), col[2].min(CLAMP_RAD)];
    if t0 > 0 {
        let k = smoothstep(p.fog0 as i64, p.fog1 as i64, t0);
        if k > 0 {
            let bd = backdrop(p, rd0);
            for i in 0..3 {
                out[i] = mix(out[i], bd[i], k);
            }
        }
    }
    out
}

/// exp(-x) for x >= 0 in Q16: halve the argument until it is small, use 1 - x + x^2/2 - x^3/6, then square back.
fn exp_neg(x: i64) -> i64 {
    if x <= 0 {
        return Q;
    }
    let mut k = 0;
    let mut y = x;
    while y > 4096 && k < 24 {
        y >>= 1;
        k += 1;
    }
    let y2 = mul16(y, y);
    let y3 = mul16(y2, y);
    let mut e = Q - y + (y2 >> 1) - y3 / 6;
    e = e.clamp(0, Q);
    for _ in 0..k {
        e = mul16(e, e);
    }
    e
}

// ------------------------------------------------------------------------------------------------ validation, cost, chunks

pub fn render3_tiles(p: &RenderParams3) -> u32 {
    (p.width as u32 / TILE as u32) * (p.height as u32 / TILE as u32)
}

pub fn render3_chunks(p: &RenderParams3) -> u32 {
    render3_tiles(p) * p.passes as u32
}

/// Evaluation units of one chunk: pixels x samples x rays x march steps x parts (the bound a judge must be able to recompute).
pub fn render3_cost(p: &RenderParams3) -> u64 {
    (TILE * TILE) as u64 * p.spp as u64 * (p.bounces as u64 + 1) * (MARCH_STEPS3 + 8) * p.parts.len() as u64
}

pub fn render3_validate(p: &RenderParams3) -> Result<(), String> {
    let (w, h) = (p.width as usize, p.height as usize);
    if w % TILE != 0 || h % TILE != 0 || w < TILE || h < TILE || p.width > MAX_SIDE3 || p.height > MAX_SIDE3 {
        return Err("bad image size".into());
    }
    if p.spp == 0 || p.spp > MAX_SPP_PASS || p.passes == 0 || p.bounces > MAX_BOUNCES3 {
        return Err("bad samples, passes or bounces".into());
    }
    if p.parts.is_empty() || p.parts.len() > MAX_PARTS || p.mats.is_empty() || p.mats.len() > MAX_MATS || p.env.boxes.len() > MAX_BOXES {
        return Err("bad number of parts, materials or soft boxes".into());
    }
    let lim = 1i64 << 23;
    let ok = |v: i32| (v as i64).abs() <= lim;
    let okv = |v: &[i32]| v.iter().all(|x| ok(*x));
    if p.parts.windows(2).any(|w| w[1].group < w[0].group) {
        return Err("part groups must not decrease".into());
    }
    for pt in p.parts.iter() {
        if pt.mat as usize >= p.mats.len() || !okv(&pt.c) || pt.rot.iter().any(|v| (*v as i64).abs() > 2 * Q) || pt.shell < 0 || !ok(pt.shell) {
            return Err("bad part".into());
        }
        let good = match &pt.geo {
            Geo3::Sphere { r } => *r > 0 && ok(*r),
            // the ellipsoid distance squares (q / r^2) in i64: with a radius below 1/16 the square overflows for the far points a ray visits
            // (the CPU wraps, the GPU compiler assumes it cannot overflow and a thread can loop forever), so such shapes are refused
            Geo3::Ellipsoid { r } => r.iter().all(|x| *x >= MIN_ELLIPSOID_R && ok(*x)),
            Geo3::Box { h, rnd } => h.iter().all(|x| *x > 0 && ok(*x)) && *rnd >= 0 && ok(*rnd),
            Geo3::Cylinder { r, h, rnd } => *r > 0 && *h > 0 && *rnd >= 0 && ok(*r) && ok(*h),
            Geo3::Capsule { h, r } => *h > 0 && *r > 0 && ok(*h) && ok(*r),
            Geo3::Torus { big, small } => *small > 0 && *big > *small && ok(*big),
            Geo3::Cone { h, r1, r2 } => *h > 0 && *r1 >= 0 && *r2 >= 0 && (*r1 + *r2) > 0 && ok(*h) && ok(*r1) && ok(*r2),
            Geo3::Plane => true,
        };
        if !good {
            return Err("bad shape".into());
        }
        if let Op3::Smooth(k) = pt.op {
            if k <= 0 || k as i64 > 4 * Q {
                return Err("bad blend radius".into());
            }
        }
    }
    for m in p.mats.iter() {
        if m.ior < 1024 || m.ior > 4096 || m.tex > 11 || m.tex_scale < 0 || (m.bump > 0 && m.bump_freq <= 0) {
            return Err("bad material".into());
        }
    }
    let cam_ok = okv(&p.cam.pos) && okv(&p.cam.look) && p.cam.focal >= (Q as i32) / 4 && p.cam.focal as i64 <= 16 * Q && p.cam.aperture >= 0 && p.cam.focus > 0;
    if !cam_ok || p.fog0 < 0 || p.fog1 < p.fog0 {
        return Err("bad camera or fog".into());
    }
    if p.env.boxes.iter().any(|b| b.hs[0] <= 0 || b.hs[1] <= 0 || b.soft < 0 || b.soft > Q as i32) {
        return Err("bad soft box".into());
    }
    if render3_chunks(p) > MAX_CHUNKS3 {
        return Err("too many chunks".into());
    }
    if render3_cost(p) > MAX_CHUNK_COST3 {
        return Err("chunk cost above the protocol bound".into());
    }
    Ok(())
}

fn seed_words(seed: &[u8; 32]) -> (u32, u32) {
    (u32::from_le_bytes([seed[0], seed[1], seed[2], seed[3]]), u32::from_le_bytes([seed[4], seed[5], seed[6], seed[7]]))
}

/// One pass of one tile: 16 x 16 pixels, three LE u16 each (mean radiance of the `spp` samples, Q11, capped at 32.0).
pub fn render3_chunk(p: &RenderParams3, chunk: u32) -> Vec<u8> {
    let ntiles = render3_tiles(p);
    let (tile, pass) = (chunk % ntiles, chunk / ntiles);
    let tiles_x = p.width as u32 / TILE as u32;
    let (tx, ty) = ((tile % tiles_x) as usize, (tile / tiles_x) as usize);
    let (w0, w1) = seed_words(&p.seed);
    let (w, h) = (p.width as i64, p.height as i64);
    let cam = &p.cam;
    let pos = v3(cam.pos);
    let fw = norm3(sub3(v3(cam.look), pos));
    let rt = norm3(cross16([0, Q, 0], fw));
    let up = cross16(fw, rt);
    let spp = p.spp as i64;
    let mut out = Vec::with_capacity(TILE * TILE * 6);
    for y in 0..TILE {
        for x in 0..TILE {
            let (px, py) = ((tx * TILE + x) as i64, (ty * TILE + y) as i64);
            let mut acc = [0i64; 3];
            for s in 0..spp {
                let mut rng = Rng(pcg((px as u32).wrapping_mul(1973).wrapping_add((py as u32).wrapping_mul(9277)).wrapping_add(chunk.wrapping_mul(26699)).wrapping_add((s as u32).wrapping_mul(612451)).wrapping_add(w0)));
                rng.0 = pcg(rng.0 ^ w1);
                let (jx, jy) = (rng.next(), rng.next());
                let (sx, sy) = (px * Q + jx, py * Q + jy);
                let u = (2 * sx - w * Q) / h;
                let v = -((2 * sy - h * Q) / h);
                let _ = pass;
                let rd = norm3([mul16(rt[0], u) + mul16(up[0], v) + mul16(fw[0], cam.focal as i64), mul16(rt[1], u) + mul16(up[1], v) + mul16(fw[1], cam.focal as i64), mul16(rt[2], u) + mul16(up[2], v) + mul16(fw[2], cam.focal as i64)]);
                let (mut ro, mut rdd) = (pos, rd);
                if cam.aperture > 0 {
                    let fd = ((cam.focus as i64) << 16) / dot16(rd, fw).max(1);
                    let fp = add3(pos, scale3(rd, fd));
                    let (lx, ly) = disc(&mut rng);
                    let lr = mul16(cam.aperture as i64, lx);
                    let lq = mul16(cam.aperture as i64, ly);
                    ro = add3(pos, add3(scale3(rt, lr), scale3(up, lq)));
                    rdd = norm3(sub3(fp, ro));
                }
                let c = trace(p, &mut rng, ro, rdd);
                for k in 0..3 {
                    acc[k] += c[k];
                }
            }
            for k in 0..3 {
                // mean radiance in Q16 -> Q11
                let mean = acc[k] / spp;
                let q11 = (mean >> 5).clamp(0, 65535);
                out.extend_from_slice(&(q11 as u16).to_le_bytes());
            }
        }
    }
    out
}

/// Off-chain assembly of the final image from the chunk outputs: per-tile average of the passes, exposure, ACES, vignette, gamma.
pub fn assemble(p: &RenderParams3, outs: &[Vec<u8>], exposure: f64) -> Vec<u8> {
    let (w, h) = (p.width as usize, p.height as usize);
    let ntiles = render3_tiles(p) as usize;
    let tiles_x = w / TILE;
    let mut img = vec![0u8; w * h * 3];
    for tile in 0..ntiles {
        let (tx, ty) = (tile % tiles_x, tile / tiles_x);
        for y in 0..TILE {
            for x in 0..TILE {
                for k in 0..3 {
                    let mut sum = 0f64;
                    for pass in 0..p.passes as usize {
                        let o = &outs[pass * ntiles + tile];
                        let i = (y * TILE + x) * 6 + k * 2;
                        sum += u16::from_le_bytes([o[i], o[i + 1]]) as f64;
                    }
                    let mut c = sum / p.passes as f64 / 2048.0 * exposure;
                    let (qx, qy) = (((tx * TILE + x) as f64 + 0.5) / w as f64 - 0.5, ((ty * TILE + y) as f64 + 0.5) / h as f64 - 0.5);
                    c *= 1.0 - 0.22 * (qx * qx + qy * qy);
                    let a = (c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14);
                    let g = a.clamp(0.0, 1.0).powf(1.0 / 2.2);
                    img[((ty * TILE + y) * w + tx * TILE + x) * 3 + k] = (g * 255.0 + 0.5) as u8;
                }
            }
        }
    }
    img
}


// ------------------------------------------------------------------------------------------------ scene building (client side)
// Floating point is fine here: it only turns a description into the integer parameters of the job.

fn f(x: f64) -> i32 {
    (x * 65536.0).round() as i32
}

pub fn rot_identity() -> [f64; 9] {
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
}

fn matmul3(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut r = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            r[i * 3 + j] = (0..3).map(|k| a[i * 3 + k] * b[k * 3 + j]).sum();
        }
    }
    r
}

/// World-to-local rotation of a part turned by `ax`, `ay`, `az` radians (x, then y, then z, in the world frame).
pub fn rot_euler(ax: f64, ay: f64, az: f64) -> [f64; 9] {
    let (sx, cx, sy, cy, sz, cz) = (ax.sin(), ax.cos(), ay.sin(), ay.cos(), az.sin(), az.cos());
    let rx = [1.0, 0.0, 0.0, 0.0, cx, -sx, 0.0, sx, cx];
    let ry = [cy, 0.0, sy, 0.0, 1.0, 0.0, -sy, 0.0, cy];
    let rz = [cz, -sz, 0.0, sz, cz, 0.0, 0.0, 0.0, 1.0];
    let world_from_local = matmul3(&rz, &matmul3(&ry, &rx));
    let mut t = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            t[i * 3 + j] = world_from_local[j * 3 + i];
        }
    }
    t
}

fn rot_q(r: [f64; 9]) -> [i32; 9] {
    let mut o = [0i32; 9];
    for i in 0..9 {
        o[i] = f(r[i]);
    }
    o
}

pub fn part(geo: Geo3, c: [f64; 3], rot: [f64; 9], mat: u8) -> Part3 {
    Part3 { geo, c: [f(c[0]), f(c[1]), f(c[2])], rot: rot_q(rot), op: Op3::Union, mat, polar: 0, mirror: false, off: [0; 3], group: 0, shell: 0 }
}

/// Assigns the groups: a part that uses a combination operator (smooth union, subtraction, intersection) joins the group of the part
/// before it; every plain union starts a new group.
pub fn autogroup(parts: &mut Vec<Part3>) {
    let mut g = 0u8;
    for (i, p) in parts.iter_mut().enumerate() {
        if i > 0 && matches!(p.op, Op3::Union) {
            g += 1;
        }
        p.group = g;
    }
}

pub fn material(alb: [u8; 3], rough: f64) -> Mat3 {
    Mat3 { alb, alb2: alb, rough: (rough * 255.0).round() as u8, metal: false, spec: 128, emit: [0; 3], trans: 0, glass: false, ior: 1536, absorb: [0; 3], coat: 0, tex: 0, tex_scale: 0, tex_p: 0, tex_q: 0, tex_r: 0, bump: 0, bump_freq: 0 }
}

pub fn metal(alb: [u8; 3], rough: f64) -> Mat3 {
    Mat3 { metal: true, ..material(alb, rough) }
}

pub fn glass(absorb: [f64; 3], ior: f64) -> Mat3 {
    let q = |v: f64| (v * 1024.0).round() as u16;
    Mat3 { glass: true, ior: q(ior), absorb: [q(absorb[0]), q(absorb[1]), q(absorb[2])], ..material([255, 255, 255], 0.0) }
}

pub fn emissive(col: [f64; 3]) -> Mat3 {
    let q = |v: f64| (v * 1024.0).round() as u16;
    Mat3 { emit: [q(col[0]), q(col[1]), q(col[2])], ..material([0, 0, 0], 1.0) }
}

fn rgb10(c: [f64; 3]) -> [u32; 3] {
    [(c[0] * 1024.0).round() as u32, (c[1] * 1024.0).round() as u32, (c[2] * 1024.0).round() as u32]
}

fn studio_env() -> Env3 {
    Env3 {
        top: rgb10([0.05, 0.055, 0.07]),
        hor: rgb10([0.03, 0.03, 0.035]),
        gnd: rgb10([0.02, 0.02, 0.02]),
        boxes: vec![
            SoftBox { dir: [f(-0.6), f(0.7), f(0.5)], hs: [f(0.55), f(0.4)], soft: f(0.5), col: rgb10([9.0, 8.6, 8.0]) },
            SoftBox { dir: [f(0.8), f(0.35), f(-0.2)], hs: [f(0.18), f(0.7)], soft: f(0.4), col: rgb10([2.2, 2.6, 3.4]) },
            SoftBox { dir: [f(0.1), f(0.9), f(-0.5)], hs: [f(0.5), f(0.2)], soft: f(0.6), col: rgb10([2.0, 2.0, 2.2]) },
        ],
        bd_top: rgb10([0.02, 0.022, 0.03]),
        bd_bot: rgb10([0.05, 0.05, 0.06]),
    }
}

fn camera(pos: [f64; 3], look: [f64; 3], fov_deg: f64, aperture: f64, focus: f64) -> Cam3 {
    Cam3 { pos: [f(pos[0]), f(pos[1]), f(pos[2])], look: [f(look[0]), f(look[1]), f(look[2])], focal: f(1.0 / (fov_deg.to_radians() / 2.0).tan()), aperture: f(aperture), focus: f(focus) }
}

/// "Meridian": a chronograph on slate (steel case, ceramic bezel with ticks, teal sunburst dial with twelve applied indices, three
/// sub-dials, hands, crown, pushers, lugs, leather strap, sapphire crystal), thin-lens camera focused on the dial.
pub fn scene3_watch(width: u16, height: u16, spp: u8, passes: u16, seed: [u8; 32]) -> RenderParams3 {
    let mut mats: Vec<Mat3> = vec![];
    let mut add = |m: Mat3| -> u8 {
        mats.push(m);
        (mats.len() - 1) as u8
    };
    let slate = add(Mat3 { bump: 51, bump_freq: f(9.0), tex: 3, tex_scale: f(3.0), alb2: [24, 26, 30], spec: 77, ..material([14, 15, 18], 0.8) });
    let steel = add(metal([232, 235, 240], 0.14));
    let bezel = add(Mat3 { coat: 255, tex: 7, tex_p: 30, tex_q: f(0.58), tex_r: f(0.66), alb2: [215, 222, 228], ..material([6, 8, 12], 0.2) });
    let dial = add(Mat3 { tex: 6, tex_p: 240, alb2: [6, 66, 62], spec: 230, ..material([2, 34, 38], 0.32) });
    let index = add(metal([238, 240, 245], 0.06));
    let sub = add(Mat3 { tex: 4, tex_scale: f(900.0), alb2: [8, 40, 42], ..material([4, 28, 30], 0.4) });
    let hand = add(metal([242, 244, 248], 0.08));
    let sec = add(metal([255, 120, 190], 0.15));
    let crystal = add(glass([0.0, 0.0, 0.0], 1.76));
    let strap = add(Mat3 { bump: 90, bump_freq: f(95.0), tex: 3, tex_scale: f(40.0), alb2: [150, 82, 38], coat: 50, ..material([112, 58, 26], 0.5) });
    let id = rot_identity();
    let mut parts: Vec<Part3> = vec![];
    parts.push(part(Geo3::Plane, [0.0, 0.0, 0.0], id, slate));
    // case with the opening for the dial
    parts.push(part(Geo3::Cylinder { r: f(0.62), h: f(0.1), rnd: f(0.045) }, [0.0, 0.1, 0.0], id, steel));
    let mut open = part(Geo3::Cylinder { r: f(0.54), h: f(0.2), rnd: 0 }, [0.0, 0.14, 0.0], id, steel);
    open.op = Op3::Sub;
    parts.push(open);
    // ceramic bezel
    parts.push(part(Geo3::Cylinder { r: f(0.64), h: f(0.03), rnd: f(0.022) }, [0.0, 0.205, 0.0], id, bezel));
    let mut bopen = part(Geo3::Cylinder { r: f(0.545), h: f(0.2), rnd: 0 }, [0.0, 0.205, 0.0], id, bezel);
    bopen.op = Op3::Sub;
    parts.push(bopen);
    // dial and sub-dials
    parts.push(part(Geo3::Cylinder { r: f(0.55), h: f(0.06), rnd: 0 }, [0.0, 0.12, 0.0], id, dial));
    for (x, z) in [(-0.22, 0.0), (0.22, 0.0), (0.0, 0.22)] {
        parts.push(part(Geo3::Cylinder { r: f(0.11), h: f(0.002), rnd: 0 }, [x, 0.181, z], id, sub));
    }
    // twelve applied indices (polar repetition around the dial centre), the 3 and 9 o'clock ones are longer
    let mut idx = part(Geo3::Box { h: [f(0.055), f(0.012), f(0.014)], rnd: f(0.006) }, [0.0, 0.0, 0.0], id, index);
    idx.polar = 12;
    idx.off = [f(0.43), f(0.19), 0];
    parts.push(idx);
    // two longer indices at 3 and 9 o'clock
    for z in [0.43, -0.43] {
        parts.push(part(Geo3::Box { h: [f(0.055), f(0.012), f(0.034)], rnd: f(0.006) }, [0.0, 0.19, z], rot_euler(0.0, -std::f64::consts::FRAC_PI_2, 0.0), index));
    }
    // hands: hour, minute and seconds (rotated boxes about the dial centre), centre cap
    let hand_part = |off: f64, half_len: f64, half_w: f64, y: f64, a: f64, thick: f64, m: u8| {
        let mut p = part(Geo3::Box { h: [f(half_len), f(thick), f(half_w)], rnd: f(thick) }, [0.0, y, 0.0], rot_euler(0.0, -a, 0.0), m);
        p.off = [f(off), 0, 0];
        p
    };
    parts.push(hand_part(0.14, 0.15, 0.022, 0.205, -0.95, 0.006, hand));
    parts.push(hand_part(0.21, 0.23, 0.016, 0.215, 1.85, 0.005, hand));
    parts.push(hand_part(0.18, 0.3, 0.005, 0.224, -2.6, 0.003, sec));
    parts.push(part(Geo3::Cylinder { r: f(0.022), h: f(0.006), rnd: 0 }, [0.0, 0.224, 0.0], id, sec));
    // sapphire crystal (glass disc over the dial)
    parts.push(part(Geo3::Cylinder { r: f(0.548), h: f(0.012), rnd: f(0.008) }, [0.0, 0.247, 0.0], id, crystal));
    // crown (turned onto the x axis) and pushers
    parts.push(part(Geo3::Cylinder { r: f(0.065), h: f(0.05), rnd: f(0.01) }, [0.67, 0.1, 0.0], rot_euler(0.0, 0.0, std::f64::consts::FRAC_PI_2), steel));
    for (z, a) in [(0.32, 0.5), (-0.32, -0.5)] {
        parts.push(part(Geo3::Box { h: [f(0.07), f(0.03), f(0.03)], rnd: f(0.015) }, [0.58, 0.1, z], rot_euler(0.0, -a, 0.0), steel));
    }
    // lugs (mirrored on x and z) and strap
    let mut lug = part(Geo3::Box { h: [f(0.05), f(0.06), f(0.14)], rnd: f(0.03) }, [0.0, 0.0, 0.0], id, steel);
    lug.mirror = true;
    lug.off = [f(0.3), f(0.09), f(0.6)];
    parts.push(lug);
    let mut band = part(Geo3::Box { h: [f(0.25), f(0.032), f(0.95)], rnd: f(0.025) }, [0.0, 0.05, 0.0], id, strap);
    band.mirror = true;
    band.off = [0, 0, f(1.6)];
    parts.push(band);
    autogroup(&mut parts);
    RenderParams3 {
        seed,
        width,
        height,
        spp,
        passes,
        bounces: 10,
        fog0: f(6.0),
        fog1: f(14.0),
        cam: camera([1.15, 2.05, 1.85], [0.02, 0.1, 0.02], 30.0, 0.025, 2.95),
        env: studio_env(),
        mats,
        parts,
    }
}


/// A small scene that uses every shape, every combination operator, polar repetition, mirroring, a glass part, an emitter, every
/// texture kind, bump mapping, coat, translucency and the thin-lens camera: the test scene of the frozen vectors.
pub fn scene3_mini(width: u16, height: u16, spp: u8, passes: u16, bounces: u8, seed: [u8; 32]) -> RenderParams3 {
    let mut mats: Vec<Mat3> = vec![];
    let mut add = |m: Mat3| -> u8 {
        mats.push(m);
        (mats.len() - 1) as u8
    };
    let floor = add(Mat3 { tex: 1, tex_scale: f(1.0), alb2: [200, 200, 205], ..material([30, 30, 40], 0.6) });
    let chrome = add(Mat3 { bump: 40, bump_freq: f(6.0), ..metal([220, 225, 235], 0.25) });
    let gl = add(glass([0.2, 0.05, 0.02], 1.5));
    let lamp = add(emissive([6.0, 5.0, 3.5]));
    let stripes = add(Mat3 { tex: 2, tex_p: 1, tex_scale: f(5.0), alb2: [240, 240, 230], ..material([180, 40, 40], 0.5) });
    let candy = add(Mat3 { tex: 3, tex_scale: f(3.0), alb2: [250, 200, 80], coat: 200, trans: 90, ..material([220, 60, 120], 0.35) });
    let wood = add(Mat3 { tex: 5, tex_scale: f(8.0), alb2: [150, 90, 40], ..material([90, 50, 25], 0.6) });
    let rings = add(Mat3 { tex: 4, tex_scale: f(10.0), alb2: [230, 230, 230], ..material([40, 90, 160], 0.4) });
    let burst = add(Mat3 { tex: 6, tex_p: 90, alb2: [10, 120, 110], bump: 30, bump_freq: f(20.0), ..material([5, 50, 55], 0.3) });
    let ticks = add(Mat3 { tex: 7, tex_p: 24, tex_q: f(0.2), tex_r: f(0.5), alb2: [250, 250, 250], ..material([10, 10, 14], 0.3) });
    let id = rot_identity();
    let mut parts: Vec<Part3> = vec![];
    parts.push(part(Geo3::Plane, [0.0, -0.5, 0.0], id, floor));
    parts.push(part(Geo3::Sphere { r: f(0.45) }, [-0.9, 0.0, 0.2], id, chrome));
    let mut blob = part(Geo3::Ellipsoid { r: [f(0.4), f(0.3), f(0.25)] }, [-0.55, 0.15, 0.2], rot_euler(0.0, 0.4, 0.2), stripes);
    blob.op = Op3::Smooth(f(0.25));
    parts.push(blob);
    parts.push(part(Geo3::Box { h: [f(0.3), f(0.3), f(0.3)], rnd: f(0.05) }, [0.2, -0.2, 0.4], rot_euler(0.3, 0.5, 0.0), gl));
    let mut cup = part(Geo3::Sphere { r: f(0.24) }, [0.2, -0.12, 0.4], id, gl);
    cup.op = Op3::Sub;
    parts.push(cup);
    let mut ring = part(Geo3::Cylinder { r: f(0.07), h: f(0.12), rnd: f(0.02) }, [0.0, -0.1, 0.0], id, rings);
    ring.polar = 6;
    ring.off = [f(0.6), 0, f(-0.6)];
    parts.push(ring);
    parts.push(part(Geo3::Capsule { h: f(0.3), r: f(0.07) }, [1.0, 0.0, 0.0], rot_euler(0.0, 0.0, 0.6), candy));
    parts.push(part(Geo3::Torus { big: f(0.35), small: f(0.08) }, [0.8, -0.35, -0.3], id, burst));
    parts.push(part(Geo3::Cone { h: f(0.3), r1: f(0.28), r2: f(0.04) }, [-0.2, -0.2, -0.5], id, wood));
    parts.push(part(Geo3::Sphere { r: f(0.14) }, [0.0, 0.7, 0.3], id, lamp));
    let mut keel = part(Geo3::Cylinder { r: f(0.4), h: f(0.03), rnd: f(0.01) }, [-1.1, -0.45, -0.5], id, ticks);
    keel.shell = 0;
    parts.push(keel);
    let mut lug = part(Geo3::Box { h: [f(0.05), f(0.08), f(0.05)], rnd: f(0.02) }, [0.0, 0.0, 0.0], id, chrome);
    lug.mirror = true;
    lug.off = [f(0.15), f(-0.4), f(-0.9)];
    parts.push(lug);
    let mut inter = part(Geo3::Sphere { r: f(0.2) }, [1.35, -0.2, 0.5], id, stripes);
    inter.op = Op3::Union;
    parts.push(inter);
    let mut cut = part(Geo3::Box { h: [f(0.25), f(0.08), f(0.25)], rnd: 0 }, [1.35, -0.2, 0.5], id, candy);
    cut.op = Op3::Inter;
    parts.push(cut);
    autogroup(&mut parts);
    RenderParams3 {
        seed,
        width,
        height,
        spp,
        passes,
        bounces,
        fog0: f(5.0),
        fog1: f(12.0),
        cam: camera([0.3, 0.9, -3.2], [0.0, -0.1, 0.1], 38.0, 0.03, 3.3),
        env: studio_env(),
        mats,
        parts,
    }
}


/// A tumbler of amber liquid with ice, an orange slice and mint on a wooden bar, with out-of-focus lights behind (round bokeh).
pub fn scene3_cocktail(width: u16, height: u16, spp: u8, passes: u16, seed: [u8; 32]) -> RenderParams3 {
    let mut mats: Vec<Mat3> = vec![];
    let mut add = |m: Mat3| -> u8 {
        mats.push(m);
        (mats.len() - 1) as u8
    };
    let wood = add(Mat3 { tex: 5, tex_scale: f(7.0), alb2: [128, 72, 34], coat: 170, bump: 25, bump_freq: f(30.0), ..material([54, 28, 14], 0.35) });
    let tumbler = add(glass([0.02, 0.012, 0.01], 1.5));
    let liquid = add(Mat3 { alb: [255, 236, 205], ..glass([0.45, 1.15, 3.4], 1.34) });
    let ice = add(glass([0.05, 0.04, 0.03], 1.31));
    let orange = add(Mat3 { tex: 3, tex_scale: f(30.0), alb2: [255, 190, 60], trans: 170, ..material([250, 112, 10], 0.4) });
    let leaf = add(Mat3 { trans: 120, ..material([28, 150, 44], 0.5) });
    let lights = [
        add(emissive([9.0, 6.5, 3.5])),
        add(emissive([3.0, 6.5, 9.0])),
        add(emissive([9.0, 3.5, 5.5])),
        add(emissive([8.0, 8.0, 5.5])),
        add(emissive([4.5, 9.0, 5.5])),
    ];
    let id = rot_identity();
    let mut parts: Vec<Part3> = vec![];
    parts.push(part(Geo3::Plane, [0.0, 0.0, 0.0], id, wood));
    parts.push(part(Geo3::Cylinder { r: f(0.4), h: f(0.425), rnd: f(0.03) }, [0.0, 0.425, 0.0], id, tumbler));
    let mut inner = part(Geo3::Cylinder { r: f(0.365), h: f(0.4), rnd: f(0.01) }, [0.0, 0.5, 0.0], id, tumbler);
    inner.op = Op3::Sub;
    parts.push(inner);
    parts.push(part(Geo3::Cylinder { r: f(0.358), h: f(0.2), rnd: f(0.012) }, [0.0, 0.29, 0.0], id, liquid));
    for (x, y, z, a, b, c) in [(0.04, 0.5, 0.02, 0.4, 0.7, 0.3), (-0.13, 0.42, 0.1, 0.9, 0.2, 0.5), (0.13, 0.36, -0.09, 0.2, 0.9, 0.8)] {
        parts.push(part(Geo3::Box { h: [f(0.1), f(0.1), f(0.1)], rnd: f(0.022) }, [x, y, z], rot_euler(a, b, c), ice));
    }
    parts.push(part(Geo3::Cylinder { r: f(0.21), h: f(0.012), rnd: f(0.004) }, [0.3, 0.86, 0.0], rot_euler(std::f64::consts::FRAC_PI_2 * 0.92, 0.0, 0.0), orange));
    // ellipsoid radii stay above MIN_ELLIPSOID_R (see render3_validate)
    parts.push(part(Geo3::Ellipsoid { r: [f(0.12), f(0.07), f(0.075)] }, [-0.06, 0.9, 0.0], rot_euler(0.3, 0.5, 0.4), leaf));
    parts.push(part(Geo3::Ellipsoid { r: [f(0.11), f(0.07), f(0.07)] }, [0.04, 0.93, -0.06], rot_euler(-0.4, 2.2, 0.3), leaf));
    for (i, (x, y, z, r)) in [(-2.6, 1.8, -5.0, 0.16), (-1.4, 2.5, -6.0, 0.2), (0.4, 1.3, -5.5, 0.14), (1.8, 2.2, -6.5, 0.22), (3.0, 0.9, -5.0, 0.15), (-3.4, 0.6, -6.0, 0.18), (2.4, 3.0, -7.0, 0.2)].iter().enumerate() {
        parts.push(part(Geo3::Sphere { r: f(*r) }, [*x, *y, *z], id, lights[i % lights.len()]));
    }
    autogroup(&mut parts);
    let mut env = studio_env();
    env.boxes = vec![
        SoftBox { dir: [f(-0.5), f(0.6), f(0.7)], hs: [f(0.4), f(0.3)], soft: f(0.5), col: rgb10([9.0, 7.8, 6.2]) },
        SoftBox { dir: [f(0.9), f(0.3), f(0.2)], hs: [f(0.15), f(0.6)], soft: f(0.4), col: rgb10([2.4, 3.2, 4.8]) },
        SoftBox { dir: [f(0.15), f(0.45), f(-0.85)], hs: [f(0.55), f(0.45)], soft: f(0.5), col: rgb10([6.5, 4.8, 3.2]) },
    ];
    env.bd_top = rgb10([0.01, 0.008, 0.012]);
    env.bd_bot = rgb10([0.03, 0.02, 0.02]);
    RenderParams3 {
        seed,
        width,
        height,
        spp,
        passes,
        bounces: 14,
        fog0: f(8.0),
        fog1: f(20.0),
        cam: camera([0.0, 0.52, 2.3], [0.0, 0.42, 0.0], 28.0, 0.045, 2.3),
        env,
        mats,
        parts,
    }
}

/// "Vearl Orb": a sphere of liquid metal (a swirl of indigo and cyan under a clear coat) on a hammered bronze plinth, circled by two neon rings
/// and their beads, glass pearls and gold balls on a dark tiled mirror floor, with out-of-focus lights behind. Textures 9, 10 and 11.
pub fn scene3_orb(width: u16, height: u16, spp: u8, passes: u16, seed: [u8; 32]) -> RenderParams3 {
    let mut mats: Vec<Mat3> = vec![];
    let mut add = |m: Mat3| -> u8 {
        mats.push(m);
        (mats.len() - 1) as u8
    };
    let floor = add(Mat3 { tex: 11, tex_scale: f(0.9), tex_p: f(0.035), tex_q: f(0.9), alb2: [90, 120, 210], bump: 14, bump_freq: f(40.0), ..metal([60, 80, 150], 0.05) });
    let orb = add(Mat3 { tex: 10, tex_scale: f(1.5), tex_p: f(2.6), tex_q: f(10.0), alb2: [255, 110, 215], coat: 255, ..metal([10, 40, 230], 0.06) });
    let plinth = add(Mat3 { tex: 9, tex_scale: f(3.2), tex_p: f(0.04), tex_q: f(0.8), alb2: [255, 200, 110], bump: 30, bump_freq: f(14.0), coat: 60, ..metal([90, 60, 32], 0.3) });
    let neon_c = add(emissive([0.8, 7.0, 10.0]));
    let neon_m = add(emissive([10.0, 1.2, 7.0]));
    let bead = add(emissive([9.0, 9.0, 10.0]));
    let pearl_a = add(glass([0.9, 0.12, 0.05], 1.52));
    let pearl_b = add(glass([0.05, 0.5, 0.9], 1.52));
    let gold = add(Mat3 { bump: 25, bump_freq: f(60.0), ..metal([245, 190, 90], 0.12) });
    let lamp_a = add(emissive([5.0, 8.0, 10.0]));
    let lamp_b = add(emissive([10.0, 3.0, 7.0]));
    let lamp_c = add(emissive([10.0, 8.0, 4.0]));
    let id = rot_identity();
    let mut parts: Vec<Part3> = vec![];
    parts.push(part(Geo3::Plane, [0.0, 0.0, 0.0], id, floor));
    parts.push(part(Geo3::Cylinder { r: f(1.0), h: f(0.07), rnd: f(0.05) }, [0.0, 0.07, 0.0], id, plinth));
    parts.push(part(Geo3::Sphere { r: f(0.8) }, [0.0, 0.95, 0.0], rot_euler(0.5, 0.3, 0.2), orb));
    // two neon rings around the orb, each carrying a ring of bright beads
    for (big, small, tilt, mat, bead_n) in [(1.3, 0.04, 0.42, neon_c, 10u8), (1.55, 0.03, -0.62, neon_m, 14)] {
        let r = rot_euler(tilt, 0.0, tilt * 0.4);
        parts.push(part(Geo3::Torus { big: f(big), small: f(small) }, [0.0, 0.95, 0.0], r, mat));
        let mut b = part(Geo3::Sphere { r: f(0.065) }, [0.0, 0.95, 0.0], r, bead);
        b.polar = bead_n;
        b.off = [f(big), 0, 0];
        parts.push(b);
    }
    for (x, y, z, r, m) in [(-1.7, 0.2, 0.9, 0.2, pearl_a), (1.9, 0.16, 0.6, 0.16, pearl_b), (-0.9, 0.12, 1.7, 0.12, pearl_b), (0.9, 0.14, 1.5, 0.14, pearl_a), (2.5, 0.22, -0.7, 0.22, pearl_a), (-2.4, 0.18, -0.6, 0.18, pearl_b)] {
        parts.push(part(Geo3::Sphere { r: f(r) }, [x, y, z], id, m));
    }
    for (x, y, z, r) in [(-1.2, 0.1, 1.2, 0.1), (1.35, 0.09, 1.05, 0.09), (0.2, 0.08, 2.0, 0.08)] {
        parts.push(part(Geo3::Sphere { r: f(r) }, [x, y, z], id, gold));
    }
    // out-of-focus lights far behind (round bokeh)
    for (i, (x, y, z, r)) in [(-4.2, 2.2, -7.0, 0.3), (-2.6, 3.4, -8.0, 0.4), (-0.6, 1.4, -7.5, 0.25), (1.6, 3.0, -8.5, 0.45), (3.4, 1.7, -7.0, 0.3), (5.0, 2.8, -9.0, 0.5), (-5.6, 1.2, -8.5, 0.4), (0.4, 4.2, -9.5, 0.45)].iter().enumerate() {
        parts.push(part(Geo3::Sphere { r: f(*r) }, [*x, *y, *z], id, [lamp_a, lamp_b, lamp_c][i % 3]));
    }
    autogroup(&mut parts);
    let mut env = studio_env();
    env.boxes = vec![
        SoftBox { dir: [f(-0.8), f(0.5), f(0.45)], hs: [f(0.3), f(0.7)], soft: f(0.5), col: rgb10([0.8, 3.2, 5.2]) },
        SoftBox { dir: [f(0.85), f(0.4), f(0.3)], hs: [f(0.25), f(0.8)], soft: f(0.5), col: rgb10([5.0, 0.9, 3.6]) },
        SoftBox { dir: [f(0.0), f(0.95), f(0.1)], hs: [f(0.6), f(0.3)], soft: f(0.6), col: rgb10([2.2, 2.4, 3.2]) },
    ];
    env.top = rgb10([0.10, 0.18, 0.50]);
    env.hor = rgb10([0.55, 0.14, 0.60]);
    env.gnd = rgb10([0.05, 0.07, 0.20]);
    env.bd_top = rgb10([0.004, 0.006, 0.02]);
    env.bd_bot = rgb10([0.012, 0.016, 0.04]);
    RenderParams3 {
        seed,
        width,
        height,
        spp,
        passes,
        bounces: 12,
        fog0: f(7.0),
        fog1: f(26.0),
        cam: camera([0.0, 1.0, 4.5], [0.0, 0.92, 0.0], 36.0, 0.03, 4.5),
        env,
        mats,
        parts,
    }
}

/// "Gallery": a still life on polished stone tiles in front of a plaster wall, lit by soft boxes: a veined marble sphere, a stack of waxed
/// walnut cubes, a brass torus, a cracked red ceramic vase and a swirl-glass bowl. Textures 3, 5, 8, 9, 10 and 11.
pub fn scene3_gallery(width: u16, height: u16, spp: u8, passes: u16, seed: [u8; 32]) -> RenderParams3 {
    let mut mats: Vec<Mat3> = vec![];
    let mut add = |m: Mat3| -> u8 {
        mats.push(m);
        (mats.len() - 1) as u8
    };
    let floor = add(Mat3 { tex: 11, tex_scale: f(0.55), tex_p: f(0.02), tex_q: f(0.9), alb2: [70, 68, 66], coat: 230, bump: 10, bump_freq: f(35.0), ..material([172, 166, 156], 0.12) });
    let wall = add(Mat3 { tex: 3, tex_scale: f(1.4), alb2: [58, 54, 50], bump: 40, bump_freq: f(22.0), ..material([104, 98, 90], 0.9) });
    let marble = add(Mat3 { tex: 8, tex_scale: f(3.4), tex_p: f(3.2), tex_q: f(0.08), alb2: [40, 48, 70], coat: 255, ..material([240, 238, 232], 0.06) });
    let walnut = add(Mat3 { tex: 5, tex_scale: f(9.0), alb2: [140, 82, 40], bump: 60, bump_freq: f(70.0), coat: 150, ..material([64, 34, 16], 0.38) });
    let brass = add(Mat3 { bump: 30, bump_freq: f(55.0), ..metal([238, 188, 92], 0.16) });
    let vase = add(Mat3 { tex: 9, tex_scale: f(4.2), tex_p: f(0.045), tex_q: f(0.8), alb2: [232, 118, 62], coat: 230, ..material([120, 18, 20], 0.22) });
    let swirl = add(Mat3 { tex: 10, tex_scale: f(2.4), tex_p: f(2.8), tex_q: f(8.0), alb2: [90, 210, 215], coat: 255, ..material([20, 60, 120], 0.1) });
    let steel = add(metal([225, 230, 238], 0.1));
    let lamp = add(emissive([9.0, 6.4, 3.4]));
    let id = rot_identity();
    let mut parts: Vec<Part3> = vec![];
    parts.push(part(Geo3::Plane, [0.0, 0.0, 0.0], id, floor));
    parts.push(part(Geo3::Box { h: [f(14.0), f(5.0), f(0.1)], rnd: 0 }, [0.0, 5.0, -3.4], id, wall));
    parts.push(part(Geo3::Sphere { r: f(0.72) }, [-1.15, 0.72, 0.0], rot_euler(0.4, 0.8, 0.2), marble));
    parts.push(part(Geo3::Box { h: [f(0.5), f(0.5), f(0.5)], rnd: f(0.05) }, [0.55, 0.5, 0.2], rot_euler(0.0, 0.42, 0.0), walnut));
    parts.push(part(Geo3::Box { h: [f(0.3), f(0.3), f(0.3)], rnd: f(0.04) }, [0.5, 1.3, 0.22], rot_euler(0.0, 0.95, 0.0), walnut));
    parts.push(part(Geo3::Sphere { r: f(0.17) }, [0.5, 1.8, 0.22], id, brass));
    parts.push(part(Geo3::Torus { big: f(0.52), small: f(0.15) }, [1.95, 0.15, 0.55], rot_euler(0.0, 0.5, 0.0), brass));
    // cracked ceramic vase: body, neck and lip fused with smooth unions
    parts.push(part(Geo3::Ellipsoid { r: [f(0.5), f(0.72), f(0.5)] }, [-2.55, 0.74, -0.6], id, vase));
    let mut neck = part(Geo3::Cylinder { r: f(0.17), h: f(0.38), rnd: f(0.02) }, [-2.55, 1.6, -0.6], id, vase);
    neck.op = Op3::Smooth(f(0.18));
    parts.push(neck);
    let mut lip = part(Geo3::Torus { big: f(0.2), small: f(0.04) }, [-2.55, 1.97, -0.6], id, vase);
    lip.op = Op3::Smooth(f(0.06));
    parts.push(lip);
    // swirl-glass bowl on the right: a sphere with a hollow cut from its top, on a steel foot
    parts.push(part(Geo3::Sphere { r: f(0.55) }, [3.0, 0.62, -0.2], id, swirl));
    let mut cut = part(Geo3::Sphere { r: f(0.45) }, [3.0, 0.82, -0.2], id, swirl);
    cut.op = Op3::Sub;
    parts.push(cut);
    parts.push(part(Geo3::Cylinder { r: f(0.2), h: f(0.05), rnd: f(0.02) }, [3.0, 0.05, -0.2], id, steel));
    // two warm light strips on the wall
    parts.push(part(Geo3::Box { h: [f(0.04), f(1.2), f(0.02)], rnd: f(0.015) }, [-4.4, 1.8, -3.25], id, lamp));
    parts.push(part(Geo3::Box { h: [f(0.04), f(1.2), f(0.02)], rnd: f(0.015) }, [4.4, 1.8, -3.25], id, lamp));
    autogroup(&mut parts);
    let mut env = studio_env();
    env.boxes = vec![
        SoftBox { dir: [f(-0.55), f(0.7), f(0.6)], hs: [f(0.55), f(0.4)], soft: f(0.5), col: rgb10([6.0, 5.6, 5.0]) },
        SoftBox { dir: [f(0.85), f(0.4), f(0.35)], hs: [f(0.2), f(0.7)], soft: f(0.45), col: rgb10([2.6, 3.2, 4.4]) },
        SoftBox { dir: [f(0.0), f(0.95), f(0.15)], hs: [f(0.5), f(0.3)], soft: f(0.6), col: rgb10([3.0, 3.0, 3.2]) },
    ];
    env.bd_top = rgb10([0.03, 0.03, 0.035]);
    env.bd_bot = rgb10([0.06, 0.06, 0.07]);
    RenderParams3 {
        seed,
        width,
        height,
        spp,
        passes,
        bounces: 10,
        fog0: f(9.0),
        fog1: f(28.0),
        cam: camera([0.2, 1.45, 6.2], [0.15, 0.7, 0.0], 36.0, 0.022, 6.2),
        env,
        mats,
        parts,
    }
}

/// A close-up of the four cell and noise textures (11 floor tiles, 8 veined marble, 9 cracked cells, 10 swirl) on three spheres: the test scene of
/// the frozen vectors for the textures added after `scene3_mini`.
pub fn scene3_textures(width: u16, height: u16, spp: u8, passes: u16, bounces: u8, seed: [u8; 32]) -> RenderParams3 {
    let mut mats: Vec<Mat3> = vec![];
    let mut add = |m: Mat3| -> u8 {
        mats.push(m);
        (mats.len() - 1) as u8
    };
    let floor = add(Mat3 { tex: 11, tex_scale: f(0.9), tex_p: f(0.06), tex_q: f(0.8), alb2: [60, 70, 90], ..material([200, 190, 170], 0.2) });
    let marble = add(Mat3 { tex: 8, tex_scale: f(3.0), tex_p: f(2.5), tex_q: f(0.08), alb2: [30, 40, 70], coat: 200, ..material([235, 232, 225], 0.1) });
    let cells = add(Mat3 { tex: 9, tex_scale: f(4.0), tex_p: f(0.06), tex_q: f(0.9), alb2: [230, 110, 60], ..material([110, 20, 30], 0.3) });
    let swirl = add(Mat3 { tex: 10, tex_scale: f(2.0), tex_p: f(2.6), tex_q: f(8.0), alb2: [120, 240, 250], ..metal([30, 40, 180], 0.15) });
    let id = rot_identity();
    let mut parts: Vec<Part3> = vec![];
    parts.push(part(Geo3::Plane, [0.0, 0.0, 0.0], id, floor));
    parts.push(part(Geo3::Sphere { r: f(0.55) }, [-1.2, 0.55, 0.0], rot_euler(0.3, 0.5, 0.1), marble));
    parts.push(part(Geo3::Sphere { r: f(0.55) }, [0.0, 0.55, 0.2], id, cells));
    parts.push(part(Geo3::Sphere { r: f(0.55) }, [1.2, 0.55, 0.0], id, swirl));
    autogroup(&mut parts);
    RenderParams3 {
        seed,
        width,
        height,
        spp,
        passes,
        bounces,
        fog0: f(8.0),
        fog1: f(20.0),
        cam: camera([0.0, 1.3, 3.6], [0.0, 0.5, 0.0], 36.0, 0.0, 3.6),
        env: studio_env(),
        mats,
        parts,
    }
}
