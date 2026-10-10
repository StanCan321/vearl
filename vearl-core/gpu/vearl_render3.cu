// Render3 on the GPU: a line-by-line port of src/render3.rs (integer arithmetic only, see that file and docs/10 section 4 ter). One CUDA
// block = one chunk (one pass of one 16x16 tile), one thread = one pixel. Division truncates toward zero and >> on signed values is
// arithmetic, exactly as in Rust; `x << 16` of a possibly negative value is written as a multiplication.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

typedef long long ll;

#define Q 65536LL
#define TILE 16
#define PI_Q 205887LL
#define TAU_Q 411775LL
#define MAXT (60LL * 65536LL)
#define EPS_Q 26LL
#define T_START 66LL
#define NORMAL_E 33LL
#define BIAS_G 131LL
#define BIAS_T 262LL
#define CLAMP_RAD (24LL * 65536LL)
#define MARCH_STEPS3 420
#define BIG (1LL << 60)

struct R3Part {
    int32_t kind, g[4], c[3], rot[9], op, k, mat, polar, mirror, off[3], group, shell;
};
struct R3Mat {
    int32_t alb[3], alb2[3], rough, metal, spec, emit[3], trans, glass, ior, absorb[3], coat, tex, tex_scale, tex_p, tex_q, tex_r, bump, bump_freq;
};
struct R3Box {
    int32_t dir[3], hs[2], soft;
    uint32_t col[3];
};
struct R3Params {
    int32_t width, height, spp, passes, bounces, fog0, fog1;
    uint32_t seed0, seed1;
    int32_t cam_pos[3], cam_look[3], focal, aperture, focus;
    uint32_t top[3], hor[3], gnd[3];
    int32_t nboxes;
    R3Box boxes[4];
    uint32_t bd_top[3], bd_bot[3];
    int32_t nparts, nmats;
    R3Mat mats[24];
    R3Part parts[48];
};

struct V3 { ll x, y, z; };
__device__ __forceinline__ V3 mk(ll x, ll y, ll z) { V3 v; v.x = x; v.y = y; v.z = z; return v; }
__device__ __forceinline__ ll mul16(ll a, ll b) { return (a * b) >> 16; }
__device__ __forceinline__ ll clampl(ll v, ll lo, ll hi) { return v < lo ? lo : (v > hi ? hi : v); }
__device__ __forceinline__ ll minl(ll a, ll b) { return a < b ? a : b; }
__device__ __forceinline__ ll maxl(ll a, ll b) { return a > b ? a : b; }
__device__ __forceinline__ ll absl(ll a) { return a < 0 ? -a : a; }
__device__ __forceinline__ ll rem_euclid(ll a, ll m) { ll r = a % m; return r < 0 ? r + m : r; }
__device__ __forceinline__ ll div_euclid(ll a, ll m) { return (a - rem_euclid(a, m)) / m; }
__device__ __forceinline__ ll comp(V3 v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }

__device__ __forceinline__ ll isqrt_ll(ll x) {
    if (x < 2) return x < 0 ? 0 : x;
    ll r = (ll)sqrt((double)x);
    while (r * r > x) r--;
    while ((r + 1) * (r + 1) <= x) r++;
    return r;
}
__device__ __forceinline__ ll len3(V3 v) { return isqrt_ll(v.x * v.x + v.y * v.y + v.z * v.z); }
__device__ __forceinline__ ll len2(ll a, ll b) { return isqrt_ll(a * a + b * b); }
__device__ __forceinline__ V3 norm3(V3 v) {
    ll l = len3(v);
    if (l == 0) return mk(0, Q, 0);
    return mk((v.x * 65536LL) / l, (v.y * 65536LL) / l, (v.z * 65536LL) / l);
}
__device__ __forceinline__ ll dot16(V3 a, V3 b) { return (a.x * b.x + a.y * b.y + a.z * b.z) >> 16; }
__device__ __forceinline__ V3 cross16(V3 a, V3 b) { return mk((a.y * b.z - a.z * b.y) >> 16, (a.z * b.x - a.x * b.z) >> 16, (a.x * b.y - a.y * b.x) >> 16); }
__device__ __forceinline__ V3 add3(V3 a, V3 b) { return mk(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ V3 sub3(V3 a, V3 b) { return mk(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ V3 scale3(V3 a, ll s) { return mk(mul16(a.x, s), mul16(a.y, s), mul16(a.z, s)); }
__device__ __forceinline__ V3 neg3(V3 a) { return mk(-a.x, -a.y, -a.z); }
__device__ __forceinline__ ll lin8(ll c) { return ((c * c) * 65536LL) / 65025LL; }
__device__ __forceinline__ V3 lin8x3(const int32_t* c) { return mk(lin8(c[0]), lin8(c[1]), lin8(c[2])); }
__device__ __forceinline__ ll q10(ll v) { return v * 64LL; }
__device__ __forceinline__ V3 q10x3(const uint32_t* c) { return mk(q10(c[0]), q10(c[1]), q10(c[2])); }
__device__ __forceinline__ V3 q10x3i(const int32_t* c) { return mk(q10(c[0]), q10(c[1]), q10(c[2])); }

__device__ __forceinline__ ll smoothstep_(ll e0, ll e1, ll x) {
    if (e1 == e0) return x >= e1 ? Q : 0;
    ll t = clampl(((x - e0) * 65536LL) / (e1 - e0), 0, Q);
    return mul16(mul16(t, t), 3 * Q - 2 * t);
}
__device__ __forceinline__ ll mix_(ll a, ll b, ll t) { return a + mul16(b - a, t); }

// ---- CORDIC
__device__ const ll CORDIC_ATAN[24] = {13176795, 7778716, 4110060, 2086331, 1047214, 524117, 262123, 131069, 65536, 32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2};
#define CORDIC_K 10188014LL

__device__ void sincos_(ll a, ll* s_out, ll* c_out) {
    a = rem_euclid(a, TAU_Q);
    if (a > PI_Q) a -= TAU_Q;
    bool flip = false;
    if (a > PI_Q / 2) { a -= PI_Q; flip = true; }
    else if (a < -(PI_Q / 2)) { a += PI_Q; flip = true; }
    ll x = CORDIC_K, y = 0, z = a * 256LL;
    for (int i = 0; i < 24; i++) {
        ll dx = y >> i, dy = x >> i;
        if (z >= 0) { x -= dx; y += dy; z -= CORDIC_ATAN[i]; }
        else { x += dx; y -= dy; z += CORDIC_ATAN[i]; }
    }
    ll s = (y + 128) >> 8, c = (x + 128) >> 8;
    if (flip) { s = -s; c = -c; }
    *s_out = s; *c_out = c;
}

__device__ ll atan2q(ll y, ll x) {
    if (x == 0 && y == 0) return 0;
    ll xx = x * 256LL, yy = y * 256LL, base = 0;
    if (xx < 0) {
        base = (yy >= 0) ? PI_Q : -PI_Q;
        xx = -xx; yy = -yy;
    }
    ll z = 0;
    for (int i = 0; i < 24; i++) {
        ll dx = yy >> i, dy = xx >> i;
        if (yy > 0) { xx += dx; yy -= dy; z += CORDIC_ATAN[i]; }
        else { xx -= dx; yy += dy; z -= CORDIC_ATAN[i]; }
    }
    return base + ((z + 128) >> 8);
}

// ---- hash, noise
__device__ __forceinline__ uint32_t pcg(uint32_t v) {
    uint32_t s = v * 747796405u + 2891336453u;
    uint32_t w = ((s >> ((s >> 28) + 4)) ^ s) * 277803737u;
    return (w >> 22) ^ w;
}
__device__ __forceinline__ ll hash3(ll x, ll y, ll z) {
    uint32_t h = pcg(((uint32_t)x * 73856093u) ^ ((uint32_t)y * 19349663u) ^ ((uint32_t)z * 83492791u));
    return (ll)(h >> 16);
}
__device__ __forceinline__ ll fsm(ll v) { ll t = v & 0xFFFF; return mul16(mul16(t, t), 3 * Q - 2 * t); }
__device__ ll vnoise(V3 p) {
    ll ix = p.x >> 16, iy = p.y >> 16, iz = p.z >> 16;
    ll fx = fsm(p.x), fy = fsm(p.y), fz = fsm(p.z);
    ll x00 = mix_(hash3(ix, iy, iz), hash3(ix + 1, iy, iz), fx);
    ll x10 = mix_(hash3(ix, iy + 1, iz), hash3(ix + 1, iy + 1, iz), fx);
    ll x01 = mix_(hash3(ix, iy, iz + 1), hash3(ix + 1, iy, iz + 1), fx);
    ll x11 = mix_(hash3(ix, iy + 1, iz + 1), hash3(ix + 1, iy + 1, iz + 1), fx);
    return mix_(mix_(x00, x10, fy), mix_(x01, x11, fy), fz);
}
__device__ ll fbm_n(V3 p, int octaves) {
    ll a = Q / 2, s = 0;
    for (int i = 0; i < octaves; i++) {
        s += mul16(a, vnoise(p));
        p = mk(p.x * 2 + (p.x >> 5) + 111411, p.y * 2 + (p.y >> 5) + 602931, p.z * 2 + (p.z >> 5) + 203162);
        a >>= 1;
    }
    return s;
}

// ---- geometry
__device__ __forceinline__ V3 rotate_(const int32_t* r, V3 p) {
    return mk(((ll)r[0] * p.x + (ll)r[1] * p.y + (ll)r[2] * p.z) >> 16, ((ll)r[3] * p.x + (ll)r[4] * p.y + (ll)r[5] * p.z) >> 16, ((ll)r[6] * p.x + (ll)r[7] * p.y + (ll)r[8] * p.z) >> 16);
}

__device__ ll geo_dist(const R3Part& pt, V3 q) {
    switch (pt.kind) {
    case 0: return len3(q) - pt.g[0];
    case 1: {
        ll r0 = pt.g[0], r1 = pt.g[1], r2 = pt.g[2];
        ll ax = (q.x * 65536LL) / r0, ay = (q.y * 65536LL) / r1, az = (q.z * 65536LL) / r2;
        ll bx = (ax * 65536LL) / r0, by = (ay * 65536LL) / r1, bz = (az * 65536LL) / r2;
        ll k0 = len3(mk(ax, ay, az));
        ll k1 = maxl(len3(mk(bx, by, bz)), 1);
        return (mul16(k0, k0 - Q) * 65536LL) / k1;
    }
    case 2: {
        ll r = pt.g[3];
        ll w0 = absl(q.x) - pt.g[0] + r, w1 = absl(q.y) - pt.g[1] + r, w2 = absl(q.z) - pt.g[2] + r;
        return len3(mk(maxl(w0, 0), maxl(w1, 0), maxl(w2, 0))) + minl(maxl(w0, maxl(w1, w2)), 0) - r;
    }
    case 3: {
        ll rr = pt.g[2];
        ll dx = len2(q.x, q.z) - pt.g[0] + rr;
        ll dy = absl(q.y) - pt.g[1] + rr;
        return len2(maxl(dx, 0), maxl(dy, 0)) + minl(maxl(dx, dy), 0) - rr;
    }
    case 4: {
        ll y = clampl(q.y, -(ll)pt.g[0], (ll)pt.g[0]);
        return len3(mk(q.x, q.y - y, q.z)) - pt.g[1];
    }
    case 5: return len2(len2(q.x, q.z) - pt.g[0], q.y) - pt.g[1];
    case 6: {
        ll h = pt.g[0], r1 = pt.g[1], r2 = pt.g[2];
        ll qx = len2(q.x, q.z), qy = q.y;
        ll k1x = r2, k1y = h, k2x = r2 - r1, k2y = 2 * h;
        ll cax = qx - minl(qx, qy < 0 ? r1 : r2);
        ll cay = absl(qy) - h;
        ll dd = mul16(k2x, k2x) + mul16(k2y, k2y);
        ll tt = clampl(dd == 0 ? 0 : ((mul16(k1x - qx, k2x) + mul16(k1y - qy, k2y)) * 65536LL) / dd, 0, Q);
        ll cbx = qx - k1x + mul16(k2x, tt), cby = qy - k1y + mul16(k2y, tt);
        ll s = (cbx < 0 && cay < 0) ? -1 : 1;
        return s * isqrt_ll(minl(cax * cax + cay * cay, cbx * cbx + cby * cby));
    }
    default: return q.y;
    }
}

__device__ ll smin3(ll a, ll b, ll k) {
    if (k <= 0) return minl(a, b);
    ll h = clampl(Q / 2 + (((b - a) * 32768LL) / k), 0, Q);
    return b + mul16(a - b, h) - mul16(mul16(k, h), Q - h);
}

__device__ ll part_dist(const R3Part& pt, V3 p) {
    V3 q = rotate_(pt.rot, sub3(p, mk(pt.c[0], pt.c[1], pt.c[2])));
    if (pt.mirror) { q.x = absl(q.x); q.z = absl(q.z); }
    if (pt.polar > 1) {
        ll n = pt.polar;
        ll sec = TAU_Q / n;
        ll ang = atan2q(q.z, q.x);
        ll k = div_euclid(ang + sec / 2, sec);
        ll s, c;
        sincos_(k * sec, &s, &c);
        q = mk(mul16(q.x, c) + mul16(q.z, s), q.y, mul16(-q.x, s) + mul16(q.z, c));
    }
    ll d = geo_dist(pt, sub3(q, mk(pt.off[0], pt.off[1], pt.off[2])));
    if (pt.shell > 0) return absl(d) - pt.shell;
    return d;
}

struct SceneHit { ll d; int mat; };

__device__ SceneHit scene_(const R3Params& P, V3 pos) {
    ll all = BIG, acc = BIG;
    int all_mat = 0, mat = 0;
    int cur = P.parts[0].group;
    for (int i = 0; i < P.nparts; i++) {
        const R3Part& pt = P.parts[i];
        if (pt.group != cur) {
            if (acc < all) { all = acc; all_mat = mat; }
            acc = BIG; mat = 0; cur = pt.group;
        }
        ll d = part_dist(pt, pos);
        if (pt.op == 0) {
            if (d < acc) { acc = d; mat = pt.mat; }
        } else if (pt.op == 1) {
            if (acc == BIG) { acc = d; mat = pt.mat; }
            else {
                int m = d < acc ? pt.mat : mat;
                acc = smin3(acc, d, pt.k);
                mat = m;
            }
        } else if (pt.op == 2) {
            if (-d > acc) { acc = -d; mat = pt.mat; }
        } else {
            if (d > acc) { acc = d; mat = pt.mat; }
        }
    }
    if (acc < all) { all = acc; all_mat = mat; }
    SceneHit h; h.d = all; h.mat = all_mat;
    return h;
}

__device__ V3 normal_at(const R3Params& P, V3 pos) {
    ll e = NORMAL_E;
    ll a = scene_(P, mk(pos.x + e, pos.y - e, pos.z - e)).d;
    ll b = scene_(P, mk(pos.x - e, pos.y - e, pos.z + e)).d;
    ll c = scene_(P, mk(pos.x - e, pos.y + e, pos.z - e)).d;
    ll d = scene_(P, mk(pos.x + e, pos.y + e, pos.z + e)).d;
    return norm3(mk(a - b - c + d, -a - b + c + d, -a + b - c + d));
}

// ---- cell textures
__device__ __forceinline__ ll cell_hash(ll ix, ll iy, ll iz, ll k) { return hash3(ix + 17 * k, iy + 31 * k, iz + 47 * k); }

// Worley noise: distances (Q16) to the nearest and to the second nearest feature point, and a hash of the nearest cell.
__device__ void voronoi(V3 p, ll* f1, ll* f2, ll* id) {
    ll ix = p.x >> 16, iy = p.y >> 16, iz = p.z >> 16;
    ll d1 = 0x7FFFFFFFFFFFFFFFLL, d2 = 0x7FFFFFFFFFFFFFFFLL, best = 0;
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                ll cx = ix + dx, cy = iy + dy, cz = iz + dz;
                ll ex = (cx << 16) + cell_hash(cx, cy, cz, 1) - p.x;
                ll ey = (cy << 16) + cell_hash(cx, cy, cz, 2) - p.y;
                ll ez = (cz << 16) + cell_hash(cx, cy, cz, 3) - p.z;
                ll d = ex * ex + ey * ey + ez * ez;
                if (d < d1) { d2 = d1; d1 = d; best = cell_hash(cx, cy, cz, 0); }
                else if (d < d2) { d2 = d; }
            }
        }
    }
    *f1 = isqrt_ll(d1); *f2 = isqrt_ll(d2); *id = best;
}

// ---- materials
struct Surf { V3 alb; ll rough; bool has_n; V3 nrm; };

__device__ V3 bump_normal(V3 pos, V3 n, ll freq, ll amp) {
    ll e = 98;
    ll b = fbm_n(scale3(pos, freq), 3);
    ll gx = ((fbm_n(scale3(mk(pos.x + e, pos.y, pos.z), freq), 3) - b) * Q) / e;
    ll gy = ((fbm_n(scale3(mk(pos.x, pos.y + e, pos.z), freq), 3) - b) * Q) / e;
    ll gz = ((fbm_n(scale3(mk(pos.x, pos.y, pos.z + e), freq), 3) - b) * Q) / e;
    V3 g = mk(gx, gy, gz);
    ll gn = dot16(g, n);
    g = mk(g.x - mul16(n.x, gn), g.y - mul16(n.y, gn), g.z - mul16(n.z, gn));
    ll k = ((amp * 256LL) * 65536LL) / maxl(freq, 1);
    return norm3(mk(n.x - mul16(g.x, k), n.y - mul16(g.y, k), n.z - mul16(g.z, k)));
}

__device__ V3 mixc(ll t, V3 a, V3 b) { return mk(mix_(a.x, b.x, t), mix_(a.y, b.y, t), mix_(a.z, b.z, t)); }

__device__ Surf surface_mat(const R3Mat& m, V3 pos, V3 ng) {
    Surf s;
    V3 alb = lin8x3(m.alb);
    V3 alb2 = lin8x3(m.alb2);
    ll rough = (ll)m.rough * 256LL;
    ll sc = m.tex_scale;
    switch (m.tex) {
    case 1: {
        ll cell = (mul16(pos.x, sc) >> 16) + (mul16(pos.y, sc) >> 16) + (mul16(pos.z, sc) >> 16);
        if (cell & 1) alb = alb2;
        break;
    }
    case 2: {
        int axis = m.tex_p < 2 ? m.tex_p : 2;
        ll cell = mul16(comp(pos, axis), sc) >> 16;
        if (cell & 1) alb = alb2;
        break;
    }
    case 3: {
        ll t = fbm_n(scale3(pos, sc), 5);
        alb = mixc(clampl(t, 0, Q), alb, alb2);
        break;
    }
    case 4: {
        ll r = len2(pos.x, pos.z);
        if ((mul16(r, sc) & 0xFFFF) < Q / 2) alb = alb2;
        break;
    }
    case 5: {
        ll u = mul16(pos.z, sc) + 2 * fbm_n(scale3(pos, 3 * Q), 3);
        ll fr = u & 0xFFFF;
        ll t = smoothstep_(0, Q / 2, fr) - smoothstep_(Q / 2, Q, fr);
        alb = mixc(clampl(t, 0, Q), alb, alb2);
        break;
    }
    case 6: {
        ll ang = atan2q(pos.z, pos.x);
        ll sn, cs;
        sincos_(ang * (ll)m.tex_p, &sn, &cs);
        ll t = (Q + sn) / 2;
        alb = mixc(clampl(t, 0, Q), alb, alb2);
        rough = minl(rough + mul16(t, Q / 4), Q);
        break;
    }
    case 7: {
        ll r = len2(pos.x, pos.z);
        ll ang = atan2q(pos.z, pos.x);
        ll sn, cs;
        sincos_(ang * (ll)m.tex_p, &sn, &cs);
        if (absl(cs) > 64553 && r >= m.tex_q && r <= m.tex_r && ng.y > 39322) alb = alb2;
        break;
    }
    case 8: {
        ll w = fbm_n(scale3(pos, sc), 4);
        ll u = mul16(pos.x + mul16(pos.y, 26214) + mul16(pos.z, 39322), sc) + mul16(w, (ll)m.tex_p);
        ll fr = u & 0xFFFF;
        ll d = minl(fr, Q - fr);
        ll vein = Q - smoothstep_(0, maxl((ll)m.tex_q, 1), d);
        V3 base = mixc(mul16(w, 16384), alb, alb2);
        alb = mixc(vein, base, alb2);
        break;
    }
    case 9: {
        ll f1, f2, id;
        voronoi(scale3(pos, sc), &f1, &f2, &id);
        ll t = mul16(id, (ll)m.tex_q);
        V3 base = mixc(t, alb, alb2);
        ll crack = Q - smoothstep_(0, maxl((ll)m.tex_p, 1), f2 - f1);
        alb = mixc(mul16(crack, 55705), base, mk(0, 0, 0));
        rough = minl(rough + mul16(crack, Q / 4), Q);
        break;
    }
    case 10: {
        V3 p = scale3(pos, sc);
        ll wx = fbm_n(p, 3);
        ll wy = fbm_n(mk(p.x + 5 * Q + 21299, p.y + Q + 19661, p.z + 2 * Q + 45875), 3);
        ll wz = fbm_n(mk(p.x + 3 * Q + 13107, p.y + 4 * Q + 52428, p.z + Q + 6554), 3);
        ll wp = m.tex_p;
        V3 q = mk(p.x + mul16(wx - Q / 4, wp), p.y + mul16(wy - Q / 4, wp), p.z + mul16(wz - Q / 4, wp));
        ll t = fbm_n(q, 4);
        ll sm = smoothstep_(Q / 5, 4 * Q / 5, t);
        V3 c = mixc(sm, alb, alb2);
        ll u = mul16(t, (ll)m.tex_q);
        ll fr = u & 0xFFFF;
        ll line = Q - smoothstep_(0, Q / 20, minl(fr, Q - fr));
        alb = mixc(mul16(line, Q / 4), c, mk(Q, Q, Q));
        break;
    }
    case 11: {
        ll u = mul16(pos.x, sc), v = mul16(pos.z, sc);
        ll fu = u & 0xFFFF, fv = v & 0xFFFF;
        ll e = minl(minl(fu, Q - fu), minl(fv, Q - fv));
        ll g = maxl((ll)m.tex_p, 2);
        ll grout = Q - smoothstep_(g / 2, g, e);
        ll tone = mul16(hash3(u >> 16, 5, v >> 16), (ll)m.tex_q);
        V3 base = mixc(tone, alb, alb2);
        alb = mixc(grout, base, scale3(base, 8000));
        rough = mix_(rough, 58982, grout);
        break;
    }
    default: break;
    }
    s.alb = alb; s.rough = rough;
    s.has_n = false; s.nrm = mk(0, 0, 0);
    if (m.bump > 0) { s.has_n = true; s.nrm = bump_normal(pos, ng, m.bump_freq, m.bump); }
    return s;
}

// ---- sampling
struct Rng { uint32_t s; };
__device__ __forceinline__ ll rnd(Rng& r) { r.s = pcg(r.s); return (ll)(r.s >> 16); }

__device__ void disc_(Rng& r, ll* ox, ll* oy) {
    for (int i = 0; i < 8; i++) {
        ll x = 2 * rnd(r) - Q;
        ll y = 2 * rnd(r) - Q;
        if (x * x + y * y <= Q * Q) { *ox = x; *oy = y; return; }
    }
    *ox = 0; *oy = 0;
}

__device__ V3 basis_t(V3 n) {
    if (absl(n.x) > Q / 2) return norm3(cross16(n, mk(0, Q, 0)));
    return norm3(cross16(n, mk(Q, 0, 0)));
}

__device__ V3 cos_hemi(V3 n, Rng& r) {
    ll x, y;
    disc_(r, &x, &y);
    ll z = isqrt_ll(maxl(Q * Q - x * x - y * y, 0));
    V3 t = basis_t(n);
    V3 b = cross16(n, t);
    return norm3(mk(mul16(t.x, x) + mul16(b.x, y) + mul16(n.x, z), mul16(t.y, x) + mul16(b.y, y) + mul16(n.y, z), mul16(t.z, x) + mul16(b.z, y) + mul16(n.z, z)));
}

__device__ V3 ggx_h(V3 n, ll a, Rng& r) {
    ll x, y;
    disc_(r, &x, &y);
    ll v = rnd(r);
    ll a2 = mul16(a, a);
    ll denom = Q + mul16(a2 - Q, v);
    ll ratio = clampl(((Q - v) * 65536LL) / maxl(denom, 1), 0, Q);
    ll ct = isqrt_ll(ratio * 65536LL);
    ll st = isqrt_ll(maxl(Q * Q - ct * ct, 0));
    ll l = len2(x, y);
    ll cx, cy;
    if (l == 0) { cx = Q; cy = 0; } else { cx = (x * 65536LL) / l; cy = (y * 65536LL) / l; }
    V3 t = basis_t(n);
    V3 b = cross16(n, t);
    ll sx = mul16(cx, st), sy = mul16(cy, st);
    return norm3(mk(mul16(t.x, sx) + mul16(b.x, sy) + mul16(n.x, ct), mul16(t.y, sx) + mul16(b.y, sy) + mul16(n.y, ct), mul16(t.z, sx) + mul16(b.z, sy) + mul16(n.z, ct)));
}

__device__ V3 reflect_(V3 i, V3 n) {
    ll d = dot16(i, n);
    return mk(i.x - 2 * mul16(d, n.x), i.y - 2 * mul16(d, n.y), i.z - 2 * mul16(d, n.z));
}

__device__ bool refract_(V3 i, V3 n, ll eta, V3* out) {
    ll d = dot16(n, i);
    ll k = Q - mul16(mul16(eta, eta), Q - mul16(d, d));
    if (k < 0) return false;
    ll sk = isqrt_ll(k * 65536LL);
    ll s = mul16(eta, d) + sk;
    *out = norm3(mk(mul16(eta, i.x) - mul16(s, n.x), mul16(eta, i.y) - mul16(s, n.y), mul16(eta, i.z) - mul16(s, n.z)));
    return true;
}

__device__ __forceinline__ ll pow5(ll x) { ll x2 = mul16(x, x); return mul16(mul16(x2, x2), x); }

// ---- environment
__device__ ll softbox(V3 d, const R3Box& b) {
    V3 dir = norm3(mk(b.dir[0], b.dir[1], b.dir[2]));
    ll c = dot16(d, dir);
    if (c <= 0) return 0;
    V3 up0 = absl(dir.y) > 62259 ? mk(Q, 0, 0) : mk(0, Q, 0);
    V3 r = norm3(cross16(dir, up0));
    V3 u = cross16(r, dir);
    ll qx = absl((dot16(d, r) * 65536LL) / c);
    ll qy = absl((dot16(d, u) * 65536LL) / c);
    ll hx = b.hs[0], hy = b.hs[1], s = b.soft;
    ll wx = Q - smoothstep_(mul16(hx, Q - s), hx, qx);
    ll wy = Q - smoothstep_(mul16(hy, Q - s), hy, qy);
    return mul16(wx, wy);
}

__device__ V3 env_(const R3Params& P, V3 d) {
    V3 top = q10x3(P.top), hor = q10x3(P.hor), gnd = q10x3(P.gnd);
    ll out[3];
    for (int i = 0; i < 3; i++) {
        ll dy = d.y;
        if (dy >= 0) { ll yy = minl(dy, Q); out[i] = mix_(comp(hor, i), comp(top, i), mul16(yy, yy)); }
        else out[i] = mix_(comp(hor, i), comp(gnd, i), minl(-dy, Q));
    }
    for (int bi = 0; bi < P.nboxes; bi++) {
        ll w = softbox(d, P.boxes[bi]);
        if (w > 0) {
            V3 c = q10x3(P.boxes[bi].col);
            out[0] += mul16(c.x, w); out[1] += mul16(c.y, w); out[2] += mul16(c.z, w);
        }
    }
    return mk(out[0], out[1], out[2]);
}

__device__ V3 backdrop_(const R3Params& P, V3 d) {
    V3 t = q10x3(P.bd_top), b = q10x3(P.bd_bot);
    ll k = clampl((d.y + Q) >> 1, 0, Q);
    return mk(mix_(b.x, t.x, k), mix_(b.y, t.y, k), mix_(b.z, t.z, k));
}

// ---- path tracing
__device__ bool march_(const R3Params& P, V3 ro, V3 rd, ll* t_out, int* mat_out) {
    ll t = T_START;
    for (int i = 0; i < MARCH_STEPS3; i++) {
        V3 pos = add3(ro, scale3(rd, t));
        SceneHit h = scene_(P, pos);
        ll d = absl(h.d);
        ll eps = maxl((EPS_Q * maxl(t, 19661)) >> 16, 1);
        if (d < eps) { *t_out = t; *mat_out = h.mat; return true; }
        t += (d * 9) / 10;
        if (t > MAXT) break;
    }
    return false;
}

__device__ ll exp_neg(ll x) {
    if (x <= 0) return Q;
    int k = 0;
    ll y = x;
    while (y > 4096 && k < 24) { y >>= 1; k++; }
    ll y2 = mul16(y, y);
    ll y3 = mul16(y2, y);
    ll e = Q - y + (y2 >> 1) - y3 / 6;
    e = clampl(e, 0, Q);
    for (int i = 0; i < k; i++) e = mul16(e, e);
    return e;
}

__device__ V3 trace_(const R3Params& P, Rng& rng, V3 ro, V3 rd0v) {
    V3 rd = rd0v;
    ll col[3] = {0, 0, 0};
    ll thr[3] = {Q, Q, Q};
    ll t0 = -1;
    bool inside = false;
    ll sig[3] = {0, 0, 0};
    for (int b = 0; b <= P.bounces; b++) {
        ll t;
        int mi;
        if (!march_(P, ro, rd, &t, &mi)) {
            V3 e = b == 0 ? backdrop_(P, rd) : env_(P, rd);
            col[0] += mul16(thr[0], e.x); col[1] += mul16(thr[1], e.y); col[2] += mul16(thr[2], e.z);
            break;
        }
        if (b == 0) t0 = t;
        if (inside) {
            for (int i = 0; i < 3; i++) thr[i] = mul16(thr[i], exp_neg(mul16(sig[i], t)));
        }
        V3 pos = add3(ro, scale3(rd, t));
        V3 ng = normal_at(P, pos);
        const R3Mat& m = P.mats[mi];
        Surf sf = surface_mat(m, pos, ng);
        V3 n = sf.has_n ? sf.nrm : ng;
        V3 emit = q10x3i(m.emit);
        col[0] += mul16(thr[0], emit.x); col[1] += mul16(thr[1], emit.y); col[2] += mul16(thr[2], emit.z);
        if ((emit.x + emit.y + emit.z) > 0 && (m.alb[0] + m.alb[1] + m.alb[2]) == 0) break;
        if (m.glass) {
            bool ent = dot16(rd, ng) < 0;
            V3 gn = ent ? ng : neg3(ng);
            V3 nn = ent ? n : neg3(n);
            if (sf.rough > 0) {
                ll r2 = mul16(sf.rough, sf.rough);
                nn = ggx_h(nn, r2, rng);
            }
            ll ior = (ll)m.ior * 64LL;
            ll eta = ent ? (Q * 65536LL) / ior : ior;
            ll ci = clampl(-dot16(rd, nn), 0, Q);
            ll f0;
            {
                ll a = ((ior - Q) * 65536LL) / (ior + Q);
                f0 = mul16(a, a);
            }
            ll fr = f0 + mul16(Q - f0, pow5(Q - ci));
            V3 tr;
            bool has_tr = refract_(rd, nn, eta, &tr);
            bool refl_choice = !has_tr ? true : (rnd(rng) < fr);
            if (refl_choice) {
                V3 r = reflect_(rd, nn);
                if (dot16(r, gn) < 0) r = reflect_(r, gn);
                rd = r;
                ro = add3(pos, scale3(gn, 164));
            } else {
                rd = tr;
                ro = sub3(pos, scale3(gn, 164));
                inside = ent;
                sig[0] = q10(m.absorb[0]); sig[1] = q10(m.absorb[1]); sig[2] = q10(m.absorb[2]);
                if (ent) {
                    V3 a = lin8x3(m.alb);
                    thr[0] = mul16(thr[0], a.x); thr[1] = mul16(thr[1], a.y); thr[2] = mul16(thr[2], a.z);
                }
            }
            continue;
        }
        ll ci = clampl(dot16(neg3(rd), n), 0, Q);
        if (m.coat > 0) {
            ll fc = 2621 + mul16(62915, pow5(Q - ci));
            if (rnd(rng) < mul16(fc, (ll)m.coat * 256LL)) {
                V3 r2 = reflect_(rd, ggx_h(n, 983, rng));
                rd = dot16(r2, ng) > 0 ? r2 : reflect_(rd, ng);
                ro = add3(pos, scale3(ng, 131));
                continue;
            }
        }
        ll spec = (ll)m.spec * 256LL;
        ll f0v[3];
        if (m.metal) { f0v[0] = sf.alb.x; f0v[1] = sf.alb.y; f0v[2] = sf.alb.z; }
        else { ll f = mul16(5243, spec); f0v[0] = f; f0v[1] = f; f0v[2] = f; }
        ll pw = pow5(Q - ci);
        ll fre[3];
        for (int i = 0; i < 3; i++) fre[i] = f0v[i] + mul16(Q - f0v[i], pw);
        ll ps = clampl(mul16((fre[0] + fre[1] + fre[2]) / 3, Q - mul16(sf.rough, Q / 2)), 2621, 62259);
        if (m.metal) ps = Q;
        if (rnd(rng) < ps) {
            ll a = mul16(sf.rough, sf.rough);
            V3 rf = reflect_(rd, ggx_h(n, maxl(a, 26), rng));
            if (dot16(rf, ng) <= 0) rf = reflect_(rd, ng);
            rd = rf;
            for (int i = 0; i < 3; i++) {
                if (m.metal) thr[i] = mul16(thr[i], comp(sf.alb, i));
                else thr[i] = (mul16(thr[i], fre[i]) * Q) / ps;
            }
        } else {
            if (m.trans > 0 && rnd(rng) < (ll)m.trans * 256LL) {
                rd = cos_hemi(neg3(n), rng);
                ro = sub3(pos, scale3(ng, BIAS_T));
                for (int i = 0; i < 3; i++) thr[i] = (mul16(thr[i], comp(sf.alb, i)) * Q) / (Q - ps);
                continue;
            }
            rd = cos_hemi(n, rng);
            if (dot16(rd, ng) < 0) rd = reflect_(rd, ng);
            for (int i = 0; i < 3; i++) thr[i] = (mul16(mul16(thr[i], comp(sf.alb, i)), Q - fre[i]) * Q) / (Q - ps);
        }
        ro = add3(pos, scale3(ng, BIAS_G));
        if (b > 2) {
            ll mx = maxl(thr[0], maxl(thr[1], thr[2]));
            ll q = maxl(Q - mx, 3277);
            if (rnd(rng) < q) break;
            for (int i = 0; i < 3; i++) thr[i] = (thr[i] * Q) / (Q - q);
        }
    }
    ll out[3] = {minl(col[0], CLAMP_RAD), minl(col[1], CLAMP_RAD), minl(col[2], CLAMP_RAD)};
    if (t0 > 0) {
        ll k = smoothstep_(P.fog0, P.fog1, t0);
        if (k > 0) {
            V3 bd = backdrop_(P, rd0v);
            for (int i = 0; i < 3; i++) out[i] = mix_(out[i], comp(bd, i), k);
        }
    }
    return mk(out[0], out[1], out[2]);
}

__global__ void render3_kernel(const R3Params* __restrict__ PP, uint32_t first, uint8_t* out) {
    const R3Params& P = *PP;
    const int x = threadIdx.x, y = threadIdx.y;
    const uint32_t chunk = first + blockIdx.x;
    const uint32_t tiles_x = P.width / TILE;
    const uint32_t ntiles = tiles_x * (P.height / TILE);
    const uint32_t tile = chunk % ntiles;
    const ll tx = tile % tiles_x, ty = tile / tiles_x;
    const ll w = P.width, h = P.height;
    V3 pos = mk(P.cam_pos[0], P.cam_pos[1], P.cam_pos[2]);
    V3 fw = norm3(sub3(mk(P.cam_look[0], P.cam_look[1], P.cam_look[2]), pos));
    V3 rt = norm3(cross16(mk(0, Q, 0), fw));
    V3 up = cross16(fw, rt);
    ll spp = P.spp;
    ll px = tx * TILE + x, py = ty * TILE + y;
    ll acc[3] = {0, 0, 0};
    for (ll s = 0; s < spp; s++) {
        Rng rng;
        rng.s = pcg((uint32_t)px * 1973u + (uint32_t)py * 9277u + chunk * 26699u + (uint32_t)s * 612451u + P.seed0);
        rng.s = pcg(rng.s ^ P.seed1);
        ll jx = rnd(rng), jy = rnd(rng);
        ll sx = px * Q + jx, sy = py * Q + jy;
        ll u = (2 * sx - w * Q) / h;
        ll v = -((2 * sy - h * Q) / h);
        ll fo = P.focal;
        V3 rd = norm3(mk(mul16(rt.x, u) + mul16(up.x, v) + mul16(fw.x, fo), mul16(rt.y, u) + mul16(up.y, v) + mul16(fw.y, fo), mul16(rt.z, u) + mul16(up.z, v) + mul16(fw.z, fo)));
        V3 ro = pos, rdd = rd;
        if (P.aperture > 0) {
            ll fd = ((ll)P.focus * 65536LL) / maxl(dot16(rd, fw), 1);
            V3 fp = add3(pos, scale3(rd, fd));
            ll lx, ly;
            disc_(rng, &lx, &ly);
            ll lr = mul16(P.aperture, lx);
            ll lq = mul16(P.aperture, ly);
            ro = add3(pos, add3(scale3(rt, lr), scale3(up, lq)));
            rdd = norm3(sub3(fp, ro));
        }
        V3 c = trace_(P, rng, ro, rdd);
        acc[0] += c.x; acc[1] += c.y; acc[2] += c.z;
    }
    uint8_t* o = out + (size_t)blockIdx.x * (TILE * TILE * 6) + (size_t)((y * TILE + x) * 6);
    for (int k = 0; k < 3; k++) {
        ll mean = acc[k] / spp;
        ll q11 = clampl(mean >> 5, 0, 65535);
        o[k * 2] = (uint8_t)(q11 & 0xFF);
        o[k * 2 + 1] = (uint8_t)((q11 >> 8) & 0xFF);
    }
}

extern "C" int vearl_gpu_render3(const R3Params* params, uint32_t first_chunk, uint32_t chunks, uint8_t* out_host) {
    if (chunks == 0) return 0;
    R3Params* d_p = nullptr;
    uint8_t* d_out = nullptr;
    if (cudaMalloc(&d_p, sizeof(R3Params)) != cudaSuccess) return 1;
    if (cudaMalloc(&d_out, (size_t)chunks * TILE * TILE * 6) != cudaSuccess) { cudaFree(d_p); return 2; }
    cudaMemcpy(d_p, params, sizeof(R3Params), cudaMemcpyHostToDevice);
    cudaDeviceSetLimit(cudaLimitStackSize, 8192);
    dim3 block(TILE, TILE), grid(chunks);
    render3_kernel<<<grid, block>>>(d_p, first_chunk, d_out);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    int rc = 0;
    if (e != cudaSuccess) {
        fprintf(stderr, "render3: %s\n", cudaGetErrorString(e));
        rc = 3;
    } else {
        cudaMemcpy(out_host, d_out, (size_t)chunks * TILE * TILE * 6, cudaMemcpyDeviceToHost);
    }
    cudaFree(d_p);
    cudaFree(d_out);
    return rc;
}
