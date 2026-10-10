// Render2 on the GPU: a line-by-line port of src/render2.rs (integer arithmetic only), so that every 16x16 tile is bit-identical to the CPU
// (and to tools/l2_ref.py). One CUDA block = one tile = one chunk; one thread = one pixel. Division truncates toward zero and >> on signed
// values is arithmetic, exactly as in Rust; `x << 16` of a possibly negative value is written as a multiplication.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

typedef long long ll;

#define Q 65536LL
#define TILE 16
#define MARCH 64
#define SOFT_STEPS 32
#define AO_N 5
#define HIT_EPS 64LL
#define T_MAX (40LL * 65536LL)
#define FAR_HIT 8192LL
#define SOFT_T0 2621LL
#define SOFT_T_MAX (14LL * 65536LL)
#define NORMAL_H 256LL
#define BIAS 2621LL
#define GLOW_R (3LL * 65536LL / 5LL)
#define EMIT_LIGHT_R (5LL * 65536LL / 2LL)
#define EMIT_LIGHT_GAIN 3LL
#define AO_STEP 5243LL
#define DIRECT_GAIN 2LL
#define SPEC_GAIN 2LL
#define AMB_GAIN 2LL
#define VIGNETTE 22000LL
#define ACES_A 164495LL
#define ACES_B 1966LL
#define ACES_C 159252LL
#define ACES_D 38666LL
#define ACES_E 9175LL

struct R2Prim {
    int32_t kind;   // 0 sphere (c, r = h[0]), 1 box (c, h, rnd), 2 torus (c, big = h[0], small = h[1]), 3 plane (y = c[1]),
                    // 4 cylinder (c, r = h[0], half height = h[1], rnd), 5 capsule (a = c, b = h, radius = rnd)
    int32_t c[3];
    int32_t h[3];
    int32_t rnd;
    uint8_t col[4];
    uint8_t col2[4];
    uint8_t refl, emit, checker, pad;
};

struct R2Params {
    int32_t width, height, spp, bounces;
    int32_t cam[3], target[3], fov, light[3];
    uint8_t light_col[4], sky_top[4], sky_hor[4];
    int32_t fog, smooth_k, exposure, nprims;
    R2Prim prims[32];
};

struct V3 { ll x, y, z; };

__device__ __forceinline__ V3 mk(ll x, ll y, ll z) { V3 v; v.x = x; v.y = y; v.z = z; return v; }
__device__ __forceinline__ ll mul16(ll a, ll b) { return (a * b) >> 16; }
__device__ __forceinline__ ll clampl(ll v, ll lo, ll hi) { return v < lo ? lo : (v > hi ? hi : v); }
__device__ __forceinline__ ll minl(ll a, ll b) { return a < b ? a : b; }
__device__ __forceinline__ ll maxl(ll a, ll b) { return a > b ? a : b; }
__device__ __forceinline__ ll absl(ll a) { return a < 0 ? -a : a; }

__device__ __forceinline__ ll isqrt_ll(ll x) {
    if (x < 2) return x;
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
__device__ __forceinline__ V3 at(V3 o, V3 d, ll t) { return mk(o.x + mul16(d.x, t), o.y + mul16(d.y, t), o.z + mul16(d.z, t)); }
__device__ __forceinline__ V3 addv(V3 a, V3 b) { return mk(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ ll lin1(ll c) { return ((c * c) * 65536LL) / 65025LL; }
__device__ __forceinline__ V3 lin3(const uint8_t* c) { return mk(lin1(c[0]), lin1(c[1]), lin1(c[2])); }
__device__ __forceinline__ ll pow_sq(ll x, int n) { for (int i = 0; i < n; i++) x = mul16(x, x); return x; }
__device__ __forceinline__ ll comp(V3 v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }

__device__ ll shape_dist(const R2Prim& s, V3 p) {
    if (s.kind == 0) {
        return len3(mk(p.x - s.c[0], p.y - s.c[1], p.z - s.c[2])) - s.h[0];
    } else if (s.kind == 3) {
        return p.y - s.c[1];
    } else if (s.kind == 1) {
        ll q0 = absl(p.x - s.c[0]) - s.h[0], q1 = absl(p.y - s.c[1]) - s.h[1], q2 = absl(p.z - s.c[2]) - s.h[2];
        ll outside = len3(mk(maxl(q0, 0), maxl(q1, 0), maxl(q2, 0)));
        return outside + minl(maxl(maxl(q0, q1), q2), 0) - s.rnd;
    } else if (s.kind == 2) {
        ll q0 = p.x - s.c[0], q1 = p.y - s.c[1], q2 = p.z - s.c[2];
        ll l = len2(q0, q2) - s.h[0];
        return len2(l, q1) - s.h[1];
    } else if (s.kind == 4) {
        ll q0 = p.x - s.c[0], q1 = p.y - s.c[1], q2 = p.z - s.c[2];
        ll dx = len2(q0, q2) - s.h[0];
        ll dy = absl(q1) - s.h[1];
        return len2(maxl(dx, 0), maxl(dy, 0)) + minl(maxl(dx, dy), 0) - s.rnd;
    } else {
        V3 pa = mk(p.x - s.c[0], p.y - s.c[1], p.z - s.c[2]);
        V3 ba = mk((ll)s.h[0] - s.c[0], (ll)s.h[1] - s.c[1], (ll)s.h[2] - s.c[2]);
        ll tq = clampl((dot16(pa, ba) * 65536LL) / dot16(ba, ba), 0, Q);
        V3 q = mk(pa.x - mul16(ba.x, tq), pa.y - mul16(ba.y, tq), pa.z - mul16(ba.z, tq));
        return len3(q) - s.rnd;
    }
}

__device__ __forceinline__ ll smin(ll a, ll b, ll k) {
    if (k <= 0) return minl(a, b);
    ll h = maxl(k - absl(a - b), 0);
    ll hq = (h * 65536LL) / k;
    return minl(a, b) - ((mul16(hq, hq) * k) >> 18);
}

struct Eval { ll d; int idx; ll emin; int eidx; ll gmin; int gidx; };

#define BIGL 0x7FFFFFFFFFFFFFFFLL

__device__ Eval eval(const R2Params& P, V3 p) {
    ll blend = BIGL, plane = BIGL, best = BIGL, emin = BIGL, gmin = BIGL;
    int idx = 0, eidx = 0, gidx = 0;
    bool first = true;
    for (int i = 0; i < P.nprims; i++) {
        const R2Prim& pr = P.prims[i];
        ll d = shape_dist(pr, p);
        if (d < best) { best = d; idx = i; }
        if (pr.emit > 0) {
            if (d < emin) { emin = d; eidx = i; }
            if (pr.kind != 0 && d < gmin) { gmin = d; gidx = i; }
        }
        if (pr.kind == 3) {
            plane = minl(plane, d);
        } else if (first) {
            blend = d;
            first = false;
        } else {
            blend = smin(blend, d, P.smooth_k);
        }
    }
    Eval e; e.d = minl(blend, plane); e.idx = idx; e.emin = emin; e.eidx = eidx; e.gmin = gmin; e.gidx = gidx;
    return e;
}

struct Ctx {
    V3 light, light_lin, top_lin, hor_lin;
};

__device__ __forceinline__ unsigned star_hash(unsigned x, unsigned y, unsigned z) {
    unsigned h = (x * 0x9E3779B1u) ^ (y * 0x85EBCA77u) ^ (z * 0xC2B2AE3Du);
    h ^= h >> 15;
    h = h * 0x2C1B3C6Du;
    return h ^ (h >> 12);
}

__device__ V3 sky(const Ctx& c, V3 d) {
    ll e = clampl(5 * d.y, 0, Q);
    ll m = mul16(e, 2 * Q - e);
    ll s = maxl(dot16(d, c.light), 0);
    ll s64 = pow_sq(s, 6);
    ll s256 = pow_sq(s64, 2);
    ll sun = (s64 >> 2) + (s256 << 2);
    ll star = 0;
    if (d.y > 0) {
        unsigned h = star_hash((unsigned)(d.x >> 6), (unsigned)(d.y >> 6), (unsigned)(d.z >> 6));
        if ((h & 0x1FFFu) == 0) star = mul16((ll)((h >> 13) & 0xFFu) << 8, Q - m) * 2;
    }
    return mk(c.hor_lin.x + mul16(c.top_lin.x - c.hor_lin.x, m) + mul16(c.light_lin.x, sun) + star,
              c.hor_lin.y + mul16(c.top_lin.y - c.hor_lin.y, m) + mul16(c.light_lin.y, sun) + star,
              c.hor_lin.z + mul16(c.top_lin.z - c.hor_lin.z, m) + mul16(c.light_lin.z, sun) + star);
}

__device__ V3 normal_at(const R2Params& P, V3 pos) {
    ll h = NORMAL_H;
    ll ex = eval(P, mk(pos.x + h, pos.y, pos.z)).d - eval(P, mk(pos.x - h, pos.y, pos.z)).d;
    ll ey = eval(P, mk(pos.x, pos.y + h, pos.z)).d - eval(P, mk(pos.x, pos.y - h, pos.z)).d;
    ll ez = eval(P, mk(pos.x, pos.y, pos.z + h)).d - eval(P, mk(pos.x, pos.y, pos.z - h)).d;
    return norm3(mk(ex, ey, ez));
}

__device__ ll soft_shadow(const R2Params& P, const Ctx& c, V3 o) {
    ll res = Q, t = SOFT_T0;
    for (int i = 0; i < SOFT_STEPS; i++) {
        ll d = eval(P, at(o, c.light, t)).d;
        if (d < HIT_EPS) { res = 0; break; }
        ll w = ((8 * d) * 65536LL) / t;
        if (w < res) res = w;
        t += clampl(d, 1311, 32768);
        if (t > SOFT_T_MAX) break;
    }
    ll r = clampl(res, 0, Q);
    return mul16(r, mul16(r, 3 * Q - 2 * r));
}

__device__ ll occlusion(const R2Params& P, V3 pos, V3 n) {
    const ll W[5] = {65536, 62259, 59146, 56188, 53379};
    ll occ = 0;
    for (int i = 0; i < AO_N; i++) {
        ll hr = (i + 1) * AO_STEP;
        V3 q = mk(pos.x + mul16(n.x, hr), pos.y + mul16(n.y, hr), pos.z + mul16(n.z, hr));
        ll dd = eval(P, q).d;
        occ += (maxl(hr - dd, 0) * W[i]) >> 16;
    }
    return clampl(Q - (occ + (occ >> 1)), 0, Q);
}

__device__ V3 surface(const R2Params& P, const Ctx& c, V3 pos, int idx, V3 d, ll t, V3* nout) {
    const R2Prim& pr = P.prims[idx];
    V3 n = normal_at(P, pos);
    V3 base;
    if (pr.checker && pr.kind == 3) {
        base = ((((pos.x >> 16) + (pos.z >> 16)) & 1) == 0) ? lin3(pr.col) : lin3(pr.col2);
    } else {
        base = lin3(pr.col);
    }
    ll ndl = maxl(dot16(n, c.light), 0);
    V3 o = mk(pos.x + mul16(n.x, BIAS), pos.y + mul16(n.y, BIAS), pos.z + mul16(n.z, BIAS));
    ll shadow = soft_shadow(P, c, o);
    ll ao = occlusion(P, pos, n);
    ll hemi = clampl((n.y + Q) >> 1, 0, Q);
    V3 half = norm3(mk(c.light.x - d.x, c.light.y - d.y, c.light.z - d.z));
    ll spec = pow_sq(maxl(dot16(n, half), 0), 5);
    ll emit = (ll)pr.emit * 512;
    ll fogf;
    {
        ll f = clampl(mul16(t, P.fog), 0, Q);
        fogf = mul16(f, f);
    }
    V3 skyc = mk(0, 0, 0);
    if (fogf > 0) skyc = sky(c, d);
    Eval nearE = eval(P, pos);
    ll ew = 0;
    V3 ecol = mk(0, 0, 0);
    if (nearE.emin < EMIT_LIGHT_R) {
        ll w = ((EMIT_LIGHT_R - maxl(nearE.emin, 0)) * 65536LL) / EMIT_LIGHT_R;
        ew = mul16(w, w);
        ecol = lin3(P.prims[nearE.eidx].col);
    }
    ll out[3];
    for (int i = 0; i < 3; i++) {
        ll hor = comp(c.hor_lin, i), top = comp(c.top_lin, i), ll_ = comp(c.light_lin, i);
        ll amb = mul16(hor + mul16(top - hor, hemi), ao) * AMB_GAIN;
        ll direct = mul16(ll_, mul16(ndl, shadow)) * DIRECT_GAIN;
        ll spec_c = mul16(ll_, mul16(spec, shadow)) * SPEC_GAIN;
        ll b = comp(base, i);
        ll col = mul16(b, direct + amb + mul16(comp(ecol, i), ew) * EMIT_LIGHT_GAIN) + spec_c + mul16(b, emit);
        out[i] = col + mul16(comp(skyc, i) - col, fogf);
    }
    *nout = n;
    return mk(out[0], out[1], out[2]);
}

__device__ V3 trace(const R2Params& P, const Ctx& c, V3 o, V3 d) {
    ll acc[3] = {0, 0, 0};
    ll thr = Q;
    for (int bounce = 0; bounce <= P.bounces; bounce++) {
        ll t = 0;
        bool hit = false, have_last = false;
        V3 hpos = mk(0, 0, 0), lpos = mk(0, 0, 0);
        int hidx = 0, lidx = 0;
        ll ldist = 0, lt = 0;
        ll emin = BIGL;
        int eidx = 0;
        for (int s = 0; s < MARCH; s++) {
            V3 pos = at(o, d, t);
            Eval e = eval(P, pos);
            if (bounce == 0 && e.gmin < emin) { emin = e.gmin; eidx = e.gidx; }
            if (e.d < HIT_EPS) { hit = true; hpos = pos; hidx = e.idx; break; }
            have_last = true; lpos = pos; lidx = e.idx; ldist = e.d; lt = t;
            t += e.d;
            if (t > T_MAX) { have_last = false; break; }
        }
        if (!hit && have_last && ldist < FAR_HIT) { hit = true; hpos = lpos; hidx = lidx; t = lt; }
        if (bounce == 0) {
            if (emin < GLOW_R) {
                ll g = ((GLOW_R - maxl(emin, 0)) * 65536LL) / GLOW_R;
                g = mul16(g, g);
                V3 ec = lin3(P.prims[eidx].col);
                acc[0] += mul16(ec.x, g) >> 1;
                acc[1] += mul16(ec.y, g) >> 1;
                acc[2] += mul16(ec.z, g) >> 1;
            }
            ll t_end = hit ? t : T_MAX;
            for (int i = 0; i < P.nprims; i++) {
                const R2Prim& pr = P.prims[i];
                if (pr.kind == 0 && pr.emit > 0) {
                    V3 oc = mk(pr.c[0] - o.x, pr.c[1] - o.y, pr.c[2] - o.z);
                    ll tc = clampl(dot16(oc, d), 0, t_end);
                    V3 q = at(o, d, tc);
                    ll dist = len3(mk(q.x - pr.c[0], q.y - pr.c[1], q.z - pr.c[2])) - pr.h[0];
                    if (dist < GLOW_R) {
                        ll g = ((GLOW_R - maxl(dist, 0)) * 65536LL) / GLOW_R;
                        g = mul16(g, g);
                        V3 ec = lin3(pr.col);
                        acc[0] += mul16(ec.x, g) >> 1;
                        acc[1] += mul16(ec.y, g) >> 1;
                        acc[2] += mul16(ec.z, g) >> 1;
                    }
                }
            }
        }
        if (!hit) {
            V3 sk = sky(c, d);
            acc[0] += mul16(thr, sk.x);
            acc[1] += mul16(thr, sk.y);
            acc[2] += mul16(thr, sk.z);
            break;
        }
        V3 n;
        V3 col = surface(P, c, hpos, hidx, d, t, &n);
        ll refl = (ll)P.prims[hidx].refl * 257;
        if (bounce == P.bounces || refl == 0) {
            acc[0] += mul16(thr, col.x);
            acc[1] += mul16(thr, col.y);
            acc[2] += mul16(thr, col.z);
            break;
        }
        ll cosv = clampl(dot16(mk(-d.x, -d.y, -d.z), n), 0, Q);
        ll x = Q - cosv;
        ll x2 = mul16(x, x);
        ll fr = mul16(mul16(x2, x2), x);
        ll f = minl(refl + mul16(Q - refl, fr), Q);
        acc[0] += mul16(thr, mul16(col.x, Q - f));
        acc[1] += mul16(thr, mul16(col.y, Q - f));
        acc[2] += mul16(thr, mul16(col.z, Q - f));
        thr = mul16(thr, f);
        ll dn = dot16(d, n);
        d = norm3(mk(d.x - 2 * mul16(dn, n.x), d.y - 2 * mul16(dn, n.y), d.z - 2 * mul16(dn, n.z)));
        o = mk(hpos.x + mul16(n.x, BIAS), hpos.y + mul16(n.y, BIAS), hpos.z + mul16(n.z, BIAS));
    }
    return mk(acc[0], acc[1], acc[2]);
}

__device__ __forceinline__ ll aces(ll c) {
    c = maxl(c, 0);
    ll num = mul16(c, mul16(ACES_A, c) + ACES_B);
    ll den = mul16(c, mul16(ACES_C, c) + ACES_D) + ACES_E;
    return clampl((num * 65536LL) / den, 0, Q);
}

// grid: one block per tile of this launch; `first` is the chunk index of block 0. jitter: 256 * spp * 4 bytes per block (only when spp > 1).
__global__ void render2_kernel(const R2Params P, uint32_t first, const uint8_t* jitter, uint8_t* out) {
    const int x = threadIdx.x, y = threadIdx.y;
    const uint32_t chunk = first + blockIdx.x;
    const int tiles_x = P.width / TILE;
    const int tx = chunk % tiles_x, ty = chunk / tiles_x;
    const int spp = P.spp;
    const ll w = P.width, h = P.height;

    Ctx c;
    c.light = norm3(mk(P.light[0], P.light[1], P.light[2]));
    c.light_lin = lin3(P.light_col);
    c.top_lin = lin3(P.sky_top);
    c.hor_lin = lin3(P.sky_hor);
    V3 cam = mk(P.cam[0], P.cam[1], P.cam[2]);
    V3 fwd = norm3(mk(P.target[0] - cam.x, P.target[1] - cam.y, P.target[2] - cam.z));
    V3 right = norm3(cross16(mk(0, Q, 0), fwd));
    V3 up = norm3(cross16(fwd, right));
    ll f = P.fov;

    ll acc[3] = {0, 0, 0};
    for (int s = 0; s < spp; s++) {
        ll jx = 32768, jy = 32768;
        if (spp > 1) {
            const uint8_t* j = jitter + (size_t)blockIdx.x * (TILE * TILE * spp * 4) + (size_t)(((y * TILE + x) * spp + s) * 4);
            jx = (ll)j[0] | ((ll)j[1] << 8);
            jy = (ll)j[2] | ((ll)j[3] << 8);
        }
        ll sx = ((ll)(tx * TILE + x)) * Q + jx, sy = ((ll)(ty * TILE + y)) * Q + jy;
        ll u = (2 * sx - w * Q) / w;
        ll v = -((2 * sy - h * Q) / w);
        V3 draw = mk(mul16(fwd.x, f) + mul16(right.x, u) + mul16(up.x, v), mul16(fwd.y, f) + mul16(right.y, u) + mul16(up.y, v), mul16(fwd.z, f) + mul16(right.z, u) + mul16(up.z, v));
        V3 col = trace(P, c, cam, norm3(draw));
        acc[0] += col.x; acc[1] += col.y; acc[2] += col.z;
    }
    ll cx = ((ll)(tx * TILE + x)) * Q + Q / 2, cy = ((ll)(ty * TILE + y)) * Q + Q / 2;
    ll nx = (2 * cx - w * Q) / w, ny = (2 * cy - h * Q) / w;
    ll vig = maxl(Q - mul16(mul16(nx, nx) + mul16(ny, ny), VIGNETTE), 16384);
    uint8_t* o = out + (size_t)blockIdx.x * (TILE * TILE * 3) + (size_t)((y * TILE + x) * 3);
    for (int k = 0; k < 3; k++) {
        ll mean = acc[k] / spp;
        ll cc = mul16(mul16(mean, P.exposure), vig);
        ll g = isqrt_ll((aces(cc) * 65025LL) >> 16);
        o[k] = (uint8_t)(g > 255 ? 255 : g);
    }
}

extern "C" int vearl_gpu_render2(const R2Params* params, uint32_t first_chunk, uint32_t chunks, const uint8_t* jitter, uint8_t* out_host) {
    if (chunks == 0) return 0;
    const int spp = params->spp;
    uint8_t *d_out = nullptr, *d_jit = nullptr;
    cudaError_t e;
    if ((e = cudaMalloc(&d_out, (size_t)chunks * TILE * TILE * 3)) != cudaSuccess) return 1;
    if (spp > 1) {
        size_t jb = (size_t)chunks * TILE * TILE * spp * 4;
        if ((e = cudaMalloc(&d_jit, jb)) != cudaSuccess) { cudaFree(d_out); return 2; }
        cudaMemcpy(d_jit, jitter, jb, cudaMemcpyHostToDevice);
    }
    dim3 block(TILE, TILE), grid(chunks);
    render2_kernel<<<grid, block>>>(*params, first_chunk, d_jit, d_out);
    e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    int rc = 0;
    if (e != cudaSuccess) {
        fprintf(stderr, "render2: %s\n", cudaGetErrorString(e));
        rc = 3;
    } else {
        cudaMemcpy(out_host, d_out, (size_t)chunks * TILE * TILE * 3, cudaMemcpyDeviceToHost);
    }
    cudaFree(d_out);
    if (d_jit) cudaFree(d_jit);
    return rc;
}
