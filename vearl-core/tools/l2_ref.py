#!/usr/bin/env python3
"""INDEPENDENT implementation of the level 2 kernels (Monte Carlo: integrand machine; rendering: ray tracing through signed
distance fields), written from docs/10-level-2.md, in Python integers (no floats). Verified against the vectors of
`vearl-l2 vectors`: the chunk outputs must be IDENTICAL BYTE FOR BYTE to those of the Rust node (this is what allows
any node to judge a fraud proof).

    ../target/release/vearl-l2 vectors > l2vec.json && python3 l2_ref.py l2vec.json      # pip install blake3
"""
import json, math, struct, sys
import blake3

ONE = 1 << 32
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1
Q = 1 << 16
MARCH_STEPS, SHADOW_STEPS, HIT_EPS, T_MAX, TILE = 48, 24, 64, 20 * Q, 16


def xof(context, parts, length):
    h = blake3.blake3(derive_key_context=context)
    for p in parts:
        h.update(p)
    return h.digest(length=length)


def sat(x):
    return max(I64_MIN, min(I64_MAX, x))


# ---------------- Monte Carlo ----------------
def mc_chunk(p, chunk):
    d, s = p["dim"], p["samples"]
    buf = xof("vearl/l2/mc/v1", [bytes(p["seed"]), struct.pack("<I", chunk)], s * d * 4)
    inp = struct.unpack("<%dI" % (s * d), buf)
    total = sq = 0
    for k in range(s):
        r = [0] * 16
        for ins in p["program"]:
            op, a, b, c, imm = ins["op"], ins["a"], ins["b"], ins["c"], ins["imm"]
            if op == 0:
                r[a] = imm
            elif op == 1:
                r[a] = inp[k * d + b]
            elif op == 2:
                r[a] = sat(r[b] + r[c])
            elif op == 3:
                r[a] = sat(r[b] - r[c])
            elif op == 4:
                r[a] = sat((r[b] * r[c]) >> 32)
            elif op == 5:
                r[a] = min(r[b], r[c])
            elif op == 6:
                r[a] = max(r[b], r[c])
            elif op == 7:
                r[a] = sat(abs(r[b]))
            elif op == 8:
                r[a] = sat(-r[b])
            elif op == 9:
                r[a] = sat(math.isqrt(max(r[b], 0) << 32))
            elif op == 10:
                r[a] = ONE if r[b] < r[c] else 0
            elif op == 11:
                r[a] = r[c] if r[b] > 0 else r[imm & 15]
            else:
                raise ValueError(op)
        v = r[p["out"]]
        total += v
        sq += (v * v) >> 32
    return total.to_bytes(16, "little", signed=True) + sq.to_bytes(16, "little", signed=True)


# ---------------- rendering ----------------
def tdiv(a, b):
    """Integer division truncated toward zero (like `/` in Rust)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def mul16(a, b):
    return (a * b) >> 16


def len3(v):
    return math.isqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def norm3(v):
    l = len3(v)
    if l == 0:
        return [0, Q, 0]
    return [tdiv(v[0] << 16, l), tdiv(v[1] << 16, l), tdiv(v[2] << 16, l)]


def prim_dist(pr, p):
    ((kind, q),) = pr.items()
    if kind == "Sphere":
        c = q["c"]
        return len3([p[0] - c[0], p[1] - c[1], p[2] - c[2]]) - q["r"]
    if kind == "Plane":
        return p[1] - q["y"]
    c, h = q["c"], q["h"]
    qq = [abs(p[0] - c[0]) - h[0], abs(p[1] - c[1]) - h[1], abs(p[2] - c[2]) - h[2]]
    outside = len3([max(qq[0], 0), max(qq[1], 0), max(qq[2], 0)])
    return outside + min(max(qq[0], qq[1], qq[2]), 0)


def sdf(prims, p):
    best = (prim_dist(prims[0], p), 0)
    for k in range(1, len(prims)):
        d = prim_dist(prims[k], p)
        if d < best[0]:
            best = (d, k)
    return best


def at(o, d, t):
    return [o[0] + mul16(d[0], t), o[1] + mul16(d[1], t), o[2] + mul16(d[2], t)]


def shade(p, o, d, light):
    prims = p["prims"]
    t, hit = 0, None
    for _ in range(MARCH_STEPS):
        pos = at(o, d, t)
        dist, k = sdf(prims, pos)
        if dist < HIT_EPS:
            hit = (pos, k)
            break
        t += dist
        if t > T_MAX:
            break
    if hit is None:
        sky = [90, 120, 180]
        return [20 + ((sky[i] * (d[1] + Q)) >> 17) for i in range(3)]
    pos, k = hit
    H = 256
    n = norm3([sdf(prims, [pos[0] + H, pos[1], pos[2]])[0] - sdf(prims, [pos[0] - H, pos[1], pos[2]])[0],
               sdf(prims, [pos[0], pos[1] + H, pos[2]])[0] - sdf(prims, [pos[0], pos[1] - H, pos[2]])[0],
               sdf(prims, [pos[0], pos[1], pos[2] + H])[0] - sdf(prims, [pos[0], pos[1], pos[2] - H])[0]])
    diffuse = max((n[0] * light[0] + n[1] * light[1] + n[2] * light[2]) >> 16, 0)
    ps = [pos[i] + mul16(n[i], 1310) for i in range(3)]
    ts, lit = 0, True
    for _ in range(SHADOW_STEPS):
        dist, _ = sdf(prims, at(ps, light, ts))
        if dist < HIT_EPS:
            lit = False
            break
        ts += dist
        if ts > T_MAX // 2:
            break
    sh = Q if lit else Q // 4
    inten = min(13107 + mul16(52429, mul16(diffuse, sh)), Q)
    ((kind, q),) = prims[k].items()
    if kind == "Plane":
        base = q["col"] if (((pos[0] >> 16) + (pos[2] >> 16)) & 1) == 0 else q["col2"]
    else:
        base = q["col"]
    return [(base[i] * inten) >> 16 for i in range(3)]


def render_chunk(p, chunk):
    w, h, spp = p["width"], p["height"], p["spp"]
    tiles_x = w // TILE
    tx, ty = chunk % tiles_x, chunk // tiles_x
    jitter = xof("vearl/l2/render/v1", [bytes(p["seed"]), struct.pack("<I", chunk)], TILE * TILE * spp * 4) if spp > 1 else b""
    light = norm3(list(p["light"]))
    o = [0, 0, -p["cam_z"]]
    out = bytearray()
    for y in range(TILE):
        for x in range(TILE):
            acc = [0, 0, 0]
            for s in range(spp):
                if spp > 1:
                    k = ((y * TILE + x) * spp + s) * 4
                    jx, jy = struct.unpack("<HH", jitter[k:k + 4])
                else:
                    jx = jy = 32768
                sx, sy = (tx * TILE + x) * Q + jx, (ty * TILE + y) * Q + jy
                u = tdiv(2 * sx - w * Q, w)
                v = -tdiv(2 * sy - h * Q, w)
                d = norm3([u, v, Q])
                c = shade(p, o, d, light)
                for i in range(3):
                    acc[i] += c[i]
            out += bytes(acc[i] // spp for i in range(3))
    return bytes(out)


# ---------------- Render2 renderer: docs/10-level-2.md section 4 bis ----------------
R2_MARCH, R2_SOFT, R2_AO_N = 64, 32, 5
R2_HIT_EPS, R2_T_MAX, R2_FAR_HIT = 64, 40 * Q, 8192
R2_SOFT_T0, R2_SOFT_T_MAX, R2_NORMAL_H, R2_BIAS = 2621, 14 * Q, 256, 2621
R2_GLOW_R, R2_EMIT_R, R2_EMIT_GAIN = 3 * Q // 5, 5 * Q // 2, 3
R2_AO_W = [65536, 62259, 59146, 56188, 53379]
R2_AO_STEP, R2_DIRECT_GAIN, R2_SPEC_GAIN, R2_AMB_GAIN, R2_VIGNETTE = 5243, 2, 2, 2, 22000
R2_ACES = (164495, 1966, 159252, 38666, 9175)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def r2_lin(c):
    return ((c * c) << 16) // 65025


def r2_lin3(c):
    return [r2_lin(c[0]), r2_lin(c[1]), r2_lin(c[2])]


def r2_dot16(a, b):
    return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) >> 16


def r2_cross16(a, b):
    return [(a[1] * b[2] - a[2] * b[1]) >> 16, (a[2] * b[0] - a[0] * b[2]) >> 16, (a[0] * b[1] - a[1] * b[0]) >> 16]


def r2_pow_sq(x, n):
    for _ in range(n):
        x = mul16(x, x)
    return x


def r2_shape_dist(shape, p):
    ((kind, q),) = shape.items()
    if kind == "Sphere":
        c = q["c"]
        return len3([p[0] - c[0], p[1] - c[1], p[2] - c[2]]) - q["r"]
    if kind == "Plane":
        return p[1] - q["y"]
    if kind == "Box":
        c, h = q["c"], q["h"]
        qq = [abs(p[0] - c[0]) - h[0], abs(p[1] - c[1]) - h[1], abs(p[2] - c[2]) - h[2]]
        outside = len3([max(qq[0], 0), max(qq[1], 0), max(qq[2], 0)])
        return outside + min(max(qq[0], qq[1], qq[2]), 0) - q["rnd"]
    if kind == "Torus":
        c = q["c"]
        qq = [p[0] - c[0], p[1] - c[1], p[2] - c[2]]
        l = math.isqrt(qq[0] * qq[0] + qq[2] * qq[2]) - q["big"]
        return math.isqrt(l * l + qq[1] * qq[1]) - q["small"]
    if kind == "Cylinder":
        c = q["c"]
        qq = [p[0] - c[0], p[1] - c[1], p[2] - c[2]]
        dx = math.isqrt(qq[0] * qq[0] + qq[2] * qq[2]) - q["r"]
        dy = abs(qq[1]) - q["h"]
        a0, b0 = max(dx, 0), max(dy, 0)
        return math.isqrt(a0 * a0 + b0 * b0) + min(max(dx, dy), 0) - q["rnd"]
    a, b = q["a"], q["b"]  # Capsule
    pa = [p[0] - a[0], p[1] - a[1], p[2] - a[2]]
    ba = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
    tq = clamp(tdiv(r2_dot16(pa, ba) << 16, r2_dot16(ba, ba)), 0, Q)
    qv = [pa[i] - mul16(ba[i], tq) for i in range(3)]
    return len3(qv) - q["r"]


def r2_smin(a, b, k):
    if k <= 0:
        return min(a, b)
    h = max(k - abs(a - b), 0)
    hq = (h << 16) // k
    return min(a, b) - ((mul16(hq, hq) * k) >> 18)


def r2_eval(prims, k, p):
    """(distance, nearest shape index, nearest emissive distance, its index, nearest NON-sphere emissive distance, its index)"""
    BIG = 1 << 62
    blend, plane, first = BIG, BIG, True
    best, idx, emin, eidx, gmin, gidx = BIG, 0, BIG, 0, BIG, 0
    for i, pr in enumerate(prims):
        d = r2_shape_dist(pr["shape"], p)
        if d < best:
            best, idx = d, i
        if pr["mat"]["emit"] > 0:
            if d < emin:
                emin, eidx = d, i
            if "Sphere" not in pr["shape"] and d < gmin:
                gmin, gidx = d, i
        if "Plane" in pr["shape"]:
            plane = min(plane, d)
        elif first:
            blend, first = d, False
        else:
            blend = r2_smin(blend, d, k)
    return min(blend, plane), idx, emin, eidx, gmin, gidx


def r2_star_hash(x, y, z):
    M = 0xFFFFFFFF
    h = ((x * 0x9E3779B1) & M) ^ ((y * 0x85EBCA77) & M) ^ ((z * 0xC2B2AE3D) & M)
    h ^= h >> 15
    h = (h * 0x2C1B3C6D) & M
    return h ^ (h >> 12)


def r2_aces(c):
    a, b, c2, d, e = R2_ACES
    c = max(c, 0)
    num = mul16(c, mul16(a, c) + b)
    den = mul16(c, mul16(c2, c) + d) + e
    return clamp((num << 16) // den, 0, Q)


class R2:
    def __init__(self, p):
        self.p, self.prims, self.k = p, p["prims"], p["smooth_k"]
        self.light = norm3(list(p["light"]))
        self.light_lin, self.top_lin, self.hor_lin = r2_lin3(p["light_col"]), r2_lin3(p["sky_top"]), r2_lin3(p["sky_hor"])

    def ev(self, pos):
        return r2_eval(self.prims, self.k, pos)

    def sky(self, d):
        e = clamp(5 * d[1], 0, Q)
        m = mul16(e, 2 * Q - e)
        s = max(r2_dot16(d, self.light), 0)
        s64 = r2_pow_sq(s, 6)
        s256 = r2_pow_sq(s64, 2)
        sun = (s64 >> 2) + (s256 << 2)
        star = 0
        if d[1] > 0:
            h = r2_star_hash((d[0] >> 6) & 0xFFFFFFFF, (d[1] >> 6) & 0xFFFFFFFF, (d[2] >> 6) & 0xFFFFFFFF)
            if h & 0x1FFF == 0:
                star = mul16(((h >> 13) & 0xFF) << 8, Q - m) * 2
        return [self.hor_lin[i] + mul16(self.top_lin[i] - self.hor_lin[i], m) + mul16(self.light_lin[i], sun) + star for i in range(3)]

    def normal(self, pos):
        H = R2_NORMAL_H
        e = lambda dx, dy, dz: self.ev([pos[0] + dx, pos[1] + dy, pos[2] + dz])[0]
        return norm3([e(H, 0, 0) - e(-H, 0, 0), e(0, H, 0) - e(0, -H, 0), e(0, 0, H) - e(0, 0, -H)])

    def soft_shadow(self, o):
        res, t = Q, R2_SOFT_T0
        for _ in range(R2_SOFT):
            d = self.ev(at(o, self.light, t))[0]
            if d < R2_HIT_EPS:
                res = 0
                break
            w = ((8 * d) << 16) // t
            if w < res:
                res = w
            t += clamp(d, 1311, 32768)
            if t > R2_SOFT_T_MAX:
                break
        r = clamp(res, 0, Q)
        return mul16(r, mul16(r, 3 * Q - 2 * r))

    def occlusion(self, pos, n):
        occ = 0
        for i in range(R2_AO_N):
            hr = (i + 1) * R2_AO_STEP
            q = [pos[0] + mul16(n[0], hr), pos[1] + mul16(n[1], hr), pos[2] + mul16(n[2], hr)]
            dd = self.ev(q)[0]
            occ += (max(hr - dd, 0) * R2_AO_W[i]) >> 16
        return clamp(Q - (occ + (occ >> 1)), 0, Q)

    def surface(self, pos, idx, d, t):
        pr = self.prims[idx]
        mat = pr["mat"]
        n = self.normal(pos)
        if mat["checker"] and "Plane" in pr["shape"]:
            base = r2_lin3(mat["col"] if (((pos[0] >> 16) + (pos[2] >> 16)) & 1) == 0 else mat["col2"])
        else:
            base = r2_lin3(mat["col"])
        ndl = max(r2_dot16(n, self.light), 0)
        o = [pos[i] + mul16(n[i], R2_BIAS) for i in range(3)]
        shadow = self.soft_shadow(o)
        ao = self.occlusion(pos, n)
        hemi = clamp((n[1] + Q) >> 1, 0, Q)
        half = norm3([self.light[i] - d[i] for i in range(3)])
        spec = r2_pow_sq(max(r2_dot16(n, half), 0), 5)
        emit = mat["emit"] * 512
        f = clamp(mul16(t, self.p["fog"]), 0, Q)
        fogf = mul16(f, f)
        skyc = self.sky(d) if fogf > 0 else [0, 0, 0]
        _, _, emin, eidx, _, _ = self.ev(pos)
        if emin < R2_EMIT_R:
            w = ((R2_EMIT_R - max(emin, 0)) << 16) // R2_EMIT_R
            ew, ecol = mul16(w, w), r2_lin3(self.prims[eidx]["mat"]["col"])
        else:
            ew, ecol = 0, [0, 0, 0]
        out = []
        for i in range(3):
            amb = mul16(self.hor_lin[i] + mul16(self.top_lin[i] - self.hor_lin[i], hemi), ao) * R2_AMB_GAIN
            direct = mul16(self.light_lin[i], mul16(ndl, shadow)) * R2_DIRECT_GAIN
            spec_c = mul16(self.light_lin[i], mul16(spec, shadow)) * R2_SPEC_GAIN
            col = mul16(base[i], direct + amb + mul16(ecol[i], ew) * R2_EMIT_GAIN) + spec_c + mul16(base[i], emit)
            out.append(col + mul16(skyc[i] - col, fogf))
        return out, n

    def trace(self, o, d):
        acc, thr = [0, 0, 0], Q
        for bounce in range(self.p["bounces"] + 1):
            t, hit, last = 0, None, None
            emin, eidx = 1 << 62, 0
            for _ in range(R2_MARCH):
                pos = at(o, d, t)
                dist, idx, _, _, gm, gi = self.ev(pos)
                if bounce == 0 and gm < emin:
                    emin, eidx = gm, gi
                if dist < R2_HIT_EPS:
                    hit = (pos, idx)
                    break
                last = (pos, idx, dist, t)
                t += dist
                if t > R2_T_MAX:
                    last = None
                    break
            if hit is None and last is not None and last[2] < R2_FAR_HIT:
                hit = (last[0], last[1])
                t = last[3]
            if bounce == 0:
                if emin < R2_GLOW_R:
                    g = ((R2_GLOW_R - max(emin, 0)) << 16) // R2_GLOW_R
                    g = mul16(g, g)
                    ec = r2_lin3(self.prims[eidx]["mat"]["col"])
                    for i in range(3):
                        acc[i] += mul16(ec[i], g) >> 1
                t_end = t if hit is not None else R2_T_MAX
                for pr in self.prims:
                    if "Sphere" in pr["shape"] and pr["mat"]["emit"] > 0:
                        sc, sr = pr["shape"]["Sphere"]["c"], pr["shape"]["Sphere"]["r"]
                        oc = [sc[0] - o[0], sc[1] - o[1], sc[2] - o[2]]
                        tc = clamp(r2_dot16(oc, d), 0, t_end)
                        q = at(o, d, tc)
                        dist = len3([q[0] - sc[0], q[1] - sc[1], q[2] - sc[2]]) - sr
                        if dist < R2_GLOW_R:
                            g = ((R2_GLOW_R - max(dist, 0)) << 16) // R2_GLOW_R
                            g = mul16(g, g)
                            ec = r2_lin3(pr["mat"]["col"])
                            for i in range(3):
                                acc[i] += mul16(ec[i], g) >> 1
            if hit is None:
                sk = self.sky(d)
                for i in range(3):
                    acc[i] += mul16(thr, sk[i])
                break
            pos, idx = hit
            col, n = self.surface(pos, idx, d, t)
            refl = self.prims[idx]["mat"]["refl"] * 257
            if bounce == self.p["bounces"] or refl == 0:
                for i in range(3):
                    acc[i] += mul16(thr, col[i])
                break
            cosv = clamp(r2_dot16([-d[0], -d[1], -d[2]], n), 0, Q)
            x = Q - cosv
            x2 = mul16(x, x)
            fr = mul16(mul16(x2, x2), x)
            f = min(refl + mul16(Q - refl, fr), Q)
            for i in range(3):
                acc[i] += mul16(thr, mul16(col[i], Q - f))
            thr = mul16(thr, f)
            dn = r2_dot16(d, n)
            d = norm3([d[i] - 2 * mul16(dn, n[i]) for i in range(3)])
            o = [pos[i] + mul16(n[i], R2_BIAS) for i in range(3)]
        return acc


def render2_chunk(p, chunk):
    w, h, spp = p["width"], p["height"], p["spp"]
    tiles_x = w // TILE
    tx, ty = chunk % tiles_x, chunk // tiles_x
    jitter = xof("vearl/l2/render/v1", [bytes(p["seed"]), struct.pack("<I", chunk)], TILE * TILE * spp * 4) if spp > 1 else b""
    r = R2(p)
    cam = list(p["cam"])
    fwd = norm3([p["target"][i] - cam[i] for i in range(3)])
    right = norm3(r2_cross16([0, Q, 0], fwd))
    up = norm3(r2_cross16(fwd, right))
    f = p["fov"]
    out = bytearray()
    for y in range(TILE):
        for x in range(TILE):
            acc = [0, 0, 0]
            for s in range(spp):
                if spp > 1:
                    k = ((y * TILE + x) * spp + s) * 4
                    jx, jy = struct.unpack("<HH", jitter[k:k + 4])
                else:
                    jx = jy = 32768
                sx, sy = (tx * TILE + x) * Q + jx, (ty * TILE + y) * Q + jy
                u = tdiv(2 * sx - w * Q, w)
                v = -tdiv(2 * sy - h * Q, w)
                draw = [mul16(fwd[i], f) + mul16(right[i], u) + mul16(up[i], v) for i in range(3)]
                c = r.trace(cam, norm3(draw))
                for i in range(3):
                    acc[i] += c[i]
            cx, cy = (tx * TILE + x) * Q + Q // 2, (ty * TILE + y) * Q + Q // 2
            nx, ny = tdiv(2 * cx - w * Q, w), tdiv(2 * cy - h * Q, w)
            vig = max(Q - mul16(mul16(nx, nx) + mul16(ny, ny), R2_VIGNETTE), 16384)
            for i in range(3):
                mean = tdiv(acc[i], spp)
                cc = mul16(mul16(mean, p["exposure"]), vig)
                g = math.isqrt((r2_aces(cc) * 65025) >> 16)
                out.append(min(g, 255))
    return bytes(out)


# ---------------- Render3 path tracer: docs/10-level-2.md section 4 ter ----------------
R3_PI, R3_TAU = 205887, 411775
R3_MAXT = 60 * Q
R3_ATAN = [13176795, 7778716, 4110060, 2086331, 1047214, 524117, 262123, 131069, 65536, 32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]
R3_K = 10188014
R3_BIG = 1 << 60
M32 = 0xFFFFFFFF


def r3_dot(a, b):
    return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) >> 16


def r3_cross(a, b):
    return [(a[1] * b[2] - a[2] * b[1]) >> 16, (a[2] * b[0] - a[0] * b[2]) >> 16, (a[0] * b[1] - a[1] * b[0]) >> 16]


def r3_add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def r3_sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def r3_scale(a, s):
    return [mul16(a[0], s), mul16(a[1], s), mul16(a[2], s)]


def r3_neg(a):
    return [-a[0], -a[1], -a[2]]


def r3_isqrt(x):
    return math.isqrt(x) if x > 0 else 0


def r3_len3(v):
    return r3_isqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def r3_len2(a, b):
    return r3_isqrt(a * a + b * b)


def r3_norm(v):
    l = r3_len3(v)
    if l == 0:
        return [0, Q, 0]
    return [tdiv(v[0] << 16, l), tdiv(v[1] << 16, l), tdiv(v[2] << 16, l)]


def r3_lin8(c):
    return ((c * c) << 16) // 65025


def r3_lin3(c):
    return [r3_lin8(c[0]), r3_lin8(c[1]), r3_lin8(c[2])]


def r3_q10(c):
    return [c[0] << 6, c[1] << 6, c[2] << 6]


def r3_smoothstep(e0, e1, x):
    if e1 == e0:
        return Q if x >= e1 else 0
    t = clamp(tdiv((x - e0) << 16, e1 - e0), 0, Q)
    return mul16(mul16(t, t), 3 * Q - 2 * t)


def r3_mix(a, b, t):
    return a + mul16(b - a, t)


def r3_sincos(a):
    a = a % R3_TAU
    if a > R3_PI:
        a -= R3_TAU
    flip = False
    if a > R3_PI // 2:
        a -= R3_PI
        flip = True
    elif a < -(R3_PI // 2):
        a += R3_PI
        flip = True
    x, y, z = R3_K, 0, a << 8
    for i in range(24):
        dx, dy = y >> i, x >> i
        if z >= 0:
            x -= dx
            y += dy
            z -= R3_ATAN[i]
        else:
            x += dx
            y -= dy
            z += R3_ATAN[i]
    s, c = (y + 128) >> 8, (x + 128) >> 8
    if flip:
        s, c = -s, -c
    return s, c


def r3_atan2(y, x):
    if x == 0 and y == 0:
        return 0
    xx, yy, base = x << 8, y << 8, 0
    if xx < 0:
        base = R3_PI if yy >= 0 else -R3_PI
        xx, yy = -xx, -yy
    z = 0
    for i in range(24):
        dx, dy = yy >> i, xx >> i
        if yy > 0:
            xx += dx
            yy -= dy
            z += R3_ATAN[i]
        else:
            xx -= dx
            yy += dy
            z -= R3_ATAN[i]
    return base + ((z + 128) >> 8)


def r3_pcg(v):
    s = (v * 747796405 + 2891336453) & M32
    w = ((((s >> ((s >> 28) + 4)) ^ s) * 277803737)) & M32
    return (w >> 22) ^ w


def r3_hash3(x, y, z):
    h = r3_pcg(((x & M32) * 73856093 & M32) ^ ((y & M32) * 19349663 & M32) ^ ((z & M32) * 83492791 & M32))
    return h >> 16


def r3_fs(v):
    t = v & 0xFFFF
    return mul16(mul16(t, t), 3 * Q - 2 * t)


def r3_vnoise(p):
    ix, iy, iz = p[0] >> 16, p[1] >> 16, p[2] >> 16
    fx, fy, fz = r3_fs(p[0]), r3_fs(p[1]), r3_fs(p[2])
    h = r3_hash3
    x00 = r3_mix(h(ix, iy, iz), h(ix + 1, iy, iz), fx)
    x10 = r3_mix(h(ix, iy + 1, iz), h(ix + 1, iy + 1, iz), fx)
    x01 = r3_mix(h(ix, iy, iz + 1), h(ix + 1, iy, iz + 1), fx)
    x11 = r3_mix(h(ix, iy + 1, iz + 1), h(ix + 1, iy + 1, iz + 1), fx)
    return r3_mix(r3_mix(x00, x10, fy), r3_mix(x01, x11, fy), fz)


def r3_fbm(p, octaves):
    a, s = Q // 2, 0
    for _ in range(octaves):
        s += mul16(a, r3_vnoise(p))
        p = [p[0] * 2 + (p[0] >> 5) + 111411, p[1] * 2 + (p[1] >> 5) + 602931, p[2] * 2 + (p[2] >> 5) + 203162]
        a >>= 1
    return s


def r3_cell_hash(ix, iy, iz, k):
    return r3_hash3(ix + 17 * k, iy + 31 * k, iz + 47 * k)


def r3_voronoi(p):
    ix, iy, iz = p[0] >> 16, p[1] >> 16, p[2] >> 16
    d1 = d2 = 1 << 62
    best = 0
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cx, cy, cz = ix + dx, iy + dy, iz + dz
                ex = (cx << 16) + r3_cell_hash(cx, cy, cz, 1) - p[0]
                ey = (cy << 16) + r3_cell_hash(cx, cy, cz, 2) - p[1]
                ez = (cz << 16) + r3_cell_hash(cx, cy, cz, 3) - p[2]
                d = ex * ex + ey * ey + ez * ez
                if d < d1:
                    d2, d1, best = d1, d, r3_cell_hash(cx, cy, cz, 0)
                elif d < d2:
                    d2 = d
    return r3_isqrt(d1), r3_isqrt(d2), best


def r3_rotate(rot, p):
    return [(rot[0] * p[0] + rot[1] * p[1] + rot[2] * p[2]) >> 16, (rot[3] * p[0] + rot[4] * p[1] + rot[5] * p[2]) >> 16, (rot[6] * p[0] + rot[7] * p[1] + rot[8] * p[2]) >> 16]


def r3_geo_dist(geo, q):
    if geo == "Plane":
        return q[1]
    ((kind, g),) = geo.items()
    if kind == "Sphere":
        return r3_len3(q) - g["r"]
    if kind == "Ellipsoid":
        r = g["r"]
        a = [tdiv(q[i] << 16, r[i]) for i in range(3)]
        b = [tdiv(a[i] << 16, r[i]) for i in range(3)]
        k0 = r3_len3(a)
        k1 = max(r3_len3(b), 1)
        return tdiv(mul16(k0, k0 - Q) << 16, k1)
    if kind == "Box":
        h, r = g["h"], g["rnd"]
        w = [abs(q[0]) - h[0] + r, abs(q[1]) - h[1] + r, abs(q[2]) - h[2] + r]
        return r3_len3([max(w[0], 0), max(w[1], 0), max(w[2], 0)]) + min(max(w[0], max(w[1], w[2])), 0) - r
    if kind == "Cylinder":
        rr = g["rnd"]
        dx = r3_len2(q[0], q[2]) - g["r"] + rr
        dy = abs(q[1]) - g["h"] + rr
        return r3_len2(max(dx, 0), max(dy, 0)) + min(max(dx, dy), 0) - rr
    if kind == "Capsule":
        y = clamp(q[1], -g["h"], g["h"])
        return r3_len3([q[0], q[1] - y, q[2]]) - g["r"]
    if kind == "Torus":
        return r3_len2(r3_len2(q[0], q[2]) - g["big"], q[1]) - g["small"]
    h, r1, r2 = g["h"], g["r1"], g["r2"]  # Cone
    qx, qy = r3_len2(q[0], q[2]), q[1]
    k1x, k1y, k2x, k2y = r2, h, r2 - r1, 2 * h
    cax = qx - min(qx, r1 if qy < 0 else r2)
    cay = abs(qy) - h
    dd = mul16(k2x, k2x) + mul16(k2y, k2y)
    tt = clamp(0 if dd == 0 else tdiv((mul16(k1x - qx, k2x) + mul16(k1y - qy, k2y)) << 16, dd), 0, Q)
    cbx, cby = qx - k1x + mul16(k2x, tt), qy - k1y + mul16(k2y, tt)
    s = -1 if (cbx < 0 and cay < 0) else 1
    return s * r3_isqrt(min(cax * cax + cay * cay, cbx * cbx + cby * cby))


def r3_smin(a, b, k):
    if k <= 0:
        return min(a, b)
    h = clamp(Q // 2 + tdiv((b - a) << 15, k), 0, Q)
    return b + mul16(a - b, h) - mul16(mul16(k, h), Q - h)


def r3_part_dist(pt, p):
    q = r3_rotate(pt["rot"], r3_sub(p, pt["c"]))
    if pt["mirror"]:
        q[0], q[2] = abs(q[0]), abs(q[2])
    if pt["polar"] > 1:
        n = pt["polar"]
        sec = R3_TAU // n
        ang = r3_atan2(q[2], q[0])
        k = (ang + sec // 2) // sec
        s, c = r3_sincos(k * sec)
        q = [mul16(q[0], c) + mul16(q[2], s), q[1], mul16(-q[0], s) + mul16(q[2], c)]
    d = r3_geo_dist(pt["geo"], r3_sub(q, pt["off"]))
    return abs(d) - pt["shell"] if pt["shell"] > 0 else d


class R3:
    def __init__(self, p):
        self.p = p
        self.parts, self.mats = p["parts"], p["mats"]
        e = p["env"]
        self.top, self.hor, self.gnd = r3_q10(e["top"]), r3_q10(e["hor"]), r3_q10(e["gnd"])
        self.bd_top, self.bd_bot = r3_q10(e["bd_top"]), r3_q10(e["bd_bot"])
        self.boxes = e["boxes"]

    def scene(self, pos):
        all_d, all_mat = R3_BIG, 0
        acc, mat = R3_BIG, 0
        cur = self.parts[0]["group"]
        for pt in self.parts:
            if pt["group"] != cur:
                if acc < all_d:
                    all_d, all_mat = acc, mat
                acc, mat = R3_BIG, 0
                cur = pt["group"]
            d = r3_part_dist(pt, pos)
            op = pt["op"]
            if op == "Union":
                if d < acc:
                    acc, mat = d, pt["mat"]
            elif op == "Sub":
                if -d > acc:
                    acc, mat = -d, pt["mat"]
            elif op == "Inter":
                if d > acc:
                    acc, mat = d, pt["mat"]
            else:
                if acc == R3_BIG:
                    acc, mat = d, pt["mat"]
                else:
                    m = pt["mat"] if d < acc else mat
                    acc = r3_smin(acc, d, op["Smooth"])
                    mat = m
        if acc < all_d:
            all_d, all_mat = acc, mat
        return all_d, all_mat

    def normal(self, pos):
        e = 33
        f = lambda dx, dy, dz: self.scene([pos[0] + dx, pos[1] + dy, pos[2] + dz])[0]
        a, b, c, d = f(e, -e, -e), f(-e, -e, e), f(-e, e, -e), f(e, e, e)
        return r3_norm([a - b - c + d, -a - b + c + d, -a + b - c + d])

    def bump(self, pos, n, freq, amp):
        e = 98
        sp = lambda p: r3_fbm(r3_scale(p, freq), 3)
        b = sp(pos)
        g = [tdiv((sp([pos[0] + e, pos[1], pos[2]]) - b) * Q, e), tdiv((sp([pos[0], pos[1] + e, pos[2]]) - b) * Q, e), tdiv((sp([pos[0], pos[1], pos[2] + e]) - b) * Q, e)]
        gn = r3_dot(g, n)
        g = [g[i] - mul16(n[i], gn) for i in range(3)]
        k = tdiv((amp << 8) << 16, max(freq, 1))
        return r3_norm([n[i] - mul16(g[i], k) for i in range(3)])

    def surf(self, m, pos, ng):
        alb, alb2 = r3_lin3(m["alb"]), r3_lin3(m["alb2"])
        rough = m["rough"] << 8
        sc = m["tex_scale"]
        mixc = lambda t, a, b: [r3_mix(a[i], b[i], t) for i in range(3)]
        tex = m["tex"]
        if tex == 1:
            cell = (mul16(pos[0], sc) >> 16) + (mul16(pos[1], sc) >> 16) + (mul16(pos[2], sc) >> 16)
            if cell & 1:
                alb = alb2
        elif tex == 2:
            axis = min(m["tex_p"], 2)
            if (mul16(pos[axis], sc) >> 16) & 1:
                alb = alb2
        elif tex == 3:
            alb = mixc(clamp(r3_fbm(r3_scale(pos, sc), 5), 0, Q), alb, alb2)
        elif tex == 4:
            r = r3_len2(pos[0], pos[2])
            if (mul16(r, sc) & 0xFFFF) < Q // 2:
                alb = alb2
        elif tex == 5:
            u = mul16(pos[2], sc) + 2 * r3_fbm(r3_scale(pos, 3 * Q), 3)
            fr = u & 0xFFFF
            t = r3_smoothstep(0, Q // 2, fr) - r3_smoothstep(Q // 2, Q, fr)
            alb = mixc(clamp(t, 0, Q), alb, alb2)
        elif tex == 6:
            ang = r3_atan2(pos[2], pos[0])
            s, _ = r3_sincos(ang * m["tex_p"])
            t = (Q + s) // 2
            alb = mixc(clamp(t, 0, Q), alb, alb2)
            rough = min(rough + mul16(t, Q // 4), Q)
        elif tex == 7:
            r = r3_len2(pos[0], pos[2])
            ang = r3_atan2(pos[2], pos[0])
            _, c = r3_sincos(ang * m["tex_p"])
            if abs(c) > 64553 and m["tex_q"] <= r <= m["tex_r"] and ng[1] > 39322:
                alb = alb2
        elif tex == 8:
            w = r3_fbm(r3_scale(pos, sc), 4)
            u = mul16(pos[0] + mul16(pos[1], 26214) + mul16(pos[2], 39322), sc) + mul16(w, m["tex_p"])
            fr = u & 0xFFFF
            d = min(fr, Q - fr)
            vein = Q - r3_smoothstep(0, max(m["tex_q"], 1), d)
            base = mixc(mul16(w, 16384), alb, alb2)
            alb = mixc(vein, base, alb2)
        elif tex == 9:
            f1, f2, cid = r3_voronoi(r3_scale(pos, sc))
            t = mul16(cid, m["tex_q"])
            base = mixc(t, alb, alb2)
            crack = Q - r3_smoothstep(0, max(m["tex_p"], 1), f2 - f1)
            alb = mixc(mul16(crack, 55705), base, [0, 0, 0])
            rough = min(rough + mul16(crack, Q // 4), Q)
        elif tex == 10:
            p = r3_scale(pos, sc)
            wx = r3_fbm(p, 3)
            wy = r3_fbm([p[0] + 5 * Q + 21299, p[1] + Q + 19661, p[2] + 2 * Q + 45875], 3)
            wz = r3_fbm([p[0] + 3 * Q + 13107, p[1] + 4 * Q + 52428, p[2] + Q + 6554], 3)
            wp = m["tex_p"]
            q = [p[0] + mul16(wx - Q // 4, wp), p[1] + mul16(wy - Q // 4, wp), p[2] + mul16(wz - Q // 4, wp)]
            t = r3_fbm(q, 4)
            sm = r3_smoothstep(Q // 5, 4 * Q // 5, t)
            c = mixc(sm, alb, alb2)
            u = mul16(t, m["tex_q"])
            fr = u & 0xFFFF
            line = Q - r3_smoothstep(0, Q // 20, min(fr, Q - fr))
            alb = mixc(mul16(line, Q // 4), c, [Q, Q, Q])
        elif tex == 11:
            u, v = mul16(pos[0], sc), mul16(pos[2], sc)
            fu, fv = u & 0xFFFF, v & 0xFFFF
            e = min(fu, Q - fu, fv, Q - fv)
            g = max(m["tex_p"], 2)
            grout = Q - r3_smoothstep(g // 2, g, e)
            tone = mul16(r3_hash3(u >> 16, 5, v >> 16), m["tex_q"])
            base = mixc(tone, alb, alb2)
            alb = mixc(grout, base, r3_scale(base, 8000))
            rough = r3_mix(rough, 58982, grout)
        nrm = self.bump(pos, ng, m["bump_freq"], m["bump"]) if m["bump"] > 0 else None
        return alb, rough, nrm

    def softbox(self, d, b):
        dirv = r3_norm(b["dir"])
        c = r3_dot(d, dirv)
        if c <= 0:
            return 0
        up0 = [Q, 0, 0] if abs(dirv[1]) > 62259 else [0, Q, 0]
        r = r3_norm(r3_cross(dirv, up0))
        u = r3_cross(r, dirv)
        qx, qy = abs(tdiv(r3_dot(d, r) << 16, c)), abs(tdiv(r3_dot(d, u) << 16, c))
        hx, hy, s = b["hs"][0], b["hs"][1], b["soft"]
        wx = Q - r3_smoothstep(mul16(hx, Q - s), hx, qx)
        wy = Q - r3_smoothstep(mul16(hy, Q - s), hy, qy)
        return mul16(wx, wy)

    def env(self, d):
        out = []
        for i in range(3):
            if d[1] >= 0:
                y = min(d[1], Q)
                out.append(r3_mix(self.hor[i], self.top[i], mul16(y, y)))
            else:
                out.append(r3_mix(self.hor[i], self.gnd[i], min(-d[1], Q)))
        for b in self.boxes:
            w = self.softbox(d, b)
            if w > 0:
                c = r3_q10(b["col"])
                for i in range(3):
                    out[i] += mul16(c[i], w)
        return out

    def backdrop(self, d):
        k = clamp((d[1] + Q) >> 1, 0, Q)
        return [r3_mix(self.bd_bot[i], self.bd_top[i], k) for i in range(3)]

    def march(self, ro, rd):
        t = 66
        for _ in range(420):
            pos = r3_add(ro, r3_scale(rd, t))
            d, mat = self.scene(pos)
            d = abs(d)
            if d < max((26 * max(t, 19661)) >> 16, 1):
                return t, mat
            t += (d * 9) // 10
            if t > R3_MAXT:
                break
        return None


class R3Rng:
    def __init__(self, s):
        self.s = s

    def next(self):
        self.s = r3_pcg(self.s)
        return self.s >> 16


def r3_disc(r):
    for _ in range(8):
        x = 2 * r.next() - Q
        y = 2 * r.next() - Q
        if x * x + y * y <= Q * Q:
            return x, y
    return 0, 0


def r3_basis_t(n):
    if abs(n[0]) > Q // 2:
        return r3_norm(r3_cross(n, [0, Q, 0]))
    return r3_norm(r3_cross(n, [Q, 0, 0]))


def r3_cos_hemi(n, r):
    x, y = r3_disc(r)
    z = r3_isqrt(Q * Q - x * x - y * y)
    t = r3_basis_t(n)
    b = r3_cross(n, t)
    return r3_norm([mul16(t[i], x) + mul16(b[i], y) + mul16(n[i], z) for i in range(3)])


def r3_ggx(n, a, r):
    x, y = r3_disc(r)
    v = r.next()
    a2 = mul16(a, a)
    denom = Q + mul16(a2 - Q, v)
    ratio = clamp(tdiv((Q - v) << 16, max(denom, 1)), 0, Q)
    ct = r3_isqrt(ratio << 16)
    st = r3_isqrt(Q * Q - ct * ct)
    l = r3_len2(x, y)
    cx, cy = (Q, 0) if l == 0 else (tdiv(x << 16, l), tdiv(y << 16, l))
    t = r3_basis_t(n)
    b = r3_cross(n, t)
    sx, sy = mul16(cx, st), mul16(cy, st)
    return r3_norm([mul16(t[i], sx) + mul16(b[i], sy) + mul16(n[i], ct) for i in range(3)])


def r3_reflect(i, n):
    d = r3_dot(i, n)
    return [i[0] - 2 * mul16(d, n[0]), i[1] - 2 * mul16(d, n[1]), i[2] - 2 * mul16(d, n[2])]


def r3_refract(i, n, eta):
    d = r3_dot(n, i)
    k = Q - mul16(mul16(eta, eta), Q - mul16(d, d))
    if k < 0:
        return None
    sk = r3_isqrt(k << 16)
    s = mul16(eta, d) + sk
    return r3_norm([mul16(eta, i[j]) - mul16(s, n[j]) for j in range(3)])


def r3_pow5(x):
    x2 = mul16(x, x)
    return mul16(mul16(x2, x2), x)


def r3_exp_neg(x):
    if x <= 0:
        return Q
    k, y = 0, x
    while y > 4096 and k < 24:
        y >>= 1
        k += 1
    y2 = mul16(y, y)
    y3 = mul16(y2, y)
    e = clamp(Q - y + (y2 >> 1) - tdiv(y3, 6), 0, Q)
    for _ in range(k):
        e = mul16(e, e)
    return e


def r3_trace(R, rng, ro, rd0):
    p = R.p
    rd = rd0
    col, thr = [0, 0, 0], [Q, Q, Q]
    t0, inside, sig = -1, False, [0, 0, 0]
    for b in range(p["bounces"] + 1):
        h = R.march(ro, rd)
        if h is None:
            e = R.backdrop(rd) if b == 0 else R.env(rd)
            for i in range(3):
                col[i] += mul16(thr[i], e[i])
            break
        t, mi = h
        if b == 0:
            t0 = t
        if inside:
            for i in range(3):
                thr[i] = mul16(thr[i], r3_exp_neg(mul16(sig[i], t)))
        pos = r3_add(ro, r3_scale(rd, t))
        ng = R.normal(pos)
        m = R.mats[mi]
        alb, rough, nrm = R.surf(m, pos, ng)
        n = nrm if nrm is not None else ng
        emit = r3_q10(m["emit"])
        for i in range(3):
            col[i] += mul16(thr[i], emit[i])
        if sum(emit) > 0 and sum(m["alb"]) == 0:
            break
        if m["glass"]:
            ent = r3_dot(rd, ng) < 0
            gn = ng if ent else r3_neg(ng)
            nn = n if ent else r3_neg(n)
            if rough > 0:
                nn = r3_ggx(nn, mul16(rough, rough), rng)
            ior = m["ior"] << 6
            eta = tdiv(Q << 16, ior) if ent else ior
            ci = clamp(-r3_dot(rd, nn), 0, Q)
            a = tdiv((ior - Q) << 16, ior + Q)
            f0 = mul16(a, a)
            fr = f0 + mul16(Q - f0, r3_pow5(Q - ci))
            tr = r3_refract(rd, nn, eta)
            refl = True if tr is None else rng.next() < fr
            if refl:
                r = r3_reflect(rd, nn)
                if r3_dot(r, gn) < 0:
                    r = r3_reflect(r, gn)
                rd = r
                ro = r3_add(pos, r3_scale(gn, 164))
            else:
                rd = tr
                ro = r3_sub(pos, r3_scale(gn, 164))
                inside = ent
                sig = r3_q10(m["absorb"])
                if ent:
                    al = r3_lin3(m["alb"])
                    for i in range(3):
                        thr[i] = mul16(thr[i], al[i])
            continue
        ci = clamp(r3_dot(r3_neg(rd), n), 0, Q)
        if m["coat"] > 0:
            fc = 2621 + mul16(62915, r3_pow5(Q - ci))
            if rng.next() < mul16(fc, m["coat"] << 8):
                r2 = r3_reflect(rd, r3_ggx(n, 983, rng))
                rd = r2 if r3_dot(r2, ng) > 0 else r3_reflect(rd, ng)
                ro = r3_add(pos, r3_scale(ng, 131))
                continue
        spec = m["spec"] << 8
        f0 = list(alb) if m["metal"] else [mul16(5243, spec)] * 3
        pw = r3_pow5(Q - ci)
        fre = [f0[i] + mul16(Q - f0[i], pw) for i in range(3)]
        ps = clamp(mul16((fre[0] + fre[1] + fre[2]) // 3, Q - mul16(rough, Q // 2)), 2621, 62259)
        if m["metal"]:
            ps = Q
        if rng.next() < ps:
            a = mul16(rough, rough)
            rf = r3_reflect(rd, r3_ggx(n, max(a, 26), rng))
            if r3_dot(rf, ng) <= 0:
                rf = r3_reflect(rd, ng)
            rd = rf
            for i in range(3):
                thr[i] = mul16(thr[i], alb[i]) if m["metal"] else tdiv(mul16(thr[i], fre[i]) * Q, ps)
        else:
            if m["trans"] > 0 and rng.next() < (m["trans"] << 8):
                rd = r3_cos_hemi(r3_neg(n), rng)
                ro = r3_sub(pos, r3_scale(ng, 262))
                for i in range(3):
                    thr[i] = tdiv(mul16(thr[i], alb[i]) * Q, Q - ps)
                continue
            rd = r3_cos_hemi(n, rng)
            if r3_dot(rd, ng) < 0:
                rd = r3_reflect(rd, ng)
            for i in range(3):
                thr[i] = tdiv(mul16(mul16(thr[i], alb[i]), Q - fre[i]) * Q, Q - ps)
        ro = r3_add(pos, r3_scale(ng, 131))
        if b > 2:
            mx = max(thr)
            q = max(Q - mx, 3277)
            if rng.next() < q:
                break
            for i in range(3):
                thr[i] = tdiv(thr[i] * Q, Q - q)
    out = [min(c, 24 * Q) for c in col]
    if t0 > 0:
        k = r3_smoothstep(p["fog0"], p["fog1"], t0)
        if k > 0:
            bd = R.backdrop(rd0)
            for i in range(3):
                out[i] = r3_mix(out[i], bd[i], k)
    return out


def render3_chunk(p, chunk):
    w, h, spp = p["width"], p["height"], p["spp"]
    tiles_x = w // TILE
    ntiles = tiles_x * (h // TILE)
    tile = chunk % ntiles
    tx, ty = tile % tiles_x, tile // tiles_x
    w0 = int.from_bytes(bytes(p["seed"][0:4]), "little")
    w1 = int.from_bytes(bytes(p["seed"][4:8]), "little")
    R = R3(p)
    cam = p["cam"]
    pos = list(cam["pos"])
    fw = r3_norm(r3_sub(list(cam["look"]), pos))
    rt = r3_norm(r3_cross([0, Q, 0], fw))
    up = r3_cross(fw, rt)
    out = bytearray()
    for y in range(TILE):
        for x in range(TILE):
            px, py = tx * TILE + x, ty * TILE + y
            acc = [0, 0, 0]
            for s in range(spp):
                rng = R3Rng(r3_pcg((px * 1973 + py * 9277 + chunk * 26699 + s * 612451 + w0) & M32))
                rng.s = r3_pcg(rng.s ^ w1)
                jx, jy = rng.next(), rng.next()
                sx, sy = px * Q + jx, py * Q + jy
                u = tdiv(2 * sx - w * Q, h)
                v = -tdiv(2 * sy - h * Q, h)
                f = cam["focal"]
                rd = r3_norm([mul16(rt[i], u) + mul16(up[i], v) + mul16(fw[i], f) for i in range(3)])
                ro, rdd = pos, rd
                if cam["aperture"] > 0:
                    fd = tdiv(cam["focus"] << 16, max(r3_dot(rd, fw), 1))
                    fp = r3_add(pos, r3_scale(rd, fd))
                    lx, ly = r3_disc(rng)
                    lr, lq = mul16(cam["aperture"], lx), mul16(cam["aperture"], ly)
                    ro = r3_add(pos, r3_add(r3_scale(rt, lr), r3_scale(up, lq)))
                    rdd = r3_norm(r3_sub(fp, ro))
                c = r3_trace(R, rng, ro, rdd)
                for k in range(3):
                    acc[k] += c[k]
            for k in range(3):
                q11 = clamp(tdiv(acc[k], spp) >> 5, 0, 65535)
                out += q11.to_bytes(2, "little")
    return bytes(out)


def chunk(case, i):
    k = case["kernel"]
    return mc_chunk(case["params"], i) if k == "mc" else render2_chunk(case["params"], i) if k == "render2" else render3_chunk(case["params"], i) if k == "render3" else render_chunk(case["params"], i)


def main(path):
    V = json.load(open(path))
    bad = 0
    for n, case in enumerate(V["cases"]):
        for c in case["chunks"]:
            got = chunk(case, c["i"]).hex()
            ok = got == c["out"]
            bad += not ok
            desc = case["kernel"] + (f" dim={case['params']['dim']}" if case["kernel"] == "mc" else f" {case['params']['width']}x{case['params']['height']} spp={case['params']['spp']}")
            print(("ok   " if ok else "FAIL"), f"cas {n} ({desc}) chunk {c['i']}")
    print("\nRESULT:", "KERNELS IDENTICAL BYTE FOR BYTE TO THE RUST NODE" if bad == 0 else f"{bad} FAILURE(S)")
    return bad == 0


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
