#!/usr/bin/env python3
"""Raster brand assets for the site: apple-touch-icon.png (180), favicon-32.png, og-image.png (1200x630 social preview).

    python3 make_images.py SITE_DIR
"""
import math, os, re, sys
from PIL import Image, ImageDraw, ImageFont

SITE = sys.argv[1] if len(sys.argv) > 1 else ".."
FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build-fonts")
BG = (7, 8, 11)
NACRE = [(217, 255, 247), (159, 240, 226), (185, 166, 255), (244, 188, 214)]
STOPS = [0, .45, .75, 1]


def nacre(t):
    t = min(1, max(0, t))
    for k in range(len(STOPS) - 1):
        if t <= STOPS[k + 1]:
            u = (t - STOPS[k]) / (STOPS[k + 1] - STOPS[k])
            return tuple(round(a + (b - a) * u) for a, b in zip(NACRE[k], NACRE[k + 1]))
    return NACRE[-1]


def draw_mark(img, x0, y0, size):
    """Pearl of dots, read back from mark.svg (single source of truth), supersampled."""
    svg = open(os.path.join(SITE, "mark.svg")).read()
    circles = [tuple(map(float, m)) for m in re.findall(r'cx="([\d.]+)" cy="([\d.]+)" r="([\d.]+)"', svg)]
    ss = 4
    big = Image.new("RGBA", (size * ss, size * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    k = size * ss / 64
    for cx, cy, r in circles:
        col = nacre((cx + cy) / 128)
        d.ellipse([(cx - r) * k, (cy - r) * k, (cx + r) * k, (cy + r) * k], fill=col + (255,))
    img.alpha_composite(big.resize((size, size), Image.LANCZOS), (x0, y0))


def icon(size, pad_ratio, radius_ratio, path):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=int(size * radius_ratio), fill=BG + (255,))
    pad = int(size * pad_ratio)
    draw_mark(img, pad, pad, size - 2 * pad)
    img.save(path)


def font(name, px):
    return ImageFont.truetype(os.path.join(FONTS, name), px)


def ascii_pearl(img, cx, cy, radius_cols, fs):
    """The hero's ASCII pearl, frozen: same shading model as the site."""
    d = ImageDraw.Draw(img)
    f = font("ibm-plex-mono-normal-500.ttf", fs)
    cw = f.getlength("M")
    ch = fs * 1.18
    aspect = ch / cw
    ramp = " .·:-=+*%#@"
    lx, ly, lz = -.55, -.5, .67
    n = math.sqrt(lx * lx + ly * ly + lz * lz)
    lx, ly, lz = lx / n, ly / n, lz / n
    cols = int(radius_cols * 2.9)
    rows = int(cols / aspect)
    for r in range(rows):
        for c in range(cols):
            dx = (c - cols / 2) / radius_cols
            dy = (r - rows / 2) * aspect / radius_cols
            d2 = dx * dx + dy * dy
            if d2 < 1:
                nz = math.sqrt(1 - d2)
                diff = max(0, dx * lx + dy * ly + nz * lz)
                hz = lz + 1
                hn = math.sqrt(lx * lx + ly * ly + hz * hz)
                spec = max(0, (dx * lx + dy * ly + nz * hz) / hn) ** 38
                swirl = math.sin(3 * math.atan2(dy, dx) + nz * 4) * .5 + .5
                v = .2 + diff * .75 + spec * .9 + swirl * .08
                hue, alpha = (nz * .6 + swirl * .5) % 1, 245
            else:
                dist = math.sqrt(d2)
                v = max(0, math.sin(dist * 9) * math.exp(-(dist - 1) * 1.1)) * .42
                hue, alpha = (dist * .25) % 1, 110
            k = min(len(ramp) - 1, int(v * (len(ramp) - 1)))
            if k <= 0:
                continue
            col = nacre(hue)
            d.text((cx + (c - cols / 2) * cw, cy + (r - rows / 2) * ch), ramp[k], font=f, fill=col + (alpha,))


def og(path):
    W, H = 1200, 630
    img = Image.new("RGBA", (W, H), BG + (255,))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ascii_pearl(layer, 905, 315, 19, 13)
    img.alpha_composite(layer)
    # left fade so the text stays readable
    fade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fade)
    for x in range(W):
        a = int(max(0, min(1, (700 - x) / 260)) * 215)
        fd.line([(x, 0), (x, H)], fill=BG + (a,))
    img.alpha_composite(fade)
    d = ImageDraw.Draw(img)
    draw_mark(img, 72, 64, 44)
    d.text((128, 66), "vearl", font=font("instrument-sans-normal-600.ttf", 36), fill=(236, 235, 230))
    d.text((74, 196), "PROOF OF USEFUL WORK · VERIFIED COMPUTE", font=font("ibm-plex-mono-normal-500.ttf", 19), fill=(159, 240, 226))
    sans, serif = font("instrument-sans-normal-600.ttf", 62), font("instrument-serif-italic-400.ttf", 92)
    d.text((70, 244), "Bitcoin mines hashes.", font=sans, fill=(236, 235, 230))
    d.text((72, 316), "Vearl mines AI.", font=serif, fill=(185, 166, 255))
    d.text((74, 452), "GPU miners run real AI models for clients.\nEvery answer is verified before anyone is paid.",
           font=font("instrument-sans-normal-400.ttf", 25), fill=(170, 176, 188), spacing=10)
    d.line([(74, 548), (W - 74, 548)], fill=(29, 34, 44), width=1)
    d.text((74, 566), "CLOSED TESTNET  ·  PUBLIC 23 OCTOBER 2026  ·  RUST + CUDA", font=font("ibm-plex-mono-normal-400.ttf", 16), fill=(110, 118, 132))
    img.convert("RGB").save(path, optimize=True)


if __name__ == "__main__":
    icon(180, .14, .0, os.path.join(SITE, "apple-touch-icon.png"))   # iOS rounds the corners itself
    icon(32, .09, .22, os.path.join(SITE, "favicon-32.png"))
    icon(512, .12, .22, os.path.join(SITE, "tools", "mark-preview.png"))
    og(os.path.join(SITE, "og-image.png"))
    print("ok")
