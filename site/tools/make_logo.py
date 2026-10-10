#!/usr/bin/env python3
"""Vearl mark: a pearl drawn as a grid of dots (a matrix whose cells follow the shading of a lit sphere).

    python3 make_logo.py OUT_DIR     -> mark.svg (site), favicon.svg (tab icon, dark disc behind)
"""
import math, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
N = 9                       # grid cells across
LIGHT = (-0.55, -0.62, 0.56)  # light from the upper left
ln = math.sqrt(sum(v * v for v in LIGHT))
LIGHT = tuple(v / ln for v in LIGHT)


def dots(size, pad):
    cell = (size - 2 * pad) / N
    r_sphere = N / 2
    out = []
    for i in range(N):
        for j in range(N):
            x, y = (j + .5 - N / 2) / r_sphere, (i + .5 - N / 2) / r_sphere
            d2 = x * x + y * y
            if d2 >= 1:
                continue
            z = math.sqrt(1 - d2)
            diff = max(0.0, x * LIGHT[0] + y * LIGHT[1] + z * LIGHT[2])
            hx, hy, hz = LIGHT[0], LIGHT[1], LIGHT[2] + 1
            hn = math.sqrt(hx * hx + hy * hy + hz * hz)
            spec = max(0.0, (x * hx + y * hy + z * hz) / hn) ** 24
            v = min(1.0, .22 + .7 * diff + .6 * spec)
            r = cell * .5 * (.28 + .68 * v)
            cx, cy = pad + (j + .5) * cell, pad + (i + .5) * cell
            out.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}"/>')
    return "".join(out)


GRAD = ('<linearGradient id="n" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#d9fff7"/><stop offset=".45" stop-color="#9ff0e2"/>'
        '<stop offset=".75" stop-color="#b9a6ff"/><stop offset="1" stop-color="#f4bcd6"/></linearGradient>')

mark = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><defs>{GRAD}</defs>'
        f'<g fill="url(#n)">{dots(64, 2)}</g></svg>')
fav = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><defs>{GRAD}</defs>'
       f'<rect width="64" height="64" rx="14" fill="#07080b"/><g fill="url(#n)">{dots(64, 7)}</g></svg>')
open(f"{OUT}/mark.svg", "w").write(mark + "\n")
open(f"{OUT}/favicon.svg", "w").write(fav + "\n")
print(len(mark), len(fav))
