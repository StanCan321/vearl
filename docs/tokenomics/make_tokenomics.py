#!/usr/bin/env python3
"""Tokenomics of VRL for the MAINNET: the numbers, a table, and an infographic for X (1920 x 1080, HTML + SVG, then PNG with Chrome).
The 7 % for the testnet contributors is a commitment of the team; the other parameters are final once published in the mainnet code; the public
testnet keeps its own test schedule (docs/04 section 4).

Plan:
  supply       CAPPED. Never more than 56,064,000 VRL. Burned VRL (10 % of every job escrow, half of every slashed bond) make the real supply lower.
  mining       R0 = 6.66666666 VRL per block (20 VRL per 30 s), 10 s blocks, halving every year (3,153,600 blocks), NO tail: 42,048,000 VRL in total, 100 % of every block to the miner.
               (integer halving of 6.66666666 VRL ends after 31 years; the issued total is a hair under 42,048,000 because of the rounding, never above)
  genesis      25 % of the maximum supply is created once, in the genesis block of the mainnet, in public and locked:
                 10 % team & foundation       12-month cliff, then linear over 36 months
                  8 % ecosystem & grants      partnerships, grants, job credits, audits, bounties; released in public tranches, decided in advance
                  7 % testnet contributors    shared by 5 public testnets: 0.7 / 1.05 / 1.4 / 1.75 / 2.1 %
    python make_tokenomics.py OUTDIR        -> OUTDIR/tokenomics.html, tokenomics.json, tokenomics.png (if Chrome is installed)"""
import json
import math
import os
import shutil
import subprocess
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))

UNIT, BPY = 10 ** 8, 3_153_600                            # units per VRL, blocks per year (10 s blocks)
R0 = 20 * UNIT * 10 // 30                                  # 20 VRL per 30 s, scaled to the block time, rounded down to the unit
SHARE = {"Mining rewards": 75.0, "Team & foundation": 10.0, "Ecosystem & grants": 8.0, "Testnet contributors": 7.0}   # % of the maximum supply
TESTNETS = [0.7, 1.05, 1.4, 1.75, 2.1]                    # % of the maximum supply, testnets 1 to 5 (sum = 7)
assert abs(sum(SHARE.values()) - 100) < 1e-9 and abs(sum(TESTNETS) - SHARE["Testnet contributors"]) < 1e-9

# what the code would issue: reward(year) = R0 >> year, integer, until it reaches 0
issued, k = 0, 0
while (R0 >> k) > 0:
    issued += (R0 >> k) * BPY
    k += 1
issued_vrl = issued / UNIT                                 # a hair under 42,048,000
MINED = 2 * 21_024_000                                     # 42,048,000 nominal: 21,024,000 VRL a year, halved every year
T = MINED / (SHARE["Mining rewards"] / 100)                # maximum supply: 56,064,000
alloc = {name: T * pct / 100 for name, pct in SHARE.items()}
genesis = T - MINED


def mined_by_month(m):
    """VRL issued by the end of month m of the mainnet (m = 0..), 1/12 of a year's blocks per month"""
    total = 0
    for mm in range(m):
        total += (R0 >> (mm // 12)) * (BPY // 12) / UNIT
    return total


year_mined = [mined_by_month(12 * y) for y in range(0, 41)]
year_supply = [genesis + v for v in year_mined]
rows = []
for y in range(1, 13):
    em = year_mined[y] - year_mined[y - 1]
    rows.append({"year": y, "reward_per_block": 20 / 3 / 2 ** (y - 1), "mined_in_year": em, "mined_total": year_mined[y], "pct_of_mining": year_mined[y] / MINED * 100,
                 "supply_total": year_supply[y], "pct_of_max": year_supply[y] / T * 100})
data = {"max_supply": T, "mining_total": MINED, "mining_issued_by_code": issued_vrl, "genesis_total": genesis, "allocations": alloc, "testnets": [T * p / 100 for p in TESTNETS],
        "testnets_pct": TESTNETS, "years": rows, "years_with_a_reward": k, "parameters": {"R0": 6.66666666, "halving_blocks": BPY, "tail": 0, "block_time_s": 10}}
json.dump(data, open(os.path.join(OUT, "tokenomics.json"), "w"), indent=1)
print(f"maximum supply {T:,.0f} VRL   (mining {MINED:,.0f} = {SHARE['Mining rewards']:g} %, genesis {genesis:,.0f})   issued by the integer schedule: {issued_vrl:,.2f}")
for name, v in alloc.items():
    print(f"  {name:22s} {v:14,.0f}  {SHARE[name]:5.2f} %")
for i, p in enumerate(TESTNETS, 1):
    print(f"    testnet {i}: {p:5.2f} %  {T * p / 100:12,.0f} VRL")
for r in rows:
    print(f"  year {r['year']:2d}: reward {r['reward_per_block']:8.4f}  mined {r['mined_in_year']:12,.0f}  total mined {r['mined_total']:12,.0f} ({r['pct_of_mining']:5.1f} % of mining)  supply {r['supply_total']:12,.0f} ({r['pct_of_max']:5.1f} % of max)")

# ------------------------------------------------------------------------------------------------ infographic (1920 x 1080, for X)
W, H = 1920, 1080
C = {"bg": "#07080b", "line": "#1b2029", "line2": "#283040", "ink": "#ecebe6", "ink2": "#b7bcc6", "mute": "#8a91a0", "dim": "#5a6272",
     "mint": "#9ff0e2", "lav": "#b9a6ff", "rose": "#f4bcd6", "eco": "#8f9bb3"}
COL = {"Mining rewards": C["lav"], "Testnet contributors": C["mint"], "Team & foundation": C["rose"], "Ecosystem & grants": C["eco"]}
fmtM = lambda v: f"{v / 1e6:.2f} M"
mark = open(os.path.join(HERE, "..", "..", "site", "mark.svg"), encoding="utf-8").read()
mark_inner = mark[mark.index(">") + 1: mark.rindex("</svg>")].replace('id="n"', 'id="mk"').replace("url(#n)", "url(#mk)")


def logo(x, y, size, opacity=1.0):
    return f'<svg x="{x}" y="{y}" width="{size}" height="{size}" viewBox="0 0 64 64" opacity="{opacity}">{mark_inner}</svg>'


# ---- supply chart (right)
YEARS = 10
cx0, cy0, cw, ch = 960, 420, 880, 280
ymax = 60e6
X = lambda m: cx0 + cw * m / (12 * YEARS)
Y = lambda v: cy0 + ch - ch * v / ymax
months = range(12 * YEARS + 1)
bands = [("Testnet contributors", [alloc["Testnet contributors"]] * len(months)), ("Team & foundation", [alloc["Team & foundation"]] * len(months)),
         ("Ecosystem & grants", [alloc["Ecosystem & grants"]] * len(months)), ("Mining rewards", [mined_by_month(m) for m in months])]
polys, base = [], [0.0] * len(months)
for name, vals in bands:
    top = [b + v for b, v in zip(base, vals)]
    pts = [f"{X(m):.1f},{Y(top[m]):.1f}" for m in months] + [f"{X(m):.1f},{Y(base[m]):.1f}" for m in reversed(months)]
    fill = "url(#gmine)" if name == "Mining rewards" else COL[name]
    polys.append(f'<polygon points="{" ".join(pts)}" fill="{fill}" fill-opacity="{0.92 if name != "Mining rewards" else 1}"/>')
    base = top
outline = " ".join(f"{X(m):.1f},{Y(base[m]):.1f}" for m in months)
grid = "".join(f'<line x1="{cx0}" x2="{cx0 + cw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{C["line"]}" stroke-width="1.5"/><text x="{cx0 - 14}" y="{Y(v) + 7:.1f}" text-anchor="end" class="ax">{int(v / 1e6)} M</text>'
               for v in range(0, int(ymax) + 1, 20_000_000))
halv = "".join(f'<line x1="{X(m):.1f}" x2="{X(m):.1f}" y1="{cy0 - 40}" y2="{cy0 + ch}" stroke="{C["line2"]}" stroke-dasharray="3 7" stroke-width="1.5"/>' for m in range(12, 12 * YEARS, 12))
xt = "".join(f'<text x="{X(m):.1f}" y="{cy0 + ch + 30}" text-anchor="middle" class="ax">{"Mainnet" if m == 0 else "Y" + str(m // 12)}</text>' for m in range(0, 12 * YEARS + 1, 24))
rew = "".join(f'<text x="{X(12 * y + 6):.1f}" y="{cy0 - 18}" text-anchor="middle" class="rw">{20 / 3 / 2 ** y:.2f}</text>' for y in range(0, 6))
rew += f'<text x="{cx0 - 14}" y="{cy0 - 18}" text-anchor="end" class="ax">VRL / block</text>'

# ---- allocations (left): donut + legend
dx, dy, R1, R2 = 390, 524, 152, 106
a0, arcs = -math.pi / 2, []
for label in ("Mining rewards", "Testnet contributors", "Team & foundation", "Ecosystem & grants"):
    a1 = a0 + 2 * math.pi * SHARE[label] / 100
    g = 0.014
    s0, s1 = a0 + g, a1 - g
    large = 1 if s1 - s0 > math.pi else 0
    arcs.append(f'<path d="M {dx + R1 * math.cos(s0):.1f} {dy + R1 * math.sin(s0):.1f} A {R1} {R1} 0 {large} 1 {dx + R1 * math.cos(s1):.1f} {dy + R1 * math.sin(s1):.1f} '
                f'L {dx + R2 * math.cos(s1):.1f} {dy + R2 * math.sin(s1):.1f} A {R2} {R2} 0 {large} 0 {dx + R2 * math.cos(s0):.1f} {dy + R2 * math.sin(s0):.1f} Z" fill="{COL[label]}"/>')
    a0 = a1
LEG = [("Mining rewards", "75 %", "100 % of every block, the reward halved every year"), ("Team & foundation", "10 %", "12-month cliff, then released over 36 months"),
       ("Ecosystem & grants", "8 %", "partnerships, job credits, audits, bounties: public tranches"), ("Testnet contributors", "7 %", "shared by five public testnets")]
leg, ly = [], 736
for name, pct, sub in LEG:
    leg.append(f'<rect x="80" y="{ly - 22}" width="20" height="20" rx="4" fill="{COL[name]}"/><text x="116" y="{ly - 3}" class="leg">{name}</text>'
               f'<text x="600" y="{ly - 3}" text-anchor="end" class="legn">{fmtM(alloc[name])}</text><text x="700" y="{ly - 3}" text-anchor="end" class="legp">{pct}</text>'
               f'<text x="116" y="{ly + 24}" class="sub">{sub}</text>')
    ly += 66

# ---- the five testnets (right, under the chart)
tn, bx0 = [], 960
for i, p in enumerate(TESTNETS):
    x = bx0 + i * 178
    h = 14 + 54 * p / TESTNETS[-1]
    first = i == 0
    tn.append(f'<rect x="{x}" y="{888 - h:.1f}" width="150" height="{h:.1f}" rx="6" fill="{C["mint"]}" fill-opacity="{1 if first else 0.34}"/>'
              f'<text x="{x}" y="920" class="tn1">Testnet {i + 1}</text><text x="{x}" y="946" class="tnp">{p:g} %</text>'
              f'<text x="{x}" y="972" class="sub">{T * p / 100:,.0f} VRL</text>')
tn.append(f'<text x="{bx0}" y="{888 - 14 - 54 * TESTNETS[0] / TESTNETS[-1] - 12:.1f}" class="now">23 OCT → 20 NOV 2026</text>')

fonts = os.path.join(HERE, "..", "..", "vearl-wallet-gui", "assets", "fonts")
for f in ("InstrumentSans-semibold.ttf", "InstrumentSans-regular.ttf", "InstrumentSerif-italic.ttf", "IBMPlexMono-regular.ttf", "IBMPlexMono-medium.ttf"):
    shutil.copy(os.path.join(fonts, f), os.path.join(OUT, f))
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>VRL tokenomics</title><style>
@font-face {{ font-family: S; src: url('InstrumentSans-semibold.ttf'); font-weight: 600; }}
@font-face {{ font-family: S; src: url('InstrumentSans-regular.ttf'); font-weight: 400; }}
@font-face {{ font-family: Se; src: url('InstrumentSerif-italic.ttf'); font-style: italic; }}
@font-face {{ font-family: M; src: url('IBMPlexMono-regular.ttf'); font-weight: 400; }}
@font-face {{ font-family: M; src: url('IBMPlexMono-medium.ttf'); font-weight: 500; }}
html, body {{ margin: 0; background: {C["bg"]}; }}
svg text {{ font-family: S; fill: {C["ink"]}; }}
.ax {{ font-family: M; font-size: 19px; fill: {C["mute"]}; }}
.rw {{ font-family: M; font-size: 20px; font-weight: 500; fill: {C["ink2"]}; }}
.leg {{ font-size: 27px; font-weight: 600; fill: {C["ink"]}; }} .legn {{ font-family: M; font-size: 23px; fill: {C["ink2"]}; }} .legp {{ font-family: M; font-size: 23px; font-weight: 500; fill: {C["ink"]}; }}
.sub {{ font-size: 20px; fill: {C["mute"]}; }}
.lab {{ font-family: M; font-size: 19px; letter-spacing: 5px; fill: {C["mint"]}; }}
.tn1 {{ font-size: 21px; font-weight: 600; }} .tnp {{ font-family: M; font-size: 21px; font-weight: 500; fill: {C["mint"]}; }}
.now {{ font-family: M; font-size: 16px; letter-spacing: 3px; fill: {C["mint"]}; }}
.fine {{ font-size: 18px; fill: {C["dim"]}; }}
</style></head><body>
<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
<defs>
<pattern id="g" width="48" height="48" patternUnits="userSpaceOnUse"><path d="M48 0H0V48" fill="none" stroke="#0d1016" stroke-width="1"/></pattern>
<radialGradient id="glA" cx="0.12" cy="0.02" r="0.7"><stop offset="0" stop-color="{C["mint"]}" stop-opacity="0.10"/><stop offset="1" stop-color="{C["mint"]}" stop-opacity="0"/></radialGradient>
<radialGradient id="glB" cx="0.95" cy="1" r="0.6"><stop offset="0" stop-color="{C["rose"]}" stop-opacity="0.09"/><stop offset="1" stop-color="{C["rose"]}" stop-opacity="0"/></radialGradient>
<linearGradient id="gr" x1="0" x2="1"><stop offset="0" stop-color="{C["mint"]}"/><stop offset="0.55" stop-color="{C["lav"]}"/><stop offset="1" stop-color="{C["rose"]}"/></linearGradient>
<linearGradient id="gmine" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#c9bbff"/><stop offset="1" stop-color="#8d78e6"/></linearGradient>
</defs>
<rect width="{W}" height="{H}" fill="{C["bg"]}"/><rect width="{W}" height="{H}" fill="url(#g)"/><rect width="{W}" height="{H}" fill="url(#glA)"/><rect width="{W}" height="{H}" fill="url(#glB)"/>
{logo(1330, -190, 740, 0.045)}
{logo(80, 40, 50)}
<text x="146" y="78" style="font-size:36px;font-weight:600;letter-spacing:-0.5px">vearl</text>
<text x="1840" y="76" text-anchor="end" class="lab">MAINNET · TOKENOMICS</text>
<text x="80" y="196" style="font-size:104px;font-weight:600;letter-spacing:-3px">VRL <tspan style="font-family:Se;font-style:italic;font-weight:400;fill:url(#gr)">tokenomics</tspan></text>
<text x="80" y="256" style="font-size:31px;fill:{C["ink2"]}">A fixed supply of <tspan style="font-weight:600;fill:{C["ink"]}">{T:,.0f} VRL</tspan>. <tspan style="font-weight:600;fill:{C["ink"]}">75 %</tspan> goes to the people who mine.</text>
<line x1="80" x2="1840" y1="292" y2="292" stroke="{C["line2"]}" stroke-width="1.5"/>
<text x="80" y="346" class="lab">ALLOCATIONS</text>
{"".join(arcs)}
<text x="{dx}" y="{dy + 4}" text-anchor="middle" style="font-size:44px;font-weight:600;letter-spacing:-1px">{T / 1e6:.2f} M</text>
<text x="{dx}" y="{dy + 34}" text-anchor="middle" style="font-size:19px;fill:{C["mute"]}">VRL, maximum</text>
{"".join(leg)}
<text x="960" y="346" class="lab">SUPPLY OVER TIME</text>
<text x="1840" y="346" text-anchor="end" style="font-size:20px;fill:{C["mute"]}">all VRL ever created, locked shares included</text>
{grid}{halv}{"".join(polys)}
<polyline points="{outline}" fill="none" stroke="{C["ink"]}" stroke-width="2.5"/>
{rew}{xt}
<text x="{X(120) - 4}" y="{Y(base[-1]) - 14:.1f}" text-anchor="end" class="rw" style="fill:{C["mint"]}">{base[-1] / 1e6:.1f} M after 10 years</text>
<text x="960" y="786" class="lab">TESTNET CONTRIBUTORS · 7 % · A COMMITMENT OF THE TEAM</text>
{"".join(tn)}
<line x1="80" x2="1840" y1="1004" y2="1004" stroke="{C["line"]}" stroke-width="1.5"/>
<text x="80" y="1040" class="fine">Testnet contributors: the 7 % is a commitment of the team. The other mainnet parameters are final once published in the mainnet code. Testnet tokens have no value.</text>
<text x="80" y="1066" class="fine">No price or value is promised. 10 % of every paid job is burned, so the real supply can only be lower than the maximum. Not investment advice.</text>
<text x="1840" y="1066" text-anchor="end" style="font-family:M;font-size:22px;fill:{C["mute"]}">vearl.org</text>
</svg></body></html>"""
path = os.path.join(OUT, "tokenomics.html")
open(path, "w", encoding="utf-8").write(html)
chrome = "C:/Program Files/Google/Chrome/Application/chrome.exe"
if os.path.exists(chrome):
    png = os.path.join(OUT, "tokenomics.png")
    subprocess.run([chrome, "--headless=new", "--hide-scrollbars", "--force-device-scale-factor=1", f"--window-size={W},{H}", f"--screenshot={png}", "file:///" + path.replace("\\", "/")], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("written", png)
