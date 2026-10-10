#!/usr/bin/env python3
"""Writes the download page of a release (site/download/index.html) from the archives built by make_release.py.

    python deploy/release/make_download_page.py --dist dist --version 0.1.0-rc9 [--base /download/files] [--out site/download/index.html]
                                                [--private-token TOKEN]

The page lists, for Windows and Linux, the main archive (mines Vearl only) and the optional Quantus add-on, with size and SHA-256 (read
from dist/SHA256SUMS.txt), the genesis file and the steps to start. --base is the URL prefix under which the archives are served
(default /download/files); --private-token makes a noindex copy meant for a private address (https://vearl.org/p-TOKEN/download/).
"""
import argparse
import html
import os
import sys

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
{robots}<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Vearl Download · Node and miner for the testnet</title>
<meta name="description" content="Download the Vearl miner and node for the public testnet. Windows and Linux, NVIDIA GPU. Test tokens have no value.">
<meta name="theme-color" content="#07080b">
<meta name="color-scheme" content="dark">
<meta name="vearl-explorer" content="/explorer/">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="icon" href="/favicon-32.png" sizes="32x32" type="image/png">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
{canonical}<meta property="og:type" content="website">
<meta property="og:site_name" content="Vearl">
<meta property="og:title" content="Vearl Download">
<meta property="og:description" content="The Vearl miner and node for the public testnet.">
<meta property="og:image" content="https://vearl.org/og-image.png?v=3">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:site" content="@vearl_org">
<link rel="preload" href="/fonts/instrument-sans-normal-600.woff2" as="font" type="font/woff2" crossorigin>
<link rel="preload" href="/fonts/ibm-plex-mono-normal-500.woff2" as="font" type="font/woff2" crossorigin>
<link rel="preload" href="/fonts/instrument-serif-italic-400.woff2" as="font" type="font/woff2" crossorigin>
<link rel="stylesheet" href="/fonts/fonts.css">
<link rel="stylesheet" href="/assets/site.css">
<link rel="stylesheet" href="/assets/explorer.css">
<link rel="stylesheet" href="/assets/compute.css">
</head>
<body class="xp cp">
<a class="skip" href="#app">Skip to content</a>

<header class="nav solid" id="nav">
  <div class="wrap">
    <a class="brand" href="/" aria-label="Vearl, home"><img src="/mark.svg" alt="" width="28" height="28">vearl<span class="x-sub">download</span></a>
    <ul class="nav-links" id="menu">
      <li><a href="/download/" aria-current="page">Download</a></li>
      <li><a href="/explorer/" data-ext="explorer">Explorer</a></li>
      <li><a href="/whitepaper">Whitepaper</a></li>
      <li><a href="/">vearl.org</a></li>
    </ul>
    <button class="burger" id="burger" aria-label="Open menu" aria-expanded="false" aria-controls="menu"><span></span><span></span><span></span></button>
  </div>
</header>

<main class="x-main" id="app" tabindex="-1">
  <section class="x-hero cp-hero"><div class="wrap">
    <div class="eyebrow">Vearl testnet · release {version}</div>
    <h1>Run a miner. <span class="serif">One download, one double-click.</span></h1>
    <p class="lede">The Vearl miner mines Vearl blocks with your NVIDIA GPU and computes the paid jobs of the network. It mines Vearl only: nothing else runs on your machine. Testnet tokens have no value and are not for sale.</p>
  </div></section>

  <div class="wrap">
    <div class="studio">
      <div>
        <section class="panel">
          <div class="panel-head"><h2>Miner and node</h2></div>
          <div class="panel-body">
<div class="kv">
{main_rows}
            </div>
            <p class="fine">Needs an NVIDIA GPU of the Ampere generation or newer (RTX 30xx, 40xx, 50xx, A-series, L-series, H100) and the NVIDIA driver (Windows 580 or newer, Linux 575 or newer). Without a GPU the miner runs on the CPU, slowly.</p>
          </div>
        </section>
        <section class="panel">
          <div class="panel-head"><h2>Network file</h2></div>
          <div class="panel-body">
<div class="kv">
{genesis_rows}
            </div>
            <p class="fine">Already inside the archives, next to <span class="mono">network.json</span>. Every miner must use the same genesis file. The testnet starts on October 23, 2026 at 16:00 UTC.</p>
          </div>
        </section>
        <section class="panel">
          <div class="panel-head"><h2>Optional: Quantus add-on</h2></div>
          <div class="panel-body">
            <p class="fine">Not needed to mine Vearl, and off unless you install it AND answer yes in the setup assistant. It adds the official open-source Quantus miner so the same GPU can also mine Quantus (QTC) to your own address. Ten percent of that Quantus mining time goes to the Vearl development address, published in the add-on&rsquo;s README. Extract it into the miner folder.</p>
<div class="kv">
{addon_rows}
            </div>
          </div>
        </section>
      </div>
      <div>
        <section class="panel">
          <div class="panel-head"><h2>Start in four steps</h2></div>
          <div class="panel-body">
            <ol class="steps">
              <li>Download the archive for your system and extract it anywhere.</li>
              <li>Windows: double-click <span class="mono">vearl-miner.exe</span>. Linux: run <span class="mono">./vearl-miner</span> in a terminal.</li>
              <li>The assistant creates your wallet and shows 24 recovery words once. Write them on paper and never share them: nobody from the project will ever ask for them.</li>
              <li>Leave it running. The first start downloads the chain from the seed node; a status line appears every 10 seconds.</li>
            </ol>
          </div>
        </section>
        <section class="panel">
          <div class="panel-head"><h2>Check what you downloaded</h2></div>
          <div class="panel-body">
            <p class="fine">Compare the SHA-256 of the file with the value shown next to it. Windows (PowerShell): <span class="mono">Get-FileHash FILE -Algorithm SHA256</span>. Linux: <span class="mono">sha256sum FILE</span>. All values are also in <a class="lnk" href="{base}/SHA256SUMS.txt">SHA256SUMS.txt</a>.</p>
            <p class="fine">Only download from this page. Anything else claiming to be the Vearl miner is not ours. The code has not been audited by a third party; run it on a machine you do not mind testing on.</p>
          </div>
        </section>
      </div>
    </div>
  </div>
</main>

<footer class="footer">
  <div class="wrap">
    <div class="footer-top">
      <div>
        <a class="brand" href="/"><img src="/mark.svg" alt="" width="28" height="28">vearl</a>
        <div class="tagline">Proof of Useful Work · Verified Compute</div>
      </div>
      <div>
        <h3>Testnet</h3>
        <ul><li><a href="/download/">Download</a></li><li><a href="/explorer/" data-ext="explorer">Explorer</a></li><li><a href="/whitepaper">Whitepaper</a></li></ul>
      </div>
      <div>
        <h3>Vearl</h3>
        <ul><li><a href="/">Website</a></li><li><a href="https://discord.gg/jya8meKnvM" target="_blank" rel="noopener">Discord</a></li><li><a href="https://x.com/vearl_org" target="_blank" rel="noopener">X</a></li></ul>
      </div>
    </div>
    <div class="footer-bottom">
      <div class="legal">Testnet. Tokens have no value and are not for sale. The code has not been audited by a third party, and the network can be reset. Nothing here is investment advice.</div>
      <div>© Vearl</div>
    </div>
  </div>
</footer>

<script src="/assets/site.js"></script>
</body>
</html>
"""

ROW = """              <div class="k">{label}</div>
              <div class="v"><div class="mono">{name}</div><div class="fine">{size}</div><div class="fine mono">SHA-256 {sha}</div><p><a class="btn" href="{href}" download>Download</a></p></div>"""


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}".replace(".0 ", " ")
        n /= 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--base", default="/download/files")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "site", "download", "index.html"))
    ap.add_argument("--private-token", help="make a noindex copy for the private address /p-TOKEN/download/")
    a = ap.parse_args()
    sums = {}
    with open(os.path.join(a.dist, "SHA256SUMS.txt")) as f:
        for line in f:
            h, n = line.split()
            sums[n] = h
    top = f"vearl-{a.version}"

    def row(label, name):
        if name not in sums:
            sys.exit(f"{name} is not listed in SHA256SUMS.txt")
        size = human(os.path.getsize(os.path.join(a.dist, name)))
        return ROW.format(label=html.escape(label), href=f"{a.base}/{name}", name=html.escape(name), size=size, sha=sums[name])

    main_rows = "\n".join([row("Windows 10/11 (x86-64)", f"{top}-windows-x86_64.zip"), row("Linux (x86-64, glibc 2.35+)", f"{top}-linux-x86_64.tar.gz")])
    addon_rows = "\n".join(
        row(label, n) for label, n in (("Windows add-on", f"{top}-quantus-addon-windows-x86_64.zip"), ("Linux add-on", f"{top}-quantus-addon-linux-x86_64.tar.gz")) if n in sums
    )
    genesis_rows = row("genesis.json", "genesis.json")
    if a.private_token:
        robots = '<meta name="robots" content="noindex, nofollow">\n'
        canonical = ""
    else:
        robots = ""
        canonical = '<link rel="canonical" href="https://vearl.org/download/">\n'
    page = PAGE.format(version=html.escape(a.version), main_rows=main_rows, addon_rows=addon_rows, genesis_rows=genesis_rows, base=a.base, robots=robots, canonical=canonical)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(page)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
