#!/usr/bin/env python3
"""Download the site's fonts once (latin subset, woff2) so the site serves them itself: no request to Google from visitors.
Also downloads TTF files used to render the social-preview image.

    python3 fetch_fonts.py OUT_DIR
"""
import os, re, sys, urllib.request

OUT = sys.argv[1]
os.makedirs(f"{OUT}/fonts", exist_ok=True)
os.makedirs(f"{OUT}/build-fonts", exist_ok=True)
FAMILIES = "family=Instrument+Sans:wght@400;500;600;700&family=Instrument+Serif:ital@0;1&family=IBM+Plex+Mono:wght@400;500;600"
WOFF2_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
TTF_UA = "curl/8"   # a non-browser client gets TTF files


def get(url, ua):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": ua}), timeout=30).read()


# 1) woff2, latin subset only
css = get(f"https://fonts.googleapis.com/css2?{FAMILIES}&display=swap", WOFF2_UA).decode()
blocks = re.findall(r"/\* ([a-z-]+) \*/\s*(@font-face \{.*?\})", css, re.S)
out = []
for subset, block in blocks:
    if subset != "latin":
        continue
    fam = re.search(r"font-family: '([^']+)'", block).group(1)
    style = re.search(r"font-style: (\w+)", block).group(1)
    weight = re.search(r"font-weight: ([\d ]+);", block).group(1).replace(" ", "-")
    url = re.search(r"url\((https://[^)]+\.woff2)\)", block).group(1)
    name = f"{fam.lower().replace(' ', '-')}-{style}-{weight}.woff2"
    with open(f"{OUT}/fonts/{name}", "wb") as f:
        f.write(get(url, WOFF2_UA))
    out.append(re.sub(r"url\(https://[^)]+\)", f"url(/fonts/{name})", block))
    print("woff2", name)
with open(f"{OUT}/fonts/fonts.css", "w") as f:
    f.write("/* Instrument Sans, Instrument Serif, IBM Plex Mono: SIL Open Font License 1.1 */\n" + "\n".join(out) + "\n")

# 2) TTF for the social-preview image (not served)
css = get(f"https://fonts.googleapis.com/css2?{FAMILIES}", TTF_UA).decode()
for block in re.findall(r"@font-face \{.*?\}", css, re.S):
    fam = re.search(r"font-family: '([^']+)'", block).group(1)
    style = re.search(r"font-style: (\w+)", block).group(1)
    weight = re.search(r"font-weight: (\d+)", block).group(1)
    m = re.search(r"url\((https://[^)]+\.ttf)\)", block)
    if not m:
        continue
    name = f"{fam.lower().replace(' ', '-')}-{style}-{weight}.ttf"
    with open(f"{OUT}/build-fonts/{name}", "wb") as f:
        f.write(get(m.group(1), TTF_UA))
    print("ttf", name)
