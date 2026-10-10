#!/usr/bin/env python3
"""Photos of the "Testnet replay" feed: CC0 photos (tools/web-photos, licence verified by fetch_cc0.py) classified by DeiT-Tiny ON THE
CHAIN (vit-photos.json: output of tools/vit_demo.py --photos --out, verified operation by operation and identical to the reference
engine). Each photo is written as the model sees it (short side 256, central crop 224 x 224); assets/photos.js carries
the chain's answer and the credits; credits.html lists author, licence and source page.

    python3 make_photo_assets.py SITE_DIR
"""
import html, json, os, sys
from PIL import Image

SITE = sys.argv[1] if len(sys.argv) > 1 else ".."
SRC = os.path.join(SITE, "tools", "web-photos")
PICK = ["king-penguin-2", "macaw-3", "monarch-butterfly-1", "golden-retriever-2", "koala-2", "hot-air-balloon-1", "red-fox-2",
        "tabby-cat-1", "tiger-3", "zebra-2", "pizza-1", "strawberry-2", "orange-fruit-3", "espresso-3", "flamingo-1", "goldfish-1",
        "lighthouse-1", "acoustic-guitar-3", "school-bus-1", "banana-2"]
# ImageNet name -> displayed label (same class, common word) when the first synonym is not very telling
DISPLAY = {"beacon": "lighthouse", "balloon": "hot-air balloon", "monarch": "monarch butterfly", "orange": "orange", "tabby": "tabby cat",
           "tiger": "tiger", "goldfish": "goldfish", "koala": "koala", "king penguin": "king penguin"}

res = {r["file"]: r for r in json.load(open(os.path.join(SRC, "vit-photos.json"), encoding="utf-8"))["rows"]}
cred = {c["file"]: c for c in json.load(open(os.path.join(SRC, "credits.json"), encoding="utf-8"))}
out_dir = os.path.join(SITE, "assets", "photos")
os.makedirs(out_dir, exist_ok=True)
items = []
for name in PICK:
    f = name + ".jpg"
    r, c = res[f], cred[f]
    assert r["bit_exact"], f
    img = Image.open(os.path.join(SRC, f)).convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BICUBIC)
    w, h = img.size
    img = img.crop(((w - 224) // 2, (h - 224) // 2, (w - 224) // 2 + 224, (h - 224) // 2 + 224))
    img.save(os.path.join(out_dir, f), quality=84, optimize=True, progressive=True)
    first = r["label"].split(",")[0].strip()
    items.append({"src": f"/assets/photos/{f}", "label": DISPLAY.get(first, first), "imagenet": r["label"], "cls": r["pred"],
                  "artist": c["artist"], "page": c["page"]})
    print(f"{f:26s} -> {items[-1]['label']}")
open(os.path.join(SITE, "assets", "photos.js"), "w", encoding="utf-8").write(
    "/* CC0 photos (Wikimedia Commons) classified by DeiT-Tiny on the testnet: verified answers, identical to the reference engine. */\n"
    "window.VEARL_PHOTOS = " + json.dumps(items, ensure_ascii=False, indent=1) + ";\n")
rows = "\n".join(
    f'      <tr><td><img src="{html.escape(i["src"])}" alt="" width="56" height="56" loading="lazy"></td><td>{html.escape(i["label"])}</td>'
    f'<td>{html.escape(i["artist"] or "Unknown author")}</td><td>CC0 1.0</td><td><a href="{html.escape(i["page"])}" rel="noopener">Wikimedia Commons</a></td></tr>'
    for i in items)
page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Photo credits · Vearl</title>
<meta name="theme-color" content="#07080b">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="/fonts/fonts.css">
<link rel="stylesheet" href="/assets/site.css">
</head>
<body>
<main class="credits wrap">
  <a class="brand" href="/"><img src="/mark.svg" alt="" width="28" height="28">vearl</a>
  <div class="eyebrow" style-free>Photo credits</div>
  <h1>Photos used in the <span class="serif">testnet replay.</span></h1>
  <p>Every photo below was classified by DeiT-Tiny running on the Vearl testnet, each operation verified before payment. The photos are
  released under CC0 1.0 (public domain dedication) on Wikimedia Commons; credit is given here as a courtesy. Images are shown exactly as the
  model sees them: 224 × 224 pixels, center crop.</p>
  <table class="credit-table">
    <thead><tr><th></th><th>Network answer</th><th>Author</th><th>License</th><th>Source</th></tr></thead>
    <tbody>
{rows}
    </tbody>
  </table>
  <a class="btn" href="/">Back to the homepage</a>
</main>
</body>
</html>
""".replace(" style-free", "")
open(os.path.join(SITE, "credits.html"), "w", encoding="utf-8").write(page)
print(len(items), "photos;", sum(os.path.getsize(os.path.join(out_dir, i["src"].split("/")[-1])) for i in items) // 1024, "KiB")
