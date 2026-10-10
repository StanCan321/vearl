"""X (Twitter) banner 1500x500 in the Vearl style. Run: python brand/make_banner.py   (uses site/tools/make_images.py)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "site", "tools"))
sys.argv = ["x", os.path.join(HERE, "..", "site")]
import make_images as m
from PIL import Image, ImageDraw

W, H = 1500, 500
img = Image.new("RGBA", (W, H), m.BG + (255,))
layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
m.ascii_pearl(layer, 1185, 250, 16, 13)
img.alpha_composite(layer)
fade = Image.new("RGBA", (W, H), (0, 0, 0, 0)); fd = ImageDraw.Draw(fade)
for x in range(W):                                  # the left side stays dark so the text reads, and the profile photo sits on calm ground
    a = int(max(0, min(1, (860 - x) / 330)) * 225)
    fd.line([(x, 0), (x, H)], fill=m.BG + (a,))
img.alpha_composite(fade)
d = ImageDraw.Draw(img)
X = 250                                             # clear of the profile photo that X puts at the bottom left
d.text((X, 108), "PROOF OF USEFUL WORK · VERIFIED COMPUTE", font=m.font("ibm-plex-mono-normal-500.ttf", 19), fill=(159, 240, 226))
d.text((X - 3, 150), "Bitcoin mines hashes.", font=m.font("instrument-sans-normal-600.ttf", 58), fill=(236, 235, 230))
d.text((X - 1, 218), "Vearl mines AI.", font=m.font("instrument-serif-italic-400.ttf", 88), fill=(185, 166, 255))
d.text((X, 352), "PUBLIC TESTNET 23 OCT 2026   ·   vearl.org", font=m.font("ibm-plex-mono-normal-500.ttf", 20), fill=(244, 188, 214))
img.convert("RGB").save(os.path.join(HERE, "vearl-x-banner-1500x500.png"), optimize=True)
print("ok")
