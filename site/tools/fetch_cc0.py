#!/usr/bin/env python3
"""CC0 photos (voluntary public domain) from Wikimedia Commons, licence VERIFIED file by file through the API (extmetadata):
only the files whose short licence is exactly "CC0" are kept. Writes the thumbnails (640 px) and credits.json.

    python3 fetch_cc0.py OUT_DIR [par_sujet]
"""
import json, os, sys, time, urllib.error, urllib.parse, urllib.request

OUT = sys.argv[1]
PER = int(sys.argv[2]) if len(sys.argv) > 2 else 3
UA = "VearlTestnetSite/0.1 (photo credits for a testnet demo page; contact via project repository)"
API = "https://commons.wikimedia.org/w/api.php"
SUBJECTS = ["zebra", "red fox", "golden retriever", "tabby cat", "king penguin", "monarch butterfly", "goldfish", "hot air balloon",
            "school bus", "sports car", "lighthouse", "espresso", "pizza", "strawberry", "acoustic guitar", "macaw", "tiger", "orange fruit",
            "teddy bear", "sailboat", "steam locomotive", "banana", "flamingo", "koala"]


def fetch(url):
    """Polite GET: 1.5 s between two requests, respectful waiting (Retry-After) on a 429."""
    for attempt in range(8):
        time.sleep(1.5)
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60).read()
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            wait = int(e.headers.get("Retry-After", "0") or 0) or 30 * (attempt + 1)
            print(f"  429: waiting {wait} s", flush=True)
            time.sleep(wait)
    raise RuntimeError("too many 429 refusals")


def get(params):
    return json.loads(fetch(API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})))


def strip_html(s):
    import re
    return re.sub(r"<[^>]+>", "", s or "").strip()


os.makedirs(OUT, exist_ok=True)
cpath = os.path.join(OUT, "credits.json")
credits = json.load(open(cpath)) if os.path.exists(cpath) else []
for subj in SUBJECTS:
    if any(c["subject"] == subj for c in credits):
        continue                                                   # resume: subject already processed
    # files whose structured data declare the CC0 licence (P275 = Q6938433), JPEG photos
    r = get({"action": "query", "list": "search", "srnamespace": 6, "srlimit": 25,
             "srsearch": f'{subj} filetype:bitmap filemime:image/jpeg haswbstatement:P275=Q6938433'})
    titles = [h["title"] for h in r["query"]["search"]]
    kept = 0
    for i in range(0, len(titles), 10):
        info = get({"action": "query", "titles": "|".join(titles[i:i + 10]), "prop": "imageinfo",
                    "iiprop": "url|extmetadata|size", "iiurlwidth": 640})
        for pg in info["query"]["pages"]:
            if kept >= PER or "imageinfo" not in pg:
                continue
            ii = pg["imageinfo"][0]
            md = ii.get("extmetadata", {})
            lic = md.get("LicenseShortName", {}).get("value", "")
            if lic.strip() != "CC0" or ii.get("width", 0) < 640 or ii.get("height", 0) < 480:
                continue
            name = f"{subj.replace(' ', '-')}-{kept + 1}.jpg"
            data = fetch(ii["thumburl"])
            open(os.path.join(OUT, name), "wb").write(data)
            credits.append({"file": name, "subject": subj, "title": pg["title"], "page": ii["descriptionurl"], "license": lic,
                            "artist": strip_html(md.get("Artist", {}).get("value", "")), "bytes": len(data)})
            kept += 1
            json.dump(credits, open(cpath, "w"), indent=1, ensure_ascii=False)
        if kept >= PER:
            break
    print(f"{subj:18s}: {kept} photo(s) CC0")
json.dump(credits, open(os.path.join(OUT, "credits.json"), "w"), indent=1, ensure_ascii=False)
print(len(credits), "photos, credits.json written")
