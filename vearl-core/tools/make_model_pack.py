#!/usr/bin/env python3
"""Builds the model packs that miners install in advance (`node --models-dir`, `vearl-miner models`).

A pack (`NAME.vblocks`) holds the weight blocks of a model: every base64 n x n int8 matrix found under a "weights" key of the model file
(gpt-qat.json, vit-qat.json, qwen3-embedding-0.6b-int8.json, or any file of tools/*.py / vearl-model). A node that has the pack answers
`HasModel` with "present" for these blocks, so a client sends only its input: a job names the model, the miner already holds it.

Layout of a pack: 8 bytes `VBLOCK01`, n (u32 LE), number of blocks (u32 LE), then the blocks (n*n int8 each, row-major, no duplicates).

  python make_model_pack.py --n 1024 --out packs --base-url https://.../ gpt=/path/gpt-qat.json vit=/path/vit-qat.json qwen3=/path/q.json[.xz]

Also writes OUT/models.json (name, file, bytes, sha256, blocks, url): the manifest `vearl-miner models` downloads from.
"""
import argparse, base64, hashlib, json, lzma, os, struct, sys

VAL_MAX = 64


def load_json(path):
    if path.endswith(".xz"):
        with lzma.open(path, "rb") as f:
            return json.load(f)
    with open(path, "rb") as f:
        return json.load(f)


def weight_strings(node, out, under_weights=False):
    """Collects the strings found under a key called "weights" (a list, or a dict whose values are the matrices)."""
    if isinstance(node, dict):
        for k, v in node.items():
            weight_strings(v, out, under_weights or k == "weights")
    elif isinstance(node, list):
        for v in node:
            weight_strings(v, out, under_weights)
    elif isinstance(node, str) and under_weights:
        out.append(node)


def build(name, src, n, out_dir, base_url):
    model = load_json(src)
    strs = []
    weight_strings(model, strs)
    del model
    seen, blocks = set(), []
    for s in strs:
        raw = base64.b64decode(s)
        if len(raw) != n * n:
            continue  # not a weight block of this size (a vector, a table...)
        if any(((b ^ 128) - 128) > VAL_MAX or ((b ^ 128) - 128) < -VAL_MAX for b in set(raw)):
            raise SystemExit(f"{name}: a block holds a value outside +-{VAL_MAX}")
        h = hashlib.sha256(raw).digest()
        if h not in seen:
            seen.add(h)
            blocks.append(raw)
    if not blocks:
        raise SystemExit(f"{name}: no {n}x{n} weight block found in {src}")
    path = os.path.join(out_dir, f"{name}.vblocks")
    with open(path, "wb") as f:
        f.write(b"VBLOCK01" + struct.pack("<II", n, len(blocks)))
        for b in blocks:
            f.write(b)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    size = os.path.getsize(path)
    print(f"{name}: {len(strs)} matrices, {len(blocks)} distinct blocks, {size / 2**20:.0f} MiB, sha256 {sha[:16]}...")
    return {"name": name, "file": f"{name}.vblocks", "bytes": size, "sha256": sha, "blocks": len(blocks), "url": (base_url + f"{name}.vblocks") if base_url else ""}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="")
    ap.add_argument("models", nargs="+", help="NAME=model.json[.xz]")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    packs = []
    for spec in a.models:
        name, _, src = spec.partition("=")
        if not src:
            sys.exit(f"expected NAME=FILE, got {spec}")
        packs.append(build(name, src, a.n, a.out, a.base_url))
    with open(os.path.join(a.out, "models.json"), "w") as f:
        json.dump({"version": 1, "n": a.n, "packs": packs}, f, indent=2)
    print("manifest:", os.path.join(a.out, "models.json"))


if __name__ == "__main__":
    main()
