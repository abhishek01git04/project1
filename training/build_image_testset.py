"""Build a frozen image test set from OpenFake's core test split (CC BY-NC 4.0).

Only the small text columns of a few test files are read to pick images, and then only the row groups
that hold the picked images are downloaded, so a few hundred MB are fetched instead of terabytes.
Fakes are spread evenly over generators and reals over sources.

Each image is saved twice:
  clean/   the original file, byte for byte (checked against OpenFake's SHA-256)
  shared/  a copy degraded like a re-shared image: shorter side 280-640 px, JPEG quality 60-85, and
           for half of them a white screenshot border. The settings are fixed per image, so a rebuild
           gives the same files.

This set is for measuring only. Never train, tune thresholds or pick models on it.

Usage (from the project root):
  ai-gateway\\.venv\\Scripts\\python training\\build_image_testset.py --index <index.json> --n 600
"""

import argparse
import hashlib
import io
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image

from common import ROOT, SEED

REPO = "datasets/ComplexDataLab/OpenFake/core"
OUT_DIR = ROOT / "data" / "image-test-openfake"
TEXT_COLUMNS = ["label", "model", "type", "release_date", "row_id"]
FETCH_COLUMNS = ["image", "row_id", "image_sha256"]
EXTENSIONS = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp", "GIF": "gif", "BMP": "bmp", "TIFF": "tif"}

README = """# Frozen image test set ({n} images)

**For measuring only. Never train, tune thresholds or choose models on these images**, or the accuracy
reported on them stops meaning anything.

- `clean/` holds the original files from OpenFake's core test split; `shared/` holds the same images degraded
  like re-shared copies (smaller, JPEG-compressed, some with a white screenshot border).
- `manifest.jsonl` lists each image's label, generator or real source, SHA-256 and degradation settings.
- Built by `training/build_image_testset.py` with a fixed seed.

Source: OpenFake (ComplexDataLab), https://huggingface.co/datasets/ComplexDataLab/OpenFake,
licensed CC BY-NC 4.0: non-commercial use, with attribution. Cite the OpenFake paper
(arXiv:2509.09495) when reporting results on this set.
"""

fs = HfFileSystem()


def with_retries(what: str, fn, attempts: int = 8):
    """The connection to the Hub drops often on this network, so every read is retried."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            print(f"  retry {attempt} ({what}): {str(exc)[:90]}", flush=True)
            time.sleep(min(60, 5 * attempt))
    raise SystemExit(f"Gave up: {what}")


def read_row_group(shard: str, rg: int, columns: list[str], block: int) -> list[dict]:
    def read():
        with fs.open(f"{REPO}/{shard}", "rb", block_size=block) as f:
            return pq.ParquetFile(f).read_row_group(rg, columns=columns).to_pylist()
    return with_retries(f"{shard} row group {rg}", read)


def build_index(path: Path, shards: int) -> list[dict]:
    """Text columns of `shards` test files spread across the split. Resumable."""
    names = sorted(p.split("/")[-1] for p in fs.ls(REPO, detail=False) if "/test-" in p)
    picked = [names[round(i * (len(names) - 1) / max(1, shards - 1))] for i in range(shards)]
    index = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    done = {(r["shard"], r["rg"]) for r in index}
    for shard in picked:
        groups = with_retries(f"{shard} metadata", lambda: pq.ParquetFile(
            fs.open(f"{REPO}/{shard}", "rb", block_size=1 << 20)).metadata.num_row_groups)
        for rg in range(groups):
            if (shard, rg) not in done:
                index += [{"shard": shard, "rg": rg, "row": i, **r}
                          for i, r in enumerate(read_row_group(shard, rg, TEXT_COLUMNS, 1 << 20))]
                path.write_text(json.dumps(index), encoding="utf-8")
        print(f"indexed {shard}: {len(index)} rows", flush=True)
    return index


def round_robin(rows: list[dict], key: str, n: int, rng: random.Random) -> list[dict]:
    """n rows spread as evenly as possible over the values of `key`."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[r[key]].append(r)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    order = sorted(buckets)
    rng.shuffle(order)
    picked = []
    while len(picked) < n and any(buckets.values()):
        for value in order:
            if buckets[value] and len(picked) < n:
                picked.append(buckets[value].pop())
    return picked


def choose(index: list[dict], n: int, max_groups: int) -> list[dict]:
    """Pick row groups until both classes can be filled and every generator in the index is covered."""
    rng = random.Random(SEED)
    by_group = defaultdict(list)
    for r in index:
        by_group[(r["shard"], r["rg"])].append(r)
    all_generators = {r["model"] for r in index if r["label"] == "fake"}
    groups = sorted(by_group)
    rng.shuffle(groups)
    chosen, rows, covered = [], [], set()
    def short():  # 1.5x the need, so the round robin can still spread over generators
        counts = Counter(r["label"] for r in rows)
        return counts["fake"] < n // 2 * 1.5 or counts["real"] < n // 2 * 1.5

    for g in groups:
        fakes = {r["model"] for r in by_group[g] if r["label"] == "fake"}
        if short() or fakes - covered:
            chosen.append(g)
            rows += by_group[g]
            covered |= fakes
        if (not short() and covered == all_generators) or len(chosen) >= max_groups:
            break
    print(f"{len(chosen)} row groups chosen; generators covered {len(covered)}/{len(all_generators)}")
    fakes = round_robin([r for r in rows if r["label"] == "fake"], "model", n // 2, rng)
    reals = round_robin([r for r in rows if r["label"] == "real"], "model", n // 2, rng)
    return fakes + reals


def degrade(img: Image.Image, row_id: str) -> tuple[Image.Image, dict]:
    """Same settings for the same image every time."""
    rng = random.Random(f"{SEED}-{row_id}")
    short_side = rng.randint(280, 640)
    quality = rng.randint(60, 85)
    border = rng.choice([0, rng.randint(2, 14)])
    img = img.convert("RGB")
    scale = min(1.0, short_side / min(img.size))
    img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    if border:
        framed = Image.new("RGB", (img.width + 2 * border, img.height + 2 * border), (255, 255, 255))
        framed.paste(img, (border, border))
        img = framed
    return img, {"shortSide": min(img.size), "jpegQuality": quality, "border": border}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, required=True, help="text-column index (built and resumed here)")
    ap.add_argument("--shards", type=int, default=3, help="test files to index")
    ap.add_argument("--n", type=int, default=600, help="images, half fake and half real")
    ap.add_argument("--max-row-groups", type=int, default=16, help="about 100 MB each")
    ap.add_argument("--index-as-is", action="store_true",
                    help="use the rows already in --index; row groups are mixed, so a partial index is enough")
    args = ap.parse_args()

    if args.index_as_is:
        index = json.loads(args.index.read_text(encoding="utf-8"))
    else:
        index = build_index(args.index, args.shards)
    print(f"index: {len(index)} rows in {len({(r['shard'], r['rg']) for r in index})} row groups")
    picked = choose(index, args.n, args.max_row_groups)
    wanted = defaultdict(dict)
    for r in picked:
        wanted[(r["shard"], r["rg"])][r["row"]] = r

    manifest_path = OUT_DIR / "manifest.jsonl"
    done = set()
    if manifest_path.exists():
        done = {json.loads(line)["rowId"] for line in manifest_path.read_text(encoding="utf-8").splitlines() if line}
    bad_hash = 0
    for (shard, rg), rows in sorted(wanted.items()):
        if all(r["row_id"] in done for r in rows.values()):
            continue
        data = read_row_group(shard, rg, FETCH_COLUMNS, 8 << 20)
        for row_no, meta in rows.items():
            if meta["row_id"] in done:
                continue
            raw = data[row_no]["image"]["bytes"]
            sha = hashlib.sha256(raw).hexdigest()
            bad_hash += sha != data[row_no]["image_sha256"]
            img = Image.open(io.BytesIO(raw))
            ext = EXTENSIONS.get(img.format, "img")
            clean = OUT_DIR / "clean" / meta["label"] / f"{meta['row_id']}.{ext}"
            shared = OUT_DIR / "shared" / meta["label"] / f"{meta['row_id']}.jpg"
            clean.parent.mkdir(parents=True, exist_ok=True)
            shared.parent.mkdir(parents=True, exist_ok=True)
            clean.write_bytes(raw)
            small, settings = degrade(img, meta["row_id"])
            small.save(shared, "JPEG", quality=settings["jpegQuality"])
            record = {"rowId": meta["row_id"], "label": meta["label"], "model": meta["model"],
                      "type": meta["type"], "releaseDate": meta["release_date"], "shard": shard, "rowGroup": rg,
                      "sha256": sha, "size": list(img.size), "clean": clean.relative_to(OUT_DIR).as_posix(),
                      "shared": shared.relative_to(OUT_DIR).as_posix(), "sharedSettings": settings}
            with manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            done.add(meta["row_id"])
        print(f"{shard} rg {rg}: {len(done)}/{len(picked)} saved", flush=True)

    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    (OUT_DIR / "README.md").write_text(README.format(n=len(records)), encoding="utf-8")
    print(f"\nSaved {len(records)} images to {OUT_DIR}; checksum mismatches: {bad_hash}")
    print("labels:", Counter(r["label"] for r in records))
    print("fake generators:", Counter(r["model"] for r in records if r["label"] == "fake").most_common())
    print("real sources:", Counter(r["model"] for r in records if r["label"] == "real").most_common())


if __name__ == "__main__":
    main()
