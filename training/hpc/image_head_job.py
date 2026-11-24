"""Training features for the image head on the college HPC (Rachel, PBS). Same output as colab_image_head.ipynb.

The compute nodes have no internet, so each part is split in two: download on the login node (internet, no
heavy work), then a GPU job that runs offline.

  julienlucas training split (fake and real):
    python image_head_job.py download          then  qsub image_head.pbs        -> image-head-train-features.zip
  extra real photos (Unsplash Lite before 2022 with camera EXIF, and Open Images validation photos):
    python image_head_job.py download-reals    then  qsub image_head_reals.pbs  -> image-head-extra-features.zip

Needs image_head_features.py in the same folder. Everything is kept in this folder: the Hugging Face cache in
hf/, the photos in reals/, the features in features/.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import random
import shutil
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(HERE / "hf"))
sys.path.insert(0, str(HERE))

DATASET = "julienlucas/midjourney-dalle-sd-nanobananapro-dataset"
DATASET_REV = "25c4fb41242b8b8e43dda7c5e6291ac74b34e16b"
DATASET_FILES = ["data/train-*.parquet"]
CLIP = "openai/clip-vit-large-patch14"
CLIP_REV = "32bd64288804d66eefd0ccbe215aa642df71cc41"
CLIP_FILES = ["config.json", "model.safetensors"]

REALS = HERE / "reals"
REAL_SOURCES = ("unsplash", "openimages")
N_REALS = 2500  # per source
SEED = 42
UNSPLASH_ZIP = "https://unsplash-datasets.s3.amazonaws.com/lite/latest/unsplash-research-dataset-lite-latest.zip"
OPENIMAGES_CSV = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
OPENIMAGES_URL = "https://open-images-dataset.s3.amazonaws.com/validation/{}.jpg"


def check_packages():
    missing = []
    for name in ("numpy", "PIL", "pyarrow", "torch", "transformers", "huggingface_hub"):
        try:
            module = __import__(name)
            print(f"  {name:<16} {getattr(module, '__version__', '?')}")
        except ImportError:
            missing.append(name)
    print(f"  python           {sys.version.split()[0]}")
    if missing:
        names = " ".join("pillow" if m == "PIL" else m for m in missing)
        raise SystemExit(f"Missing packages: {', '.join(missing)}. On the login node run:  pip install --user {names}")


def dataset_dir(offline: bool) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(DATASET, repo_type="dataset", revision=DATASET_REV,
                                  allow_patterns=DATASET_FILES, local_files_only=offline)) / "data"


def download():
    from huggingface_hub import snapshot_download

    print("packages:")
    check_packages()
    data = dataset_dir(offline=False)
    snapshot_download(CLIP, revision=CLIP_REV, allow_patterns=CLIP_FILES)
    files = sorted(data.glob("train-*.parquet"))
    size = sum(f.stat().st_size for f in files) / 1e9
    print(f"\ndataset: {len(files)} train files, {size:.1f} GB in {data}")
    print(f"CLIP: {CLIP} saved in {os.environ['HF_HOME']}")
    if len(files) != 9:
        raise SystemExit("Expected 9 train files; run the download again")
    print("DOWNLOAD OK - now submit the GPU job:  qsub image_head.pbs")


# ------------------------------------------------------------ extra real photos

def fetch(url: str, path: Path, attempts: int = 4) -> bool:
    tmp = path.with_name(path.name + ".part")
    error = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "truthchain-research"})
            with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as f:
                shutil.copyfileobj(response, f)
            tmp.replace(path)
            return True
        except Exception as exc:
            error = exc
            time.sleep(2 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    print(f"  failed {url}: {error}", flush=True)
    return False


def unsplash_list(rng: random.Random) -> list[tuple[str, str]]:
    """Unsplash Lite photos submitted before 2022 (before AI images were common) that carry camera EXIF."""
    zip_path = REALS / "unsplash-lite.zip"
    if not zip_path.exists() and not fetch(UNSPLASH_ZIP, zip_path):
        raise SystemExit("Could not download the Unsplash Lite dataset")
    csv.field_size_limit(10 ** 8)
    with zipfile.ZipFile(zip_path) as z:
        member = next(m for m in z.namelist() if m.startswith("photos.tsv"))
        with z.open(member) as f:
            rows = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t"))
    need = {"photo_id", "photo_image_url", "photo_submitted_at", "exif_camera_make"}
    if need - set(rows[0]):
        raise SystemExit(f"Unsplash columns changed; missing {need - set(rows[0])}; have {sorted(rows[0])}")
    ok = sorted((r for r in rows
                 if (r["photo_image_url"] or "").startswith("https://images.unsplash.com/")
                 and (r["photo_submitted_at"] or "")[:4].isdigit() and r["photo_submitted_at"] < "2022"
                 and (r["exif_camera_make"] or "").strip()),
                key=lambda r: r["photo_id"])
    print(f"unsplash: {len(rows)} photos listed, {len(ok)} submitted before 2022 with camera EXIF")
    rng.shuffle(ok)
    sep = lambda url: "&" if "?" in url else "?"
    return [(r["photo_id"], f"{r['photo_image_url']}{sep(r['photo_image_url'])}w=1600&q=90&fm=jpg") for r in ok]


def openimages_list(rng: random.Random) -> list[tuple[str, str]]:
    """Open Images validation photos (Flickr, CC BY 2.0, collected before 2018)."""
    csv_path = REALS / "openimages-validation.csv"
    if not csv_path.exists() and not fetch(OPENIMAGES_CSV, csv_path):
        raise SystemExit("Could not download the Open Images list")
    with open(csv_path, encoding="utf-8") as f:
        ids = sorted(r["ImageID"] for r in csv.DictReader(f))
    print(f"openimages: {len(ids)} validation photos listed")
    rng.shuffle(ids)
    return [(i, OPENIMAGES_URL.format(i)) for i in ids]


def valid_image(path: Path) -> bool:
    from PIL import Image
    try:
        with Image.open(path) as img:
            return img.format in ("JPEG", "PNG", "WEBP") and min(img.size) >= 200
    except Exception:
        return False


def download_reals():
    from huggingface_hub import snapshot_download

    print("packages:")
    check_packages()
    snapshot_download(CLIP, revision=CLIP_REV, allow_patterns=CLIP_FILES)  # the GPU job is offline
    print(f"CLIP: {CLIP} saved in {os.environ['HF_HOME']}")
    REALS.mkdir(parents=True, exist_ok=True)
    for source, lister in (("unsplash", unsplash_list), ("openimages", openimages_list)):
        folder = REALS / source
        folder.mkdir(exist_ok=True)
        candidates = lister(random.Random(f"{SEED}-{source}"))
        have = sum(1 for _ in folder.glob("*.jpg"))
        position = 0
        while have < N_REALS and position < len(candidates):
            batch = candidates[position:position + (N_REALS - have) + 20]
            position += len(batch)
            jobs = [(url, folder / f"{photo_id}.jpg") for photo_id, url in batch if not (folder / f"{photo_id}.jpg").exists()]
            with ThreadPoolExecutor(8) as pool:
                list(pool.map(lambda job: fetch(*job), jobs))
            for p in folder.glob("*.jpg"):
                if not valid_image(p):
                    p.unlink()
            have = sum(1 for _ in folder.glob("*.jpg"))
            print(f"{source}: {have}/{N_REALS} photos", flush=True)
        for extra in sorted(folder.glob("*.jpg"))[N_REALS:]:
            extra.unlink()
        size = sum(p.stat().st_size for p in folder.glob("*.jpg")) / 1e9
        print(f"{source}: {min(have, N_REALS)} photos, {size:.2f} GB in {folder}", flush=True)
    print("REALS DOWNLOAD OK - now submit the GPU job:  qsub image_head_reals.pbs")


# ------------------------------------------------------------------- GPU jobs

def start_gpu_job(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    from multiprocessing import Pool

    workers = args.workers or int(os.environ.get("NCPUS", "2"))
    pool = Pool(workers)  # before CUDA starts; the workers only resize and recompress images

    import torch
    import image_head_features as features

    torch.set_num_threads(workers)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("No GPU visible to this job (check the queue and CUDA_VISIBLE_DEVICES)")
    if args.device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print(f"workers: {workers}", flush=True)
    encoder = features.Encoder(CLIP, device=args.device, batch=32, revision=CLIP_REV, local_files_only=True)
    return pool, features, encoder


def finish(pool, out: Path, zip_name: str, start: float):
    pool.close()
    zip_path = shutil.make_archive(str(HERE / zip_name), "zip", str(out))
    print(f"total {(time.time() - start) / 60:.0f} minutes; wrote {zip_path}", flush=True)
    print("EXTRACT DONE", flush=True)


def extract(args):
    pool, features, encoder = start_gpu_job(args)
    data = dataset_dir(offline=True)
    print(f"data: {data}", flush=True)
    if args.dry_run:
        print("dry run OK: dataset and CLIP found offline")
        return
    start = time.time()
    out = HERE / "features" / "train"
    features.extract_train(encoder, data, out, pool=pool)
    finish(pool, out, "image-head-train-features", start)


def extract_reals(args):
    pool, features, encoder = start_gpu_job(args)
    start = time.time()
    out = HERE / "features" / "extra"
    for source in REAL_SOURCES:
        features.extract_folder(encoder, REALS / source, out, prefix=source, label=0, pool=pool)
    finish(pool, out, "image-head-extra-features", start)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["download", "extract", "download-reals", "extract-reals"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=0, help="image workers (default: the job's NCPUS)")
    ap.add_argument("--dry-run", action="store_true", help="extract: only check that everything loads offline")
    args = ap.parse_args()
    {"download": lambda: download(), "extract": lambda: extract(args),
     "download-reals": lambda: download_reals(), "extract-reals": lambda: extract_reals(args)}[args.step]()


if __name__ == "__main__":
    main()
