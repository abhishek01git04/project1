"""Image copies, CLIP preprocessing and feature extraction for the image head (train_image_head.py).

Standalone on purpose (numpy, Pillow, pyarrow, torch and transformers only): the same file runs on this
laptop, in Google Colab (colab_image_head.ipynb embeds a copy of it) and on the college HPC (hpc/), so features
made in any of them match. Kept compatible with Python 3.8+ for the HPC's environment.
"""

from __future__ import annotations

import io
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 42
SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)  # CLIP normalisation
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
CHUNK = 256            # images per saved feature file
CLEAN_MAX_SIDE = 1536  # CLIP sees 224 px, so larger clean copies only cost time


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def clean_copy(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((CLEAN_MAX_SIDE, CLEAN_MAX_SIDE), Image.LANCZOS)
    return jpeg(img, 95)


def shared_copy(img: Image.Image, key: str) -> Image.Image:
    """A re-shared copy. The settings are fixed per image and deliberately wider than the frozen test set's
    recipe (short side 280-640, quality 60-85, white border), so the head cannot just learn that recipe."""
    rng = random.Random(f"train-{SEED}-{key}")
    img = img.convert("RGB")
    scale = min(1.0, rng.randint(200, 800) / min(img.size))
    if scale < 1.0:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    if rng.random() < 0.4:
        border = rng.randint(2, 24)
        colour = rng.choice([(255, 255, 255), (0, 0, 0), (240, 240, 240)])
        framed = Image.new("RGB", (img.width + 2 * border, img.height + 2 * border), colour)
        framed.paste(img, (border, border))
        img = framed
    qualities = [rng.randint(50, 92)] + ([rng.randint(60, 90)] if rng.random() < 0.3 else [])
    for quality in qualities:
        img = jpeg(img, quality)
    return img


def preprocess(img: Image.Image) -> np.ndarray:
    """Standard CLIP input: shorter side resized to 224, centre crop, CLIP normalisation."""
    img = img.convert("RGB")
    scale = SIZE / min(img.size)
    img = img.resize((max(SIZE, round(img.width * scale)), max(SIZE, round(img.height * scale))), Image.BICUBIC)
    left, top = (img.width - SIZE) // 2, (img.height - SIZE) // 2
    pixels = np.asarray(img.crop((left, top, left + SIZE, top + SIZE)), dtype=np.float32) / 255.0
    return ((pixels - MEAN) / STD).transpose(2, 0, 1)


def fake_label_value(parquet_file) -> int:
    """Which label value means fake. The dataset's own metadata decides; its documentation says 0."""
    meta = (parquet_file.schema_arrow.metadata or {}).get(b"huggingface")
    if meta:
        names = json.loads(meta)["info"]["features"]["label"].get("names")
        if names and "fake" in names:
            return names.index("fake")
    return 0


def prepare(job):
    """One training image (raw bytes, key) -> preprocessed clean and shared copies.
    Top level, so a process pool can run it."""
    raw, key = job
    img = Image.open(io.BytesIO(raw))
    return preprocess(clean_copy(img)), preprocess(shared_copy(img, key))


class Encoder:
    """Frozen CLIP image encoder; returns L2-normalised image embeddings."""

    def __init__(self, name: str, device: str = "cpu", batch: int = 16, revision: str | None = None,
                 local_files_only: bool = True):
        import torch
        from transformers import CLIPVisionModelWithProjection

        self.torch, self.device, self.batch = torch, device, batch
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            name, revision=revision, local_files_only=local_files_only).eval().to(device)

    def encode(self, pixels: np.ndarray) -> np.ndarray:
        """pixels: (n, 3, 224, 224) from preprocess()."""
        out = []
        for i in range(0, len(pixels), self.batch):
            x = self.torch.from_numpy(np.ascontiguousarray(pixels[i:i + self.batch])).to(self.device)
            with self.torch.inference_mode():
                emb = self.model(pixel_values=x).image_embeds.float()
            out.append((emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32))
        return np.concatenate(out)

    def __call__(self, images: list) -> np.ndarray:
        return self.encode(np.stack([preprocess(im) for im in images]))


def extract_testset(encoder: Encoder, folder: Path, out_file: Path):
    """Both copies of every image of a frozen test set (build_image_testset.py); ids match the Smaug scores."""
    folder, out_file = Path(folder), Path(out_file)
    if out_file.exists():
        print(f"{out_file.name} already extracted", flush=True)
        return
    records = [json.loads(line) for line in (folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    ids, labels, sources, features = [], [], [], []
    start = time.time()
    for variant in ("clean", "shared"):
        for i in range(0, len(records), CHUNK):
            part = records[i:i + CHUNK]
            features.append(encoder([Image.open(folder / r[variant]) for r in part]))
            ids += [f"{variant}-{r['rowId']}" for r in part]
            labels += [int(r["label"] == "fake") for r in part]
            sources += [r["model"] for r in part]
            print(f"test set {variant}: {min(i + CHUNK, len(records))}/{len(records)} "
                  f"({time.time() - start:.0f}s)", flush=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_file, ids=np.array(ids), labels=np.array(labels, dtype=np.int8), sources=np.array(sources),
             features=np.concatenate(features))
    print(f"saved {out_file}", flush=True)


def extract_train(encoder: Encoder, folder: Path, out_dir: Path, pool=None):
    """Clean and shared copies of every training image, saved CHUNK images per file. Resumable."""
    import pyarrow.parquet as pq

    files = sorted(Path(folder).glob("train-*.parquet"))
    if not files:
        raise SystemExit(f"No train-*.parquet in {folder}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = sum(pq.ParquetFile(p).metadata.num_rows for p in files)
    done = 0
    for path in files:
        pf = pq.ParquetFile(path)
        fake_value = fake_label_value(pf)
        offset = 0
        for batch in pf.iter_batches(batch_size=CHUNK, columns=["image", "label"]):
            rows = batch.num_rows
            out = out_dir / f"{path.stem}-{offset:05d}.npz"
            if not out.exists():
                start = time.time()
                keys = [f"{path.stem}:{offset + j}" for j in range(rows)]
                jobs = [(cell["bytes"], key) for cell, key in zip(batch.column("image").to_pylist(), keys)]
                pairs = pool.map(prepare, jobs, chunksize=4) if pool else [prepare(job) for job in jobs]
                labels = [int(v == fake_value) for v in batch.column("label").to_pylist()]
                tmp = out.with_name(out.stem + ".tmp.npz")
                np.savez(tmp, ids=np.array(keys), labels=np.array(labels, dtype=np.int8),
                         clean=encoder.encode(np.stack([c for c, _ in pairs])),
                         shared=encoder.encode(np.stack([s for _, s in pairs])))
                tmp.replace(out)
                seconds = time.time() - start
                print(f"{out.name}: {rows} images in {seconds:.0f}s; {done + rows}/{total}", flush=True)
            offset += rows
            done += rows
    print("train features done", flush=True)


def extract_folder(encoder: Encoder, folder: Path, out_dir: Path, prefix: str, label: int, pool=None):
    """Clean and shared copies of every image file in a folder, all with one label (1 = fake, 0 = real).
    Saved CHUNK images per file as <prefix>-<offset>.npz, in the same format as extract_train. Resumable."""
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    if not files:
        raise SystemExit(f"No images in {folder}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(files), CHUNK):
        part = files[offset:offset + CHUNK]
        out = out_dir / f"{prefix}-{offset:05d}.npz"
        if out.exists():
            continue
        start = time.time()
        keys = [f"{prefix}:{p.stem}" for p in part]
        jobs = [(p.read_bytes(), key) for p, key in zip(part, keys)]
        pairs = pool.map(prepare, jobs, chunksize=4) if pool else [prepare(job) for job in jobs]
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, ids=np.array(keys), labels=np.full(len(part), label, dtype=np.int8),
                 clean=encoder.encode(np.stack([c for c, _ in pairs])),
                 shared=encoder.encode(np.stack([s for _, s in pairs])))
        tmp.replace(out)
        print(f"{out.name}: {len(part)} images in {time.time() - start:.0f}s; "
              f"{offset + len(part)}/{len(files)}", flush=True)
    print(f"{prefix} features done", flush=True)
