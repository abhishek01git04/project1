"""Benchmark one image model at a time on labelled images, exactly as the gateway runs it.

Each model is scored independently; fit_image_fusion.py then reads the scores. Two sets are used,
because the models look for different things:

  ai    whole images, AI-generated vs real (Hugging Face parquet with `image` and `label` columns,
        e.g. julienlucas/midjourney-dalle-sd-nanobananapro-dataset)
  edit  real photos vs the same photos with one region inpainted by AI (CocoGlide, from the TruFor authors)

Usage (from the project root; one process per model keeps memory low):
  ai-gateway\\.venv\\Scripts\\python training\\benchmark_image_models.py --model univfd ^
      --parquet <folder with test-*.parquet> --cocoglide <folder containing fake\\ and real\\>

Writes models/image-fusion/scores-<model>.jsonl and can be stopped and resumed.

With --testset <folder> it instead scores a frozen test set built by build_image_testset.py (both the
clean and the shared copy of every image) and writes models/image-fusion/frozen-test/<folder>-<model>.jsonl.
Those scores are for reporting only: fit_image_fusion.py never reads them, and nothing may be tuned on them.
"""

import argparse
import io
import json
import random
import re
import time
from itertools import chain
from pathlib import Path

from common import MODELS_DIR, SEED
from app import detectors

OUT_DIR = MODELS_DIR / "image-fusion"
TEST_DIR = OUT_DIR / "frozen-test"


def fake_label_value(parquet_file) -> int:
    """Which label value means fake. The dataset's own metadata decides; its documentation says 0."""
    meta = (parquet_file.schema_arrow.metadata or {}).get(b"huggingface")
    if meta:
        names = json.loads(meta)["info"]["features"]["label"].get("names")
        if names and "fake" in names:
            return names.index("fake")
    return 0


def parquet_items(folder: Path, n: int):
    """Yield n images (half fake, half real) as (id, group, label, PIL image). Only picked rows are decoded."""
    import pyarrow.parquet as pq
    from PIL import Image

    files = sorted(folder.rglob("test-*.parquet"))
    if not files:
        raise SystemExit(f"No test-*.parquet in {folder}")
    index = []  # (file number, row number, is fake)
    for f_no, path in enumerate(files):
        pf = pq.ParquetFile(path)
        fake_value = fake_label_value(pf)
        labels = pf.read(columns=["label"]).column("label").to_pylist()
        index += [(f_no, row, int(value == fake_value)) for row, value in enumerate(labels)]
    rng = random.Random(SEED)
    picked = []
    for is_fake in (1, 0):
        pool = [i for i in index if i[2] == is_fake]
        picked += rng.sample(pool, min(n // 2, len(pool)))
    wanted = {(f, r): fake for f, r, fake in picked}

    for f_no, path in enumerate(files):
        row = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=["image"]):
            column = batch.column("image")
            for j in range(batch.num_rows):
                if (f_no, row) in wanted:
                    img = Image.open(io.BytesIO(column[j].as_py()["bytes"])).convert("RGB")
                    item_id = f"ai-{f_no}-{row}"
                    yield item_id, item_id, wanted[(f_no, row)], img
                row += 1


def cocoglide_items(folder: Path, n: int):
    """Yield n images as matched pairs: the same COCO photo, real and with an inpainted region.
    A pair shares one group, so the fitting script never splits it across train and test."""
    from PIL import Image

    number = re.compile(r"_(\d+)(?:_up)?\.png$")
    fake_dir = next(folder.rglob("fake"), None)
    real_dir = next(folder.rglob("real"), None)
    if fake_dir is None or real_dir is None:
        raise SystemExit(f"Expected fake and real folders in {folder}")
    fakes = {m.group(1): p for p in fake_dir.glob("*.png") if (m := number.search(p.name))}
    reals = {m.group(1): p for p in real_dir.glob("*.png") if (m := number.search(p.name))}
    common_ids = sorted(fakes.keys() & reals.keys())
    picked = random.Random(SEED).sample(common_ids, min(n // 2, len(common_ids)))
    for key in picked:
        for label, path in ((1, fakes[key]), (0, reals[key])):
            yield f"edit-{'fake' if label else 'real'}-{key}", f"edit-{key}", label, Image.open(path).convert("RGB")


def tagged(set_name: str, items):
    for item_id, group, label, img in items:
        yield {"id": item_id, "group": group, "set": set_name, "label": label}, img


def testset_items(folder: Path):
    """Every image of a frozen test set, its shared copy and then its clean copy. The order is shuffled
    with a fixed seed, so a run that stops early is still a fair sample of both labels and both copies."""
    from PIL import Image

    manifest = folder / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"No manifest.jsonl in {folder}")
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    random.Random(SEED).shuffle(records)
    for r in records:
        for variant in ("shared", "clean"):
            record = {"id": f"{variant}-{r['rowId']}", "group": r["rowId"], "set": variant,
                      "label": int(r["label"] == "fake"), "source": r["model"]}
            yield record, Image.open(folder / r[variant]).convert("RGB")


def load_predictor(name: str):
    if name == "smaug":
        return next((p for src, p in detectors._image_models() if "smaug" in src.lower()), None)
    return detectors.forensic_model(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["smaug", "univfd", "trufor"])
    ap.add_argument("--parquet", type=Path)
    ap.add_argument("--cocoglide", type=Path)
    ap.add_argument("--testset", type=Path, help="frozen test set folder; replaces --parquet and --cocoglide")
    ap.add_argument("--n-ai", type=int, default=200, help="whole images, half fake and half real")
    ap.add_argument("--n-edit", type=int, default=150, help="locally edited images, as real/edited pairs")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for torch (0 = default)")
    args = ap.parse_args()

    if args.testset:
        out = TEST_DIR / f"{args.testset.name}-{args.model}.jsonl"
        items = testset_items(args.testset)
    elif args.parquet and args.cocoglide:
        out = OUT_DIR / f"scores-{args.model}.jsonl"
        items = chain(tagged("ai", parquet_items(args.parquet, args.n_ai)),
                      tagged("edit", cocoglide_items(args.cocoglide, args.n_edit)))
    else:
        ap.error("give --testset, or both --parquet and --cocoglide")

    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    predict = load_predictor(args.model)
    if predict is None:
        raise SystemExit(f"Could not load {args.model}; see the log above")

    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    print(f"{args.model}: {len(done)} already scored, writing {out}", flush=True)

    count = 0
    for record, img in items:
        if record["id"] in done:
            continue
        start = time.time()
        score = predict(img)[0]
        record |= {"score": score, "seconds": round(time.time() - start, 2)}
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        count += 1
        print(f"[{count}] {record['id']} label={record['label']} score={score:.3f} {record['seconds']}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
