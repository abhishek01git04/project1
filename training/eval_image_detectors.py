"""Evaluate image detectors exactly as the gateway runs them, on a labelled test set.

Test set: a Hugging Face parquet file with `image` and `label` (0 = fake, 1 = real) columns,
e.g. julienlucas/midjourney-dalle-sd-nanobananapro-dataset (real photos vs Midjourney,
DALL-E, Stable Diffusion and Nano Banana images).

    python training/eval_image_detectors.py --parquet <file.parquet> --n 200 ^
        --models abacusai/Smaug-AI-Image-Detector,Ateeqq/ai-vs-human-image-detector

Metrics are saved to models/pretrained-<name>/ so compare.py includes them.
"""

import argparse
import io
import random
import time

from common import MODELS_DIR, SEED, evaluate
from app import detectors


def load_samples(parquet, n):
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(parquet, columns=["image", "label"])
    labels = table.column("label").to_pylist()
    rng = random.Random(SEED)
    picked = []
    for value in (0, 1):  # balanced: n/2 fake, n/2 real
        idx = [i for i, label in enumerate(labels) if label == value]
        picked += rng.sample(idx, min(n // 2, len(idx)))
    images = table.column("image").to_pylist()
    return [(Image.open(io.BytesIO(images[i]["bytes"])).convert("RGB"), int(labels[i] == 0)) for i in picked]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--n", type=int, default=200, help="images to evaluate (half real, half fake)")
    ap.add_argument("--models", required=True, help="comma-separated, as in the gateway's IMAGE_MODEL")
    args = ap.parse_args()

    samples = load_samples(args.parquet, args.n)
    y_true = [y for _, y in samples]
    print(f"{len(samples)} images: {sum(y_true)} AI-generated, {len(y_true) - sum(y_true)} real")
    for src in args.models.split(","):
        predict = detectors._predictor(src, detectors._image_loader(src))
        if predict is None:
            print(f"Skipping {src}: could not be loaded")
            continue
        t = time.time()
        probs = [predict(img)[0] for img, _ in samples]
        per_image = (time.time() - t) / len(samples)
        name = src.split("/")[-1]
        evaluate(f"{src} (pretrained)", y_true, probs, MODELS_DIR / f"pretrained-{name}",
                 {"seconds_per_image_cpu": round(per_image, 2), "test_set": args.parquet.split("datasets--")[-1][:80]})


if __name__ == "__main__":
    main()
