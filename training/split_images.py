"""Split a flat image dataset (one folder per class) into train/valid/test folders.

    python training/split_images.py --src "data/faces-raw/Final Dataset" --dst data/faces

Class folder names are lower-cased (Fake -> fake), as the training scripts expect.
"""

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from common import SEED

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUG_SUFFIX = re.compile(r"_aug_\d+$")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="folder containing one subfolder per class")
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    args = ap.parse_args()

    classes = [d for d in sorted(args.src.iterdir()) if d.is_dir()]
    if not classes:
        raise SystemExit(f"No class folders in {args.src}")
    for cls in classes:
        # Augmented copies (photo_aug_0.jpg, photo_aug_1.jpg, ...) of one photo stay in the
        # same split; otherwise the test set contains near-duplicates of training images.
        groups = defaultdict(list)
        for f in sorted(cls.iterdir()):
            if f.suffix.lower() in IMAGE_EXT:
                groups[AUG_SUFFIX.sub("", f.stem)].append(f)
        keys = sorted(groups)
        random.Random(SEED).shuffle(keys)
        n_val, n_test = round(len(keys) * args.val), round(len(keys) * args.test)
        parts = {"test": keys[:n_test], "valid": keys[n_test:n_test + n_val], "train": keys[n_test + n_val:]}
        counts = {}
        for split, split_keys in parts.items():
            out = args.dst / split / cls.name.lower()
            out.mkdir(parents=True, exist_ok=True)
            files = [f for k in split_keys for f in groups[k]]
            for f in files:
                shutil.copy2(f, out / f.name)
            counts[split] = len(files)
        print(f"{cls.name.lower():>6}: {len(keys)} distinct photos; " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print(f"Done -> {args.dst}")


if __name__ == "__main__":
    main()
