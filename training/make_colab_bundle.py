"""Package the training code for Google Colab as training/colab_bundle.zip.

    python training/make_colab_bundle.py               # code only
    python training/make_colab_bundle.py --with-faces  # + data/faces as training/colab_faces.zip
"""

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    *sorted((ROOT / "training").glob("*.py")),
    ROOT / "ai-gateway/app/__init__.py",
    ROOT / "ai-gateway/app/textclean.py",
    ROOT / "ai-gateway/app/cnn.py",
]

out = ROOT / "training" / "colab_bundle.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.write(f, f.relative_to(ROOT).as_posix())
print(f"Wrote {out} ({len(FILES)} files). Upload it in the first cell of colab_finetune.ipynb.")

if "--with-faces" in sys.argv:
    faces = ROOT / "data" / "faces"
    if not faces.is_dir():
        raise SystemExit(f"{faces} not found - run training/split_images.py first")
    images = sorted(p for p in faces.rglob("*") if p.is_file())
    out = ROOT / "training" / "colab_faces.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:  # JPEGs are already compressed
        for f in images:
            z.write(f, f.relative_to(ROOT).as_posix())
    print(f"Wrote {out} ({len(images)} images, {out.stat().st_size / 1e6:.0f} MB). Upload it too.")
