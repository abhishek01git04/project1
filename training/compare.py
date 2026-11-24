"""Print a comparison table of every trained model (paste it into your report).

    python training/compare.py
"""

import json

from common import MODELS_DIR

COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc", "test_samples"]


def main():
    rows = [json.loads(p.read_text()) for p in sorted(MODELS_DIR.glob("*/metrics.json"))]
    if not rows:
        raise SystemExit(f"No metrics found in {MODELS_DIR} - train a model first")
    print("| Model | " + " | ".join(COLUMNS) + " |")
    print("|---" * (len(COLUMNS) + 1) + "|")
    for r in rows:
        print(f"| {r['model']} | " + " | ".join(str(r.get(c, "")) for c in COLUMNS) + " |")


if __name__ == "__main__":
    main()
