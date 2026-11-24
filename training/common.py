"""Shared helpers for the training scripts: dataset loading, metrics and plots."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
SEED = 42

# Code that must match between training and inference lives in the gateway's app package.
sys.path.insert(0, str(ROOT / "ai-gateway"))
from app.textclean import clean_text  # noqa: E402


# ---------------------------------------------------------------- text data

def _detect_fake_label(df):
    """WELFake's documentation of which label means fake is inconsistent, so infer it:
    real news is mostly Reuters copy, fake news almost never is."""
    reuters = df["text"].str.contains("(Reuters)", regex=False).groupby(df["label"]).mean()
    print("Share of articles with a '(Reuters)' dateline per label:", reuters.round(3).to_dict())
    if reuters.max() < 0.05:
        raise SystemExit("Cannot infer which label is fake for this dataset; pass --fake-label")
    real = reuters.idxmax()
    fake = next(label for label in reuters.index if label != real)
    print(f"-> treating label {fake} as FAKE and {real} as REAL (override with --fake-label)")
    return fake


def load_welfake(csv_path, fake_label="auto", train_limit=None):
    """Load WELFake (or any CSV with title/text/label columns) as fixed train/val/test splits.

    Returns {"train": (texts, y), "val": ..., "test": ...} where y = 1 means fake.
    The split depends only on SEED, and train_limit subsamples just the training split,
    so every text model is scored on the same test set.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path).dropna(subset=["text", "label"])
    if "title" not in df:
        df["title"] = ""
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].astype(str)
    fake_label = _detect_fake_label(df) if fake_label == "auto" else int(fake_label)

    data = pd.DataFrame({
        "text": (df["title"] + ". " + df["text"]).map(clean_text),
        "y": (df["label"] == fake_label).astype(int),
    })
    # WELFake merges several datasets and contains duplicates; drop them so no
    # article appears in both train and test.
    data = data[data["text"].str.len() > 20].drop_duplicates("text")

    train, rest = train_test_split(data, test_size=0.2, stratify=data["y"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["y"], random_state=SEED)
    if train_limit and train_limit < len(train):
        train = train.sample(train_limit, random_state=SEED)

    splits = {"train": train, "val": val, "test": test}
    for name, part in splits.items():
        print(f"{name:>5}: {len(part):>6} articles, {part['y'].mean():.1%} fake")
    return {name: (part["text"].tolist(), part["y"].to_numpy()) for name, part in splits.items()}


# --------------------------------------------------------------- image data

def find_split(root, name):
    """Locate train/valid/test folders of an ImageFolder-style dataset."""
    aliases = {"train": ["train"], "valid": ["valid", "val", "validation"], "test": ["test"]}
    for alias in aliases[name]:
        path = Path(root) / alias
        if path.is_dir():
            return path
    raise SystemExit(f"No '{name}' folder in {root} (expected {root}/{name}/fake and {root}/{name}/real)")


# ------------------------------------------------------------------ metrics

def evaluate(model_name, y_true, prob_fake, out_dir, extra=None):
    """Score predictions (y = 1 means fake), then save metrics.json and confusion_matrix.png."""
    import numpy as np
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support, roc_auc_score)

    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob_fake, dtype=float)
    y_pred = (prob >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "model": model_name,
        "test_samples": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc_score(y_true, prob)), 4) if len(set(y_true)) > 1 else None,
        "confusion_matrix": {"labels": ["real", "fake"], "rows_true_cols_pred": cm.tolist()},
        **(extra or {}),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _plot_confusion(cm, model_name, out_dir / "confusion_matrix.png")

    print(f"\n=== {model_name} - test set ({len(y_true)} samples) ===")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{key:>10}: {metrics[key]}")
    print(f"Confusion matrix (rows = true real/fake, cols = predicted):\n{cm}")
    print(f"Saved metrics and plots to {out_dir}")
    return metrics


def _plot_confusion(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["real", "fake"])
    ax.set_yticks([0, 1], ["real", "fake"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history, title, path):
    """Training loss and validation accuracy per epoch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    ax1.plot(epochs, [h["train_loss"] for h in history], "o-", label="train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [h["val_accuracy"] for h in history], "s-", color="tab:green", label="val accuracy")
    ax2.set_ylabel("Validation accuracy")
    fig.legend(loc="upper center", ncol=2, fontsize=8)
    ax1.set_title(title, fontsize=9, pad=22)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    (Path(path).parent / "history.json").write_text(json.dumps(history, indent=2))
