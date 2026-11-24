"""Benchmark each image model on its own, then fit and test the calibrated Image Prediction Fusion.

Reads the scores from benchmark_image_models.py. Images are split 60/40 before anything is fitted, with
real/edited pairs kept together, so the numbers reported for the fusion come from images it never saw.
Two fusion methods compete on cross-validated log-loss on the training part:

  logistic   one logistic regression over the three models' logits
  noisy_or   each model calibrated alone (Platt scaling); fake if any calibrated model says so

A plain average of the raw scores is reported only for comparison, and is never deployed.

Usage (from the project root):
  ai-gateway\\.venv\\Scripts\\python training\\fit_image_fusion.py

Writes models/image-fusion/benchmark.json. Writes calibration.json, which the gateway uses at once,
only if the fusion beats Smaug alone on training log-loss. Pass --no-write to only measure.
"""

import argparse
import json
import random
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from common import MODELS_DIR, SEED
from app.image_fusion import LOGIT_LIMIT, combine, logit

OUT_DIR = MODELS_DIR / "image-fusion"
MODELS = ("smaug", "univfd", "trufor")
TRAIN_SHARE = 0.6


def load_rows() -> list[dict]:
    """Images that all three models scored."""
    rows = {}
    for model in MODELS:
        path = OUT_DIR / f"scores-{model}.jsonl"
        if not path.exists():
            raise SystemExit(f"Missing {path}: run benchmark_image_models.py --model {model}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                row = rows.setdefault(r["id"], {"id": r["id"], "group": r["group"], "set": r["set"],
                                                "label": r["label"], "scores": {}, "seconds": {}})
                row["scores"][model] = r["score"]
                row["seconds"][model] = r["seconds"]
    complete = [r for r in rows.values() if len(r["scores"]) == len(MODELS)]
    if len(complete) < len(rows):
        print(f"Using {len(complete)} of {len(rows)} images: the others are not scored by every model yet")
    return complete


def split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Fixed split by group, keeping each set and class balanced."""
    strata = defaultdict(set)
    for r in rows:
        key = "edit" if r["set"] == "edit" else f"ai-{r['label']}"
        strata[key].add(r["group"])
    rng = random.Random(SEED)
    train_groups = set()
    for key in sorted(strata):
        groups = sorted(strata[key])
        rng.shuffle(groups)
        train_groups.update(groups[: round(len(groups) * TRAIN_SHARE)])
    return [r for r in rows if r["group"] in train_groups], [r for r in rows if r["group"] not in train_groups]


def features(rows: list[dict], models=MODELS) -> np.ndarray:
    return np.array([[logit(r["scores"][m]) for m in models] for r in rows])


def labels(rows: list[dict]) -> np.ndarray:
    return np.array([r["label"] for r in rows])


def fit_logistic(rows: list[dict]) -> dict:
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(features(rows), labels(rows))
    return {"method": "logistic", "models": list(MODELS), "intercept": float(clf.intercept_[0]),
            "coef": {m: float(c) for m, c in zip(MODELS, clf.coef_[0])}}


def fit_platt(rows: list[dict], model: str) -> list[float]:
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(features(rows, (model,)), labels(rows))
    return [float(clf.coef_[0][0]), float(clf.intercept_[0])]


def fit_noisy_or(rows: list[dict]) -> dict:
    return {"method": "noisy_or", "models": list(MODELS), "platt": {m: fit_platt(rows, m) for m in MODELS}}


def fit_smaug_only(rows: list[dict]) -> dict:
    return {"method": "noisy_or", "models": ["smaug"], "platt": {"smaug": fit_platt(rows, "smaug")}}


def predict(calibration: dict, rows: list[dict]) -> np.ndarray:
    return np.array([combine(calibration, r["scores"]) for r in rows])


def cv_log_loss(fit, rows: list[dict], folds: int = 5) -> float:
    y, groups = labels(rows), [r["group"] for r in rows]
    losses = []
    for train_idx, test_idx in GroupKFold(n_splits=folds).split(rows, y, groups):
        calibration = fit([rows[i] for i in train_idx])
        p = np.clip(predict(calibration, [rows[i] for i in test_idx]), 1e-6, 1 - 1e-6)
        losses.append(log_loss(y[test_idx], p, labels=[0, 1]))
    return float(np.mean(losses))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        chosen = (p >= low) & (p < high if high < 1 else p <= high)
        if chosen.any():
            error += chosen.mean() * abs(y[chosen].mean() - p[chosen].mean())
    return float(error)


def scorecard(rows: list[dict], p: np.ndarray) -> dict:
    """Metrics at the 0.5 threshold, overall and for each set."""
    y = labels(rows)
    pred = p >= 0.5
    card = {
        "n": len(rows),
        "accuracy": round(float((pred == y).mean()), 3),
        "roc_auc": round(float(roc_auc_score(y, p)), 3) if len(set(y)) > 1 else None,
        "log_loss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])), 3),
        "brier": round(float(brier_score_loss(y, p)), 3),
        "ece": round(expected_calibration_error(y, p), 3),
    }
    for set_name in ("ai", "edit"):
        chosen = np.array([r["set"] == set_name for r in rows])
        if chosen.any():
            fakes, reals = chosen & (y == 1), chosen & (y == 0)
            card[f"{set_name}_caught"] = f"{int(pred[fakes].sum())}/{int(fakes.sum())}"
            card[f"{set_name}_false_alarms"] = f"{int(pred[reals].sum())}/{int(reals.sum())}"
    return card


def show(name: str, card: dict):
    print(f"{name:26} acc {card['accuracy']:.3f}  auc {card['roc_auc']}  logloss {card['log_loss']:.3f}  "
          f"ece {card['ece']:.3f}  AI caught {card.get('ai_caught')}  false {card.get('ai_false_alarms')}  "
          f"edits caught {card.get('edit_caught')}  false {card.get('edit_false_alarms')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-write", action="store_true", help="measure only; leave calibration.json alone")
    args = ap.parse_args()

    rows = load_rows()
    train, test = split(rows)
    print(f"{len(rows)} images: {len(train)} to fit, {len(test)} held out\n")

    report = {"images": len(rows), "train": len(train), "heldOut": len(test), "logitLimit": LOGIT_LIMIT, "models": {}}
    print("Each model alone, raw scores, on the held-out images:")
    for model in MODELS:
        p = np.array([r["scores"][model] for r in test])
        report["models"][model] = scorecard(test, p)
        seconds = [r["seconds"][model] for r in rows]
        report["models"][model]["median_seconds"] = round(float(np.median(seconds)), 2)
        show(model, report["models"][model])

    print("\nCombinations on the held-out images:")
    average = np.array([np.mean([r["scores"][m] for m in MODELS]) for r in test])
    report["plainAverage"] = scorecard(test, average)
    show("plain average (not used)", report["plainAverage"])

    candidates = {"logistic": fit_logistic, "noisy_or": fit_noisy_or}
    cv = {name: cv_log_loss(fit, train) for name, fit in candidates.items()}
    smaug_cv = cv_log_loss(fit_smaug_only, train)
    report["cvLogLoss"] = {**{k: round(v, 4) for k, v in cv.items()}, "smaug_calibrated": round(smaug_cv, 4)}
    chosen = min(cv, key=cv.get)

    smaug_card = scorecard(test, predict(fit_smaug_only(train), test))
    report["smaugCalibrated"] = smaug_card
    show("Smaug alone, calibrated", smaug_card)
    for name, fit in candidates.items():
        card = scorecard(test, predict(fit(train), test))
        report[name] = card
        show(f"{name}{' (chosen)' if name == chosen else ''}", card)
    report["chosen"] = chosen
    print(f"\nCross-validated log-loss on the fitting images: {report['cvLogLoss']}")

    beats_smaug = cv[chosen] < smaug_cv
    report["deployed"] = bool(beats_smaug and not args.no_write)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {OUT_DIR / 'benchmark.json'}")

    if args.no_write:
        print("--no-write: calibration.json not changed")
    elif not beats_smaug:
        print("The fusion did not beat Smaug alone: calibration.json not written")
    else:
        # The numbers above come from the 60% fit. The deployed calibration also learns from the held-out images.
        calibration = candidates[chosen](rows)
        calibration.update(version="image-fusion-1", fittedOn=len(rows), logitLimit=LOGIT_LIMIT)
        (OUT_DIR / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        print(f"Saved {OUT_DIR / 'calibration.json'} ({chosen})")


if __name__ == "__main__":
    main()
