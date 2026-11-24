"""Image Prediction Fusion: combine the image models into one calibrated fake probability.

Smaug, UnivFD and TruFor measure different things on different scales, so their raw scores are
never averaged. training/fit_image_fusion.py benchmarks each model, fits the calibration on labelled
images and writes models/image-fusion/calibration.json.

Without that file, or when one of the models it needs did not load, fuse() returns None and the
caller keeps its uncalibrated result. The other models are then reported but not counted.
"""

import json
import logging
import math
from pathlib import Path

from . import config

log = logging.getLogger("image_fusion")

LOGIT_LIMIT = 6.0  # a model's most extreme outputs must not be able to outvote every other model

_cached: tuple[float, dict] | None = None


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return max(-LOGIT_LIMIT, min(LOGIT_LIMIT, math.log(p / (1 - p))))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def combine(calibration: dict, scores: dict[str, float]) -> float:
    """Apply a calibration to raw model scores. Used by both the gateway and the fitting script."""
    models = calibration["models"]
    if calibration["method"] == "logistic":
        z = calibration["intercept"] + sum(calibration["coef"][m] * logit(scores[m]) for m in models)
        return _sigmoid(z)
    if calibration["method"] == "noisy_or":
        # Each score is calibrated alone; the image counts as fake if any calibrated model says so.
        all_clean = 1.0
        for m in models:
            a, b = calibration["platt"][m]
            all_clean *= 1 - _sigmoid(a * logit(scores[m]) + b)
        return 1 - all_clean
    raise ValueError(f"Unknown fusion method: {calibration['method']}")


def load() -> dict | None:
    """The saved calibration, or None. The file is re-read when it changes, so a refit needs no restart."""
    global _cached
    path = Path(config.resolve_model(config.IMAGE_FUSION_FILE))
    try:
        stamp = path.stat().st_mtime
        if _cached is None or _cached[0] != stamp:
            _cached = (stamp, json.loads(path.read_text(encoding="utf-8")))
        return _cached[1]
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        log.warning("Could not read image fusion calibration %s: %s", path, exc)
        return None


def fuse(scores: dict[str, float]) -> float | None:
    """Calibrated probability, or None unless exactly the calibrated set of models produced a score."""
    calibration = load()
    if calibration is None or set(calibration["models"]) != set(scores):
        return None
    return combine(calibration, scores)
