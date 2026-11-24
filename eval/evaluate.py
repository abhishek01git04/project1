"""Score the verification pipeline on the collected claims and tune the fusion weights offline.

Usage (from the project root, after eval/collect.py):
  ai-gateway\\.venv\\Scripts\\python eval\\evaluate.py           # measure current weights
  ai-gateway\\.venv\\Scripts\\python eval\\evaluate.py --tune    # also search for better weights

Tuning uses a fixed half of the claims; the other half is only scored, so the reported
improvement is on claims the search never saw. Results go to eval/runs/report.json.

Metrics:
  accuracy     correct / all claims (an Uncertain verdict counts as not correct)
  decided      share of claims that got Fake or Real instead of Uncertain
  precision    when the system says Fake or Real, how often it is right
  false-fake   real claims called Fake (the most harmful error for a registry)
  false-real   fake claims called Real
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai-gateway"))

from app import fusion  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "components.jsonl"
REPORT = HERE / "runs" / "report.json"

# Each search dimension: fusion.WEIGHTS key -> candidate values
GRID = {
    "detector_scale": [0.25, 0.5, 0.75, 1.0],
    "llm_refuted": [2.0, 3.0, 4.0],
    "llm_supported": [1.5, 2.5, 3.5],
    "news_support": [0.0, 0.25, 0.5],
    "fc_refute": [3.0, 4.5, 6.0],
    "min_confidence": [0.4, 0.5, 0.6],
    "fake_at": [0.6, 0.7],
    "real_at": [0.3, 0.4],
}


def load(path: Path = RUNS) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def predict(row: dict, mode: str, weights: dict | None) -> str:
    det = row["detector"]
    ev = row["evidence"]
    if mode == "detector":
        ev = {"claims": ev.get("claims")}
    elif mode == "no-ai":
        ev = {**ev, "llm": None}
    return fusion.decide(det["prob"], det["isModel"], ev, weights).verdict


def score(rows: list[dict], mode: str = "full", weights: dict | None = None) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    want = {"fake": "Fake", "real": "Real"}
    preds = [predict(r, mode, weights) for r in rows]
    correct = sum(p == want[r["label"]] for r, p in zip(rows, preds))
    decided = sum(p != "Uncertain" for p in preds)
    n_real = sum(r["label"] == "real" for r in rows)
    n_fake = n - n_real
    false_fake = sum(r["label"] == "real" and p == "Fake" for r, p in zip(rows, preds))
    false_real = sum(r["label"] == "fake" and p == "Real" for r, p in zip(rows, preds))
    return {
        "n": n,
        "accuracy": round(correct / n, 3),
        "decided": round(decided / n, 3),
        "precision": round(correct / decided, 3) if decided else None,
        "false_fake": false_fake, "n_real": n_real,
        "false_real": false_real, "n_fake": n_fake,
    }


def objective(s: dict) -> float:
    """Accuracy, with calling a real item Fake penalised more than other errors."""
    return s["accuracy"] - 0.5 * s["false_fake"] / max(1, s["n_real"]) - 0.25 * s["false_real"] / max(1, s["n_fake"])


def tune(rows: list[dict]) -> dict:
    """Coordinate descent over GRID (each weight in turn), repeated until nothing improves."""
    best: dict = {}
    best_score = objective(score(rows, "full", best))
    for _ in range(4):
        improved = False
        for key, values in GRID.items():
            for v in values:
                trial = {**best, key: v}
                s = objective(score(rows, "full", trial))
                if s > best_score + 1e-9:
                    best, best_score, improved = trial, s, True
        if not improved:
            break
    return best


def line(name: str, s: dict) -> str:
    if not s.get("n"):
        return f"{name:34} (no claims)"
    prec = f"{s['precision']:.0%}" if s["precision"] is not None else "-"
    return (f"{name:34} n={s['n']:<4} accuracy {s['accuracy']:>4.0%} | decided {s['decided']:>4.0%} | "
            f"precision {prec:>4} | real->Fake {s['false_fake']}/{s['n_real']} | fake->Real {s['false_real']}/{s['n_fake']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--runs", type=Path, default=RUNS, help="result file from collect.py or rerun_llm.py")
    args = ap.parse_args()

    rows = load(args.runs)
    # sanity: the same claim-level evidence is reused, nothing here touches the network
    errors = sum((r["evidence"].get("llm") or {}).get("status") == "error" for r in rows)
    print(f"{len(rows)} claims collected; AI model errors on {errors}\n")

    report = {"claims": len(rows), "llmErrors": errors, "overall": {}}
    print("Current weights, all claims:")
    for mode, name in (("detector", "detector only"), ("no-ai", "detector + news + fact-checks"), ("full", "full pipeline (+ local AI)")):
        s = score(rows, mode)
        report["overall"][mode] = s
        print(" ", line(name, s))

    print("\nFull pipeline by group:")
    groups = {
        "fake: statements": [r for r in rows if r["label"] == "fake" and r["kind"] == "statement"],
        "fake: photo/video captions": [r for r in rows if r["label"] == "fake" and r["kind"] == "media-caption"],
        "real: rated true by fact-checkers": [r for r in rows if r["label"] == "real" and r["origin"] == "fact-check"],
        "real: credible headlines": [r for r in rows if r["label"] == "real" and r["origin"] == "news-headline"],
    }
    report["groups"] = {}
    for name, part in groups.items():
        s = score(part)
        report["groups"][name] = s
        print(" ", line(name, s))

    if args.tune:
        # stratified split, fixed by seed, so tuning never sees the held-out half
        rng = random.Random(args.seed)
        tune_set, held = [], []
        for label in ("fake", "real"):
            part = [r for r in rows if r["label"] == label]
            rng.shuffle(part)
            tune_set += part[: len(part) // 2]
            held += part[len(part) // 2:]
        best = tune(tune_set)
        print(f"\nTuned on {len(tune_set)} claims, scored on {len(held)} unseen claims:")
        print("  changed weights:", best or "(none)")
        before, after = score(held), score(held, "full", best)
        print(" ", line("held-out, current weights", before))
        print(" ", line("held-out, tuned weights", after))
        report["tuning"] = {"weights": best, "tuneSet": len(tune_set), "heldOut": len(held),
                            "before": before, "after": after}

    path = REPORT if args.runs == RUNS else args.runs.with_suffix(".report.json")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
