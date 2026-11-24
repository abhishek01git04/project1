"""Compare evidence guards ("a supported verdict needs a source that states the claim") on a saved
tuning-set run, using the source scores from support_features.py.

Usage (from the project root):
  ai-gateway\\.venv\\Scripts\\python eval\\support_eval.py

Guards that need a threshold are chosen on one half of the claims and scored on the other half
(both ways round), so the numbers are not flattered by picking the best threshold after the fact.
"""

import argparse
import collections
import copy
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv, _argv = sys.argv[:1], sys.argv  # evaluate.py parses its own arguments on import
import evaluate as E  # noqa: E402

sys.argv = _argv
WANT = {"fake": "Fake", "real": "Real"}
PTS = {"right": 1, "uncertain": 0, "wrong": -1}


def outcome(row: dict) -> str:
    p = E.predict(row, "full", None)
    return "uncertain" if p == "Uncertain" else ("right" if p == WANT[row["label"]] else "wrong")


def best(feat: dict, measure: str, kinds=("article", "headline")) -> float:
    return max((s[measure] or 0.0 for s in feat["sources"] if s["kind"] in kinds and s[measure] is not None),
               default=0.0)


def guarded(row: dict, ok: bool) -> dict:
    """The run row with a 'supported' verdict turned into 'unverifiable' when the guard fails."""
    llm = (row["evidence"].get("llm") or {})
    if llm.get("verdict") != "supported" or ok:
        return row
    row = copy.deepcopy(row)
    row["evidence"]["llm"]["verdict"] = "unverifiable"
    row["evidence"]["llm"]["confidence"] = min(llm.get("confidence", 0.0), 0.5)
    return row


def guard_rule(measure, kinds):
    return lambda feat, t: best(feat, measure, kinds) >= t


GUARDS = {
    # name: (rule(feature, threshold) -> passes, thresholds to try; one value = fixed, not tuned)
    "no guard": (lambda f, t: True, [0]),
    "keywords >=50% (live)": (guard_rule("keyword", ("article", "headline")), [0.5]),
    "keywords (tuned)": (guard_rule("keyword", ("article", "headline")), [0.34, 0.5, 0.67, 0.84]),
    "embedding (tuned)": (guard_rule("embed", ("article", "headline")), [0.7, 0.75, 0.8, 0.85, 0.9]),
    "MiniCheck >=0.5": (guard_rule("minicheck", ("article", "headline")), [0.5]),
    "MiniCheck (tuned)": (guard_rule("minicheck", ("article", "headline")), [0.1, 0.3, 0.5, 0.7, 0.9]),
    "MiniCheck, articles only >=0.5": (guard_rule("minicheck", ("article",)), [0.5]),
}


def run_guard(rows, feats, rule, t):
    return [guarded(r, rule(feats[r["id"]], t)) for r in rows]


def total(rows) -> int:
    return sum(PTS[outcome(r)] for r in rows)


def tuned_outcomes(rows, feats, rule, thresholds, seed=0):
    """Pick the threshold on half A, score half B, and the other way round."""
    if len(thresholds) == 1:
        return {r["id"]: outcome(g) for r, g in zip(rows, run_guard(rows, feats, rule, thresholds[0]))}, [thresholds[0]]
    ids = sorted(r["id"] for r in rows)
    random.Random(seed).shuffle(ids)
    halves = [set(ids[: len(ids) // 2]), set(ids[len(ids) // 2:])]
    out, picked = {}, []
    for a, b in ((0, 1), (1, 0)):
        fit = [r for r in rows if r["id"] in halves[a]]
        t = max(thresholds, key=lambda t: (total(run_guard(fit, feats, rule, t)), -abs(t - thresholds[len(thresholds) // 2])))
        picked.append(t)
        test = [r for r in rows if r["id"] in halves[b]]
        out.update({r["id"]: outcome(g) for r, g in zip(test, run_guard(test, feats, rule, t))})
    return out, picked


def auc(pos, neg) -> float:
    """Chance that a random real claim's best source scores higher than a random fake claim's."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=HERE / "runs" / "7b-v3-articles.jsonl")
    ap.add_argument("--features", type=Path, default=HERE / "runs" / "support-features.jsonl")
    args = ap.parse_args()
    feats = {f["id"]: f for f in E.load(args.features)}
    rows = [r for r in E.load(args.base) if r["id"] in feats]
    print(f"{len(rows)} claims with source scores ({sum(r['label'] == 'real' for r in rows)} real)\n")

    print("How well each measure separates real from fake claims (best source per claim; 0.5 = no better than chance)")
    for m in ("keyword", "embed", "minicheck"):
        pos = [best(feats[r["id"]], m) for r in rows if r["label"] == "real"]
        neg = [best(feats[r["id"]], m) for r in rows if r["label"] == "fake"]
        print(f"  {m:10} AUC {auc(pos, neg):.2f}")

    sup = [r for r in rows if (r["evidence"].get("llm") or {}).get("verdict") == "supported"]
    print(f"\nWhere the AI said 'supported' ({len(sup)} claims): best MiniCheck score")
    for label in ("real", "fake"):
        print(f"  {label}: {sorted(round(best(feats[r['id']], 'minicheck'), 2) for r in sup if r['label'] == label)}")

    print("\nGuard                               right wrong uncertain  real->Fake fake->Real  threshold")
    live, _ = tuned_outcomes(rows, feats, *GUARDS["keywords >=50% (live)"])
    results = {}
    for name, (rule, ths) in GUARDS.items():
        res, picked = tuned_outcomes(rows, feats, rule, ths)
        results[name] = res
        c = collections.Counter(res.values())
        ff = sum(1 for r in rows if r["label"] == "real" and res[r["id"]] == "wrong")
        fr = sum(1 for r in rows if r["label"] == "fake" and res[r["id"]] == "wrong")
        print(f"  {name:33} {c['right']:5} {c['wrong']:5} {c['uncertain']:9} {ff:11} {fr:10}  {picked}")

    print("\nChange against the live keyword guard (right=+1, wrong=-1; 95% range by paired bootstrap)")
    ids = [r["id"] for r in rows]
    for name, res in results.items():
        diff = [PTS[res[i]] - PTS[live[i]] for i in ids]
        rng = random.Random(1)
        boots = sorted(sum(diff[rng.randrange(len(ids))] for _ in ids) for _ in range(4000))
        print(f"  {name:33} net {sum(diff):+3}  ({boots[100]:+} to {boots[3899]:+})  "
              f"better {sum(d > 0 for d in diff)}, worse {sum(d < 0 for d in diff)}")


if __name__ == "__main__":
    main()
