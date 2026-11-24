"""Does MiniCheck help as an extra fusion signal (not a guard)? Simulated on the saved tuning-set run.

When the best source the AI saw (articles + headlines, scored by support_features.py) reaches a
MiniCheck threshold, fusion moves towards Real by a weight (fusion.WEIGHTS "minicheck_support"),
optionally also lifting the "found nothing" confidence penalty. The baseline is the live setup:
fusion-v3 with the keyword evidence guard.

Weight, threshold and the penalty switch are chosen on one half of the claims and scored on the
other half (both ways round), as in support_eval.py.

Usage (from the project root):
  ai-gateway\\.venv\\Scripts\\python eval\\minicheck_fusion.py
"""

import collections
import copy
import itertools
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import support_eval as S  # noqa: E402  (also imports evaluate as S.E)

BASE = HERE / "runs" / "7b-v3-articles.jsonl"
FEATURES = HERE / "runs" / "support-features.jsonl"
GRID = {
    "minicheck_support": [1.0, 1.5, 2.0, 2.5, 3.0],
    "minicheck_at": [0.5, 0.7, 0.8, 0.9],
    "minicheck_clears_penalty": [0.0, 1.0],
}
CONFIGS = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]


def outcome(row: dict, weights: dict | None) -> str:
    p = S.E.fusion.decide(row["detector"]["prob"], row["detector"]["isModel"], row["evidence"], weights).verdict
    return "uncertain" if p == "Uncertain" else ("right" if p == S.WANT[row["label"]] else "wrong")


def points(rows, weights) -> tuple[int, int]:
    res = [outcome(r, weights) for r in rows]
    return sum(S.PTS[x] for x in res), -res.count("wrong")


def summary(name, rows, res, picked=""):
    c = collections.Counter(res.values())
    ff = sum(r["label"] == "real" and res[r["id"]] == "wrong" for r in rows)
    fr = sum(r["label"] == "fake" and res[r["id"]] == "wrong" for r in rows)
    print(f"  {name:34} {c['right']:5} {c['wrong']:5} {c['uncertain']:9} {ff:10} {fr:10}  {picked}")


def main():
    feats = {f["id"]: f for f in S.E.load(FEATURES)}
    rows = []
    for r in S.E.load(BASE):
        if r["id"] not in feats:
            continue
        f = feats[r["id"]]
        r = S.guarded(r, S.best(f, "keyword") >= 0.5)  # the live keyword guard
        r = copy.deepcopy(r)
        r["evidence"]["minicheck"] = {"best": S.best(f, "minicheck")}
        rows.append(r)
    print(f"{len(rows)} tuning claims ({sum(r['label'] == 'real' for r in rows)} real); baseline = live keyword guard\n")

    live = {r["id"]: outcome(r, None) for r in rows}

    # Honest number: choose on half A, score on half B, and the other way round.
    ids = sorted(r["id"] for r in rows)
    random.Random(0).shuffle(ids)
    halves = [set(ids[: len(ids) // 2]), set(ids[len(ids) // 2:])]
    tuned, picked = {}, []
    for a, b in ((0, 1), (1, 0)):
        fit = [r for r in rows if r["id"] in halves[a]]
        # ties: fewer wrong, then the smallest weight and the strictest threshold (least change)
        cfg = max(CONFIGS, key=lambda c: (*points(fit, c), -c["minicheck_support"], c["minicheck_at"]))
        picked.append(cfg)
        tuned.update({r["id"]: outcome(r, cfg) for r in rows if r["id"] in halves[b]})

    print("Setup                               right wrong uncertain real->Fake fake->Real")
    summary("live (keyword guard)", rows, live)
    summary("+ MiniCheck signal (tuned on halves)", rows, tuned)
    for i, cfg in enumerate(picked):
        print(f"    half {i + 1} picked {cfg}")

    print("\nEvery setting on all 120 claims (chosen after the fact, so optimistic):")
    print("  weight  at   clears  right wrong uncertain  net vs live")
    base_pts = sum(S.PTS[x] for x in live.values())
    for cfg in CONFIGS:
        res = [outcome(r, cfg) for r in rows]
        c = collections.Counter(res)
        net = sum(S.PTS[x] for x in res) - base_pts
        print(f"  {cfg['minicheck_support']:6} {cfg['minicheck_at']:4} {cfg['minicheck_clears_penalty']:7}  "
              f"{c['right']:5} {c['wrong']:5} {c['uncertain']:9}  {net:+}")

    print("\nTuned signal against live (right=+1, wrong=-1; 95% range by paired bootstrap)")
    diff = [S.PTS[tuned[i]] - S.PTS[live[i]] for i in ids]
    rng = random.Random(1)
    boots = sorted(sum(diff[rng.randrange(len(ids))] for _ in ids) for _ in range(4000))
    print(f"  net {sum(diff):+}  ({boots[100]:+} to {boots[3899]:+})  "
          f"better {sum(d > 0 for d in diff)}, worse {sum(d < 0 for d in diff)}")
    changed = [r for r in rows if tuned[r["id"]] != live[r["id"]]]
    for r in changed:
        print(f"    {r['label']:4} {live[r['id']]:9} -> {tuned[r['id']]:9}  MiniCheck {r['evidence']['minicheck']['best']:.2f}  "
              f"AI said {(r['evidence'].get('llm') or {}).get('verdict')}")


if __name__ == "__main__":
    main()
