"""Directional filter for matched fact-checks, tested offline on the tuning run.

Problem: keyword matching links a submitted claim to fact-checks of *different* claims about the same person
("narendra modi is the pm of india" matched four false-rated Modi fact-checks -> Fake). A false rating says
the fact-checked claim F is false; that makes the submitted claim C false only if C implies F. A true rating
makes C true only if F implies C. The filter keeps a review only when the trained stance model
(training/train_stance.py) gives that entailment at least `t`.

Baseline: the live keyword rule (>= 60% of key terms, either direction) applied to the saved reviews.
Every run is replayed through the same fusion (evaluate.py), so only the fact-check signal changes.

Usage (from the project root):
  ai-gateway\\.venv\\Scripts\\python eval\\factcheck_direction.py
Tuning set only. Thresholds are also chosen on one half of the claims and scored on the other.
"""

import collections
import copy
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "eval"
sys.path.insert(0, str(ROOT / "ai-gateway"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "training"))
sys.argv, _argv = sys.argv[:1], sys.argv
import evaluate as E  # noqa: E402

sys.argv = _argv
from app.claims import overlap  # noqa: E402

RUN = HERE / "runs" / "7b-v3-articles.jsonl"
CACHE = HERE / "runs" / "factcheck-direction-scores.json"
GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
WANT = {"fake": "Fake", "real": "Real"}
PTS = {"right": 1, "uncertain": 0, "wrong": -1}
MODI = "narendra modi is the pm of india"
MODI_FC = ["PM Narendra Modi said “Israel is fatherland and India is motherland,” referring to himself during his speech in Israel.",
           "Video shows PM Modi saying \"Today, India is the world's second largest beef producer\"",
           "Narendra Modi is India's first OBC Prime Minister",
           "Video shows PM Modi raising slogans for the opposition's INDIA bloc."]


def claim_of(r):
    return (r["evidence"].get("claims") or [r["claim"]])[0]


def live_match(claim, fc):
    return max(overlap(claim, fc, top=6), overlap(fc, claim, top=6)) >= 0.6


def outcome(r):
    p = E.predict(r, "full", None)
    return "uncertain" if p == "Uncertain" else ("right" if p == WANT[r["label"]] else "wrong")


def scores(rows):
    """{(claim, fc): (claim implies fc, fc implies claim)}, cached."""
    pairs = {(claim_of(r), rv["claim"]) for r in rows for rv in ((r["evidence"].get("factChecks") or {}).get("reviews") or [])}
    pairs |= {(MODI, f) for f in MODI_FC}
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [p for p in pairs if json.dumps(p) not in cache]
    if todo:
        import train_stance as T

        tok = T.AutoTokenizer.from_pretrained(T.OUT)
        model = T.load_model(T.OUT, "cpu")
        fwd = T.predict(model, tok, [{"evidence": c, "claim": f} for c, f in todo], 256, "cpu")  # claim -> fc
        bwd = T.predict(model, tok, [{"evidence": f, "claim": c} for c, f in todo], 256, "cpu")  # fc -> claim
        for p, a, b in zip(todo, fwd, bwd):
            cache[json.dumps(p)] = [a["e"], b["e"]]
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return {tuple(json.loads(k)): v for k, v in cache.items()}


def filtered(r, sc, t):
    """The run row with only the fact-check reviews the rule keeps (t=None: live keyword rule only)."""
    r = copy.deepcopy(r)
    fc = r["evidence"].get("factChecks") or {}
    c = claim_of(r)
    kept = []
    for rv in fc.get("reviews") or []:
        if not live_match(c, rv["claim"]):
            continue
        if t is not None and rv["stance"] in ("refutes", "supports"):
            c2f, f2c = sc[(c, rv["claim"])]
            if (c2f if rv["stance"] == "refutes" else f2c) < t:
                continue
        kept.append(rv)
    if fc:
        fc["reviews"] = kept
    return r


def has_refuting(r):
    return any(rv["stance"] == "refutes" for rv in ((r["evidence"].get("factChecks") or {}).get("reviews") or []))


def main():
    rows = E.load(RUN)
    sc = scores(rows)
    base = {r["id"]: filtered(r, sc, None) for r in rows}
    base_out = {i: outcome(r) for i, r in base.items()}
    fake_fc = [i for i, r in base.items() if r["label"] == "fake" and has_refuting(r)]
    real_fc = [i for i, r in base.items() if r["label"] == "real" and has_refuting(r)]
    print(f"{len(rows)} tuning claims. Live keyword rule: a false-rated fact-check matches {len(fake_fc)} fake claims "
          f"(genuine signal) and {len(real_fc)} real claims (false match).")

    print("\nModi example (claim implies fact-checked claim, the score the filter uses for a false rating):")
    for f in MODI_FC:
        print(f"  {sc[(MODI, f)][0]:.2f}  {f[:80]}")

    print(f"\n{'rule':16} {'fake kept':>10} {'false removed':>14} {'right':>6} {'wrong':>6} {'uncertain':>10} "
          f"{'real->Fake':>11} {'fake->Real':>11}  Modi FCs kept")
    results = {}
    for t in [None] + GRID:
        runs = {i: filtered(r, sc, t) for i, r in base.items()}
        out = {i: outcome(r) for i, r in runs.items()}
        results[t] = out
        c = collections.Counter(out.values())
        kept = sum(has_refuting(runs[i]) for i in fake_fc)
        removed = sum(not has_refuting(runs[i]) for i in real_fc)
        ff = sum(runs[i]["label"] == "real" and v == "wrong" for i, v in out.items())
        fr = sum(runs[i]["label"] == "fake" and v == "wrong" for i, v in out.items())
        modi = "4/4" if t is None else f"{sum(sc[(MODI, f)][0] >= t for f in MODI_FC)}/4"
        name = "live keywords" if t is None else f"directional {t}"
        print(f"{name:16} {kept:>6}/{len(fake_fc):<3} {removed:>10}/{len(real_fc):<3} {c['right']:>6} {c['wrong']:>6} "
              f"{c['uncertain']:>10} {ff:>11} {fr:>11}  {modi}")

    # threshold chosen on one half, scored on the other (both ways)
    ids = sorted(base)
    random.Random(0).shuffle(ids)
    halves = [set(ids[: len(ids) // 2]), set(ids[len(ids) // 2:])]
    mixed, picked = {}, []
    for a, b in ((0, 1), (1, 0)):
        t = max(GRID, key=lambda t: (sum(PTS[results[t][i]] for i in halves[a]), -abs(t - 0.5)))
        picked.append(t)
        mixed.update({i: results[t][i] for i in halves[b]})
    diff = [PTS[mixed[i]] - PTS[base_out[i]] for i in ids]
    rng = random.Random(1)
    boots = sorted(sum(diff[rng.randrange(len(ids))] for _ in ids) for _ in range(4000))
    print(f"\nThreshold chosen on half the claims (picked {picked}), scored on the other half: net {sum(diff):+} "
          f"against live (95% range {boots[100]:+} to {boots[3899]:+}), better {sum(d > 0 for d in diff)}, "
          f"worse {sum(d < 0 for d in diff)}")

    print("\nFake claims whose genuine fact-check signal the filter removes at 0.5 (audit these):")
    for i in fake_fc:
        r = filtered(base[i], sc, 0.5)
        if not has_refuting(r):
            c = claim_of(base[i])
            print(f"  [{base_out[i]} -> {results[0.5][i]}] {c[:100]}")
            for rv in base[i]["evidence"]["factChecks"]["reviews"]:
                if rv["stance"] == "refutes" and live_match(c, rv["claim"]):
                    print(f"      claim->fc {sc[(c, rv['claim'])][0]:.2f}  [{rv['rating']}] {rv['claim'][:100]}")


if __name__ == "__main__":
    main()
