"""Would the AVeriTeC-trained stance model make a better evidence guard on our own labelled claims?

Scores every source the AI judge saw in the live-config tuning run (eval/runs/7b-v3-articles.jsonl) with
the model fine-tuned by training/train_stance.py and with the same NLI model before training, then
replays the run with each guard and counts right / wrong / uncertain verdicts, as support_eval.py does
for the keyword guard and MiniCheck:

  support guard   a "supported" verdict needs a source the model says entails the claim
  refute guard    a "refuted" verdict needs a source the model says contradicts it (nothing checks
                  "refuted" today)

A failed guard turns the verdict into "unverifiable" before fusion. Thresholds are either fixed from
AVeriTeC (the fine-tuned model's validation split, or the AVeriTeC train sample for the untrained
model), or chosen on one half of our claims and scored on the other half, both ways round.

Usage (from the project root, after training):
  ai-gateway\\.venv\\Scripts\\python eval\\stance_tuning.py score
  ai-gateway\\.venv\\Scripts\\python eval\\stance_tuning.py report
Tuning set only: never the independent holdout.
"""

import argparse
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
sys.argv, _argv = sys.argv[:1], sys.argv  # evaluate.py parses its own arguments on import
import evaluate as E  # noqa: E402
import support_eval as S  # noqa: E402

sys.argv = _argv
import averitec_stance as A  # noqa: E402
from support_features import shown_sources  # noqa: E402

BASE_RUN = HERE / "runs" / "7b-v3-articles.jsonl"
SUPPORT = HERE / "runs" / "support-features.jsonl"
FEATURES = HERE / "runs" / "stance-features.jsonl"
GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def score(args) -> None:
    import train_stance as T

    rows = E.load(BASE_RUN)
    feats = {f["id"]: f for f in E.load(SUPPORT)}
    items, where = [], []
    for r in rows:
        f = feats.get(r["id"])
        if f is None:
            continue
        ev = r["evidence"]
        claim = (ev.get("claims") or [r["claim"]])[0]
        srcs = shown_sources(ev)
        assert [s["url"] for s in srcs] == [s["url"] for s in f["sources"]], f"sources differ for claim {r['id']}"
        for k, s in enumerate(srcs):
            items.append({"claim": claim, "evidence": s["text"]})
            where.append((r["id"], k))
    print(f"{len(items)} sources of {len(feats)} claims", flush=True)
    if FEATURES.exists():  # resume: keep the models already scored
        saved = {f["id"]: f for f in E.load(FEATURES)}
        for rid, f in saved.items():
            if rid in feats and [s["url"] for s in f["sources"]] == [s["url"] for s in feats[rid]["sources"]]:
                feats[rid] = f
    import gc

    for prefix, path in (("ft", T.OUT), ("nli", T.BASE)):
        if all(prefix + "E" in feats[rid]["sources"][k] for rid, k in where):
            print(f"{prefix} already scored", flush=True)
            continue
        tok = T.AutoTokenizer.from_pretrained(path)
        model = T.load_model(path, "cpu")
        probs = T.predict(model, tok, items, 384, "cpu")
        for (rid, k), p in zip(where, probs):
            src = feats[rid]["sources"][k]
            src[prefix + "E"], src[prefix + "N"], src[prefix + "C"] = p["e"], p["n"], p["c"]
        del model, tok
        gc.collect()  # free the first model's RAM before loading the second
        with FEATURES.open("w", encoding="utf-8") as f:
            for rid in sorted(feats):
                f.write(json.dumps(feats[rid], ensure_ascii=False) + "\n")
        print(f"scored with {prefix} ({path}); saved {FEATURES}", flush=True)


def best(feat: dict, measure: str) -> float:
    return max((s.get(measure) or 0.0 for s in feat["sources"]), default=0.0)


def replay(row: dict, feat: dict, support=None, refute=None) -> dict:
    """The run row with a failed guard's verdict turned into "unverifiable"."""
    llm = row["evidence"].get("llm") or {}
    rule = support if llm.get("verdict") == "supported" else refute if llm.get("verdict") == "refuted" else None
    if rule is None or rule(feat):
        return row
    row = copy.deepcopy(row)
    row["evidence"]["llm"]["verdict"] = "unverifiable"
    row["evidence"]["llm"]["confidence"] = min(llm.get("confidence", 0.0), 0.5)
    return row


def outcomes(rows, feats, make, thresholds, seed=0):
    """make(t) -> (support rule, refute rule). One threshold: fixed. Several: chosen on half A, scored on
    half B and the other way round (the same halves as support_eval.py)."""
    def run(part, t):
        sup, ref = make(t)
        return {r["id"]: S.outcome(replay(r, feats[r["id"]], sup, ref)) for r in part}

    if len(thresholds) == 1:
        return run(rows, thresholds[0]), [thresholds[0]]
    ids = sorted(r["id"] for r in rows)
    random.Random(seed).shuffle(ids)
    halves = [set(ids[: len(ids) // 2]), set(ids[len(ids) // 2:])]
    out, picked = {}, []
    for a, b in ((0, 1), (1, 0)):
        fit = [r for r in rows if r["id"] in halves[a]]
        t = max(thresholds, key=lambda t: (sum(S.PTS[v] for v in run(fit, t).values()), -abs(t - 0.5)))
        picked.append(t)
        out.update(run([r for r in rows if r["id"] in halves[b]], t))
    return out, picked


def report(args) -> None:
    feats = {f["id"]: f for f in E.load(FEATURES)}
    rows = [r for r in E.load(BASE_RUN) if r["id"] in feats]
    labels = {r["id"]: r["label"] for r in rows}

    # fixed thresholds from AVeriTeC, never from these claims
    va, tr = collections.defaultdict(list), collections.defaultdict(list)
    for r in A.load(A.STANCE_VAL):
        va[r["kind"]].append(r)
    for r in A.load(A.SCORES["train"]):
        tr[r["kind"]].append(r)
    get = lambda rs, m: [x[m] for x in rs]  # noqa: E731
    th = {"ftE": A.best_threshold(get(va["supports"], "ftE"), get(va["refutes"], "ftE") + get(va["topic"], "ftE")),
          "ftC": A.best_threshold(get(va["refutes"], "ftC"), get(va["supports"], "ftC") + get(va["topic"], "ftC")),
          "nliE": A.best_threshold(get(tr["supports"], "nliE"), get(tr["refutes"], "nliE") + get(tr["topic"], "nliE")),
          "nliC": A.best_threshold(get(tr["refutes"], "nliC"), get(tr["supports"], "nliC") + get(tr["topic"], "nliC"))}
    print(f"{len(rows)} claims ({sum(v == 'real' for v in labels.values())} real); thresholds from AVeriTeC: "
          + ", ".join(f"{k} {v:.3f}" for k, v in th.items()))

    print("\nHow well the best source separates real from fake claims (AUC, 0.5 = chance)")
    for m in ("keyword", "embed", "minicheck", "nliE", "ftE", "nliC", "ftC"):
        pos = [best(feats[r["id"]], m) for r in rows if r["label"] == "real"]
        neg = [best(feats[r["id"]], m) for r in rows if r["label"] == "fake"]
        flip = m.endswith("C")  # contradiction should be higher for fake claims
        print(f"  {m:10} AUC {S.auc(neg, pos) if flip else S.auc(pos, neg):.2f}"
              + ("  (fake above real)" if flip else ""))

    for verdict, m in (("supported", "ftE"), ("refuted", "ftC")):
        said = [r for r in rows if (r["evidence"].get("llm") or {}).get("verdict") == verdict]
        print(f"\nWhere the AI said '{verdict}' ({len(said)} claims): best fine-tuned {m} per claim")
        for label in ("real", "fake"):
            print(f"  {label}: {sorted(round(best(feats[r['id']], m), 2) for r in said if r['label'] == label)}")

    def at(m, t):
        return lambda f: best(f, m) >= t

    guards = {
        "no guard": (lambda t: (None, None), [0]),
        "keywords >=50% (live)": (lambda t: (at("keyword", 0.5), None), [0]),
        "MiniCheck >=0.5 (rejected before)": (lambda t: (at("minicheck", 0.5), None), [0]),
        "support: NLI, AVeriTeC threshold": (lambda t: (at("nliE", th["nliE"]), None), [0]),
        "support: fine-tuned, AVeriTeC thr.": (lambda t: (at("ftE", th["ftE"]), None), [0]),
        "support: fine-tuned, tuned": (lambda t: (at("ftE", t), None), GRID),
        "refute: NLI, AVeriTeC threshold": (lambda t: (at("keyword", 0.5), at("nliC", th["nliC"])), [0]),
        "refute: fine-tuned, AVeriTeC thr.": (lambda t: (at("keyword", 0.5), at("ftC", th["ftC"])), [0]),
        "refute: fine-tuned, tuned": (lambda t: (at("keyword", 0.5), at("ftC", t)), GRID),
        "both: fine-tuned, AVeriTeC thr.": (lambda t: (at("ftE", th["ftE"]), at("ftC", th["ftC"])), [0]),
    }
    print("\n(refute guards keep the live keyword guard for 'supported')")
    print("Guard                                right wrong uncertain  real->Fake fake->Real  threshold")
    results = {}
    for name, (make, ths) in guards.items():
        res, picked = outcomes(rows, feats, make, ths)
        results[name] = res
        c = collections.Counter(res.values())
        ff = sum(labels[i] == "real" and v == "wrong" for i, v in res.items())
        fr = sum(labels[i] == "fake" and v == "wrong" for i, v in res.items())
        print(f"  {name:34} {c['right']:5} {c['wrong']:5} {c['uncertain']:9} {ff:11} {fr:10}  "
              f"{picked if len(ths) > 1 else ''}")

    print("\nChange against the live keyword guard (right=+1, wrong=-1; 95% range by paired bootstrap)")
    live = results["keywords >=50% (live)"]
    ids = [r["id"] for r in rows]
    for name, res in results.items():
        diff = [S.PTS[res[i]] - S.PTS[live[i]] for i in ids]
        rng = random.Random(1)
        boots = sorted(sum(diff[rng.randrange(len(ids))] for _ in ids) for _ in range(4000))
        print(f"  {name:34} net {sum(diff):+3}  ({boots[100]:+} to {boots[3899]:+})  "
              f"better {sum(d > 0 for d in diff)}, worse {sum(d < 0 for d in diff)}")


def harm(args) -> None:
    """The refute guard at each threshold: what it does to the AI's 'refuted' verdicts and to the final ones.
    A 'refuted' verdict on a fake claim is a correct contradiction; on a real claim it is a false one."""
    feats = {f["id"]: f for f in E.load(FEATURES)}
    rows = [r for r in E.load(BASE_RUN) if r["id"] in feats]
    ref = [r for r in rows if (r["evidence"].get("llm") or {}).get("verdict") == "refuted"]
    good = [r for r in ref if r["label"] == "fake"]
    bad = [r for r in ref if r["label"] == "real"]
    print(f"AI 'refuted' verdicts: {len(good)} on fake claims (correct), {len(bad)} on real claims (false)")
    print(f"\n{'threshold':>9} {'correct kept':>13} {'false removed':>14} {'false left':>11}   final verdicts: "
          f"right wrong uncertain  real->Fake")
    at = lambda m, t: (lambda f: best(f, m) >= t)  # noqa: E731
    for t in [None] + GRID:
        keep = (lambda f: True) if t is None else at("ftC", t)
        res = {r["id"]: S.outcome(replay(r, feats[r["id"]], at("keyword", 0.5), keep)) for r in rows}
        c = collections.Counter(res.values())
        ff = sum(r["label"] == "real" and res[r["id"]] == "wrong" for r in rows)
        kept = sum(keep(feats[r["id"]]) for r in good)
        removed = sum(not keep(feats[r["id"]]) for r in bad)
        print(f"{'none' if t is None else t:>9} {kept:>9}/{len(good)} {removed:>10}/{len(bad)} {len(bad) - removed:>11}"
              f"   {c['right']:>21} {c['wrong']:5} {c['uncertain']:9} {ff:10}")
    print("\nFinal verdict of the real claims the AI wrongly refuted, with no guard:")
    for r in bad:
        print(f"  {S.outcome(r):9} best ftC {best(feats[r['id']], 'ftC'):.2f}  {r['claim'][:90]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["score", "report", "harm"])
    args = ap.parse_args()
    {"score": score, "report": report, "harm": harm}[args.step](args)


if __name__ == "__main__":
    main()
