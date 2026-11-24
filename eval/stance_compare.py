"""Old stance model (models/stance-averitec) against the RobustFND-trained one (models/stance-averitec-robust,
training/train_stance.py --aug) on three tests:

  1. AVeriTeC dev pairs (data/averitec/pairs-dev.jsonl): accuracy and macro F1 of the top answer
  2. hard set (eval/hard_stance.jsonl, written by hand, not by Qwen): same measures
  3. end to end on the 120 tuning claims: the run eval/runs/7b-v3-articles.jsonl replayed through fusion with
     each model as the evidence guard ("supported" needs a source the model says entails the claim,
     "refuted" needs one it says contradicts it; a failed check becomes "unverifiable"). Thresholds come
     from each model's own AVeriTeC validation scores, fixed before looking at these claims.

Rule agreed for the gateway: the new model goes in only if its end-to-end net against the live keyword
guard is >= 0.

hard_stance.jsonl, one JSON object per line (see eval/hard_stance.example.jsonl):
  {"id": 1, "claim": "...", "evidence": "...", "label": "supports" | "refutes" | "neutral", "note": "..."}

Usage (from the project root): ai-gateway\\.venv\\Scripts\\python eval\\stance_compare.py
"""

import collections
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
import support_eval as S  # noqa: E402

sys.argv = _argv
import averitec_stance as A  # noqa: E402
import stance_tuning as ST  # noqa: E402
from support_features import shown_sources  # noqa: E402

MODELS = {"old": ROOT / "models" / "stance-averitec", "new": ROOT / "models" / "stance-averitec-robust"}
DEV = ROOT / "data" / "averitec" / "pairs-dev.jsonl"
HARD = HERE / "hard_stance.jsonl"
CACHE = HERE / "runs" / "stance-compare"
GOLD = {"supports": "e", "refutes": "c", "nei": "n", "topic": "n", "neutral": "n"}


def load(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def scored(name: str, what: str, items: list[dict]) -> list[dict]:
    """Probabilities {"e","n","c"} of model `name` for each {"claim","evidence"}, cached on disk."""
    path = CACHE / f"{name}-{what}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if len(cached) == len(items):
            return cached
    import train_stance as T

    tok = T.AutoTokenizer.from_pretrained(MODELS[name])
    model = T.load_model(MODELS[name], "cpu")
    probs = T.predict(model, tok, items, 384, "cpu")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(probs), encoding="utf-8")
    return probs


def classification(gold: list[str], probs: list[dict]):
    pred = [max(p, key=p.get) for p in probs]
    f1s = []
    for c in "enc":
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        prec, rec = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    acc = sum(g == p for g, p in zip(gold, pred)) / len(gold)
    rng = random.Random(7)
    boots = []
    for _ in range(1000):
        idx = [rng.randrange(len(gold)) for _ in gold]
        boots.append(sum(gold[i] == pred[i] for i in idx) / len(idx))
    boots.sort()
    return acc, sum(f1s) / 3, (boots[25], boots[974]), collections.Counter(zip(gold, pred))


def show(title, gold, results):
    print(f"\n{title}: {len(gold)} pairs ({', '.join(f'{k} {v}' for k, v in sorted(collections.Counter(gold).items()))})")
    for name, (acc, f1, (lo, hi), conf) in results.items():
        print(f"  {name}: accuracy {acc:.1%} (95% range {lo:.1%}-{hi:.1%}), macro F1 {f1:.1%}")
        for g in "enc":
            row = [conf[(g, p)] for p in "enc"]
            if sum(row):
                print(f"      gold {g}: answered e/n/c {row[0]}/{row[1]}/{row[2]}")


def main():
    have = {n: p for n, p in MODELS.items() if (p / "config.json").exists()}
    if "new" not in have:
        print("models/stance-averitec-robust is missing: train it first (training/train_stance.py --aug ... "
              "--save-dir models/stance-averitec-robust)")
    # 1. AVeriTeC dev
    dev = [r for r in load(DEV) if r["kind"] in GOLD]
    gold = [GOLD[r["kind"]] for r in dev]
    show("1. AVeriTeC dev", gold, {n: classification(gold, scored(n, "dev", dev)) for n in have})
    # 2. hand-written hard set
    if HARD.exists():
        hard = load(HARD)
        gold = [GOLD[r["label"]] for r in hard]
        show("2. Hard set (hand-written)", gold, {n: classification(gold, scored(n, "hard", hard)) for n in have})
    else:
        print(f"\n2. Hard set: {HARD} not written yet (format: eval/hard_stance.example.jsonl)")
    # 3. end to end on the tuning claims
    feats = {f["id"]: f for f in E.load(ST.SUPPORT)}
    runs = [r for r in E.load(ST.BASE_RUN) if r["id"] in feats]
    items, where = [], []
    for r in runs:
        claim = (r["evidence"].get("claims") or [r["claim"]])[0]
        for k, s in enumerate(shown_sources(r["evidence"])):
            items.append({"claim": claim, "evidence": s["text"]})
            where.append((r["id"], k))
    print(f"\n3. End to end, {len(runs)} tuning claims, {len(items)} sources")
    live = {r["id"]: S.outcome(ST.replay(r, feats[r["id"]], lambda f: ST.best(f, "keyword") >= 0.5, None)) for r in runs}
    labels = {r["id"]: r["label"] for r in runs}
    verdict = {}
    for name in have:
        probs = scored(name, "tuning", items)
        for (rid, k), p in zip(where, probs):
            src = feats[rid]["sources"][k]
            src[name + "E"], src[name + "C"] = p["e"], p["c"]
        va = collections.defaultdict(list)
        for r in load(MODELS[name] / "val-scores.jsonl"):
            va[r["kind"]].append(r)
        get = lambda rs, m: [x[m] for x in rs]  # noqa: E731
        te = A.best_threshold(get(va["supports"], "ftE"), get(va["refutes"], "ftE") + get(va["topic"], "ftE"))
        tc = A.best_threshold(get(va["refutes"], "ftC"), get(va["supports"], "ftC") + get(va["topic"], "ftC"))
        res = {r["id"]: S.outcome(ST.replay(r, feats[r["id"]], lambda f, m=name, t=te: ST.best(f, m + "E") >= t,
                                            lambda f, m=name, t=tc: ST.best(f, m + "C") >= t)) for r in runs}
        c = collections.Counter(res.values())
        ff = sum(labels[i] == "real" and v == "wrong" for i, v in res.items())
        fr = sum(labels[i] == "fake" and v == "wrong" for i, v in res.items())
        diff = [S.PTS[res[i]] - S.PTS[live[i]] for i in res]
        verdict[name] = sum(diff)
        print(f"  {name} model as guard (support >= {te:.3f}, refute >= {tc:.3f}): {c['right']} right / {c['wrong']} wrong / "
              f"{c['uncertain']} uncertain, real->Fake {ff}, fake->Real {fr}; net {sum(diff):+} against the keyword guard "
              f"(better {sum(d > 0 for d in diff)}, worse {sum(d < 0 for d in diff)})")
    c = collections.Counter(live.values())
    print(f"  live keyword guard: {c['right']} right / {c['wrong']} wrong / {c['uncertain']} uncertain")
    if "new" in verdict:
        print(f"\nGateway rule (new model net >= 0): {'PASS' if verdict['new'] >= 0 else 'FAIL'} (net {verdict['new']:+})")


if __name__ == "__main__":
    main()
