"""Does the evidence check tell "states the claim" apart from "same topic"? Measured on AVeriTeC.

AVeriTeC (Schlichtkrull et al., NeurIPS 2023, CC BY-NC 4.0; github.com/MichSchli/AVeriTeC) has real
fact-checked claims, each with the evidence a fact-checker collected (question-answer pairs with source
links) and a verdict. Every claim gives up to two (claim, evidence) pairs:

  own evidence   the claim's own evidence; kind = its verdict: supports, refutes, nei (not enough
                 evidence) or conflicting
  topic          the evidence of a different claim, picked at random among those whose evidence passes
                 the live keyword guard against this claim (at least half of its 6 key terms): same
                 topic, but not about this claim

Steps (from the project root; train.json and dev.json in data/averitec/):
  ai-gateway\\.venv\\Scripts\\python eval\\averitec_stance.py build
  ai-gateway\\.venv\\Scripts\\python eval\\averitec_stance.py score          # keyword, embedding, NLI
  ai-gateway\\.venv\\Scripts\\python eval\\averitec_stance.py judge --n 40   # live Qwen judge, slow, resumable
  ai-gateway\\.venv\\Scripts\\python eval\\averitec_stance.py report

Thresholds are chosen on train pairs; dev pairs are only measured. The AVeriTeC test split is not used.
"""

import argparse
import collections
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai-gateway"))

import numpy as np  # noqa: E402

from app import config, credibility, reasoner  # noqa: E402
from app.claims import _WORD, _stem, keywords, overlap  # noqa: E402
from app.reasoner import SUPPORT_MATCH  # noqa: E402

DATA = ROOT / "data" / "averitec"
RUNS = ROOT / "eval" / "runs"
PAIRS = {s: DATA / f"pairs-{s}.jsonl" for s in ("train", "dev")}
SCORES = {s: RUNS / f"averitec-scores-{s}.jsonl" for s in ("train", "dev")}
JUDGE = RUNS / "averitec-judge-dev.jsonl"
# the model fine-tuned by training/train_stance.py: its dev scores, and its validation scores (10% of train)
STANCE_DEV = RUNS / "averitec-stance-dev.jsonl"
STANCE_VAL = ROOT / "models" / "stance-averitec" / "val-scores.jsonl"
KIND = {"Supported": "supports", "Refuted": "refutes", "Not Enough Evidence": "nei",
        "Conflicting Evidence/Cherrypicking": "conflicting"}
KINDS = ["supports", "refutes", "nei", "conflicting", "topic"]
TRAIN_CLAIMS = 1000      # train claims used (NLI on the laptop CPU costs about a second per pair)
MAX_CHARS = 1500         # the live passage limit is 1200 per article
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
SEED = 7
ARCHIVE = re.compile(r"web\.archive\.org/web/(\d{8})\d*[^/]*/(.+)")


def unarchive(url: str) -> tuple[str, str]:
    """(original url, archive date YYYY-MM-DD or "")."""
    m = ARCHIVE.search(url or "")
    if not m:
        return url or "", ""
    d = m.group(1)
    return m.group(2), f"{d[:4]}-{d[4:6]}-{d[6:]}"


def domain(url: str) -> str:
    m = re.match(r"(?:https?://)?([^/:?#]+)", url or "")
    return (m.group(1) if m else "").lower().removeprefix("www.")


def qa_items(item: dict) -> list[dict]:
    out = []
    for q in item.get("questions") or []:
        for a in q.get("answers") or []:
            answer = (a.get("answer") or "").strip()
            if a.get("answer_type") == "Boolean" and a.get("boolean_explanation"):
                answer = f"{answer}. {a['boolean_explanation'].strip()}"
            url, date = unarchive(a.get("source_url") or "")
            out.append({"question": q["question"].strip(), "answer": answer, "url": url, "date": date})
    return out


def evidence_text(qas: list[dict]) -> str:
    return "\n".join(f"Q: {x['question']} A: {x['answer']}" for x in qas)[:MAX_CHARS]


# ------------------------------------------------------------------ build

def build(args) -> None:
    """Measurement pairs (pairs-dev, pairs-train from a 1000-claim train sample) and training pairs for
    training/train_stance.py (stance-train and stance-val: every train claim, split 90/10 by claim)."""
    train_items = json.loads((DATA / "train.json").read_text(encoding="utf-8"))
    dev_items = json.loads((DATA / "dev.json").read_text(encoding="utf-8"))
    order = list(range(len(train_items)))
    random.Random(f"{SEED}-stance").shuffle(order)
    cut = len(order) // 10
    jobs = [("dev", dev_items, list(range(len(dev_items))), PAIRS["dev"]),
            ("train", train_items, sorted(random.Random(SEED).sample(range(len(train_items)), TRAIN_CLAIMS)),
             PAIRS["train"]),
            ("stance-val", train_items, sorted(order[:cut]), DATA / "stance-val.jsonl"),
            ("stance-train", train_items, sorted(order[cut:]), DATA / "stance-train.jsonl")]
    for split, items, idx, out in jobs:
        make_pairs(split, items, idx, out)


def make_pairs(split: str, items: list[dict], idx: list[int], out: Path) -> None:
    """Own-evidence and same-topic pairs for the claims items[i], i in idx (partners come from idx too)."""
    from app.support import _embedder

    claims = [items[i]["claim"].strip() for i in idx]
    qas = [qa_items(items[i]) for i in idx]
    evid = [evidence_text(q) for q in qas]
    # the live guard's measure, vectorised: share of the claim's 6 key terms (stemmed) found in the text
    keys = [{_stem(w) for w in keywords(c, 6)} for c in claims]
    words = [{_stem(w) for w in _WORD.findall(e)} for e in evid]
    for k in random.Random(SEED).sample(range(len(idx)), 30):  # same numbers as claims.overlap
        j = (k + 1) % len(idx)
        mine = len(keys[k] & words[j]) / len(keys[k]) if keys[k] else 0.0
        assert abs(mine - overlap(claims[k], evid[j], top=6, stem=True)) < 1e-9
    ce = _embedder().encode(claims, normalize_embeddings=True, batch_size=32)
    near = (ce @ ce.T) >= 0.8  # near-duplicate claims (meaning similarity)
    articles = [items[i].get("fact_checking_article") or f"none-{i}" for i in idx]
    # The partner is drawn at random from every claim whose evidence passes the guard, not the one with
    # the most overlap: picking by keyword or embedding score would make those measures look worse.
    rng = random.Random(f"{SEED}-{split}")
    pairs, with_topic, counts = [], 0, []
    for k, i in enumerate(idx):
        item = items[i]
        if not evid[k] or item["label"] not in KIND:
            continue
        base = {"split": split, "claimIdx": i, "claim": claims[k], "claimDate": item.get("claim_date") or ""}
        pairs.append({**base, "pid": f"{split}-{i}-own", "kind": KIND[item["label"]], "evidence": evid[k],
                      "qas": qas[k]})
        # a different claim, not a near-duplicate of this one, from a different fact-check
        need = SUPPORT_MATCH * len(keys[k])
        cands = [j for j in range(len(idx))
                 if keys[k] and len(keys[k] & words[j]) >= need - 1e-9
                 and j != k and evid[j] and articles[j] != articles[k] and not near[k, j]]
        counts.append(len(cands))
        if cands:
            j = rng.choice(cands)
            with_topic += 1
            pairs.append({**base, "pid": f"{split}-{i}-topic", "kind": "topic", "evidence": evid[j],
                          "qas": qas[j], "partnerIdx": idx[j], "partnerClaim": claims[j],
                          "partnerLabel": items[idx[j]]["label"], "candidates": len(cands)})
    with out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    c = collections.Counter(p["kind"] for p in pairs)
    print(f"{split}: {len(idx)} claims, {len(pairs)} pairs {dict(c)}; "
          f"{with_topic} claims have a same-topic partner that passes the keyword guard "
          f"(median {int(np.median(counts))} such partners per claim)", flush=True)


# ------------------------------------------------------------------ fast scores

class NLI:
    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        # float32: the checkpoint is stored in float16, which transformers 5 keeps, and float16 runs
        # about 17 times slower on this CPU (same probabilities to 3 decimals)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).float().eval()
        lab = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.e, self.n, self.c = lab["entailment"], lab["neutral"], lab["contradiction"]

    def __call__(self, premises: list[str], hypotheses: list[str], max_length: int, batch: int = 8) -> list[tuple]:
        out = []
        for i in range(0, len(premises), batch):
            enc = self.tok(premises[i:i + batch], hypotheses[i:i + batch], truncation="only_first",
                           max_length=max_length, padding=True, return_tensors="pt")
            with self.torch.inference_mode():
                probs = self.model(**enc).logits.softmax(-1)
            out += [(float(p[self.e]), float(p[self.n]), float(p[self.c])) for p in probs]
        return out


def score(args) -> None:
    from app.support import _embedder

    keep_awake()
    nli = NLI()
    emb = _embedder()
    for split in ("dev", "train"):
        pairs = [json.loads(x) for x in PAIRS[split].read_text(encoding="utf-8").splitlines() if x.strip()]
        done = set()
        if SCORES[split].exists():
            done = {json.loads(x)["pid"] for x in SCORES[split].read_text(encoding="utf-8").splitlines() if x.strip()}
        todo = [p for p in pairs if p["pid"] not in done]
        print(f"{split}: {len(done)} scored, {len(todo)} to go", flush=True)
        start = time.time()
        for n in range(0, len(todo), 32):
            part = todo[n:n + 32]
            ce = emb.encode([p["claim"] for p in part], normalize_embeddings=True, batch_size=32)
            ee = emb.encode([p["evidence"] for p in part], normalize_embeddings=True, batch_size=32)
            full = nli([p["evidence"] for p in part], [p["claim"] for p in part], max_length=512)
            lines = [(k, f"Q: {x['question']} A: {x['answer']}") for k, p in enumerate(part) for x in p["qas"]]
            per = nli([t for _, t in lines], [part[k]["claim"] for k, _ in lines], max_length=256)
            best = collections.defaultdict(lambda: [0.0, 0.0])
            for (k, _), (e, _n, c) in zip(lines, per):
                best[k][0], best[k][1] = max(best[k][0], e), max(best[k][1], c)
            with SCORES[split].open("a", encoding="utf-8") as f:
                for k, p in enumerate(part):
                    f.write(json.dumps({
                        "pid": p["pid"], "kind": p["kind"],
                        "keyword": round(overlap(p["claim"], p["evidence"], top=6, stem=True), 4),
                        "embed": round(float(ce[k] @ ee[k]), 4),
                        "nliE": round(full[k][0], 4), "nliN": round(full[k][1], 4), "nliC": round(full[k][2], 4),
                        "lineE": round(best[k][0], 4), "lineC": round(best[k][1], 4),
                    }) + "\n")
            print(f"  {split}: {n + len(part)}/{len(todo)} ({time.time() - start:.0f}s)", flush=True)


# ------------------------------------------------------------------ live Qwen judge

def keep_awake() -> None:
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x1 | 0x2)


def judge(args) -> None:
    """The live judge (Ollama, prompt v3) reads each pair's evidence as one source: all its answers
    together, about as long as one article passage in the gateway (so the guard sees the same text the
    pairs were built on). Its verdict is recorded before and after the live keyword guard."""
    keep_awake()
    pairs = [json.loads(x) for x in PAIRS["dev"].read_text(encoding="utf-8").splitlines() if x.strip()]
    rng = random.Random(SEED)
    sample = []
    for kind in ("supports", "refutes", "topic"):
        group = [p for p in pairs if p["kind"] == kind]
        sample += rng.sample(group, min(args.n, len(group)))
    rng.shuffle(sample)
    done = {r["pid"] for r in load(JUDGE) if r["status"] == "ok"}
    todo = [p for p in sample if p["pid"] not in done]
    print(f"judge: {len(done)} done, {len(todo)} to go, model {config.OLLAMA_MODEL}, prompt {config.LLM_PROMPT_VERSION}",
          flush=True)
    guard = reasoner._require_matching_evidence
    reasoner._require_matching_evidence = lambda data, claim, passages, headlines: data  # raw verdict first
    try:
        for n, p in enumerate(todo, 1):
            domains = collections.Counter(domain(x["url"]) for x in p["qas"] if x["url"])
            cred = credibility.rate(domains.most_common(1)[0][0] if domains else "")
            dates = sorted(x["date"] for x in p["qas"] if x["date"])
            passages = [{"url": f"averitec:{p['pid']}", "title": "", "domain": cred["domain"] or "unknown",
                         "credibility": cred["score"], "published": dates[-1] if dates else "",
                         "passage": p["evidence"]}]
            t = time.time()
            for attempt in range(10):  # a dropped SSH tunnel to the HPC reconnects within a minute or so
                raw = reasoner._assess_ollama([p["claim"]], p["claim"], "", passages)
                if raw.get("status") == "ok":
                    break
                print(f"  judge error ({raw.get('reason')}); retrying in 20 s", flush=True)
                reasoner._ollama_check = (0.0, False)
                time.sleep(20)
            if raw.get("status") != "ok":
                print(f"  giving up on {p['pid']} for now (not saved, so a rerun tries it again)", flush=True)
                continue
            guarded = guard(dict(raw), p["claim"], passages, None)
            rec = {"pid": p["pid"], "kind": p["kind"], "claim": p["claim"], "status": raw.get("status"),
                   "reason": raw.get("reason", ""), "verdict": raw.get("verdict"), "confidence": raw.get("confidence"),
                   "guarded": guarded.get("verdict"), "downgraded": guarded.get("downgraded", ""),
                   "cited": [s.get("stance") for s in raw.get("sources") or []],
                   "summary": (raw.get("summary") or "")[:400], "seconds": round(time.time() - t, 1),
                   "backend": config.OLLAMA_URL}
            with JUDGE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{n}/{len(todo)}] {p['kind']:8} -> {rec['verdict']} / guarded {rec['guarded']} "
                  f"({rec['seconds']:.0f}s)", flush=True)
    finally:
        reasoner._require_matching_evidence = guard
    print("done", flush=True)


# ------------------------------------------------------------------ report

def auc(pos: list[float], neg: list[float]) -> float:
    """Chance that a random positive scores above a random negative (ties count half)."""
    if not pos or not neg:
        return float("nan")
    a, b = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((a > b).mean() + 0.5 * (a == b).mean())


def best_threshold(pos: list[float], neg: list[float]) -> float:
    """Threshold with the best balanced accuracy (chosen on train)."""
    grid = sorted(set(round(v, 3) for v in pos + neg))
    return max(grid, key=lambda t: (np.mean([v >= t for v in pos]) + np.mean([v < t for v in neg])) / 2)


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def report(args) -> None:
    sc = {s: load(SCORES[s]) for s in ("train", "dev")}
    ft = {r["pid"]: r for r in load(STANCE_DEV)}
    for r in sc["dev"]:
        r.update({k: v for k, v in ft.get(r["pid"], {}).items() if k.startswith("ft")})
    has_ft = bool(sc["dev"]) and all("ftE" in r for r in sc["dev"])
    by = {s: collections.defaultdict(list) for s in sc}
    for s in sc:
        for r in sc[s]:
            by[s][r["kind"]].append(r)
    for s in ("train", "dev"):
        print(f"{s}: " + ", ".join(f"{k} {len(by[s][k])}" for k in KINDS))
    tr, dv = by["train"], by["dev"]

    def get(rows, m):
        return [r[m] for r in rows]

    measures = [("keyword", "keyword guard overlap"), ("embed", "embedding similarity"),
                ("nliE", "NLI entailment, all evidence"), ("lineE", "NLI entailment, best answer"),
                ("nliC", "NLI contradiction, all evidence"), ("lineC", "NLI contradiction, best answer")]
    if has_ft:
        measures += [("ftE", "fine-tuned entailment"), ("ftC", "fine-tuned contradiction")]
    print("\nAUC on dev (0.5 = cannot tell them apart, 1.0 = perfect)")
    print(f"  {'measure':34} supports>topic  refutes>topic  supports>refutes  refutes>supports")
    for m, name in measures:
        print(f"  {name:34} {auc(get(dv['supports'], m), get(dv['topic'], m)):14.2f} "
              f"{auc(get(dv['refutes'], m), get(dv['topic'], m)):14.2f} "
              f"{auc(get(dv['supports'], m), get(dv['refutes'], m)):17.2f} "
              f"{auc(get(dv['refutes'], m), get(dv['supports'], m)):17.2f}")
    # thresholds chosen on train only: "states the claim" (supports vs the rest), "contradicts" (refutes vs the rest)
    th = {}
    if tr["supports"]:
        for m in ("nliE", "lineE"):
            th[m] = best_threshold(get(tr["supports"], m), get(tr["refutes"], m) + get(tr["topic"], m))
        for m in ("nliC", "lineC"):
            th[m] = best_threshold(get(tr["refutes"], m), get(tr["supports"], m) + get(tr["topic"], m))
    if has_ft:  # the fine-tuned model's thresholds come from its own validation split, not from dev
        va = collections.defaultdict(list)
        for r in load(STANCE_VAL):
            va[r["kind"]].append(r)
        th["ftE"] = best_threshold(get(va["supports"], "ftE"), get(va["refutes"], "ftE") + get(va["topic"], "ftE"))
        th["ftC"] = best_threshold(get(va["refutes"], "ftC"), get(va["supports"], "ftC") + get(va["topic"], "ftC"))
    rules = [("keyword guard >= 0.5 (live)", "keyword", SUPPORT_MATCH),
             ("embedding >= EMBED_MIN", "embed", config.EMBED_MIN)]
    rules += [(f"{m} >= {t:.3f} (from {'val' if m.startswith('ft') else 'train'})", m, t) for m, t in th.items()]
    print("\nShare of dev pairs each rule accepts")
    print(f"  {'rule':34}" + "".join(f"{k:>12}" for k in KINDS))
    for name, m, t in rules:
        cells = "".join(f"{np.mean([v >= t for v in get(dv[k], m)]) if dv[k] else float('nan'):12.0%}" for k in KINDS)
        print(f"  {name:34}{cells}")

    # the NLI models' own three-way answer, no threshold
    heads = [("NLI, not trained on AVeriTeC", ("nliE", "nliN", "nliC"))]
    if has_ft:
        heads.append(("fine-tuned on AVeriTeC", ("ftE", "ftN", "ftC")))
    for name, cols in heads:
        print(f"\n{name}: highest of entailment / neutral / contradiction, share of dev pairs")
        print(f"  {'evidence':12} {'n':>4} {'entailment':>11} {'neutral':>8} {'contradiction':>14}")
        for k in KINDS:
            if dv[k]:
                picks = collections.Counter(max(range(3), key=lambda i: r[cols[i]]) for r in dv[k])
                print(f"  {k:12} {len(dv[k]):4} " + " ".join(
                    f"{picks[i] / len(dv[k]):{w}.0%}" for i, w in zip(range(3), (11, 8, 14))))

    # how far the fast measures together can tell the three kinds apart (fitted on train, scored on dev)
    if tr["supports"]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, confusion_matrix
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        three = ("supports", "refutes", "topic")
        sets = {"keyword + embedding": ["keyword", "embed"],
                "NLI only": ["nliE", "nliN", "nliC", "lineE", "lineC"],
                "all": ["keyword", "embed", "nliE", "nliN", "nliC", "lineE", "lineC"]}

        def xy(rows_by, cols):
            rows = [r for k in three for r in rows_by[k]]
            return np.array([[r[c] for c in cols] for r in rows]), np.array([r["kind"] for r in rows])

        print("\nSupports / refutes / topic, logistic regression fitted on train, balanced accuracy on dev "
              "(chance 33%)")
        for name, cols in sets.items():
            (xt, yt), (xd, yd) = xy(tr, cols), xy(dv, cols)
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
            pred = clf.fit(xt, yt).predict(xd)
            cm = confusion_matrix(yd, pred, labels=list(three))
            rows = "; ".join(f"{k}: " + "/".join(str(v) for v in cm[i]) for i, k in enumerate(three))
            print(f"  {name:22} {balanced_accuracy_score(yd, pred):5.0%}   predicted supports/refutes/topic -> {rows}")

    jr = [r for r in load(JUDGE) if r["status"] == "ok"]
    errors = len(load(JUDGE)) - len(jr)
    if jr:
        print(f"\nLive Qwen judge on {len(jr)} dev pairs ({errors} errors left out)")
        print(f"  {'evidence':10} {'n':>3} {'supported':>10} {'refuted':>8} {'mixed':>6} {'unverifiable':>13}")
        for k in ("supports", "refutes", "topic"):
            rows = [r for r in jr if r["kind"] == k]
            if rows:
                c = collections.Counter(r["verdict"] for r in rows)
                print(f"  {k:10} {len(rows):3} {c['supported']:10} {c['refuted']:8} {c['mixed']:6} {c['unverifiable']:13}")
        sd = {r["pid"]: r for r in sc["dev"]}
        print("\n  'supported' verdicts each check would let through (the rest become unverifiable)")
        checks = [("keyword guard (live)", "keyword", SUPPORT_MATCH)] + \
                 [(f"{m} >= {th[m]:.3f}", m, th[m]) for m in ("nliE", "lineE", "ftE") if m in th]
        for name, m, t in checks:
            cells = []
            for k in ("supports", "refutes", "topic"):
                said = [r for r in jr if r["kind"] == k and r["verdict"] == "supported"]
                cells.append(f"{k} {sum(sd[r['pid']][m] >= t for r in said)}/{len(said)}")
            print(f"    {name:24} " + "   ".join(cells))
        print("  'refuted' verdicts a contradiction check would let through (today nothing checks them)")
        for m in ("nliC", "lineC", "ftC"):
            if m not in th:
                continue
            cells = []
            for k in ("supports", "refutes", "topic"):
                said = [r for r in jr if r["kind"] == k and r["verdict"] == "refuted"]
                cells.append(f"{k} {sum(sd[r['pid']][m] >= th[m] for r in said)}/{len(said)}")
            print(f"    {m + ' >= ' + format(th[m], '.3f'):24} " + "   ".join(cells))


def final(args) -> None:
    """Final test of the stance models on the AVeriTeC dev pairs (held out: never used for training or for
    any choice). Labels: supports -> entailment, refutes -> contradiction, nei and topic -> neutral;
    conflicting pairs are shown apart. No thresholds: each model's highest output is its answer."""
    rows = {r["pid"]: r for r in load(SCORES["dev"])}
    for r in load(STANCE_DEV):
        rows[r["pid"]].update({k: v for k, v in r.items() if k.startswith("ft")})
    gold_of = {"supports": "E", "refutes": "C", "nei": "N", "topic": "N"}
    names = {"E": "supports (entailment)", "N": "neutral / topic only", "C": "refutes (contradiction)"}
    rng = random.Random(SEED)
    for title, cols in (("NLI before training", ("nliE", "nliN", "nliC")),
                        ("fine-tuned on AVeriTeC", ("ftE", "ftN", "ftC"))):
        pred = {pid: "ENC"[max(range(3), key=lambda i: r[cols[i]])] for pid, r in rows.items()}
        test = [pid for pid, r in rows.items() if r["kind"] in gold_of]
        gold = {pid: gold_of[rows[pid]["kind"]] for pid in test}

        def scores(ids):
            f1s, per = [], {}
            for c in "ENC":
                tp = sum(gold[i] == c and pred[i] == c for i in ids)
                fp = sum(gold[i] != c and pred[i] == c for i in ids)
                fn = sum(gold[i] == c and pred[i] != c for i in ids)
                p = tp / (tp + fp) if tp + fp else 0.0
                rc = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
                per[c] = (p, rc, f1, tp + fn)
                f1s.append(f1)
            acc = sum(gold[i] == pred[i] for i in ids) / len(ids)
            return acc, float(np.mean(f1s)), per

        acc, macro, per = scores(test)
        boots = sorted(scores([test[rng.randrange(len(test))] for _ in test])[1] for _ in range(1000))
        print(f"\n{title}: {len(test)} dev pairs, accuracy {acc:.1%}, macro F1 {macro:.1%} "
              f"(95% range {boots[25]:.1%} to {boots[974]:.1%})")
        print(f"  {'class':26} {'n':>4} {'precision':>10} {'recall':>8} {'F1':>6}")
        for c in "ENC":
            p, rc, f1, n = per[c]
            print(f"  {names[c]:26} {n:4} {p:10.1%} {rc:8.1%} {f1:6.1%}")
        print(f"  confusion (rows = evidence kind, columns = answer E / N / C):")
        for k in ("supports", "refutes", "nei", "topic", "conflicting"):
            ids = [pid for pid, r in rows.items() if r["kind"] == k]
            c = collections.Counter(pred[i] for i in ids)
            print(f"    {k:12} {len(ids):4}   " + " / ".join(f"{c[x]:3}" for x in "ENC"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["build", "score", "judge", "report", "final"])
    ap.add_argument("--n", type=int, default=40, help="judge: pairs per kind (supports, refutes, topic)")
    args = ap.parse_args()
    {"build": build, "score": score, "judge": judge, "report": report, "final": final}[args.step](args)


if __name__ == "__main__":
    main()
