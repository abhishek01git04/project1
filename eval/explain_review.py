"""Explanation quality of the AI's "refuted" verdicts: does the evidence it relies on really contradict the
claim, and does the trained stance guard (training/train_stance.py) remove the false contradictions?

This measures the explanation, not the final verdict (on the tuning set the guard changes no final verdict;
see eval/stance_tuning.py harm). The labels must come from a person reading each source:

  valid       the source contradicts the claim
  topic       same topic, no contradiction
  supports    the source supports the claim
  unclear     not enough to tell, or conflicting

Steps (from the project root, after eval/stance_tuning.py score):
  ai-gateway\\.venv\\Scripts\\python eval\\explain_review.py make    # writes eval/review/refuted-sources.csv
  ... fill the "human_label" column in Excel (one of the four words above) ...
  ai-gateway\\.venv\\Scripts\\python eval\\explain_review.py score

Rows: every source the AI cited as refuting; when a "refuted" verdict cites no source, the shown source the
trained model rates most contradicting (marked cited=no). The guard never turns a rejected contradiction into
"supported": a verdict whose contradiction is rejected becomes "unverifiable".
The thresholds 0.5 and 0.7 were seen on these claims (stance_tuning.py harm), so a final explanation test
needs new claims.
"""

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "eval"
sys.path.insert(0, str(ROOT / "ai-gateway"))
sys.path.insert(0, str(HERE))
from support_features import shown_sources  # noqa: E402

RUN = HERE / "runs" / "7b-v3-articles.jsonl"
FEATURES = HERE / "runs" / "stance-features.jsonl"
SHEET = HERE / "review" / "refuted-sources.csv"
LABELS = ("valid", "topic", "supports", "unclear")
THRESHOLDS = (0.5, 0.7)
FIELDS = ["row", "claim_id", "true_label", "claim", "ai_summary", "cited", "source_title", "source_text", "ftC",
          *[f"guard_keeps_at_{t}" for t in THRESHOLDS], "human_label", "notes"]


def load(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def make(args) -> None:
    if SHEET.exists() and not args.force:
        sys.exit(f"{SHEET} exists (it may hold your labels); use --force to overwrite")
    feats = {f["id"]: f for f in load(FEATURES)}
    out = []
    for r in load(RUN):
        llm = r["evidence"].get("llm") or {}
        if llm.get("verdict") != "refuted" or r["id"] not in feats:
            continue
        shown = shown_sources(r["evidence"])
        scores = {s["url"]: s for s in feats[r["id"]]["sources"]}
        cited = [s["url"] for s in llm.get("sources") or [] if s.get("stance") == "refutes"]
        picks = [(u, "yes") for u in dict.fromkeys(cited) if u in scores]
        if not picks and shown:
            top = max(shown, key=lambda s: scores[s["url"]].get("ftC", 0))
            picks = [(top["url"], "no")]
        for url, was_cited in picks:
            src = next(s for s in shown if s["url"] == url)
            ftc = scores[url]["ftC"]
            out.append({"claim_id": r["id"], "true_label": r["label"], "claim": r["claim"],
                        "ai_summary": (llm.get("summary") or "")[:400], "cited": was_cited,
                        "source_title": src["title"], "source_text": src["text"][:900], "ftC": round(ftc, 3),
                        **{f"guard_keeps_at_{t}": "yes" if ftc >= t else "no" for t in THRESHOLDS},
                        "human_label": "", "notes": ""})
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    with SHEET.open("w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig: Excel shows Hindi and quotes
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for n, row in enumerate(out, 1):
            w.writerow({"row": n, **row})
    c = collections.Counter(r["cited"] for r in out)
    print(f"{len(out)} rows for {len({r['claim_id'] for r in out})} 'refuted' verdicts "
          f"({c['yes']} cited sources, {c['no']} verdicts that cited none) -> {SHEET}")
    print(f"Fill human_label with one of: {', '.join(LABELS)}")


def score(args) -> None:
    with SHEET.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    bad = [r["row"] for r in rows if r["human_label"].strip().lower() not in LABELS]
    if bad:
        print(f"{len(bad)} rows have no valid human_label yet (rows {', '.join(bad[:20])}"
              f"{'...' if len(bad) > 20 else ''}); scoring the labelled ones only")
    rows = [r for r in rows if r["human_label"].strip().lower() in LABELS]
    if not rows:
        return
    lab = collections.Counter(r["human_label"].strip().lower() for r in rows)
    print(f"{len(rows)} labelled sources: " + ", ".join(f"{k} {lab[k]}" for k in LABELS))
    for cited in ("yes", "no"):
        part = [r for r in rows if r["cited"] == cited]
        if part:
            c = collections.Counter(r["human_label"].strip().lower() for r in part)
            print(f"  cited={cited}: " + ", ".join(f"{k} {c[k]}" for k in LABELS))
    print(f"\n{'guard':14} {'valid kept':>11} {'false removed':>14} {'false left':>11}   (false = topic/supports/unclear)")
    for t in THRESHOLDS:
        keep = [r[f"guard_keeps_at_{t}"] == "yes" for r in rows]
        valid = [k for r, k in zip(rows, keep) if r["human_label"].strip().lower() == "valid"]
        false = [k for r, k in zip(rows, keep) if r["human_label"].strip().lower() != "valid"]
        print(f"  ftC >= {t:<6} {sum(valid):>6}/{len(valid):<4} {sum(not k for k in false):>9}/{len(false):<4} "
              f"{sum(false):>11}")
    by_label = collections.defaultdict(list)
    for r in rows:
        by_label[r["true_label"]].append(r["human_label"].strip().lower())
    print("\nHuman label of the sources, by the claim's true label:")
    for k, v in by_label.items():
        c = collections.Counter(v)
        print(f"  {k} claims: " + ", ".join(f"{x} {c[x]}" for x in LABELS))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["make", "score"])
    ap.add_argument("--force", action="store_true", help="make: overwrite an existing sheet")
    args = ap.parse_args()
    {"make": make, "score": score}[args.step](args)


if __name__ == "__main__":
    main()
