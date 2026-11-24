"""Score saved evidence with an NLI model: does each headline, passage or fact-checked claim
entail or contradict the claim being checked?

Usage (from the project root):
  ai-gateway\\.venv\\Scripts\\python eval\\nli_features.py --runs eval\\runs\\7b-v3.jsonl --out eval\\runs\\nli-dev.jsonl

Uses no network after the model download and can be stopped and resumed. Only use the tuning set
(eval/claims.jsonl) for choosing how to apply these scores.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


class NLI:
    def __init__(self, name: str = MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.entail, self.neutral, self.contradict = labels["entailment"], labels["neutral"], labels["contradiction"]

    @torch.no_grad()
    def score(self, premises: list[str], hypothesis: str, batch: int = 8) -> list[dict]:
        out = []
        for i in range(0, len(premises), batch):
            part = premises[i : i + batch]
            enc = self.tokenizer(part, [hypothesis] * len(part), truncation=True, max_length=256,
                                 padding=True, return_tensors="pt")
            probs = self.model(**enc).logits.softmax(-1)
            for p in probs:
                out.append({"e": round(float(p[self.entail]), 4), "n": round(float(p[self.neutral]), 4),
                            "c": round(float(p[self.contradict]), 4)})
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2, help="keep low when other CPU work is running")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    rows = [json.loads(x) for x in args.runs.read_text(encoding="utf-8").splitlines() if x.strip()]
    done = set()
    if args.out.exists():
        done = {json.loads(x)["id"] for x in args.out.read_text(encoding="utf-8").splitlines() if x.strip()}
    todo = [r for r in rows if r["id"] not in done]
    t0 = time.time()
    nli = NLI()
    print(f"model loaded in {time.time() - t0:.0f}s; {len(done)} done, {len(todo)} to go", flush=True)

    for n, r in enumerate(todo, 1):
        t = time.time()
        ev = r["evidence"]
        claim = r["claim"]
        rec = {"id": r["id"], "label": r["label"], "origin": r["origin"], "kind": r["kind"], "news": [], "passages": [], "reviews": []}
        articles = (ev.get("news") or {}).get("articles") or []
        for a, s in zip(articles, nli.score([a["title"] for a in articles], claim)):
            rec["news"].append({**s, "cred": a["credibility"], "rel": a["relevance"], "stance": a["stance"]})
        passages = ev.get("passages") or []
        for p, s in zip(passages, nli.score([p["passage"] for p in passages], claim)):
            rec["passages"].append({**s, "cred": p["credibility"]})
        reviews = (ev.get("factChecks") or {}).get("reviews") or []
        for x, s in zip(reviews, nli.score([x["claim"] for x in reviews], claim)):
            rec["reviews"].append({**s, "stance": x["stance"], "rel": x["relevance"]})
        rec["seconds"] = round(time.time() - t, 1)
        with args.out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{n}/{len(todo)}] {r['label']:4} pairs {len(articles) + len(passages) + len(reviews)} {rec['seconds']}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
