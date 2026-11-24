"""Compare "AI reads full articles" with headlines only, on the evidence saved for the same claims.

Keeps each claim's saved news, fact-checks and detector score, downloads articles through Bing News RSS,
and runs the AI again. Each claim's own source is excluded so it cannot look up the answer.

Usage (from the project root):
  ai-gateway\\.venv\\Scripts\\python eval\\rerun_articles.py --base eval\\runs\\7b-v3.jsonl ^
      --claims eval\\claims.jsonl --name 7b-v3-articles

Writes eval/runs/<name>.jsonl and can be stopped and resumed. Score it with eval/evaluate.py --runs.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai-gateway"))

from app import config, evidence, reasoner  # noqa: E402
from app.urlfetch import domain_of  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--claims", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=600)
    args = ap.parse_args()

    config.ARTICLE_SEARCH = True
    config.OLLAMA_TIMEOUT = args.timeout

    meta = {r["id"]: r for r in map(json.loads, filter(str.strip, args.claims.read_text(encoding="utf-8").splitlines()))}
    rows = [json.loads(x) for x in args.base.read_text(encoding="utf-8").splitlines() if x.strip()]
    out = HERE / "runs" / f"{args.name}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(x)["id"] for x in out.read_text(encoding="utf-8").splitlines() if x.strip()}
    todo = [r for r in rows if r["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{args.name}: {len(done)} done, {len(todo)} to go | model {config.OLLAMA_MODEL}, prompt {config.LLM_PROMPT_VERSION}", flush=True)

    for n, row in enumerate(todo, 1):
        t = time.time()
        rec = copy.deepcopy(row)
        ev = rec["evidence"]
        src = meta[row["id"]]
        exclude = {"urls": [src["sourceUrl"]], "texts": [src["sourceTitle"]]}
        if src["origin"] == "fact-check":  # the publisher's other pages may carry the same verdict
            exclude["domains"] = [domain_of(src["sourceUrl"])]
        claims = ev.get("claims") or [row["claim"]]
        articles = (ev.get("news") or {}).get("articles") or []
        ev["passages"] = evidence.retrieve_passages(articles, claims[0], exclude=exclude)
        fetch_s = time.time() - t
        ev["llm"] = reasoner._assess_ollama(claims, row["claim"], "", ev["passages"], articles)
        rec["llmSeconds"] = round(time.time() - t - fetch_s, 1)
        rec["fetchSeconds"] = round(fetch_s, 1)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{n}/{len(todo)}] {row['label']:4} read {len(ev['passages'])} articles ({fetch_s:.0f}s) | "
              f"ai {ev['llm'].get('verdict') or ev['llm'].get('status')} ({rec['llmSeconds']}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
