"""Score every source the AI saw, for each claim of a saved run, with three "does this source state
the claim?" measures: keyword overlap (the live guard), embedding similarity and MiniCheck.

Usage (from the project root, Ollama running with bespoke-minicheck pulled):
  ai-gateway\\.venv\\Scripts\\python eval\\support_features.py --base eval\\runs\\7b-v3-articles.jsonl

Writes eval/runs/<name>.jsonl (default support-features) and can be stopped and resumed.
Analyse it with eval/support_eval.py. Use tuning-set runs only: never the independent holdout.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai-gateway"))

from app import support  # noqa: E402
from app.claims import overlap  # noqa: E402

HERE = Path(__file__).resolve().parent


def shown_sources(ev: dict) -> list[dict]:
    """The articles and headlines the AI prompt contained (same de-duplication as reasoner.py)."""
    passages = ev.get("passages") or []
    headlines = (ev.get("news") or {}).get("articles") or []
    read = {p["url"] for p in passages}
    read_titles = {" ".join(p["title"].lower().split()) for p in passages}
    heads = [h for h in headlines
             if h["url"] not in read and " ".join(h["title"].lower().split()) not in read_titles][:8]
    return ([{"kind": "article", "url": p["url"], "title": p["title"], "domain": p.get("domain", ""),
              "credibility": p.get("credibility", 0.5), "text": p["title"] + "\n" + p["passage"]} for p in passages]
            + [{"kind": "headline", "url": h["url"], "title": h["title"], "domain": h.get("domain", ""),
                "credibility": h.get("credibility", 0.5), "text": h["title"]} for h in heads])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="saved run (tuning set) whose evidence is scored")
    ap.add_argument("--name", default="support-features")
    args = ap.parse_args()
    if "fresh" in args.base.name:
        sys.exit("refusing to score the independent holdout: it must not be used to choose settings")

    out = HERE / "runs" / f"{args.name}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(x)["id"] for x in out.read_text(encoding="utf-8").splitlines() if x.strip()}
    rows = [json.loads(x) for x in args.base.read_text(encoding="utf-8").splitlines() if x.strip()]
    todo = [r for r in rows if r["id"] not in done]
    print(f"{args.name}: {len(done)} done, {len(todo)} to go", flush=True)

    for n, row in enumerate(todo, 1):
        t = time.time()
        ev = row["evidence"]
        claim = (ev.get("claims") or [row["claim"]])[0]
        sources = shown_sources(ev)
        sims = support.similarity(claim, [s["text"] for s in sources])
        for s, sim in zip(sources, sims):
            s["keyword"] = round(overlap(claim, s["text"], top=6, stem=True), 3)
            s["embed"] = round(sim, 4)
            p = support.minicheck(claim, s["text"])
            s["minicheck"] = None if p is None else round(p, 4)
            del s["text"]
        rec = {"id": row["id"], "label": row["label"], "claim": claim, "sources": sources,
               "seconds": round(time.time() - t, 1)}
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        best = max((s["minicheck"] or 0 for s in sources), default=0)
        print(f"[{n}/{len(todo)}] {row['label']:4} sources {len(sources):2} best MiniCheck {best:.2f} "
              f"{rec['seconds']}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
