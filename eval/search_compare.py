"""Does the AI-written search find better evidence than keyword search? The arms run on the same
claim minutes apart, so the news available is the same; the AI judge is not run.

  live      the claim as the old sentence splitter cut it (at "U.S.", "Oct." ...), keyword search
  keywords  the full claim (abbreviation fix), keyword search, keyword relevance and paragraphs
  ai-query  the full claim, query.rewrite (qwen2.5:7b) -> 3 queries, embedding relevance and paragraphs
"live" only differs from "keywords" for claims the old splitter cut; elsewhere it is a copy.

Every source each arm would show the AI (articles read + up to 8 headlines) is scored with
MiniCheck against the original claim. Good search: for real claims a source that states the
claim (MiniCheck >= 0.5) is found more often; for fake claims, not more often.

Usage (from the project root, Ollama running):
  ai-gateway\\.venv\\Scripts\\python eval\\search_compare.py            # tuning set, resumable
  ai-gateway\\.venv\\Scripts\\python eval\\search_compare.py --report   # summary only

GDELT is switched off for both arms: its rate limit would hit whichever arm runs second.
Uses the tuning set only; the independent holdout must not be used to choose settings.
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai-gateway"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import re  # noqa: E402

from app import claims as claims_mod, config, evidence, support  # noqa: E402
from app.claims import extract_claims, keywords, overlap  # noqa: E402
from app.query import is_english, rewrite  # noqa: E402
from collect import wait_for_internet  # noqa: E402
from support_features import shown_sources  # noqa: E402

HERE = Path(__file__).resolve().parent
REWRITES = HERE / "runs" / "rewrites.jsonl"
OUT = HERE / "runs" / "search-compare.jsonl"


def _no_gdelt(query):
    raise RuntimeError("GDELT switched off for this comparison")


def old_split_claim(text: str) -> str:
    """The main claim as the sentence splitter cut it before the abbreviation fix."""
    saved, claims_mod._ABBR = claims_mod._ABBR, re.compile(r"(?!x)x")
    try:
        return (extract_claims(text) or [text])[0]
    finally:
        claims_mod._ABBR = saved


def run_arm(arm: str, claim: str, exclude: dict, rw: dict | None) -> dict:
    """claim: the claim this arm searches with (cut, for "live"). Mirrors evidence.gather."""
    t = time.time()
    if arm == "ai-query" and rw:
        config.RELEVANCE = "embedding"
        search_claim = claim if is_english(claim) else rw["claim"]
        queries = rw["queries"]
    else:
        config.RELEVANCE = "keywords"
        search_claim, queries = claim, None
    news = evidence.search_news(search_claim, queries)
    urls, texts = set(exclude["urls"]), {evidence._norm(t) for t in exclude["texts"]}
    news["articles"] = [a for a in news.get("articles") or []
                        if a.get("url") not in urls and evidence._norm(a.get("title")) not in texts]
    passages = evidence.retrieve_passages(news["articles"], search_claim, exclude=exclude, queries=queries)
    config.RELEVANCE = "keywords"
    if queries:  # AI queries without the "fact check" suffix, then the live keyword query
        words = keywords(search_claim, 5)
        fc = fact_check_items([q for q in queries if "fact check" not in q.lower()][:2] + [" ".join(words)],
                              exclude, union=True)
    else:
        words = keywords(search_claim, 5)
        fc = fact_check_items([" ".join(words)] + ([" ".join(words[:3])] if len(words) > 3 else []),
                              exclude, union=False)
    return {"engine": news.get("engine"), "status": news.get("status"), "reason": news.get("reason", ""),
            "evidence": {"news": news, "passages": passages}, "factChecks": fc,
            "searchSeconds": round(time.time() - t, 1)}


def fact_check_items(queries: list[str], exclude: dict, union: bool) -> dict:
    """Raw Google Fact Check results, without the relevance filter, own source removed.
    union=False: stop at the first query that finds something (the live rule); True: all queries."""
    if not config.GOOGLE_FACTCHECK_KEY:
        return {"status": "disabled", "items": []}
    items, used, err = [], [], ""
    for q in queries:
        try:
            got = evidence._factcheck_query(q)
        except Exception as exc:
            err = str(exc)[:160]
            continue
        used.append(q)
        items += got
        if got and not union:
            break
    out, seen = [], set()
    urls = set(exclude["urls"])
    for it in items:
        rev = (it.get("claimReview") or [{}])[0]
        url = rev.get("url", "")
        if not it.get("text") or url in seen or url in urls:
            continue
        seen.add(url)
        rating = rev.get("textualRating", "")
        stance = ("refutes" if evidence.FALSE_RATINGS.search(rating)
                  else "supports" if evidence.TRUE_RATINGS.search(rating) else "mixed")
        out.append({"text": it["text"], "rating": rating, "stance": stance, "url": url,
                    "publisher": (rev.get("publisher") or {}).get("name", "")})
    return {"status": "error" if err and not used else "ok", "queries": used, "error": err, "items": out}


LOCAL_OLLAMA = config.OLLAMA_URL
# Optional remote Ollama (e.g. a GPU node behind an SSH tunnel). Each claim is scored there when it
# answers, otherwise on the local Ollama; a claim the remote drops mid-way is rescored locally.
REMOTE_OLLAMA = os.getenv("MINICHECK_REMOTE", "").rstrip("/")


class _RemoteDown(Exception):
    pass


def _pick_backend() -> str:
    if REMOTE_OLLAMA:
        try:
            httpx.get(f"{REMOTE_OLLAMA}/api/tags", timeout=3).raise_for_status()
            config.OLLAMA_URL = REMOTE_OLLAMA
            return "remote"
        except Exception:
            pass
    config.OLLAMA_URL = LOCAL_OLLAMA
    return "local"


def _minicheck(claim: str, document: str) -> float:
    """MiniCheck with retries. A missing score would silently count as "no stating source", so if
    Ollama stays unreachable (e.g. an SSH tunnel to a remote GPU dropped) stop before saving the claim;
    re-running resumes from the last saved claim."""
    if REMOTE_OLLAMA and config.OLLAMA_URL == REMOTE_OLLAMA:
        for _ in range(2):
            score = support.minicheck(claim, document)
            if score is not None:
                return score
            time.sleep(5)
        raise _RemoteDown
    for wait in (10, 30, 60, 120):
        score = support.minicheck(claim, document)
        if score is not None:
            return score
        print(f"  MiniCheck unavailable at {config.OLLAMA_URL}; retrying in {wait}s", flush=True)
        time.sleep(wait)
    score = support.minicheck(claim, document)
    if score is None:
        sys.exit(f"MiniCheck still unavailable at {config.OLLAMA_URL}; stopped without saving this claim")
    return score


def score_fact_checks(claim: str, fc: dict, cache: dict, limit: int = 6) -> None:
    """Is each fact-checked claim the same claim? keyword (live rule: >=0.6 either direction),
    embedding, and MiniCheck both ways (a fact-checked claim may be a shorter part of this one)."""
    items = fc["items"]
    sims = support.similarity(claim, [it["text"] for it in items])
    for it, sim in zip(items, sims):
        it["keyword"] = round(max(overlap(claim, it["text"], top=6), overlap(it["text"], claim, top=6)), 3)
        it["embed"] = round(sim, 4)
        it["minicheck"] = None
    for it in sorted(items, key=lambda it: -it["embed"])[:limit]:
        if it["embed"] < 0.55 and it["keyword"] < 0.4:
            continue
        for key in ((it["text"], claim), (claim, it["text"])):
            if key not in cache:
                cache[key] = _minicheck(key[1], key[0])  # minicheck(claim, document)
        a, b = cache[(it["text"], claim)], cache[(claim, it["text"])]
        it["minicheck"] = None if a is None and b is None else round(max(a or 0, b or 0), 4)


def score_sources(claim: str, ev: dict, cache: dict) -> list[dict]:
    """MiniCheck (cached: both arms often find the same source), keyword and embedding scores."""
    sources = shown_sources(ev)
    sims = support.similarity(claim, [s["text"] for s in sources])
    for s, sim in zip(sources, sims):
        key = s["text"]
        if key not in cache:
            cache[key] = _minicheck(claim, s["text"])
        s["minicheck"] = None if cache[key] is None else round(cache[key], 4)
        s["keyword"] = round(overlap(claim, s["text"], top=6, stem=True), 3)
        s["embed"] = round(sim, 4)
        del s["text"]
    return sources


def collect(rows):
    # Phase 1: all query rewrites first, so Ollama does not swap qwen and MiniCheck for every claim.
    rewrites = {}
    if REWRITES.exists():
        rewrites = {r["id"]: r for r in map(json.loads, REWRITES.read_text(encoding="utf-8").splitlines()) if r}
    todo = [r for r in rows if r["id"] not in rewrites]
    print(f"rewrites: {len(rewrites)} done, {len(todo)} to go", flush=True)
    for n, row in enumerate(todo, 1):
        claim = (extract_claims(row["claim"]) or [row["claim"]])[0]
        t = time.time()
        rec = {"id": row["id"], "claim": claim, "oldClaim": old_split_claim(row["claim"]),
               "rewrite": rewrite(claim), "seconds": round(time.time() - t, 1)}
        with REWRITES.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        rewrites[row["id"]] = rec
        print(f"[rewrite {n}/{len(todo)}] {rec['seconds']}s {(rec['rewrite'] or {}).get('queries')}", flush=True)

    # Phase 2: both searches back to back (alternating which goes first), then MiniCheck.
    evidence._gdelt = _no_gdelt
    done = set()
    if OUT.exists():
        done = {json.loads(x)["id"] for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip()}
    todo = [r for r in rows if r["id"] not in done]
    print(f"searches: {len(done)} done, {len(todo)} to go", flush=True)
    for n, row in enumerate(todo, 1):
        wait_for_internet()
        t = time.time()
        claim, old = rewrites[row["id"]]["claim"], rewrites[row["id"]]["oldClaim"]
        rw = rewrites[row["id"]]["rewrite"]
        exclude = {"urls": [row["sourceUrl"]], "texts": [row["sourceTitle"]]}
        order = ["keywords", "ai-query"] if row["id"] % 2 == 0 else ["ai-query", "keywords"]
        if old != claim:
            order.insert(row["id"] % 3, "live")
        found = {arm: run_arm(arm, old if arm == "live" else claim, exclude, rw) for arm in order}
        while True:
            backend = _pick_backend()
            arms = copy.deepcopy(found)  # scoring consumes the evidence; keep it for a rescore
            try:
                cache = {}
                for arm in arms.values():  # every arm is judged against the full claim
                    arm["sources"] = score_sources(claim, arm.pop("evidence"), cache)
                    score_fact_checks(claim, arm["factChecks"], cache)
                break
            except _RemoteDown:
                print(f"  remote MiniCheck dropped mid-claim; rescoring claim {row['id']} locally", flush=True)
                config.OLLAMA_URL = LOCAL_OLLAMA
        if "live" not in arms:
            arms["live"] = {**arms["keywords"], "copyOf": "keywords"}
        rec = {"id": row["id"], "label": row["label"], "claim": claim, "rewrite": rw, "arms": arms,
               "seconds": round(time.time() - t, 1), "scoredOn": backend}
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        summary = " | ".join(f"{a}: {len(v['sources'])} src, best {best(v):.2f}" for a, v in arms.items())
        print(f"[{n}/{len(todo)}] {row['label']:4} {summary} | {rec['seconds']}s on {backend}", flush=True)
    print("done", flush=True)


def best(arm: dict, kinds=("article", "headline")) -> float:
    return max((s["minicheck"] or 0.0 for s in arm["sources"] if s["kind"] in kinds), default=0.0)


def report():
    recs = [json.loads(x) for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip()]
    print(f"{len(recs)} claims ({sum(r['label'] == 'real' for r in recs)} real)\n")
    print("Share of claims where a source states the claim (MiniCheck >= 0.5)")
    print("  arm         real (higher=better)   fake (lower=better)   avg sources   avg articles read")
    for arm in ("live", "keywords", "ai-query"):
        real = [r for r in recs if r["label"] == "real"]
        fake = [r for r in recs if r["label"] == "fake"]
        hit = lambda rs: sum(best(r["arms"][arm]) >= 0.5 for r in rs)
        n_src = sum(len(r["arms"][arm]["sources"]) for r in recs) / len(recs)
        n_art = sum(sum(s["kind"] == "article" for s in r["arms"][arm]["sources"]) for r in recs) / len(recs)
        print(f"  {arm:10}  {hit(real):3}/{len(real):<3} {hit(real) / max(1, len(real)):5.0%}"
              f"            {hit(fake):3}/{len(fake):<3} {hit(fake) / max(1, len(fake)):5.0%}"
              f"          {n_src:5.1f}         {n_art:5.1f}")
    print("\nPaired: claims where one arm found a stating source and the other did not")
    cut = sum(1 for r in recs if "copyOf" not in r["arms"]["live"])
    for a, b, note in (("keywords", "live", f"abbreviation fix; differs on {cut} cut claims"),
                       ("ai-query", "keywords", "AI queries + embedding"),
                       ("ai-query", "live", "everything new against live")):
        for label in ("real", "fake"):
            rs = [r for r in recs if r["label"] == label]
            diffs = [(best(r["arms"][a]) >= 0.5) - (best(r["arms"][b]) >= 0.5) for r in rs]
            rng = random.Random(0)
            boots = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) for _ in range(4000)) if diffs else [0]
            print(f"  {a} vs {b:8} {label}: only {a} {sum(d > 0 for d in diffs)}, only {b} {sum(d < 0 for d in diffs)}"
                  f" (net {sum(diffs):+}, 95% range {boots[100]:+} to {boots[3899]:+})  [{note}]")
    print("\nFact-checks: claims with a matching fact-check rated false (fake: higher=better; real: lower=better)")
    rules = {"keyword>=0.6 (live)": lambda it: it["keyword"] >= 0.6,
             "embed>=0.8": lambda it: it["embed"] >= 0.8,
             "MiniCheck>=0.5": lambda it: (it["minicheck"] or 0) >= 0.5}
    for arm in ("live", "keywords", "ai-query"):
        for name, rule in rules.items():
            hit = lambda r, stance: any(rule(it) and it["stance"] == stance
                                        for it in r["arms"][arm]["factChecks"]["items"])
            fake = [r for r in recs if r["label"] == "fake"]
            real = [r for r in recs if r["label"] == "real"]
            print(f"  {arm:9} {name:20} fake refuted {sum(hit(r, 'refutes') for r in fake):3}/{len(fake):<3} "
                  f"real refuted {sum(hit(r, 'refutes') for r in real):3}/{len(real):<3} "
                  f"real confirmed {sum(hit(r, 'supports') for r in real):3}")

    fails = sum(1 for r in recs if not r["rewrite"])
    print(f"\nquery rewrite failed on {fails} claims (those fell back to keyword search)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", type=Path, default=HERE / "claims.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if "fresh" in args.claims.name:
        sys.exit("refusing to use the independent holdout: it must not be used to choose settings")
    if args.report:
        return report()
    rows = [json.loads(x) for x in args.claims.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit:
        rows = rows[: args.limit]
    collect(rows)
    report()


if __name__ == "__main__":
    main()
