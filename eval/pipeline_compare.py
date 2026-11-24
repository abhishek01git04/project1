"""Do the new search features improve the verdicts? The full text pipeline (detector, evidence, local
AI judge with the keyword evidence guard, fusion) runs on each tuning claim twice, minutes apart:

  current  all flags off: what the gateway does after a restart (abbreviation fix, keyword search,
           keyword relevance, keyword fact-check matching)
  new      QUERY_REWRITE=on, RELEVANCE=embedding, FACTCHECK_MATCH=embedding (AI-written queries for
           news, articles and fact-checks; matching by meaning)

GDELT is switched off for both (its rate limit would hit whichever arm runs second), so news comes
from Google News RSS, as it does in the gateway whenever GDELT is rate-limited.

Usage (from the project root, Ollama running):
  ai-gateway\\.venv\\Scripts\\python eval\\pipeline_compare.py              # tuning set, resumable
  ai-gateway\\.venv\\Scripts\\python eval\\pipeline_compare.py --report     # summary only
Set OLLAMA_REMOTE=http://127.0.0.1:11934 to run the models on a remote GPU (an SSH tunnel) whenever
it answers; a claim the remote drops mid-way is redone on the local Ollama.

Writes eval/runs/pipeline-current.jsonl and pipeline-new.jsonl in collect.py's format, so
evaluate.py can read them too. Uses the tuning set only; the holdout must not choose settings.
"""

import argparse
import collections
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai-gateway"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402

from app import config, detectors, evidence, query, reasoner  # noqa: E402
from app.claims import extract_claims  # noqa: E402
from collect import wait_for_internet  # noqa: E402

sys.argv, _argv = sys.argv[:1], sys.argv  # evaluate.py parses its own arguments on import
import evaluate as E  # noqa: E402

sys.argv = _argv

HERE = Path(__file__).resolve().parent
ARMS = {
    "current": {"QUERY_REWRITE": False, "RELEVANCE": "keywords", "FACTCHECK_MATCH": "keywords"},
    "new": {"QUERY_REWRITE": True, "RELEVANCE": "embedding", "FACTCHECK_MATCH": "embedding"},
}
OUT = {arm: HERE / "runs" / f"pipeline-{arm}.jsonl" for arm in ARMS}
LOCAL = config.OLLAMA_URL
REMOTE = os.getenv("OLLAMA_REMOTE", "").rstrip("/")
WANT = {"fake": "Fake", "real": "Real"}
PTS = {"right": 1, "uncertain": 0, "wrong": -1}


def _no_gdelt(q):
    raise RuntimeError("GDELT switched off for this comparison")


def reachable(url: str) -> bool:
    try:
        httpx.get(f"{url}/api/tags", timeout=3).raise_for_status()
        return True
    except Exception:
        return False


def use(url: str) -> None:
    config.OLLAMA_URL = url
    reasoner._ollama_check = (0.0, False)  # force a fresh "is Ollama running" check
    query.rewrite.cache_clear()  # a rewrite that failed on a dropped tunnel must not be reused


def run_arm(arm: str, row: dict, det) -> dict:
    for key, value in ARMS[arm].items():
        setattr(config, key, value)
    t = time.time()
    text = row["claim"]
    ev = evidence.gather(extract_claims(text), text=text,
                         exclude={"urls": [row["sourceUrl"]], "texts": [row["sourceTitle"]]})
    return {
        "id": row["id"], "label": row["label"], "origin": row["origin"], "kind": row.get("kind", "statement"),
        "claim": text,
        "detector": {"prob": det.fake_probability, "modelId": det.model_id,
                     "isModel": not det.model_id.startswith("heuristic"), "signals": det.signals},
        "evidence": ev, "seconds": round(time.time() - t, 1),
    }


def llm_ok(rec: dict) -> bool:
    return ((rec["evidence"].get("llm") or {}).get("status")) == "ok"


def outcome(rec: dict) -> str:
    p = E.predict(rec, "full", None)
    return "uncertain" if p == "Uncertain" else ("right" if p == WANT[rec["label"]] else "wrong")


def keep_awake() -> None:
    """Stop Windows from sleeping or entering Modern Standby (which cuts the network) while this
    process runs; released automatically when it exits. The lid must still stay open."""
    if os.name == "nt":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x1, 0x2
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)


def collect(rows: list[dict]) -> None:
    keep_awake()
    evidence._gdelt = _no_gdelt
    done = set.intersection(*[
        {json.loads(x)["id"] for x in OUT[arm].read_text(encoding="utf-8").splitlines() if x.strip()}
        if OUT[arm].exists() else set() for arm in ARMS])
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go; remote: {REMOTE or 'none'}", flush=True)
    for n, row in enumerate(todo, 1):
        wait_for_internet()
        t = time.time()
        det = detectors.analyze_text(row["claim"])
        order = list(ARMS) if row["id"] % 2 == 0 else list(ARMS)[::-1]
        attempts = 0
        while True:
            attempts += 1
            backend = "remote" if REMOTE and reachable(REMOTE) else "local"
            use(REMOTE if backend == "remote" else LOCAL)
            recs = {arm: run_arm(arm, row, det) for arm in order}
            failed = not all(llm_ok(r) for r in recs.values())
            if backend == "remote" and (failed or not reachable(REMOTE)):
                print(f"  remote dropped during claim {row['id']}; redoing it", flush=True)
                continue
            if failed and attempts < 3:  # e.g. the PC slept or Ollama restarted mid-claim
                print(f"  AI judge failed on claim {row['id']} ({backend}); retrying in 30 s", flush=True)
                time.sleep(30)
                wait_for_internet()
                continue
            break
        for arm in ARMS:  # write both arms together, so a stop never leaves a claim half done
            recs[arm]["backend"] = backend
        for arm in ARMS:
            with OUT[arm].open("a", encoding="utf-8") as f:
                f.write(json.dumps(recs[arm], ensure_ascii=False) + "\n")
        parts = []
        for arm in ARMS:
            ev = recs[arm]["evidence"]
            parts.append(f"{arm}: {E.predict(recs[arm], 'full', None)} "
                         f"(ai {(ev.get('llm') or {}).get('verdict') or (ev.get('llm') or {}).get('status')}, "
                         f"fc {len((ev.get('factChecks') or {}).get('reviews') or [])})")
        print(f"[{n}/{len(todo)}] {row['label']:4} " + " | ".join(parts)
              + f" | {round(time.time() - t)}s on {backend}", flush=True)
    print("done", flush=True)


def report() -> None:
    recs = {arm: {r["id"]: r for r in E.load(OUT[arm])} for arm in ARMS}
    ids = sorted(set(recs["current"]) & set(recs["new"]))
    if not ids:
        print("nothing collected yet")
        return
    res = {arm: {i: outcome(recs[arm][i]) for i in ids} for arm in ARMS}
    labels = {i: recs["current"][i]["label"] for i in ids}
    print(f"{len(ids)} claims ({sum(v == 'real' for v in labels.values())} real), both setups run minutes apart\n")
    print("Setup      right wrong uncertain  accuracy  right when decided  real->Fake  fake->Real")
    for arm in ARMS:
        c = collections.Counter(res[arm].values())
        decided = c["right"] + c["wrong"]
        ff = sum(labels[i] == "real" and res[arm][i] == "wrong" for i in ids)
        fr = sum(labels[i] == "fake" and res[arm][i] == "wrong" for i in ids)
        print(f"  {arm:8} {c['right']:5} {c['wrong']:5} {c['uncertain']:9}  {c['right'] / len(ids):8.0%}  "
              f"{(c['right'] / decided if decided else 0):18.0%}  {ff:10}  {fr:10}")
    diff = [PTS[res["new"][i]] - PTS[res["current"][i]] for i in ids]
    rng = random.Random(1)
    boots = sorted(sum(diff[rng.randrange(len(ids))] for _ in ids) for _ in range(4000))
    lo, hi = boots[int(0.025 * 4000)], boots[int(0.975 * 4000) - 1]
    print(f"\nnew vs current (right=+1, wrong=-1): net {sum(diff):+} (95% range {lo:+} to {hi:+}), "
          f"better on {sum(d > 0 for d in diff)}, worse on {sum(d < 0 for d in diff)}")
    for i in ids:
        if res["new"][i] != res["current"][i]:
            print(f"  {labels[i]:4} {res['current'][i]:9} -> {res['new'][i]:9}  {recs['current'][i]['claim'][:80]}")
    backends = collections.Counter(recs["new"][i].get("backend", "?") for i in ids)
    print(f"\nmodels ran on: {dict(backends)}")


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
