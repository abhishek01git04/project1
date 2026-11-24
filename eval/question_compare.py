"""Does AVeriTeC-style question search (QUESTION_SEARCH, app/questions.py) give better verdicts?
The full text pipeline (detector, evidence, local AI judge with the keyword evidence guard, fusion) runs
on each claim twice, minutes apart:

  current    the live setup (one keyword search for the claim)
  questions  QUESTION_SEARCH=on: the local model writes 3-5 questions, each is searched (GDELT first),
             and the judge sees the evidence grouped by question

Claim sets: the tuning set (eval/claims.jsonl, each claim's own source excluded) and the hand-written
short claims (eval/claims_short.jsonl: 15 true and 15 false one-line facts, e.g. "narendra modi is the
pm of india", where keyword matching is weakest). The independent holdout is refused.

Usage (from the project root, Ollama running):
  ai-gateway\\.venv\\Scripts\\python eval\\question_compare.py            # both sets, resumable
  ai-gateway\\.venv\\Scripts\\python eval\\question_compare.py --report
Set OLLAMA_REMOTE=http://127.0.0.1:11934 to use the HPC GPU through the SSH tunnel whenever it answers.
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
HERE = ROOT / "eval"
sys.path.insert(0, str(ROOT / "ai-gateway"))
sys.path.insert(0, str(HERE))

import httpx  # noqa: E402

from app import config, detectors, evidence, query, questions, reasoner  # noqa: E402
from app.claims import extract_claims  # noqa: E402
from collect import wait_for_internet  # noqa: E402

sys.argv, _argv = sys.argv[:1], sys.argv
import evaluate as E  # noqa: E402

sys.argv = _argv

ARMS = {"current": {"QUESTION_SEARCH": False}, "questions": {"QUESTION_SEARCH": True}}
SETS = {"short": HERE / "claims_short.jsonl", "tuning": HERE / "claims.jsonl"}
OUT = {arm: HERE / "runs" / f"questions-{arm}.jsonl" for arm in ARMS}
LOCAL = config.OLLAMA_URL
REMOTE = os.getenv("OLLAMA_REMOTE", "").rstrip("/")
WANT = {"fake": "Fake", "real": "Real"}
PTS = {"right": 1, "uncertain": 0, "wrong": -1}


def reachable(url: str) -> bool:
    try:
        httpx.get(f"{url}/api/tags", timeout=3).raise_for_status()
        return True
    except Exception:
        return False


def use(url: str) -> None:
    config.OLLAMA_URL = url
    reasoner._ollama_check = (0.0, False)
    query.rewrite.cache_clear()
    questions.generate.cache_clear()  # a generation that failed on a dropped tunnel must not be reused


def keep_awake() -> None:
    if os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x1 | 0x2)


def run_arm(arm: str, row: dict, det) -> dict:
    for key, value in ARMS[arm].items():
        setattr(config, key, value)
    t = time.time()
    text = row["claim"]
    exclude = {"urls": [row["sourceUrl"]], "texts": [row["sourceTitle"]]} if row.get("sourceUrl") else None
    ev = evidence.gather(extract_claims(text), text=text, exclude=exclude)
    ev.pop("questionEvidence", None)  # passages are kept in ev["passages"]; the groups are summarised in ev["questions"]
    return {"id": row["id"], "set": row["set"], "label": row["label"], "origin": row["origin"],
            "kind": row.get("kind", "statement"), "claim": text,
            "detector": {"prob": det.fake_probability, "modelId": det.model_id,
                         "isModel": not det.model_id.startswith("heuristic"), "signals": det.signals},
            "evidence": ev, "seconds": round(time.time() - t, 1)}


def llm_ok(rec: dict) -> bool:
    return ((rec["evidence"].get("llm") or {}).get("status")) == "ok"


def outcome(rec: dict) -> str:
    p = E.predict(rec, "full", None)
    return "uncertain" if p == "Uncertain" else ("right" if p == WANT[rec["label"]] else "wrong")


def load_rows() -> list[dict]:
    rows = []
    for name, path in SETS.items():
        if "fresh" in path.name:
            sys.exit("refusing to use the independent holdout")
        for x in path.read_text(encoding="utf-8").splitlines():
            if x.strip():
                rows.append({**json.loads(x), "set": name})
    return rows


def collect(rows: list[dict]) -> None:
    keep_awake()
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
            if failed and attempts < 3:
                print(f"  AI judge failed on claim {row['id']} ({backend}); retrying in 30 s", flush=True)
                time.sleep(30)
                continue
            break
        for arm in ARMS:
            recs[arm]["backend"] = backend
            with OUT[arm].open("a", encoding="utf-8") as f:
                f.write(json.dumps(recs[arm], ensure_ascii=False) + "\n")
        parts = [f"{arm}: {E.predict(recs[arm], 'full', None)} "
                 f"(ai {(recs[arm]['evidence'].get('llm') or {}).get('verdict')}, {recs[arm]['seconds']:.0f}s)"
                 for arm in ARMS]
        print(f"[{n}/{len(todo)}] {row['set']:6} {row['label']:4} " + " | ".join(parts)
              + f" | {round(time.time() - t)}s on {backend}", flush=True)
    print("done", flush=True)


def report() -> None:
    recs = {arm: {r["id"]: r for r in E.load(OUT[arm])} for arm in ARMS}
    for name in SETS:
        ids = sorted(i for i in set(recs["current"]) & set(recs["questions"]) if recs["current"][i]["set"] == name)
        if not ids:
            continue
        res = {arm: {i: outcome(recs[arm][i]) for i in ids} for arm in ARMS}
        labels = {i: recs["current"][i]["label"] for i in ids}
        print(f"\n{name} set: {len(ids)} claims ({sum(v == 'real' for v in labels.values())} real)")
        print("  setup      right wrong uncertain  real->Fake  fake->Real  AI supported/refuted/unverifiable")
        for arm in ARMS:
            c = collections.Counter(res[arm].values())
            ff = sum(labels[i] == "real" and res[arm][i] == "wrong" for i in ids)
            fr = sum(labels[i] == "fake" and res[arm][i] == "wrong" for i in ids)
            ai = collections.Counter((recs[arm][i]["evidence"].get("llm") or {}).get("verdict") for i in ids)
            print(f"  {arm:10} {c['right']:5} {c['wrong']:5} {c['uncertain']:9} {ff:11} {fr:11}  "
                  f"{ai['supported']}/{ai['refuted']}/{ai['unverifiable']}")
        diff = [PTS[res["questions"][i]] - PTS[res["current"][i]] for i in ids]
        rng = random.Random(1)
        boots = sorted(sum(diff[rng.randrange(len(ids))] for _ in ids) for _ in range(4000))
        lo, hi = boots[int(0.025 * 4000)], boots[int(0.975 * 4000) - 1]
        print(f"  questions vs current (right=+1, wrong=-1): net {sum(diff):+} (95% range {lo:+} to {hi:+}), "
              f"better on {sum(d > 0 for d in diff)}, worse on {sum(d < 0 for d in diff)}")
        for i in ids:
            if res["questions"][i] != res["current"][i]:
                print(f"    {labels[i]:4} {res['current'][i]:9} -> {res['questions'][i]:9}  {recs['current'][i]['claim'][:80]}")
        secs = {arm: sum(recs[arm][i]["seconds"] for i in ids) / len(ids) for arm in ARMS}
        print("  mean seconds per claim: " + ", ".join(f"{a} {s:.0f}" for a, s in secs.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.report:
        return report()
    rows = load_rows()
    if args.limit:
        rows = rows[: args.limit]
    collect(rows)
    report()


if __name__ == "__main__":
    main()
