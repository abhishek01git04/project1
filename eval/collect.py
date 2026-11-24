"""Run the verification pipeline once per labelled claim and cache every component's output.

Usage (from the project root):
  ai-gateway\\.venv\\Scripts\\python eval\\collect.py            # all claims in eval/claims.jsonl
  ai-gateway\\.venv\\Scripts\\python eval\\collect.py --limit 5  # quick trial

Writes eval/runs/components.jsonl (one line per claim). Already-collected claims are skipped,
so the run can be stopped and resumed. evaluate.py then scores and tunes the fusion offline,
without touching the network again.

Leakage control: each claim's own fact-check / headline (the source it was taken from) is
removed from the evidence before the AI model reads it, so the system cannot "look up the answer".
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai-gateway"))

import httpx  # noqa: E402

from app import detectors, evidence, reasoner  # noqa: E402
from app.claims import extract_claims  # noqa: E402

HERE = Path(__file__).resolve().parent
CLAIMS = HERE / "claims.jsonl"
OUT = HERE / "runs" / "components.jsonl"


NETWORK_ERRORS = ("getaddrinfo", "forcibly closed", "handshake", "timed out", "connecterror", "connection")


def hit_network_failure(rec: dict) -> bool:
    """True if a connection problem (not a rate limit or an empty result) starved this claim of evidence."""
    ev = rec["evidence"]
    for block in ("news", "factChecks"):
        b = ev.get(block) or {}
        reason = ((b.get("reason") or "") + " " + (b.get("primaryError") or "")).lower()
        if b.get("status") == "error" and "rate limit" not in reason and any(e in reason for e in NETWORK_ERRORS):
            return True
    return False


def wait_for_internet() -> None:
    """Pause while the connection is down, so an outage does not produce evidence-free records."""
    announced = False
    while True:
        try:
            httpx.get("https://news.google.com/rss", timeout=10, follow_redirects=True)
            if announced:
                print("internet back, continuing", flush=True)
            return
        except httpx.HTTPError:
            if not announced:
                print("internet down, waiting...", flush=True)
                announced = True
            time.sleep(15)


def main():
    global CLAIMS, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", type=Path, default=CLAIMS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redo-network-failures", action="store_true",
                    help="drop records affected by a connection outage and collect them again")
    args = ap.parse_args()
    CLAIMS, OUT = args.claims, args.out

    if args.redo_network_failures and OUT.exists():
        recs = [json.loads(line) for line in OUT.read_text(encoding="utf-8").splitlines() if line.strip()]
        keep = [r for r in recs if not hit_network_failure(r)]
        OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
        print(f"dropped {len(recs) - len(keep)} records affected by network failures", flush=True)

    rows = [json.loads(line) for line in CLAIMS.read_text(encoding="utf-8").splitlines() if line.strip()]
    OUT.parent.mkdir(exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(line)["id"] for line in OUT.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [r for r in rows if r["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} already collected, {len(todo)} to go. AI model: {reasoner.provider() or 'none'}", flush=True)

    for n, row in enumerate(todo, 1):
        wait_for_internet()
        t = time.time()
        text = row["claim"]
        det = detectors.analyze_text(text)
        claims = extract_claims(text)
        ev = evidence.gather(claims, text=text, exclude={"urls": [row["sourceUrl"]], "texts": [row["sourceTitle"]]})
        rec = {
            "id": row["id"], "label": row["label"], "origin": row["origin"], "kind": row.get("kind", "statement"),
            "claim": text,
            "detector": {"prob": det.fake_probability, "modelId": det.model_id,
                         "isModel": not det.model_id.startswith("heuristic"), "signals": det.signals},
            "evidence": ev,
            "seconds": round(time.time() - t, 1),
        }
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        news = ev.get("news") or {}
        llm = ev.get("llm") or {}
        print(f"[{n}/{len(todo)}] {row['label']:4} det {det.fake_probability:.2f} | news {news.get('engine')}:"
              f"{len(news.get('articles') or [])} | fc {len((ev.get('factChecks') or {}).get('reviews') or [])} | "
              f"ai {llm.get('verdict') or llm.get('status')} | {rec['seconds']}s | {text[:50]}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
