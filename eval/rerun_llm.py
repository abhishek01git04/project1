"""Re-run only the AI model over the evidence saved by collect.py, so model and prompt variants are
compared on identical news, passages and fact-checks.

Usage (from the project root):
  ai-gateway\\.venv\\Scripts\\python eval\\rerun_llm.py --model qwen2.5:3b --hint strict --name 3b-strict

Writes eval/runs/<name>.jsonl and can be stopped and resumed. Score it with:
  ai-gateway\\.venv\\Scripts\\python eval\\evaluate.py --runs eval\\runs\\<name>.jsonl
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai-gateway"))

from app import config, reasoner  # noqa: E402

HERE = Path(__file__).resolve().parent
BASE = HERE / "runs" / "components.jsonl"

HINTS = {
    "original": "Several independent reliable outlets reporting the same event is evidence that it happened.",
    "strict": reasoner.EVIDENCE_HINT,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hint", choices=HINTS, default="strict")
    ap.add_argument("--name", required=True)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--prompt", choices=["v2", "v3", "v4"], default="v3", help="LLM_PROMPT_VERSION to test")
    ap.add_argument("--base", type=Path, default=BASE, help="saved run whose evidence the AI should re-read")
    args = ap.parse_args()

    config.LLM_PROMPT_VERSION = args.prompt
    config.OLLAMA_MODEL = args.model
    config.OLLAMA_TIMEOUT = args.timeout
    reasoner.EVIDENCE_HINT = HINTS[args.hint]

    out = HERE / "runs" / f"{args.name}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(x)["id"] for x in out.read_text(encoding="utf-8").splitlines() if x.strip()}
    rows = [json.loads(x) for x in args.base.read_text(encoding="utf-8").splitlines() if x.strip()]
    todo = [r for r in rows if r["id"] not in done]
    print(f"{args.name}: {len(done)} done, {len(todo)} to go", flush=True)

    for n, row in enumerate(todo, 1):
        t = time.time()
        rec = copy.deepcopy(row)
        ev = rec["evidence"]
        ev["llm"] = reasoner._assess_ollama(ev.get("claims") or [row["claim"]], row["claim"], "",
                                            ev.get("passages") or [], (ev.get("news") or {}).get("articles") or [])
        rec["llmSeconds"] = round(time.time() - t, 1)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{n}/{len(todo)}] {row['label']:4} {ev['llm'].get('verdict') or ev['llm'].get('status')} "
              f"{rec['llmSeconds']}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
