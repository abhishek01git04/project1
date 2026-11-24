"""Does a source actually state the claim? Judged by meaning, not by shared keywords.

Two free local models:
  embedding  BAAI/bge-small-en-v1.5 (sentence-transformers): how close in meaning a headline or
             paragraph is to the claim, used to rank them. Fast on CPU.
  MiniCheck  bespoke-minicheck through Ollama: given a document and a claim, answers whether the
             document supports the claim. It cannot say that a document contradicts a claim.

Keyword overlap (claims.overlap) links same-topic headlines to a claim ("Maharashtra cabinet
discusses new districts" shares 2 of 6 terms with "15 new districts confirmed by CM Fadnavis").
These checks are meant to replace it once the evaluation shows they do better.
"""

import logging
import math
from functools import lru_cache

from . import config

log = logging.getLogger("truthchain.support")


# ------------------------------------------------------------------ embedding similarity

@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL, device="cpu")


def similarity(claim: str, texts: list[str]) -> list[float]:
    """Cosine similarity (about 0.3 unrelated .. 1.0 same meaning) of each text to the claim."""
    if not texts:
        return []
    vecs = _embedder().encode([claim] + list(texts), normalize_embeddings=True, batch_size=32)
    return [float(vecs[0] @ v) for v in vecs[1:]]


# ------------------------------------------------------------------ MiniCheck

def minicheck(claim: str, document: str) -> float | None:
    """Probability that the document supports the claim (0..1), or None if MiniCheck is unavailable.

    The model answers "Yes" or "No"; when Ollama returns token probabilities the probability of
    "Yes" is used, otherwise the answer counts as 1.0 or 0.0."""
    import httpx

    body = {
        "model": config.MINICHECK_MODEL,
        "prompt": f"Document: {document.strip()}\nClaim: {claim.strip()}",
        "stream": False,
        "logprobs": True,
        "top_logprobs": 5,
        "options": {"temperature": 0, "num_predict": 1, "num_ctx": 4096},
    }
    try:
        r = httpx.post(f"{config.OLLAMA_URL}/api/generate", json=body, timeout=config.MINICHECK_TIMEOUT)
        r.raise_for_status()
        reply = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.info("MiniCheck unavailable: %s", exc)
        return None
    answer = reply.get("response", "").strip().lower()
    for tok in reply.get("logprobs") or []:
        # "Yes", " Yes" and "yes" are separate tokens: add up the probability of each answer
        p = {"yes": 0.0, "no": 0.0}
        for alt in tok.get("top_logprobs") or []:
            word = alt["token"].strip().lower()
            if word in p:
                p[word] += math.exp(alt["logprob"])
        if p["yes"] + p["no"] > 0:
            return p["yes"] / (p["yes"] + p["no"])
        break
    if answer.startswith("yes"):
        return 1.0
    if answer.startswith("no"):
        return 0.0
    return None


def check_sources(claim: str, passages: list[dict], headlines: list[dict] | None = None,
                  max_headlines: int = 8) -> list[dict]:
    """MiniCheck every article passage read and the headlines shown to the AI.

    Returns [{"url", "title", "kind": "article"|"headline", "support": 0..1 or None}]."""
    out = []
    read = {p["url"] for p in passages}
    for p in passages:
        out.append({"url": p["url"], "title": p["title"], "kind": "article",
                    "support": minicheck(claim, p["title"] + "\n" + p["passage"])})
    for h in [h for h in headlines or [] if h["url"] not in read][:max_headlines]:
        out.append({"url": h["url"], "title": h["title"], "kind": "headline",
                    "support": minicheck(claim, h["title"])})
    return out
