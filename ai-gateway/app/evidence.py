"""Internet Evidence Engine: live news search, fact-check search, reverse image search and
source credibility. Each source is optional and fails soft - a missing key, timeout or
outage is recorded in the report instead of failing the verification.

Sources:
  news         GDELT DOC 2.0 (free, no key); NewsAPI.org with NEWS_ENGINE=newsapi + NEWSAPI_KEY
  passages     the top articles are downloaded and their relevant paragraphs extracted
  reasoning    Claude judges the claims against those passages (reasoner.py)
  fact checks  Google Fact Check Tools API (GOOGLE_FACTCHECK_KEY)
  reverse img  Google Cloud Vision web detection (GOOGLE_VISION_KEY)
"""

import base64
import io
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from . import config, credibility, reasoner
from .claims import keywords, overlap
from . import urlfetch
from .urlfetch import domain_of

log = logging.getLogger("evidence")

DEBUNK = re.compile(r"\b(fake|hoax|false|debunk\w*|fact[- ]check\w*|misleading|no evidence|doctored|"
                    r"fabricated|satire|not true|misinformation|disinformation|rumou?r)\b", re.I)

FALSE_RATINGS = re.compile(r"false|fake|pants on fire|incorrect|misleading|no evidence|fabricated|"
                           r"wrong|scam|hoax|distorted|unproven|unsupported|altered|satire|miscaptioned", re.I)
TRUE_RATINGS = re.compile(r"^(true|correct|accurate|mostly true|mostly correct|verified|legit)", re.I)


def enabled() -> bool:
    return config.EVIDENCE_MODE != "off"


def _client() -> httpx.Client:
    return httpx.Client(timeout=config.EVIDENCE_TIMEOUT, headers={"User-Agent": config.FETCH_USER_AGENT})


# ------------------------------------------------------------------ news search (GDELT)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_INTERVAL = 5.5     # documented limit: one request every 5 seconds per client
GDELT_CACHE_TTL = 900    # identical queries within 15 minutes are served from memory
_gdelt_lock = threading.Lock()
_gdelt_last = 0.0
_gdelt_cooldown_until = 0.0
_gdelt_cache: dict[str, tuple[float, list[dict]]] = {}


def _gdelt(query: str) -> list[dict]:
    """GDELT DOC 2.0 article search (free, no key).

    GDELT rate-limits per IP, often more strictly than its documented one request per 5 seconds.
    A retry seconds later rarely succeeds, so on a 429 the gateway stops calling GDELT for
    GDELT_COOLDOWN seconds and fails fast, instead of making every verification wait."""
    global _gdelt_last, _gdelt_cooldown_until
    cached = _gdelt_cache.get(query)
    if cached and time.monotonic() - cached[0] < GDELT_CACHE_TTL:
        return cached[1]
    params = {
        "query": query + " sourcelang:english", "mode": "artlist", "format": "json",
        "maxrecords": 25, "sort": "hybridrel", "timespan": config.GDELT_TIMESPAN,
    }
    with _gdelt_lock:  # serialise requests and keep them spaced out
        remaining = _gdelt_cooldown_until - time.monotonic()
        if remaining > 0:
            raise RuntimeError(f"GDELT rate limit - paused for another {remaining:.0f} s")
        wait = _gdelt_last + GDELT_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            with httpx.Client(timeout=config.GDELT_TIMEOUT,
                              headers={"User-Agent": config.FETCH_USER_AGENT}) as c:
                r = c.get(GDELT_URL, params=params)
        finally:
            _gdelt_last = time.monotonic()
        if r.status_code == 429:
            _gdelt_cooldown_until = time.monotonic() + config.GDELT_COOLDOWN
            raise RuntimeError(f"GDELT rate limit (429) - news search paused for {config.GDELT_COOLDOWN:.0f} s")
    r.raise_for_status()
    try:
        data = r.json()
    except ValueError:  # GDELT answers errors (e.g. "query too short") as plain text
        raise RuntimeError(r.text.strip()[:120]) from None
    articles = [{"title": a.get("title", ""), "url": a.get("url", ""), "published": a.get("seendate", ""),
                 "country": a.get("sourcecountry", "")} for a in data.get("articles", [])]
    _gdelt_cache[query] = (time.monotonic(), articles)
    return articles


def _newsapi(query: str) -> list[dict]:
    with _client() as c:
        r = c.get("https://newsapi.org/v2/everything", params={
            "q": query, "language": "en", "sortBy": "relevancy", "pageSize": 15, "apiKey": config.NEWSAPI_KEY,
        })
    r.raise_for_status()
    return [{"title": a.get("title") or "", "url": a.get("url", ""), "published": a.get("publishedAt", "")}
            for a in r.json().get("articles", [])]


def _google_news(query: str) -> list[dict]:
    """Google News RSS search (free, no key, fast). Used when GDELT is rate-limited or finds nothing.
    Links point to news.google.com redirects, so articles are judged by headline and publisher."""
    import xml.etree.ElementTree as ET

    with _client() as c:
        r = c.get("https://news.google.com/rss/search", follow_redirects=True, params={
            "q": query, "hl": config.GNEWS_LANG, "gl": config.GNEWS_REGION,
            "ceid": f"{config.GNEWS_REGION}:{config.GNEWS_LANG.split('-')[0]}",
        })
    r.raise_for_status()
    articles = []
    for item in ET.fromstring(r.content).findall("./channel/item")[:30]:
        source = item.find("source")
        publisher = source.text if source is not None else ""
        title = item.findtext("title", "")
        if publisher and title.endswith(" - " + publisher):  # "Headline - Publisher"
            title = title[: -len(publisher) - 3]
        articles.append({
            "title": title.strip(), "url": item.findtext("link", ""),
            "sourceUrl": source.get("url", "") if source is not None else "",
            "publisher": publisher, "published": item.findtext("pubDate", ""),
        })
    return articles


def _relevance(claim: str, texts: list[str], top: int = 6) -> tuple[list[float], float | None]:
    """How closely each text matches the claim, and the cut-off to use (None = the caller's).

    RELEVANCE=keywords: share of the claim's `top` key terms in the text. RELEVANCE=embedding:
    meaning similarity (support.py), so same-topic headlines rank below one stating the same event."""
    if config.RELEVANCE == "embedding" and texts:
        try:
            from .support import similarity

            return similarity(claim, texts), config.EMBED_MIN
        except Exception as exc:  # model missing or failed to load: keyword matching still works
            log.info("Embedding relevance unavailable, using keywords: %s", exc)
    return [overlap(claim, t, top=top, stem=True) for t in texts], None


def _tag(claim: str, raw: list[dict], min_relevance: float) -> list[dict]:
    articles, seen = [], set()
    raw = [a for a in raw if a.get("title")]
    scores, cutoff = _relevance(claim, [a["title"] for a in raw])
    min_relevance = min_relevance if cutoff is None else cutoff
    for a, relevance in zip(raw, scores):
        key = a["title"].lower().strip()
        if relevance < min_relevance or not a["url"] or key in seen:
            continue
        seen.add(key)
        cred = credibility.rate(domain_of(a.get("sourceUrl") or a["url"]))
        stance = "refutes" if DEBUNK.search(a["title"]) else "supports"
        articles.append({**a, "domain": cred["domain"], "credibility": cred["score"],
                         "relevance": round(relevance, 2), "stance": stance})
    articles.sort(key=lambda a: (-a["relevance"], -a["credibility"]))
    return articles[:8]


def _search_queries(claim: str, queries: list[str]) -> dict:
    """News search with the AI-written queries (QUERY_REWRITE=on): GDELT or NewsAPI for the first
    query (GDELT allows one request per 5 s), and Google News RSS for every query."""
    raw, engines, errors = [], [], []
    try:
        if config.NEWS_ENGINE == "newsapi" and config.NEWSAPI_KEY:
            raw += _newsapi(queries[0])
            engines.append("newsapi")
        else:
            words = [w for w in queries[0].split() if len(w) >= 3]  # GDELT rejects shorter keywords
            if len(words) >= 2:
                raw += _gdelt(" ".join(words[:6]))
                engines.append("gdelt")
    except Exception as exc:
        log.info("News search failed: %s", exc)
        errors.append(str(exc)[:160])
    if config.GOOGLE_NEWS_FALLBACK:
        for q in queries:
            try:
                raw += _google_news(q)
                if "google-news" not in engines:
                    engines.append("google-news")
            except Exception as exc:
                log.info("Google News search failed: %s", exc)
                errors.append(str(exc)[:160])
    articles = _tag(claim, raw, 0.3)
    status = "ok" if engines else "error"
    return {"engine": "+".join(engines) or "none", "status": status, "queries": queries,
            "reason": "; ".join(errors)[:300], "articles": articles}


def search_news(claim: str, queries: list[str] | None = None) -> dict:
    """Articles about the claim, each tagged with source credibility and stance.

    Order: GDELT (or NewsAPI with NEWS_ENGINE=newsapi). If GDELT is rate-limited, fails or finds no
    relevant article: NewsAPI when NEWSAPI_KEY is set, then Google News RSS (free, no key).
    With queries (from query.rewrite) those are searched instead of the claim's keywords."""
    if queries:
        return _search_queries(claim, queries)
    words = keywords(claim, 6)
    engine = "newsapi" if config.NEWS_ENGINE == "newsapi" and config.NEWSAPI_KEY else "gdelt"
    gdelt_words = [w for w in words if len(w) >= 3]  # GDELT rejects keywords shorter than 3 characters
    if len(words) < 2:
        return {"engine": engine, "status": "skipped", "reason": "not enough keywords", "articles": []}

    primary_error, used = "", ""
    try:
        if engine == "newsapi":
            used = " ".join(words[:4])
            articles = _tag(claim, _newsapi(used), 0.3)
        elif len(gdelt_words) >= 2:
            articles = []
            queries = [" ".join(gdelt_words[:4])] + ([" ".join(gdelt_words[:2])] if len(gdelt_words) > 2 else [])
            for query in queries:
                raw = _gdelt(query)
                articles, used = _tag(claim, raw, 0.3), query
                if raw:  # each GDELT call costs ~10 s, so only broaden when the search found nothing
                    break
        else:
            articles, primary_error = [], "too few usable keywords for GDELT"
        if articles:
            return {"engine": engine, "status": "ok", "query": used, "articles": articles}
        primary_error = primary_error or "no relevant articles"
    except Exception as exc:
        log.info("News search failed (%s): %s", engine, exc)
        primary_error = str(exc)[:160]

    # Fallbacks, so one rate-limited source does not leave the claim without evidence.
    fallbacks = ([("newsapi", _newsapi)] if config.NEWSAPI_KEY and engine != "newsapi" else []) \
        + ([("google-news", _google_news)] if config.GOOGLE_NEWS_FALLBACK else [])
    query = " ".join(words[:5])
    for name, search in fallbacks:
        try:
            articles = _tag(claim, search(query), 0.3)
        except Exception as exc:
            log.info("%s fallback failed: %s", name, exc)
            continue
        if articles:
            return {"engine": name, "status": "ok", "query": query, "fallbackFrom": engine,
                    "primaryError": primary_error, "articles": articles}
    status = "ok" if primary_error == "no relevant articles" else "error"
    return {"engine": engine, "status": status, "reason": primary_error, "query": used or query, "articles": []}


# ------------------------------------------------------------- passage retrieval

def _bing_news(query: str) -> list[dict]:
    """Bing News RSS (free, no key). Its links carry the publisher's own URL, so unlike Google News
    results the article itself can be downloaded and read."""
    import xml.etree.ElementTree as ET
    from urllib.parse import parse_qs, urlsplit

    with _client() as c:
        r = c.get("https://www.bing.com/news/search", follow_redirects=True,
                  params={"q": query, "format": "rss", "mkt": config.GNEWS_LANG})
    r.raise_for_status()
    articles = []
    for item in ET.fromstring(r.content).findall("./channel/item"):
        link = item.findtext("link", "")
        url = (parse_qs(urlsplit(link).query).get("url") or [link])[0]
        if urlsplit(url).scheme in ("http", "https") and "bing.com" not in domain_of(url):
            articles.append({"title": item.findtext("title", "").strip(), "url": url,
                             "published": item.findtext("pubDate", "")})
    return articles


def _best_paragraphs(text: str, claim: str, limit: int = 1200) -> str:
    """An article's opening paragraph plus the paragraphs sharing the most keywords with the claim."""
    paragraphs = [p for p in text.split("\n") if len(p.split()) >= 8]
    if not paragraphs:
        return ""
    scored = 60 if config.RELEVANCE == "embedding" else len(paragraphs)  # embeddings: first 60 only
    scores, _ = _relevance(claim, paragraphs[1:scored], top=12)
    scores += [0.0] * (len(paragraphs) - 1 - len(scores))
    ranked = sorted(range(1, len(paragraphs)), key=lambda i: -scores[i - 1])
    picked, size = [], 0
    for i in [0] + ranked[:5]:
        if size + len(paragraphs[i]) > limit and picked:
            break
        picked.append(i)
        size += len(paragraphs[i])
    return "\n".join(paragraphs[i] for i in sorted(picked))[:limit]


def _is_excluded(article: dict, exclude: dict | None) -> bool:
    """True for a test claim's own source, which must not be used as evidence for it."""
    if not exclude:
        return False
    url = article["url"].split("?")[0]
    return (url in {u.split("?")[0] for u in exclude.get("urls") or []}
            or _norm(article["title"]) in {_norm(t) for t in exclude.get("texts") or []}
            or domain_of(url) in set(exclude.get("domains") or []))


def retrieve_passages(articles: list[dict], claim: str, count: int | None = None, exclude: dict | None = None,
                      queries: list[str] | None = None) -> list[dict]:
    """Download news articles and keep the passages relevant to the claim. Sites that refuse or
    return little text are skipped, and the next most credible article is used instead."""
    count = count or config.RETRIEVE_ARTICLES
    # Google News links are redirects to a script page, so only direct article links are downloaded.
    candidates = [a for a in articles if "news.google.com" not in a["url"]]
    if config.ARTICLE_SEARCH:
        for q in (queries or [" ".join(keywords(claim, 5))])[:2]:
            try:
                candidates += _tag(claim, _bing_news(q), 0.3)
            except Exception as exc:
                log.info("Article search failed: %s", exc)
    seen, unique = set(), []
    for a in sorted(candidates, key=lambda a: (-a["credibility"], -a["relevance"])):
        key = _norm(a["title"])
        if key not in seen and not _is_excluded(a, exclude):
            seen.add(key)
            unique.append(a)
    picked = unique[: count * 2]

    def one(a):
        try:
            page = urlfetch.fetch(a["url"], max_bytes=3 * 1024 * 1024)
            passage = _best_paragraphs(page.text, claim)
            if len(passage.split()) < 40:  # a cookie wall or script page, not an article
                return None
            return {"url": a["url"], "title": a["title"], "domain": a["domain"], "credibility": a["credibility"],
                    "published": page.published or a.get("published", ""), "passage": passage}
        except Exception as exc:  # paywalls, bot blocks, timeouts: skip that article
            log.info("Could not read %s: %s", a["url"], exc)
            return None

    if not picked:
        return []
    with ThreadPoolExecutor(max_workers=len(picked)) as pool:
        return [p for p in pool.map(one, picked) if p][:count]


# ------------------------------------------------------------------ fact checks

def _factcheck_query(query: str) -> list[dict]:
    with _client() as c:
        r = c.get("https://factchecktools.googleapis.com/v1alpha1/claims:search", params={
            "query": query, "languageCode": "en", "pageSize": 15, "key": config.GOOGLE_FACTCHECK_KEY,
        })
    r.raise_for_status()
    return r.json().get("claims", [])


def _factcheck_relevance(claim: str, texts: list[str]) -> tuple[list[float], float]:
    """Same claim, not just the same topic. Keywords: at least 60% of the key terms match in either
    direction (a short fact-checked claim can be fully contained in a longer submitted one).
    Embedding (FACTCHECK_MATCH=embedding): cosine similarity of the meaning."""
    if config.FACTCHECK_MATCH == "embedding" and texts:
        try:
            from .support import similarity

            return similarity(claim, texts), config.FACTCHECK_EMBED_MIN
        except Exception as exc:  # model missing or failed to load: fall back to the live rule
            log.info("Embedding fact-check matching unavailable, using keywords: %s", exc)
    return [max(overlap(claim, t, top=6), overlap(t, claim, top=6)) for t in texts], 0.6


def search_fact_checks(claim: str, queries: list[str] | None = None) -> dict:
    """Published fact-checks of matching claims (Google Fact Check Tools API).

    Fact-checked claims are worded differently from the submitted text, so the search uses the
    claim's 5 key terms (names and numbers first) and falls back to 3 if nothing is found.
    With AI-written queries (QUERY_REWRITE=on) it runs the first two (not the "... fact check" one,
    which the API does not need) plus the key-term query, and keeps everything they find."""
    if not config.GOOGLE_FACTCHECK_KEY:
        return {"engine": "google-factcheck", "status": "disabled", "reason": "GOOGLE_FACTCHECK_KEY not set", "reviews": []}
    words = keywords(claim, 5)
    if queries:
        plan = [q for q in queries if "fact check" not in q.lower()][:2] + [" ".join(words)]
        union = True
    else:
        plan = [" ".join(words)] + ([" ".join(words[:3])] if len(words) > 3 else [])
        union = False
    items, used, errors = [], [], []
    for query in plan:
        try:
            got = _factcheck_query(query)
        except Exception as exc:
            if not union:  # live behaviour: the first failure ends the search
                return {"engine": "google-factcheck", "status": "error", "reason": str(exc)[:160], "reviews": []}
            errors.append(str(exc)[:160])
            continue
        used.append(query)
        items += got
        if got and not union:
            break
    if errors and not used:
        return {"engine": "google-factcheck", "status": "error", "reason": errors[0], "reviews": []}
    used = " | ".join(used) if union else used[-1]
    texts = [item.get("text", "") for item in items]
    scores, cutoff = _factcheck_relevance(claim, texts)
    reviews, seen = [], set()
    for item, text, relevance in zip(items, texts, scores):
        if relevance < cutoff:
            continue
        for rev in item.get("claimReview", [])[:1]:
            url = rev.get("url", "")
            if url in seen:
                continue
            seen.add(url)
            rating = rev.get("textualRating", "")
            stance = "refutes" if FALSE_RATINGS.search(rating) else "supports" if TRUE_RATINGS.search(rating) else "mixed"
            reviews.append({
                "claim": item.get("text", ""), "rating": rating, "stance": stance,
                "publisher": (rev.get("publisher") or {}).get("name") or domain_of(url),
                "url": url, "date": rev.get("reviewDate", ""), "relevance": round(relevance, 2),
            })
    reviews.sort(key=lambda r: -r["relevance"])
    return {"engine": "google-factcheck", "status": "ok", "query": used, "reviews": reviews[:6]}


# ------------------------------------------------------------- reverse image

def reverse_image(image) -> dict:
    """Pages on the web showing the same image (Google Vision web detection)."""
    if not config.GOOGLE_VISION_KEY:
        return {"engine": "google-vision", "status": "disabled", "reason": "GOOGLE_VISION_KEY not set", "pages": []}
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG", quality=90)
    body = {"requests": [{
        "image": {"content": base64.b64encode(buf.getvalue()).decode()},
        "features": [{"type": "WEB_DETECTION", "maxResults": 20}],
    }]}
    try:
        with _client() as c:
            r = c.post("https://vision.googleapis.com/v1/images:annotate",
                       params={"key": config.GOOGLE_VISION_KEY}, json=body)
        r.raise_for_status()
        web = r.json()["responses"][0].get("webDetection", {})
    except Exception as exc:
        return {"engine": "google-vision", "status": "error", "reason": str(exc)[:160], "pages": []}
    pages = []
    for p in web.get("pagesWithMatchingImages", [])[:10]:
        cred = credibility.rate(domain_of(p.get("url", "")))
        title = re.sub("<[^>]+>", "", p.get("pageTitle", ""))
        pages.append({
            "url": p.get("url", ""), "title": title, "domain": cred["domain"], "credibility": cred["score"],
            "factCheck": cred["domain"] in credibility.FACT_CHECKERS or bool(DEBUNK.search(title)),
            "exact": bool(p.get("fullMatchingImages")),
        })
    return {
        "engine": "google-vision", "status": "ok", "pages": pages,
        "bestGuess": [g.get("label") for g in web.get("bestGuessLabels", [])],
        "exactMatches": len(web.get("fullMatchingImages", [])),
    }


# ------------------------------------------------------------------ orchestration

def _norm(t: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (t or "").lower()))


def gather(claims: list[str], image=None, source_url: str = "", text: str = "", exclude: dict | None = None) -> dict:
    """Run every applicable evidence source in parallel, then let the AI model reason over the claims.

    exclude={"urls": [...], "texts": [...]} drops matching news articles and fact-checks before
    anything uses them. The evaluation uses it so a test claim is not judged by its own source."""
    out = {"claims": claims, "news": None, "factChecks": None, "reverseImage": None, "source": None, "llm": None}
    if source_url:
        out["source"] = credibility.rate(domain_of(source_url))
    if not enabled():
        out["status"] = "disabled (EVIDENCE_MODE=off)"
        return out
    main_claim = claims[0] if claims else ""
    queries = None
    if config.QUERY_REWRITE and main_claim:
        from .query import is_english, rewrite

        rewritten = rewrite(main_claim)
        if rewritten:
            queries = rewritten["queries"]
            out["queryRewrite"] = {"original": main_claim, **rewritten}
            # An English claim is ranked and judged as written (a restatement can lose detail);
            # one in Hindi or Hinglish is ranked and judged in its English restatement.
            if not is_english(main_claim):
                main_claim = rewritten["claim"]
                claims = [main_claim] + claims[1:]
    with ThreadPoolExecutor(max_workers=3) as pool:
        news = pool.submit(search_news, main_claim, queries) if main_claim else None
        facts = pool.submit(search_fact_checks, main_claim, queries) if main_claim else None
        rev = pool.submit(reverse_image, image) if image is not None else None
        out["news"] = news.result() if news else None
        out["factChecks"] = facts.result() if facts else None
        out["reverseImage"] = rev.result() if rev else None
    if exclude:
        urls = set(exclude.get("urls") or [])
        texts = {_norm(t) for t in exclude.get("texts") or []}
        for block, key, field in (("news", "articles", "title"), ("factChecks", "reviews", "claim")):
            if out[block] and out[block].get(key):
                kept = [x for x in out[block][key] if x.get("url") not in urls and _norm(x.get(field)) not in texts]
                out[block]["excluded"] = len(out[block][key]) - len(kept)
                out[block][key] = kept
    # Retrieval + reasoning: read the articles GDELT found, then let Claude judge the claims.
    if claims and reasoner.enabled():
        retrieval = config.LLM_RETRIEVAL
        which = reasoner.provider()
        if config.QUESTION_SEARCH and which == "ollama" and main_claim:
            # AVeriTeC-style: search each question a fact-checker would ask (questions.py)
            from . import questions

            asked = questions.generate(main_claim)
            if asked:
                groups = questions.research(main_claim, asked, out["news"] or {}, exclude)
                out["questionEvidence"] = groups
                out["questions"] = [{"question": g["question"], "engine": g["engine"],
                                     "read": len(g["passages"]), "headlines": len(g["headlines"])}
                                    for g in groups[1:]]
                out["passages"] = [p for g in groups for p in g["passages"]]
                out["llm"] = reasoner.assess(claims, text, source_url, prior=out, retrieval=retrieval)
                out["status"] = "ok"
                return out
        if retrieval == "gdelt" and which:
            news = out["news"] or {}
            if news.get("status") == "error" and config.LLM_WEB_FALLBACK and which == "anthropic":
                # GDELT failed (usually its rate limit): let Claude run its own web search this time.
                retrieval = "web_search"
            else:
                out["passages"] = retrieve_passages(news.get("articles") or [], main_claim, exclude=exclude,
                                                    queries=queries)
        out["llm"] = reasoner.assess(claims, text, source_url, prior=out, retrieval=retrieval)
    out["status"] = "ok"
    return out


def gdelt_status() -> str:
    remaining = _gdelt_cooldown_until - time.monotonic()
    if remaining > 0:
        return f"gdelt - paused {remaining:.0f} s after a rate limit"
    return "gdelt (keyless)"


def llm_status() -> str:
    which = reasoner.provider()
    if which == "anthropic":
        return f"claude ({config.CLAUDE_MODEL}) reading {config.LLM_RETRIEVAL} results"
    if which == "ollama":
        return f"local ollama ({config.OLLAMA_MODEL}, free) reading gdelt results"
    if not reasoner.enabled():
        return "off"
    return "no model - start Ollama (free) or set ANTHROPIC_API_KEY"


def engines_status() -> dict:
    if not enabled():
        return {"evidence": "off"}
    return {
        "news": "newsapi" if config.NEWS_ENGINE == "newsapi" and config.NEWSAPI_KEY else gdelt_status(),
        "newsFallback": ", ".join(filter(None, [
            "newsapi" if config.NEWSAPI_KEY else "",
            "claude web search" if config.LLM_WEB_FALLBACK and reasoner.provider() == "anthropic" else "",
        ])) or "none - add NEWSAPI_KEY or ANTHROPIC_API_KEY",
        "factCheck": "google-factcheck" if config.GOOGLE_FACTCHECK_KEY else "disabled - set GOOGLE_FACTCHECK_KEY",
        "reverseImage": "google-vision" if config.GOOGLE_VISION_KEY else "disabled - set GOOGLE_VISION_KEY",
        "sourceCredibility": "built-in list" + (" + custom file" if config.SOURCE_RATINGS_FILE else ""),
        "llmReasoning": llm_status(),
        "searchQueries": "AI rewrite (" + config.OLLAMA_MODEL + ")" if config.QUERY_REWRITE else "keywords",
        "relevance": config.EMBED_MODEL if config.RELEVANCE == "embedding" else "keywords",
        "supportCheck": config.MINICHECK_MODEL if config.SUPPORT_CHECK == "minicheck" else "keywords",
    }
