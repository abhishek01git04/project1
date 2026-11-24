"""Source credibility: a trust score (0..1) for a publisher's domain.

A small built-in list of well-known outlets, fact-checkers, satire sites and known
misinformation sites, plus simple rules for unknown domains. Extend or override it
with SOURCE_RATINGS_FILE (JSON object {"domain": score}).
"""

import json
import logging

from . import config

log = logging.getLogger("credibility")

HIGH = 0.9
GOOD = 0.75
LOW = 0.15

_BUILTIN = {
    # wire services and established newsrooms
    **dict.fromkeys([
        "reuters.com", "apnews.com", "afp.com", "bbc.com", "bbc.co.uk", "npr.org", "pbs.org",
        "nytimes.com", "washingtonpost.com", "wsj.com", "ft.com", "economist.com", "bloomberg.com",
        "theguardian.com", "aljazeera.com", "dw.com", "france24.com", "cbc.ca", "abc.net.au",
        "thehindu.com", "indianexpress.com", "hindustantimes.com", "ndtv.com", "livemint.com",
        "scroll.in", "theprint.in", "nature.com", "science.org", "who.int", "un.org",
    ], HIGH),
    **dict.fromkeys([
        "cnn.com", "nbcnews.com", "cbsnews.com", "abcnews.go.com", "usatoday.com", "latimes.com",
        "politico.com", "axios.com", "time.com", "newsweek.com", "independent.co.uk", "sky.com",
        "news.sky.com", "timesofindia.indiatimes.com", "indiatoday.in", "news18.com", "thewire.in",
        "wikipedia.org", "en.wikipedia.org",
    ], GOOD),
    # fact-checkers
    **dict.fromkeys([
        "snopes.com", "politifact.com", "factcheck.org", "fullfact.org", "altnews.in", "boomlive.in",
        "factly.in", "leadstories.com", "checkyourfact.com", "healthfeedback.org", "sciencefeedback.co",
    ], HIGH),
    # satire (not news, even if it reads like it)
    **dict.fromkeys([
        "theonion.com", "babylonbee.com", "clickhole.com", "thebeaverton.com", "fakingnews.com",
        "worldnewsdailyreport.com", "newsthump.com",
    ], 0.05),
    # sites repeatedly rated as spreading fabricated stories
    **dict.fromkeys([
        "infowars.com", "naturalnews.com", "beforeitsnews.com", "newspunch.com", "yournewswire.com",
        "thegatewaypundit.com", "zerohedge.com", "globalresearch.ca", "rt.com", "sputniknews.com",
        "postcard.news",
    ], LOW),
}

FACT_CHECKERS = {
    "snopes.com", "politifact.com", "factcheck.org", "fullfact.org", "altnews.in", "boomlive.in",
    "factly.in", "leadstories.com", "checkyourfact.com", "healthfeedback.org", "sciencefeedback.co",
}
SATIRE = {d for d, s in _BUILTIN.items() if s == 0.05}

_ratings: dict | None = None


def _load() -> dict:
    global _ratings
    if _ratings is None:
        _ratings = dict(_BUILTIN)
        if config.SOURCE_RATINGS_FILE:
            try:
                with open(config.SOURCE_RATINGS_FILE, encoding="utf-8") as f:
                    _ratings.update({k.lower(): float(v) for k, v in json.load(f).items()})
            except (OSError, ValueError) as exc:
                log.warning("Could not read SOURCE_RATINGS_FILE: %s", exc)
    return _ratings


def rate(domain: str) -> dict:
    """{"domain", "score" 0..1, "label", "reason"} for a host name like "www.bbc.co.uk"."""
    domain = (domain or "").lower().removeprefix("www.")
    ratings = _load()
    # match the domain or any parent domain (news.bbc.co.uk -> bbc.co.uk)
    parts = domain.split(".")
    for i in range(len(parts) - 1):
        candidate = ".".join(parts[i:])
        if candidate in ratings:
            score = ratings[candidate]
            if candidate in SATIRE:
                reason = "Known satire site - stories are intentionally fictional"
            elif candidate in FACT_CHECKERS:
                reason = "Recognised fact-checking organisation"
            elif score >= GOOD:
                reason = "Established news organisation"
            else:
                reason = "Repeatedly rated as publishing false or fabricated stories"
            return _result(domain, score, reason)

    tld = parts[-1] if parts else ""
    if domain.endswith((".com.co", ".com.de", ".news.co")) or domain.count("-") >= 3:
        return _result(domain, 0.25, "Look-alike or unusual domain pattern often used by imitation news sites")
    if tld in ("gov", "mil", "edu") or domain.endswith((".gov.in", ".nic.in", ".gov.uk", ".ac.uk", ".ac.in")):
        return _result(domain, 0.85, "Government or academic domain")
    return _result(domain, 0.5, "Unknown source - no rating available")


def _result(domain: str, score: float, reason: str) -> dict:
    label = "high" if score >= GOOD else "low" if score <= 0.3 else "unknown" if score == 0.5 else "medium"
    return {"domain": domain, "score": score, "label": label, "reason": reason}
