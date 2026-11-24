"""Claim extraction: pick the check-worthy sentences of a text and build search queries."""

import re

from .textclean import clean_text

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more most
my myself no nor not now of off on once only or other our ours ourselves out over own same she should
so some such than that the their theirs them themselves then there these they this those through to
too under until up very was we were what when where which while who whom why will with would you your
yours yourself yourselves says said say also new news report reports breaking according just like get
got one two may might must many much us via amid people year years today yesterday week time
shocking share shared deleted urgent viral watch video exclusive unbelievable amazing omg wow
""".split())

CLAIM_VERBS = re.compile(
    r"\b(is|are|was|were|will|has|have|had|claims?|confirm\w*|announc\w*|reveal\w*|found|finds|"
    r"cause[sd]?|cures?|kill\w*|ban\w*|arrest\w*|died|dies|dead|won|wins|launch\w*|declar\w*|"
    r"approv\w*|deni\w*|admit\w*|shows?|prov\w*|leak\w*)\b",
    re.I,
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"”’])\s+|\n+")
# Abbreviations whose full stop does not end a sentence. Without this "The official poverty rate in
# the U.S. hit a record low" was cut to "The official poverty rate in the U.S." before searching
# (13 of 120 tuning claims lost a fifth or more of their words this way).
_ABBR = re.compile(r"\b(?:[A-Z]\.){2,}|\b(?:Mr|Mrs|Ms|Dr|Prof|St|Gen|Gov|Sen|Rep|Lt|Col|Capt|Sgt|Jr|Sr|Inc|Ltd|"
                   r"Corp|Co|vs|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.|\b[A-Z]\.(?=\s+[A-Z][a-z])")
_DOT = "․"  # one-dot leader: stands in for a protected full stop while splitting
QUOTES = "\"\u201c\u2018'"  # straight and curly opening quotes
FIRST_PERSON = re.compile(r"\b(I|we|my|our)\b")
_WORD = re.compile(r"\d+(?:[.,]\d+)+%?|[A-Za-z0-9](?:[A-Za-z0-9'\-]*[A-Za-z0-9%])?")


def sentences(text: str) -> list[str]:
    protected = _ABBR.sub(lambda m: m.group(0).replace(".", _DOT), text.replace(_DOT, "."))
    return [s.strip().replace(_DOT, ".") for s in _SENT_SPLIT.split(protected) if 5 <= len(s.split()) <= 60]


def _score(sentence: str, position: int) -> float:
    words = sentence.split()
    capitalised = sum(1 for w in words[1:] if w[:1].isupper())
    return (
        2.0 * bool(re.search(r"\d", sentence))
        + 1.5 * bool(CLAIM_VERBS.search(sentence))
        + min(capitalised, 4) * 0.5
        + (1.5 if position == 0 else 0)          # headlines / first sentences carry the claim
        - (1.0 if sentence.endswith("?") else 0)  # questions are rarely claims
        - (2.0 if sentence[:1] in QUOTES else 0)  # quotes are testimony, not the story's claim
        - (1.0 if FIRST_PERSON.search(sentence) else 0)  # first-person accounts
    )


def extract_claims(text: str, title: str = "", limit: int = 3) -> list[str]:
    """Return up to `limit` check-worthy claims, most important first."""
    text = clean_text(text)
    headline = title.strip() if title and len(title.split()) >= 4 else ""
    candidates = [s for s in sentences(text) if s != headline]
    if not candidates and not headline and text:
        candidates = [" ".join(text.split()[:40])]
    ranked = sorted(enumerate(candidates), key=lambda p: -_score(p[1], p[0]))
    if headline:  # an article's headline states its main claim
        ranked.insert(0, (-1, headline))
    seen, claims = set(), []
    for _, s in ranked:
        key = frozenset(keywords(s, 6))
        if key and key not in seen:
            seen.add(key)
            claims.append(s)
        if len(claims) == limit:
            break
    return claims


def keywords(text: str, limit: int = 8) -> list[str]:
    """Most informative words of a claim: names and numbers first, then other non-stopwords."""
    words = _WORD.findall(text)
    out, seen = [], set()
    for prefer in (True, False):
        for w in words:
            lw = w.lower().strip(".,%")
            if lw in STOPWORDS or lw in seen or (len(lw) < 3 and not any(c.isdigit() for c in lw)):
                continue
            is_key = w[:1].isupper() or w[:1].isdigit()
            if is_key == prefer:
                seen.add(lw)
                out.append(w.strip(".,"))
    return out[:limit]


def _stem(word: str) -> str:
    """Crude stem so word forms match: retired / retires / retirement -> "retir"."""
    w = word.lower().strip(".,'%").removesuffix("'s")
    return w[:5] if len(w) > 5 and not any(c.isdigit() for c in w) else w


def overlap(a: str, b: str, top: int = 12, stem: bool = False) -> float:
    """Share of a's top keywords that also appear in b (0..1) - how well a result matches a claim.
    stem=True matches word forms (retired / retires); used for news headlines, not fact-checks,
    where looser matching links same-topic but different claims."""
    norm = _stem if stem else (lambda w: w.lower())
    ka = {norm(w) for w in keywords(a, top)}
    kb = {norm(w) for w in _WORD.findall(b)}
    return len(ka & kb) / len(ka) if ka else 0.0
