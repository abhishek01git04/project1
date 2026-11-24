"""Build a labelled set of recent claims for measuring the verification pipeline.

  fake  claims that professional fact-checkers rated false (Google Fact Check Tools API)
  real  recent headlines from established news outlets (Google News RSS)

Usage (from the project root):
  ai-gateway\\.venv\\Scripts\\python eval\\build_claims.py --per-class 60

Writes eval/claims.jsonl. Needs GOOGLE_FACTCHECK_KEY in ai-gateway/.env.

Caveats, also printed in the evaluation report:
  - "real" means "headline of an established outlet", not independently verified.
  - Each claim's own fact-check / headline is recorded (sourceUrl, sourceTitle) so that
    evaluate.py can remove it from the evidence and measure without that leakage.
"""

import argparse
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai-gateway"))

import httpx  # noqa: E402

from app import config, credibility  # noqa: E402
from app.evidence import FALSE_RATINGS, TRUE_RATINGS  # noqa: E402
from app.urlfetch import domain_of  # noqa: E402

OUT = Path(__file__).resolve().parent / "claims.jsonl"
MEDIA = re.compile(r"\b(image|images|photo|photos|picture|pictures|video|videos|clip|footage|visuals?|screenshot)\b"
                   r".{0,40}\b(shows?|showing|depicts?|of)\b", re.I)

FACT_CHECKERS = [
    "altnews.in", "boomlive.in", "factly.in", "thequint.com", "factcheck.afp.com", "politifact.com",
    "snopes.com", "fullfact.org", "leadstories.com", "reuters.com", "indiatoday.in", "newschecker.in",
]
NEWS_FEEDS = [  # Google News top stories for several regions (English)
    ("en-IN", "IN"), ("en-US", "US"), ("en-GB", "GB"), ("en-AU", "AU"),
]


def fact_checked_claims(max_age_days: int) -> list[dict]:
    key = config.GOOGLE_FACTCHECK_KEY
    if not key:
        sys.exit("GOOGLE_FACTCHECK_KEY is not set in ai-gateway/.env")
    out = []
    with httpx.Client(timeout=20) as c:
        for site in FACT_CHECKERS:
            token = None
            for _ in range(3):  # up to 3 pages per publisher
                params = {"reviewPublisherSiteFilter": site, "languageCode": "en", "maxAgeDays": max_age_days,
                          "pageSize": 50, "key": key}
                if token:
                    params["pageToken"] = token
                r = c.get("https://factchecktools.googleapis.com/v1alpha1/claims:search", params=params)
                if r.status_code != 200:
                    print(f"  {site}: HTTP {r.status_code}")
                    break
                data = r.json()
                for item in data.get("claims", []):
                    text = " ".join((item.get("text") or "").split())
                    for rev in item.get("claimReview", [])[:1]:
                        rating = rev.get("textualRating", "")
                        if FALSE_RATINGS.search(rating):
                            label = "fake"
                        elif TRUE_RATINGS.search(rating):
                            label = "real"
                        else:
                            continue  # mixed / unclear ratings are not usable labels
                        out.append({"claim": text, "label": label, "origin": "fact-check", "rating": rating,
                                    "publisher": (rev.get("publisher") or {}).get("name") or site,
                                    "sourceUrl": rev.get("url", ""), "sourceTitle": text,
                                    "date": rev.get("reviewDate", "")})
                token = data.get("nextPageToken")
                if not token:
                    break
            print(f"  {site}: {sum(1 for x in out if site in x['sourceUrl'])} usable")
    return out


def credible_headlines() -> list[dict]:
    out = []
    with httpx.Client(timeout=20, headers={"User-Agent": config.FETCH_USER_AGENT}, follow_redirects=True) as c:
        for lang, region in NEWS_FEEDS:
            r = c.get("https://news.google.com/rss", params={"hl": lang, "gl": region, "ceid": f"{region}:en"})
            if r.status_code != 200:
                print(f"  feed {region}: HTTP {r.status_code}")
                continue
            for item in ET.fromstring(r.content).findall("./channel/item"):
                source = item.find("source")
                publisher = source.text if source is not None else ""
                site = source.get("url", "") if source is not None else ""
                title = item.findtext("title", "")
                if publisher and title.endswith(" - " + publisher):
                    title = title[: -len(publisher) - 3]
                if credibility.rate(domain_of(site))["score"] < 0.75:
                    continue  # only established outlets count as "real"
                out.append({"claim": title.strip(), "label": "real", "origin": "news-headline", "rating": "",
                            "publisher": publisher, "sourceUrl": item.findtext("link", ""),
                            "sourceTitle": title.strip(), "date": item.findtext("pubDate", "")})
    return out


def usable(text: str) -> bool:
    words = text.split()
    # skip non-claims: very short, questions, live blogs, quizzes
    return 5 <= len(words) <= 60 and not text.endswith("?") and not re.search(r"\blive\b|\bquiz\b|\bwatch\b", text, re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--max-age-days", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[],
                    help="existing claim files: their claims and sources are left out, for a fresh test set")
    args = ap.parse_args()

    print("Fact-checked claims:")
    checked = fact_checked_claims(args.max_age_days)
    print("Credible headlines:")
    heads = credible_headlines()

    seen, seen_urls, pool = set(), set(), []
    for path in args.exclude:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                old = json.loads(line)
                seen.add(old["claim"].lower()[:80])
                seen_urls.add(old["sourceUrl"])
    print(f"Excluding {len(seen)} earlier claims")
    for x in checked + heads:
        key = x["claim"].lower()[:80]
        if usable(x["claim"]) and key not in seen and x["sourceUrl"] not in seen_urls:
            seen.add(key)
            pool.append(x)
    rng = random.Random(args.seed)
    fake = [x for x in pool if x["label"] == "fake"]
    real_fc = [x for x in pool if x["label"] == "real" and x["origin"] == "fact-check"]
    real_news = [x for x in pool if x["label"] == "real" and x["origin"] == "news-headline"]
    rng.shuffle(fake)
    rng.shuffle(real_news)
    n = args.per_class
    rng.shuffle(real_fc)
    half = n // 2
    real = real_fc[:half] + real_news[: n - min(half, len(real_fc))]  # half rated true, half headlines
    data = fake[:n] + real
    for x in data:  # captions of photos/videos are hard to judge from text alone: reported separately
        x["kind"] = "media-caption" if MEDIA.search(x["claim"]) else "statement"
    rng.shuffle(data)
    for i, x in enumerate(data):
        x["id"] = i
    with args.out.open("w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(data)} claims to {args.out}: {sum(x['label'] == 'fake' for x in data)} fake, "
          f"{sum(x['label'] == 'real' for x in data)} real "
          f"({sum(x['origin'] == 'fact-check' for x in real)} rated true by fact-checkers, "
          f"{sum(x['origin'] == 'news-headline' for x in real)} credible headlines); "
          f"{sum(x['kind'] == 'media-caption' for x in data)} are photo/video captions")


if __name__ == "__main__":
    main()
