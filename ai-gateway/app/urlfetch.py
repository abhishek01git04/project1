"""Fetch a user-supplied URL safely and extract the article from HTML pages.

The gateway fetches URLs on behalf of anonymous users, so every hop (including redirects)
is checked to resolve to a public IP address: no localhost, private networks or cloud
metadata endpoints (SSRF protection).
"""

import ipaddress
import socket
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx

from . import config

MAX_REDIRECTS = 5
USER_AGENT = config.FETCH_USER_AGENT


class UnsafeURL(ValueError):
    pass


@dataclass
class Fetched:
    url: str                 # final URL after redirects
    content_type: str
    body: bytes
    title: str = ""
    text: str = ""
    published: str = ""
    image_url: str = ""
    meta: dict = field(default_factory=dict)


def _check_public(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise UnsafeURL("only http(s) URLs with a host name are allowed")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise UnsafeURL(f"cannot resolve host {parts.hostname}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast:
            raise UnsafeURL(f"{parts.hostname} resolves to a non-public address")


def fetch(url: str, max_bytes: int | None = None) -> Fetched:
    """GET a public URL, following up to MAX_REDIRECTS redirects, with a size cap."""
    max_bytes = max_bytes or config.MAX_VIDEO_MB * 1024 * 1024
    with httpx.Client(timeout=10, follow_redirects=False, headers={"User-Agent": USER_AGENT}) as client:
        for _ in range(MAX_REDIRECTS + 1):
            _check_public(url)
            with client.stream("GET", url) as resp:
                if resp.is_redirect:
                    url = urljoin(url, resp.headers.get("location", ""))
                    continue
                resp.raise_for_status()
                body = bytearray()
                for chunk in resp.iter_bytes():
                    body += chunk
                    if len(body) > max_bytes:
                        raise ValueError(f"content larger than {max_bytes // (1024 * 1024)} MB")
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                fetched = Fetched(url=str(resp.url), content_type=ctype, body=bytes(body))
                if ctype in ("text/html", "application/xhtml+xml") or not ctype:
                    _extract_article(fetched, resp.encoding or "utf-8")
                return fetched
    raise ValueError("too many redirects")


class _ArticleParser(HTMLParser):
    SKIP = {"script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta, self.paragraphs, self.title = {}, [], ""
        self._skip = 0
        self._in_title = self._in_p = False
        self._buf = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in ("p", "h1") and not self._skip:
            self._in_p, self._buf = True, []
        elif tag == "meta":
            key = (a.get("property") or a.get("name") or "").lower()
            if key and a.get("content"):
                self.meta.setdefault(key, a["content"].strip())

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in ("p", "h1") and self._in_p:
            self._in_p = False
            text = " ".join("".join(self._buf).split())
            if len(text.split()) >= 6:
                self.paragraphs.append(text)

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self._in_p and not self._skip:
            self._buf.append(data)


def _extract_article(fetched: Fetched, encoding: str) -> None:
    parser = _ArticleParser()
    parser.feed(fetched.body.decode(encoding, errors="replace"))
    m = parser.meta
    fetched.meta = m
    fetched.title = (m.get("og:title") or parser.title or "").strip()
    fetched.text = "\n".join(parser.paragraphs)[:50_000] or m.get("og:description") or m.get("description", "")
    fetched.published = m.get("article:published_time") or m.get("date") or m.get("pubdate") or ""
    if m.get("og:image"):
        fetched.image_url = urljoin(fetched.url, m["og:image"])


def domain_of(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host.removeprefix("www.")
