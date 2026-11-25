"""
backend/web/wikipedia.py
Permission-based Wikipedia fallback.

Called only after the user has approved the exact search term (enforced in backend/app.py).
Uses the official MediaWiki API on a fixed host, never an arbitrary URL. Crawled pages are linked by
their real hyperlinks and scored with the same hand-written HITS used for the corpus. Nothing fetched
here is added to the indexed corpus.
"""

import logging
import re
from typing import Any, Dict, List

import requests

import config
from backend.data.chunker import create_chunks_for_record
from backend.ir.bm25 import BM25Retriever
from backend.ir.inverted_index import InvertedIndex
from backend.ir.linkgraph import hits
from backend.ir.text import detect_language

logger = logging.getLogger(__name__)
_SKIP_SECTIONS = re.compile(r"^==+\s*(references|external links|see also|further reading|notes|bibliography|sources|संदर्भ|सन्दर्भ|बाहरी कड़ियाँ)\s*==+\s*$", re.I)


class WikipediaError(RuntimeError):
    pass


def host_for(query: str) -> str:
    return "hi.wikipedia.org" if detect_language(query) == "hindi" else "en.wikipedia.org"


def _api(host: str, **params) -> Dict[str, Any]:
    try:
        res = requests.get(f"https://{host}/w/api.php", params={"action": "query", "format": "json", "formatversion": 2, **params},
                           headers={"User-Agent": config.WIKIPEDIA_USER_AGENT}, timeout=config.WIKIPEDIA_TIMEOUT_SECONDS)
        res.raise_for_status()
        return res.json()
    except (requests.RequestException, ValueError) as exc:
        raise WikipediaError("Wikipedia could not be reached.") from exc


def _clean(extract: str) -> str:
    out, skip = [], False
    for line in extract.splitlines():
        if line.startswith("=="):
            skip = bool(_SKIP_SECTIONS.match(line.strip()))
            continue
        if not skip and line.strip():
            out.append(line.strip())
    return "\n".join(out)


def crawl(query: str) -> List[Dict[str, Any]]:
    """Search Wikipedia and fetch the best pages: [{title, url, text, links}]."""
    host = host_for(query)
    found = _api(host, list="search", srsearch=query, srlimit=config.WIKIPEDIA_MAX_PAGES, srnamespace=0)
    pages = []
    for hit in found.get("query", {}).get("search", []):
        data = _api(host, prop="extracts|links|info|pageprops", titles=hit["title"], explaintext=1, redirects=1,
                    inprop="url", plnamespace=0, pllimit="max", ppprop="disambiguation")
        for pg in data.get("query", {}).get("pages", []):
            if pg.get("missing") or "disambiguation" in pg.get("pageprops", {}) or not pg.get("extract"):
                continue
            url = pg.get("fullurl", "")
            if not url.startswith(f"https://{host}/"):
                continue
            pages.append({"title": pg["title"], "url": url, "text": _clean(pg["extract"]), "links": {l["title"] for l in pg.get("links", [])}})
    return pages


def search(query: str, expanded: str, top_k: int) -> Dict[str, Any]:
    """Crawl, then rank passages with BM25 and break near-ties with HITS authority."""
    pages = crawl(query)
    titles = {p["title"] for p in pages}
    auth, hub = hits({p["title"]: p["links"] & titles for p in pages}, titles) if pages else ({}, {})
    chunks = []
    for i, p in enumerate(pages):
        rec = {"qa_id": f"wiki_{i}", "question": p["title"], "answer": p["text"]}
        for c in create_chunks_for_record(rec):
            # the title is part of each passage so that BM25 sees what page it came from
            chunks.append({**c, "url": p["url"], "origin": "wikipedia"})
    if not chunks:
        return {"results": [], "pages": []}
    index = InvertedIndex()
    index.index_chunks([{**c, "text": f"{c['question']}. {c['text']}"} for c in chunks])
    by_id = {c["chunk_id"]: c for c in chunks}
    res = BM25Retriever(index).search(expanded, top_k=max(10, top_k * 2))
    top = max((r["score"] for r in res), default=1.0) or 1.0
    max_auth = max(auth.values(), default=0.0) or 1.0
    for r in res:
        r["chunk"] = by_id[r["chunk_id"]]
        t = r["chunk"]["question"]
        r["link"] = {"topic": t, "authority": round(auth.get(t, 0.0), 4), "hub": round(hub.get(t, 0.0), 4)}
        r["_o"] = (1 - config.LINK_WEIGHT) * r["score"] / top + config.LINK_WEIGHT * auth.get(t, 0.0) / max_auth
    res.sort(key=lambda r: -r.pop("_o"))
    for i, r in enumerate(res, 1):
        r["rank"] = i
    return {"results": res[:top_k], "pages": [{"title": p["title"], "url": p["url"], "authority": round(auth.get(p["title"], 0.0), 4),
                                               "hub": round(hub.get(p["title"], 0.0), 4)} for p in pages]}
