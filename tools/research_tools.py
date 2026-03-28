from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from langchain_core.documents import Document

from harness.base import PermissionDeniedError, RecoverableHarnessError
from schemas.common import Citation
from tools.rate_limit import TokenBucket


def _hash_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class ResearchToolbox:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        allowed_tools: list[str],
        rate_limit_per_minute: int,
        tavily_api_key_present: bool,
    ) -> None:
        self._logger = logger
        self._allowed = set(allowed_tools)
        self._bucket = TokenBucket.per_minute(limit=rate_limit_per_minute)
        self._tavily_key_present = tavily_api_key_present

    def _require(self, tool_name: str) -> None:
        if tool_name not in self._allowed:
            raise PermissionDeniedError(f"tool_not_allowed: {tool_name}")
        if not self._bucket.consume(tokens=1):
            raise RecoverableHarnessError("tool_rate_limited")

    def tavily_search(self, *, query: str, max_results: int) -> list[Citation]:
        self._require("tavily")
        if not self._tavily_key_present:
            return self._github_search(query=query, max_results=max_results)

        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
        except Exception:
            return self._github_search(query=query, max_results=max_results)

        try:
            tool = TavilySearchResults(max_results=max_results)
        except Exception:
            return self._github_search(query=query, max_results=max_results)
        try:
            results = tool.invoke({"query": query})
        except Exception:
            return self._github_search(query=query, max_results=max_results)

        citations: list[Citation] = []
        now = datetime.now(tz=timezone.utc)
        for r in results if isinstance(results, list) else []:
            title = str(r.get("title") or r.get("content") or "Tavily result")[:400]
            url = str(r.get("url") or "").strip()
            if not url:
                continue
            citations.append(
                Citation(
                    source_type="webpage",
                    title=title,
                    url=url,
                    published_date=None,
                    authors=[],
                    organization=None,
                    accessed_at=now,
                    excerpt=str(r.get("content") or "")[:1800] or None,
                    reliability_score=0.55,
                )
            )
        return citations

    def _github_search(self, *, query: str, max_results: int) -> list[Citation]:
        max_results = max(1, min(10, int(max_results)))
        q_raw = query.strip()
        if not q_raw:
            return []

        def _fetch(q_text: str) -> list[dict]:
            q = quote_plus(q_text)
            url = f"https://api.github.com/search/repositories?q={q}&per_page={max_results}"
            req = Request(
                url,
                headers={
                    "User-Agent": "myteam-blog/0.1 (+local dev)",
                    "Accept": "application/vnd.github+json",
                },
                method="GET",
            )
            try:
                with urlopen(req, timeout=12) as resp:
                    raw = resp.read().decode("utf-8", errors="ignore")
            except Exception:
                return []
            try:
                payload = json.loads(raw)
            except Exception:
                return []
            items = payload.get("items") if isinstance(payload, dict) else None
            return items if isinstance(items, list) else []

        items = _fetch(q_raw)
        if not items:
            ascii_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]{2,}", q_raw)
            ascii_q = " ".join(ascii_tokens)
            if ascii_q and ascii_q.lower() != q_raw.lower():
                items = _fetch(ascii_q)
        if not items:
            return []

        now = datetime.now(tz=timezone.utc)
        out: list[Citation] = []
        for it in items[:max_results]:
            if not isinstance(it, dict):
                continue
            u = str(it.get("html_url") or "").strip()
            if not u:
                continue
            full_name = str(it.get("full_name") or "GitHub repository").strip()
            desc = str(it.get("description") or "").strip()
            org = None
            owner = it.get("owner")
            if isinstance(owner, dict):
                org = str(owner.get("login") or "").strip() or None
            official_orgs = {"microsoft", "google", "meta", "langchain-ai", "langchain"}
            is_official = bool(org and org.lower() in official_orgs)
            out.append(
                Citation(
                    source_type="official_doc" if is_official else "webpage",
                    title=full_name[:400],
                    url=u,
                    published_date=None,
                    authors=[],
                    organization=org or "GitHub",
                    accessed_at=now,
                    excerpt=desc[:1800] or None,
                    reliability_score=0.75 if is_official else 0.6,
                )
            )
        return out

    def _wikipedia_opensearch(self, *, query: str, max_results: int) -> list[Citation]:
        max_results = max(1, min(10, int(max_results)))
        q = quote_plus(query.strip()) if query.strip() else ""
        if not q:
            return []
        url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={q}&limit={max_results}&namespace=0&format=json"
        req = Request(
            url,
            headers={
                "User-Agent": "myteam-blog/0.1 (+local dev)",
                "Accept": "application/json",
            },
            method="GET",
        )
        try:
            with urlopen(req, timeout=12) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
        except Exception:
            return []
        try:
            payload = json.loads(raw)
        except Exception:
            return []
        if not isinstance(payload, list) or len(payload) < 4:
            return []

        titles = payload[1] if isinstance(payload[1], list) else []
        descs = payload[2] if isinstance(payload[2], list) else []
        urls = payload[3] if isinstance(payload[3], list) else []

        now = datetime.now(tz=timezone.utc)
        out: list[Citation] = []
        for i in range(min(len(titles), len(urls), max_results)):
            t = str(titles[i])[:400].strip() or f"Wikipedia: {query}"
            u = str(urls[i]).strip()
            if not (u.startswith("http://") or u.startswith("https://")):
                continue
            excerpt = str(descs[i])[:1800].strip() if i < len(descs) else None
            out.append(
                Citation(
                    source_type="wikipedia",
                    title=t,
                    url=u,
                    published_date=None,
                    authors=[],
                    organization="Wikipedia",
                    accessed_at=now,
                    excerpt=excerpt or None,
                    reliability_score=0.65,
                )
            )
        return out

    def arxiv_search(self, *, query: str, max_results: int) -> list[Citation]:
        self._require("arxiv")
        out = self._arxiv_api_search(query=query, max_results=max_results)
        if out:
            return out
        return self._github_search(query=query, max_results=max_results)

    def _arxiv_api_search(self, *, query: str, max_results: int) -> list[Citation]:
        max_results = max(1, min(10, int(max_results)))
        q = quote_plus(query.strip()) if query.strip() else ""
        if not q:
            return []
        url = f"https://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results={max_results}"
        req = Request(
            url,
            headers={
                "User-Agent": "myteam-blog/0.1 (+local dev)",
                "Accept": "application/atom+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            method="GET",
        )
        try:
            with urlopen(req, timeout=12) as resp:
                raw = resp.read()
        except Exception:
            return []
        try:
            root = ElementTree.fromstring(raw)
        except Exception:
            return []

        ns = {"a": "http://www.w3.org/2005/Atom"}
        now = datetime.now(tz=timezone.utc)
        out: list[Citation] = []
        for entry in root.findall("a:entry", ns):
            title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
            published = (entry.findtext("a:published", default="", namespaces=ns) or "").strip()
            authors = [
                ((a.findtext("a:name", default="", namespaces=ns) or "").strip())
                for a in entry.findall("a:author", ns)
            ]
            authors = [a for a in authors if a][:12]

            link = ""
            for l in entry.findall("a:link", ns):
                if l.attrib.get("rel") == "alternate" and l.attrib.get("href"):
                    link = str(l.attrib["href"])
                    break
            if not link:
                continue

            pub_date = None
            try:
                pub_date = datetime.fromisoformat(published.replace("Z", "+00:00")).date() if published else None
            except Exception:
                pub_date = None

            out.append(
                Citation(
                    source_type="paper",
                    title=title[:400] or f"arXiv: {query}",
                    url=link,
                    published_date=pub_date,
                    authors=authors,
                    organization="arXiv",
                    accessed_at=now,
                    excerpt=summary[:1800] if summary else None,
                    reliability_score=0.75,
                )
            )
        return out

    def wikipedia_search(self, *, query: str) -> list[Citation]:
        self._require("wikipedia")
        opensearch = self._wikipedia_opensearch(query=query, max_results=3)
        if opensearch:
            return opensearch
        try:
            from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
            from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
        except Exception:
            return self._github_search(query=query, max_results=3)

        try:
            wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
            tool = WikipediaQueryRun(api_wrapper=wrapper)
        except Exception:
            return self._github_search(query=query, max_results=3)
        try:
            raw = tool.invoke({"query": query})
        except Exception:
            return self._github_search(query=query, max_results=3)

        now = datetime.now(tz=timezone.utc)
        text = str(raw)
        if not text.strip():
            return []

        return [
            Citation(
                source_type="wikipedia",
                title=f"Wikipedia: {query}",
                url="https://en.wikipedia.org/",
                published_date=None,
                authors=[],
                organization="Wikipedia",
                accessed_at=now,
                excerpt=text[:1800],
                reliability_score=0.5,
            )
        ]

    def web_load(self, *, url: str) -> list[Document]:
        self._require("web_loader")
        try:
            from langchain_community.document_loaders.web_base import WebBaseLoader
        except Exception:
            return []

        loader = WebBaseLoader(web_paths=[url])
        try:
            return loader.load()
        except Exception:
            return []


def infer_source_type(url: str) -> Literal["official_doc", "blog", "webpage", "other"]:
    u = url.lower()
    if "docs." in u or "/docs" in u:
        return "official_doc"
    if "blog" in u or "medium.com" in u:
        return "blog"
    if u.startswith("http"):
        return "webpage"
    return "other"
