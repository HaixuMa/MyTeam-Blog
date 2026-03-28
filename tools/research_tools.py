from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Literal

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
            return []

        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
        except Exception as e:
            raise RecoverableHarnessError(f"tavily_tool_import_failed: {e}") from e

        tool = TavilySearchResults(max_results=max_results)
        try:
            results = tool.invoke({"query": query})
        except Exception as e:
            raise RecoverableHarnessError(f"tavily_invoke_failed: {e}") from e

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

    def arxiv_search(self, *, query: str, max_results: int) -> list[Citation]:
        self._require("arxiv")
        try:
            from langchain_community.tools.arxiv.tool import ArxivQueryRun
        except Exception as e:
            raise RecoverableHarnessError(f"arxiv_tool_import_failed: {e}") from e

        tool = ArxivQueryRun()
        try:
            raw = tool.invoke({"query": query, "max_results": max_results})
        except Exception as e:
            raise RecoverableHarnessError(f"arxiv_invoke_failed: {e}") from e

        now = datetime.now(tz=timezone.utc)
        text = str(raw)
        if not text.strip():
            return []

        return [
            Citation(
                source_type="paper",
                title=f"arXiv search: {query}",
                url=f"https://arxiv.org/search/?query={_hash_id(query)}&searchtype=all",
                published_date=None,
                authors=[],
                organization="arXiv",
                accessed_at=now,
                excerpt=text[:1800],
                reliability_score=0.8,
            )
        ]

    def wikipedia_search(self, *, query: str) -> list[Citation]:
        self._require("wikipedia")
        try:
            from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
            from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
        except Exception as e:
            raise RecoverableHarnessError(f"wikipedia_tool_import_failed: {e}") from e

        wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
        tool = WikipediaQueryRun(api_wrapper=wrapper)
        try:
            raw = tool.invoke({"query": query})
        except Exception as e:
            raise RecoverableHarnessError(f"wikipedia_invoke_failed: {e}") from e

        now = datetime.now(tz=timezone.utc)
        text = str(raw)
        if not text.strip():
            return []

        return [
            Citation(
                source_type="wikipedia",
                title=f"Wikipedia: {query}",
                url=f"https://en.wikipedia.org/wiki/{_hash_id(query)}",
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
        except Exception as e:
            raise RecoverableHarnessError(f"web_loader_import_failed: {e}") from e

        loader = WebBaseLoader(web_paths=[url])
        try:
            return loader.load()
        except Exception as e:
            raise RecoverableHarnessError(f"web_loader_failed: {e}") from e


def infer_source_type(url: str) -> Literal["official_doc", "blog", "webpage", "other"]:
    u = url.lower()
    if "docs." in u or "/docs" in u:
        return "official_doc"
    if "blog" in u or "medium.com" in u:
        return "blog"
    if u.startswith("http"):
        return "webpage"
    return "other"

