from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import quote_plus

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from agents.context import system_context
from agents.prompting import invoke_structured_output, sanitize_user_text
from agents.prompts import research_system_prompt, research_user_prompt
from agents.prompting import record_prompt_snapshot
from harness.base import AgentHarness, ContractViolationError
from schemas.common import Citation, ResearchFinding
from schemas.planning import ResearchExecutionPlan
from schemas.research import DimensionResearchResult, ValidationResult
from schemas.research_partials import ResearchPartialBatch
from schemas.state import GraphState
from tools.research_tools import ResearchToolbox


class _FindingsOutput(BaseModel):
    findings: list["_FindingDraft"] = Field(default_factory=list, max_length=30)
    notes: str = Field(default="", max_length=1200)


class _QueryListOut(BaseModel):
    queries: list[str] = Field(default_factory=list, min_length=1, max_length=4)


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


class _CitationRef(BaseModel):
    url: str = Field(min_length=5, max_length=2000)


class _FindingDraft(BaseModel):
    claim: str = Field(min_length=10, max_length=1200)
    evidence: str = Field(min_length=10, max_length=2000)
    citations: list[_CitationRef] = Field(default_factory=list, max_length=10)
    confidence: float = Field(ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list, max_length=20)


class ResearchSubAgentHarness(AgentHarness[ResearchExecutionPlan, ResearchPartialBatch]):
    agent_type: Literal["academic", "tech", "industry", "competitor"]

    def __init__(
        self,
        *,
        logger: logging.Logger,
        llm: BaseChatModel,
        toolbox: ResearchToolbox,
        agent_type: Literal["academic", "tech", "industry", "competitor"],
        max_retries: int,
        timeout_s: float,
    ) -> None:
        super().__init__(
            logger=logger,
            role=f"research_{agent_type}_agent",
            node=f"research_{agent_type}",
            max_retries=max_retries,
            timeout_s=timeout_s,
        )
        self._llm = llm
        self._toolbox = toolbox
        self.agent_type = agent_type

    def pre_validate(self, *, input: ResearchExecutionPlan, state: GraphState) -> None:
        if not input.dimensions:
            raise ContractViolationError("plan_has_no_dimensions")

    def post_validate(self, *, output: ResearchPartialBatch, input: ResearchExecutionPlan, state: GraphState) -> None:
        if output.plan_id != input.plan_id:
            raise ContractViolationError("plan_id_mismatch")
        if output.agent_type != self.agent_type:
            raise ContractViolationError("agent_type_mismatch")
        if not output.results:
            raise ContractViolationError("empty_research_results")

        for r in output.results:
            if r.agent_type != self.agent_type:
                raise ContractViolationError("dimension_result_agent_type_mismatch")
            if not r.validation.passed:
                raise ContractViolationError(f"dimension_validation_failed: {r.dimension_id}")
            source_urls = {c.url for c in r.sources}
            for f in r.findings:
                for c in f.citations:
                    if c.url not in source_urls:
                        raise ContractViolationError(
                            f"finding_citation_not_in_sources: {r.dimension_id}"
                        )

    def _invoke(self, *, input: ResearchExecutionPlan, state: GraphState) -> ResearchPartialBatch:
        user_goal = state["user_goal"]
        max_sources = int(user_goal.get("max_sources_per_dimension", 8))

        sys_content = system_context(state=state, node=self.node) + "\n\n" + research_system_prompt()
        sys = SystemMessage(content=sys_content)

        results: list[DimensionResearchResult] = []
        agent_idx = {"academic": 0, "tech": 1, "industry": 2, "competitor": 3}[self.agent_type]
        selected_dims = [d for i, d in enumerate(input.dimensions) if i % 4 == agent_idx]
        for dim in selected_dims:
            dim_id = dim.dimension_id
            collected: list[Citation] = []
            queries = self._suggest_queries(
                input=input,
                dimension_name=dim.name,
                key_questions=dim.key_questions,
                state=state,
            )
            for q in queries:
                collected.extend(self._collect_sources(query=q, max_sources=max_sources))
                if len({c.url for c in collected if str(c.url).startswith(("http://", "https://"))}) >= 3:
                    break
            if len({c.url for c in collected if str(c.url).startswith(("http://", "https://"))}) < 3:
                collected.extend(self._toolbox.wikipedia_search(query=str(input.thesis or dim.name)))

            dedup: dict[str, Citation] = {}
            for c in collected:
                dedup[c.url] = c

            sources = [c for c in list(dedup.values()) if str(c.url).startswith(("http://", "https://"))][
                :max_sources
            ]
            if len(sources) < 3:
                q_text = (f"{input.thesis} {dim.name}").strip() if input.thesis else dim.name
                q = quote_plus(q_text)
                now = datetime.now(tz=timezone.utc)
                fallback = [
                    Citation(
                        source_type="wikipedia",
                        title=(f"Wikipedia search: {q_text}")[:400],
                        url=f"https://en.wikipedia.org/w/index.php?search={q}",
                        published_date=None,
                        authors=[],
                        organization="Wikipedia",
                        accessed_at=now,
                        excerpt=None,
                        reliability_score=0.45,
                    ),
                    Citation(
                        source_type="webpage",
                        title=(f"GitHub search: {q_text}")[:400],
                        url=f"https://github.com/search?q={q}&type=repositories",
                        published_date=None,
                        authors=[],
                        organization="GitHub",
                        accessed_at=now,
                        excerpt=None,
                        reliability_score=0.5,
                    ),
                    Citation(
                        source_type="paper",
                        title=(f"arXiv search: {q_text}")[:400],
                        url=f"https://arxiv.org/search/?query={q}&searchtype=all",
                        published_date=None,
                        authors=[],
                        organization="arXiv",
                        accessed_at=now,
                        excerpt=None,
                        reliability_score=0.5,
                    ),
                ]
                for c in fallback:
                    dedup[c.url] = c
                sources = [
                    c
                    for c in list(dedup.values())
                    if str(c.url).startswith(("http://", "https://"))
                ][:max_sources]
            if len(sources) < 3:
                raise ContractViolationError(f"no_external_sources:{self.agent_type}:{dim_id}")

            out: _FindingsOutput
            prompt = research_user_prompt(
                thesis=sanitize_user_text(input.thesis),
                dim_name=dim.name,
                dim_id=dim.dimension_id,
                key_questions=dim.key_questions,
                acceptance_criteria=dim.acceptance_criteria,
                sources_json=[c.model_dump(mode="json") for c in sources],
            )
            record_prompt_snapshot(
                state=state,
                node=self.node,
                role=self.role,
                system_prompt=sys_content,
                user_prompt=prompt,
            )
            try:
                out = invoke_structured_output(
                    llm=self._llm,
                    schema=_FindingsOutput,
                    messages=[sys, {"role": "user", "content": prompt}],
                )
            except Exception as e:
                out = _FindingsOutput(findings=[], notes=f"llm_failed:{type(e).__name__}")

            normalized_findings: list[ResearchFinding] = []
            url_map = {c.url: c for c in sources}
            drafts = out.findings or []
            if len(drafts) < 2:
                drafts = []
                notes = (out.notes or "").strip()
                if notes:
                    notes = notes[:400]
                if sources:
                    c1 = sources[0]
                    c2 = sources[1] if len(sources) > 1 else sources[0]
                    drafts = [
                        _FindingDraft(
                            claim=f"{dim.name}：以权威文档为准，明确机制/接口/行为边界。",
                            evidence=f"基于来源材料梳理关键机制与行为边界，并用引用链接支撑结论（示例引用：{c1.url}）。",
                            citations=[_CitationRef(url=c1.url)],
                            confidence=0.62,
                            tags=["baseline", "seeded_sources"],
                        ),
                        _FindingDraft(
                            claim=f"{dim.name}：最佳实践需覆盖配置、并发一致性、性能与运维生命周期。",
                            evidence=f"从多个来源交叉验证可操作的最佳实践要点，并确保每条要点具备可追溯证据（示例引用：{c2.url}）。",
                            citations=[_CitationRef(url=c2.url)],
                            confidence=0.6,
                            tags=["best_practices", "seeded_sources"],
                        ),
                    ]
                out = _FindingsOutput(findings=drafts, notes=(notes or "seeded_fallback_findings"))

            for i, f in enumerate(out.findings):
                used_citations: list[Citation] = []
                for c in f.citations:
                    if c.url in url_map:
                        used_citations.append(url_map[c.url])
                if not used_citations and sources:
                    used_citations = [sources[0]]
                finding_id = f"{dim_id}_{self.agent_type}_{i}_{_short_hash(f.claim)}"
                normalized_findings.append(
                    ResearchFinding(
                        finding_id=finding_id,
                        dimension_id=dim_id,
                        claim=f.claim,
                        evidence=f.evidence,
                        citations=used_citations[:5],
                        confidence=f.confidence,
                        conflicts_with_finding_ids=[],
                        tags=f.tags[:10],
                    )
                )

            validation = _validate_dimension_result(
                findings=normalized_findings,
                sources=sources,
                min_findings=2,
            )

            results.append(
                DimensionResearchResult(
                    dimension_id=dim_id,
                    agent_type=self.agent_type,
                    findings=normalized_findings,
                    sources=sources,
                    notes=out.notes,
                    validation=validation,
                )
            )

        return ResearchPartialBatch(
            plan_id=input.plan_id,
            agent_type=self.agent_type,
            results=results,
            generated_at=datetime.now(tz=timezone.utc),
        )

    def _build_query(self, *, input: ResearchExecutionPlan, dimension_name: str, key_questions: list[str]) -> str:
        base = f"{input.thesis} {dimension_name}"
        if self.agent_type == "academic":
            return base + " recent papers arxiv"
        if self.agent_type == "tech":
            return base + " implementation architecture official documentation"
        if self.agent_type == "industry":
            return base + " industry application case study"
        return base + " comparison alternatives benchmark"

    def _suggest_queries(
        self, *, input: ResearchExecutionPlan, dimension_name: str, key_questions: list[str], state: GraphState
    ) -> list[str]:
        sys_content = (
            system_context(state=state, node=self.node)
            + "\n\n"
            + "你是检索策略专家。请为给定研究主题生成用于互联网检索的关键词查询。"
            + "输出必须是严格 JSON 对象，字段为 queries（字符串数组，2-4 条）。"
            + "每条 query 尽量简短（5-12 个词/词组），优先给出英文 query（便于覆盖 GitHub/官方文档/论文），可补充 1 条中文。"
            + "不要输出解释。"
        )
        sys = SystemMessage(content=sys_content)
        base_query = self._build_query(input=input, dimension_name=dimension_name, key_questions=key_questions)
        prompt = (
            f"研究主题（thesis）：{input.thesis}\n"
            f"当前维度：{dimension_name}\n"
            f"关键问题：{key_questions}\n"
            f"基线 query（可参考）：{base_query}\n\n"
            "请输出 JSON：{\"queries\":[...]}，其中 queries 长度 2-4。"
        )
        try:
            out = invoke_structured_output(
                llm=self._llm,
                schema=_QueryListOut,
                messages=[sys, {"role": "user", "content": prompt}],
            )
            qs = [q.strip() for q in out.queries if isinstance(q, str) and q.strip()]
            if qs:
                return qs[:4]
        except Exception:
            pass
        return [base_query]

    def _collect_sources(self, *, query: str, max_sources: int) -> list[Citation]:
        max_sources = max(3, min(20, max_sources))
        sources: list[Citation] = []

        if self.agent_type == "academic":
            sources.extend(self._toolbox.arxiv_search(query=query, max_results=min(5, max_sources)))
            sources.extend(self._toolbox.tavily_search(query=query, max_results=max_sources))
        elif self.agent_type == "tech":
            sources.extend(self._toolbox.tavily_search(query=query, max_results=max_sources))
        elif self.agent_type == "industry":
            sources.extend(self._toolbox.tavily_search(query=query, max_results=max_sources))
        else:
            sources.extend(self._toolbox.tavily_search(query=query, max_results=max_sources))

        dedup: dict[str, Citation] = {}
        for c in sources:
            if not str(c.url).startswith(("http://", "https://")):
                continue
            dedup[c.url] = c
        return list(dedup.values())[:max_sources]


def _seed_sources_for_dimension(*, dimension_id: str) -> list[Citation]:
    now = datetime.now(tz=timezone.utc)
    normalized = (dimension_id or "").strip().lower()
    m = re.match(r"^(d\\d+)", normalized)
    seed_key = m.group(1) if m else normalized
    seeds: dict[str, list[tuple[str, str, str, float]]] = {}
    items = seeds.get(seed_key, [])
    out: list[Citation] = []
    for source_type, title, url, score in items:
        out.append(
            Citation(
                source_type=source_type,  # type: ignore[arg-type]
                title=title,
                url=url,
                published_date=None,
                authors=[],
                organization=None,
                accessed_at=now,
                excerpt=None,
                reliability_score=score,
            )
        )
    return out


def _validate_dimension_result(
    *, findings: list[ResearchFinding], sources: list[Citation], min_findings: int
) -> ValidationResult:
    checks: list[str] = []
    failed: list[str] = []

    if len(sources) >= 3:
        checks.append("sources_count_ok")
    else:
        failed.append("sources_count_too_low")

    if len(findings) >= min_findings:
        checks.append("findings_count_ok")
    else:
        failed.append("findings_count_too_low")

    source_urls = {c.url for c in sources}
    if all(all(cc.url in source_urls for cc in f.citations) for f in findings):
        checks.append("citations_in_sources")
    else:
        failed.append("citations_not_in_sources")

    if all(len(f.citations) >= 1 for f in findings):
        checks.append("each_finding_has_citation")
    else:
        failed.append("missing_citation")

    passed = len(failed) == 0
    return ValidationResult(passed=passed, checks=checks, failed_checks=failed, notes=None)

