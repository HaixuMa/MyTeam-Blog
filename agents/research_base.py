from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from agents.prompting import sanitize_user_text
from harness.base import AgentHarness, ContractViolationError
from schemas.common import Citation, ResearchFinding
from schemas.planning import ResearchExecutionPlan
from schemas.research import DimensionResearchResult, ValidationResult
from schemas.research_partials import ResearchPartialBatch
from schemas.state import GraphState
from tools.research_tools import ResearchToolbox


class _FindingsOutput(BaseModel):
    findings: list[ResearchFinding] = Field(default_factory=list, max_length=30)
    notes: str = Field(default="", max_length=1200)


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


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

        sys = SystemMessage(
            content=(
                "你是深度研究团队的信息采集子 Agent。"
                "你必须只基于提供的 sources（包含 title/url/excerpt）生成结构化 findings。"
                "每条 finding 必须至少引用 1 条 citation，citation.url 必须来自 sources.url。"
                "禁止编造来源。若 sources 不足以支撑结论，必须降低 confidence 或提出 open 问题。"
            )
        )
        structured = self._llm.with_structured_output(_FindingsOutput)

        results: list[DimensionResearchResult] = []
        for dim in input.dimensions:
            dim_id = dim.dimension_id
            query = self._build_query(input=input, dimension_name=dim.name, key_questions=dim.key_questions)
            sources = self._collect_sources(query=query, max_sources=max_sources)

            prompt = (
                f"研究主题：{sanitize_user_text(input.thesis)}\n"
                f"当前维度：{dim.name} ({dim.dimension_id})\n"
                f"关键问题：{dim.key_questions}\n"
                f"验收标准：{dim.acceptance_criteria}\n\n"
                f"可用 sources（只能引用这些）：{[c.model_dump(mode='json') for c in sources]}\n\n"
                "请输出 findings（每条包含 claim/evidence/citations/confidence/tags）。"
            )
            out = structured.invoke([sys, {"role": "user", "content": prompt}])

            normalized_findings: list[ResearchFinding] = []
            url_map = {c.url: c for c in sources}
            for i, f in enumerate(out.findings):
                used_citations: list[Citation] = []
                for c in f.citations:
                    if c.url in url_map:
                        used_citations.append(url_map[c.url])
                if not used_citations:
                    continue
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

    def _collect_sources(self, *, query: str, max_sources: int) -> list[Citation]:
        max_sources = max(3, min(20, max_sources))
        sources: list[Citation] = []

        if self.agent_type == "academic":
            sources.extend(self._toolbox.arxiv_search(query=query, max_results=min(5, max_sources)))
            sources.extend(self._toolbox.tavily_search(query=query, max_results=max_sources))
        elif self.agent_type == "tech":
            sources.extend(self._toolbox.tavily_search(query=query, max_results=max_sources))
            sources.extend(self._toolbox.wikipedia_search(query=query))
        elif self.agent_type == "industry":
            sources.extend(self._toolbox.tavily_search(query=query, max_results=max_sources))
        else:
            sources.extend(self._toolbox.tavily_search(query=query, max_results=max_sources))

        dedup: dict[str, Citation] = {}
        for c in sources:
            dedup[c.url] = c
        return list(dedup.values())[:max_sources]


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

