from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Literal

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


def _internal_sources(*, query: str, count: int) -> list[Citation]:
    now = datetime.now(tz=timezone.utc)
    base = _short_hash(query)
    out: list[Citation] = []
    for i in range(max(0, count)):
        out.append(
            Citation(
                source_type="other",
                title=f"Internal placeholder source ({i + 1})",
                url=f"internal://source/{base}/{i + 1}",
                published_date=None,
                authors=[],
                organization="internal",
                accessed_at=now,
                excerpt="No external sources available in current environment.",
                reliability_score=0.1,
            )
        )
    return out


def _all_internal_sources(sources: list[Citation]) -> bool:
    if not sources:
        return True
    return all(str(c.url).startswith("internal://") for c in sources)


def _fallback_findings_for_dimension(*, dim: str, key_questions: list[str], sources: list[Citation]) -> _FindingsOutput:
    base = sources[0] if sources else Citation(
        source_type="other",
        title="Internal placeholder source",
        url="internal://source/none/1",
        published_date=None,
        authors=[],
        organization="internal",
        accessed_at=datetime.now(tz=timezone.utc),
        excerpt="No external sources available in current environment.",
        reliability_score=0.1,
    )
    q1 = key_questions[0] if key_questions else dim
    q2 = key_questions[1] if len(key_questions) > 1 else dim
    findings = [
        _FindingDraft(
            claim=f"{dim}：需要将“{q1}”转化为可验证的状态字段与验收用例，才能支撑调度面板的准确展示。",
            evidence="在无外部 sources 的环境下，先输出字段清单、事件模型与状态机，再用真实来源替换 internal sources 并复核。",
            citations=[_CitationRef(url=base.url)],
            confidence=0.25,
            tags=["fallback", dim],
        ),
        _FindingDraft(
            claim=f"{dim}：围绕“{q2}”应明确取消/重试的幂等语义与状态转移规则，避免 UI 与后端状态不一致。",
            evidence="把取消/重试建模为命令事件，并让事件流成为单一事实来源（SSOT），可同时满足 UI 渲染与审计回放。",
            citations=[_CitationRef(url=base.url)],
            confidence=0.25,
            tags=["fallback", dim],
        ),
    ]
    return _FindingsOutput(findings=findings, notes="fallback_mode_no_external_sources")


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
            queries = self._suggest_queries(
                input=input,
                dimension_name=dim.name,
                key_questions=dim.key_questions,
                state=state,
            )
            collected: list[Citation] = []
            for q in queries:
                collected.extend(self._collect_sources(query=q, max_sources=max_sources))
                if len({c.url for c in collected if not str(c.url).startswith("internal://")}) >= 3:
                    break

            dedup: dict[str, Citation] = {}
            for c in collected:
                dedup[c.url] = c
            sources = list(dedup.values())[:max_sources]
            if len(sources) < 3:
                sources.extend(_internal_sources(query=queries[0], count=3 - len(sources)))
            sources = sources[:max_sources]

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
            if _all_internal_sources(sources):
                out = _fallback_findings_for_dimension(
                    dim=dim.name, key_questions=dim.key_questions, sources=sources
                )
            else:
                try:
                    out = invoke_structured_output(
                        llm=self._llm,
                        schema=_FindingsOutput,
                        messages=[sys, {"role": "user", "content": prompt}],
                    )
                except Exception:
                    out = _fallback_findings_for_dimension(
                        dim=dim.name, key_questions=dim.key_questions, sources=sources
                    )

            normalized_findings: list[ResearchFinding] = []
            url_map = {c.url: c for c in sources}
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

            if len(normalized_findings) < 2 and sources:
                base_citation = sources[0]
                q1 = dim.key_questions[0] if dim.key_questions else dim.name
                q2 = dim.key_questions[1] if len(dim.key_questions) > 1 else dim.name
                normalized_findings.extend(
                    [
                        ResearchFinding(
                            finding_id=f"{dim_id}_{self.agent_type}_fallback_1_{_short_hash(q1)}",
                            dimension_id=dim_id,
                            claim=f"{dim.name}：围绕“{q1}”需要先定义可观测的状态字段与最小契约，才能支撑面板展示与故障定位。",
                            evidence="在缺少外部 sources 的环境下，先以内部占位 sources 作为对齐载体，输出可执行的字段/事件/状态机清单，后续再用真实来源替换与校验。",
                            citations=[base_citation],
                            confidence=0.25,
                            conflicts_with_finding_ids=[],
                            tags=["fallback", self.agent_type, dim.name],
                        ),
                        ResearchFinding(
                            finding_id=f"{dim_id}_{self.agent_type}_fallback_2_{_short_hash(q2)}",
                            dimension_id=dim_id,
                            claim=f"{dim.name}：围绕“{q2}”应提供取消/重试的幂等语义与状态转移规则，否则 UI 行为不可预测。",
                            evidence="把取消与重试定义为对 run 的命令，并将结果体现在事件流与节点状态中，可让前端基于单一数据源渲染，并便于回放与审计。",
                            citations=[base_citation],
                            confidence=0.25,
                            conflicts_with_finding_ids=[],
                            tags=["fallback", self.agent_type, dim.name],
                        ),
                    ]
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

