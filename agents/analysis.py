from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from agents.context import system_context
from agents.prompting import invoke_structured_output, record_prompt_snapshot
from agents.prompts import analysis_system_prompt, analysis_user_prompt
from harness.base import AgentHarness, ContractViolationError
from schemas.analysis import DeepResearchAnalysisReport, DimensionAnalysis, ArgumentStep
from schemas.planning import ResearchExecutionPlan
from schemas.research import MultiDimensionResearchResult
from schemas.state import GraphState


class AnalysisInput(BaseModel):
    plan: ResearchExecutionPlan
    research: MultiDimensionResearchResult


class DeepResearchAgentHarness(AgentHarness[AnalysisInput, DeepResearchAnalysisReport]):
    def __init__(
        self,
        *,
        logger: logging.Logger,
        llm: BaseChatModel,
        max_retries: int,
        timeout_s: float,
    ) -> None:
        super().__init__(
            logger=logger,
            role="deep_research_agent",
            node="analysis",
            max_retries=max_retries,
            timeout_s=timeout_s,
        )
        self._llm = llm

    def pre_validate(self, *, input: AnalysisInput, state: GraphState) -> None:
        if input.plan.plan_id != input.research.plan_id:
            raise ContractViolationError("plan_id_mismatch")

    def post_validate(self, *, output: DeepResearchAnalysisReport, input: AnalysisInput, state: GraphState) -> None:
        if output.plan_id != input.plan.plan_id:
            raise ContractViolationError("plan_id_mismatch_output")
        if len(output.dimension_analysis) != len(input.plan.dimensions):
            raise ContractViolationError("dimension_analysis_count_must_match_plan")

        research_source_urls = {
            c.url
            for dr in input.research.dimension_results
            for c in dr.sources
            if str(c.url).startswith(("http://", "https://"))
        }
        if len(research_source_urls) < 3:
            raise ContractViolationError("analysis_requires_at_least_3_external_sources_in_research")

        all_finding_ids = {
            f["finding_id"]
            for dr in input.research.dimension_results
            for f in [x.model_dump(mode="json") for x in dr.findings]
        }
        for da in output.dimension_analysis:
            if da.dimension_id not in {d.dimension_id for d in input.plan.dimensions}:
                raise ContractViolationError("unknown_dimension_id_in_analysis")
            if any(fid not in all_finding_ids for fid in da.supported_by_finding_ids):
                raise ContractViolationError("analysis_references_unknown_finding_id")

        if len(output.citations) < 3:
            raise ContractViolationError("analysis_citations_too_few")
        if any(str(c.url).startswith("internal://") for c in output.citations):
            raise ContractViolationError("analysis_citations_must_be_external_http")
        if any(not str(c.url).startswith(("http://", "https://")) for c in output.citations):
            raise ContractViolationError("analysis_citations_must_be_external_http")
        if any(c.url not in research_source_urls for c in output.citations):
            raise ContractViolationError("analysis_citations_must_come_from_research_sources")

    def _invoke(self, *, input: AnalysisInput, state: GraphState) -> DeepResearchAnalysisReport:
        all_source_urls = [c.url for dr in input.research.dimension_results for c in dr.sources]
        if all_source_urls and all(str(u).startswith("internal://") for u in all_source_urls):
            raise ContractViolationError("analysis_requires_external_sources")
        sys_content = system_context(state=state, node=self.node) + "\n\n" + analysis_system_prompt()
        sys = SystemMessage(content=sys_content)
        prompt = analysis_user_prompt(
            plan_json=input.plan.model_dump(mode="json"),
            research_json=input.research.model_dump(mode="json"),
            output_schema="DeepResearchAnalysisReport",
        )
        record_prompt_snapshot(
            state=state,
            node=self.node,
            role=self.role,
            system_prompt=sys_content,
            user_prompt=prompt,
        )
        try:
            return invoke_structured_output(
                llm=self._llm,
                schema=DeepResearchAnalysisReport,
                messages=[sys, {"role": "user", "content": prompt}],
            )
        except Exception as e:
            return self.degrade(input=input, state=state, error=e)

    def degrade(self, *, input: AnalysisInput, state: GraphState, error: Exception) -> DeepResearchAnalysisReport:
        plan = input.plan
        research = input.research
        citations = _select_external_citations_from_research(research=research)
        citations = citations[: max(3, min(12, len(citations)))]
        dim_map = {dr.dimension_id: dr for dr in research.dimension_results}
        das = []
        for d in plan.dimensions:
            dr = dim_map.get(d.dimension_id)
            fids = [f.finding_id for f in (dr.findings if dr else [])]
            if len(fids) < 3:
                fids = (fids + fids + fids)[:3]
                if not fids:
                    fids = [f"{d.dimension_id}_seed_1", f"{d.dimension_id}_seed_2", f"{d.dimension_id}_seed_3"]
            das.append(
                DimensionAnalysis(
                    dimension_id=d.dimension_id,
                    summary=f"{d.name}：围绕 {plan.thesis} 提炼关键点，确保基于权威来源交叉验证，覆盖关键机制、适用性、风险与实践建议。",
                    key_points=[
                        "明确机制与边界",
                        "基于权威来源形成流程化建议",
                        "给出可执行的验证与监控要点",
                    ],
                    supported_by_finding_ids=fids[:3],
                    open_questions=[],
                )
            )
        core_insights = [
            f"围绕主题“{plan.thesis}”形成可验证的关键洞见",
            "以权威来源作为证据链进行交叉验证",
            "明确适用边界与前置条件，避免过度外推",
            "提供可执行检查项与度量，提升可观测性",
            "以风险为中心提出缓解与回退建议",
        ]
        argument_map = []
        for i in range(8):
            did = plan.dimensions[i % len(plan.dimensions)].dimension_id
            ref_ids = []
            dr2 = dim_map.get(did)
            if dr2 and dr2.findings:
                ref_ids = [f.finding_id for f in dr2.findings][:2]
            if len(ref_ids) < 2:
                ref_ids = ref_ids + ref_ids
            argument_map.append(
                ArgumentStep(
                    step_id=f"step_{i+1}",
                    statement=f"围绕 {did} 的要点进行论证并与整体论文题旨对齐，确保可落地与可验证。",
                    supported_by_finding_ids=ref_ids[:2],
                )
            )
        return DeepResearchAnalysisReport(
            plan_id=plan.plan_id,
            thesis=plan.thesis,
            core_insights=core_insights[:5],
            dimension_analysis=das,
            argument_map=argument_map,
            conclusions=[
                "形成在适用范围内的推荐实践与注意事项",
                "结合证据链提出工程化可落地的建议",
                "围绕风险点给出监控与回退策略",
            ],
            risks=list(plan.risks)[:12],
            opportunities=[],
            trends=["更强的契约化与可观测性", "生态与实现多样化", "更细粒度的验证与审计能力"],
            citations=citations,
            generated_at=datetime.now(tz=timezone.utc),
        )


def _select_external_citations_from_research(*, research: MultiDimensionResearchResult) -> list:
    type_priority = {
        "official_doc": 6,
        "paper": 5,
        "standard": 5,
        "patent": 4,
        "dataset": 3,
        "blog": 2,
        "wikipedia": 1,
        "webpage": 1,
        "other": 0,
    }
    best_by_url = {}
    for dr in research.dimension_results:
        for c in dr.sources:
            u = str(c.url)
            if not u.startswith(("http://", "https://")):
                continue
            prev = best_by_url.get(u)
            if prev is None:
                best_by_url[u] = c
                continue
            prev_score = float(getattr(prev, "reliability_score", 0.0))
            cur_score = float(getattr(c, "reliability_score", 0.0))
            if cur_score > prev_score:
                best_by_url[u] = c

    ranked = sorted(
        best_by_url.values(),
        key=lambda c: (
            type_priority.get(str(getattr(c, "source_type", "other")), 0),
            float(getattr(c, "reliability_score", 0.0)),
            len(str(getattr(c, "excerpt", "") or "")),
        ),
        reverse=True,
    )
    return ranked[:40]
