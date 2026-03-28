from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from agents.context import system_context
from agents.prompting import record_prompt_snapshot
from agents.prompts import analysis_system_prompt, analysis_user_prompt
from harness.base import AgentHarness, ContractViolationError
from schemas.analysis import DeepResearchAnalysisReport
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

    def _invoke(self, *, input: AnalysisInput, state: GraphState) -> DeepResearchAnalysisReport:
        sys_content = system_context(state=state, node=self.node) + "\n\n" + analysis_system_prompt()
        sys = SystemMessage(content=sys_content)
        structured = self._llm.with_structured_output(DeepResearchAnalysisReport)

        prompt = analysis_user_prompt(
            plan_json=input.plan.model_dump(mode="json"),
            research_json=input.research.model_dump(mode="json"),
            output_schema=DeepResearchAnalysisReport.__name__,
        )
        record_prompt_snapshot(
            state=state,
            node=self.node,
            role=self.role,
            system_prompt=sys_content,
            user_prompt=prompt,
        )
        out = structured.invoke([sys, {"role": "user", "content": prompt}])
        out.plan_id = input.plan.plan_id
        out.thesis = input.plan.thesis
        out.generated_at = datetime.now(tz=timezone.utc)
        return out
