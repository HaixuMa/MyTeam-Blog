from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from pydantic import BaseModel

from agents.context import system_context
from agents.prompting import record_prompt_snapshot
from agents.prompting import enforce_markdown_no_html
from agents.prompts import writing_system_prompt, writing_user_prompt
from harness.base import AgentHarness, ContractViolationError
from schemas.analysis import DeepResearchAnalysisReport
from schemas.planning import ResearchExecutionPlan
from schemas.state import GraphState
from schemas.writing import TechnicalArticleDraft


class WritingInput(BaseModel):
    plan: ResearchExecutionPlan
    analysis: DeepResearchAnalysisReport


class TechnicalWritingAgentHarness(AgentHarness[WritingInput, TechnicalArticleDraft]):
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
            role="technical_writing_agent",
            node="writing",
            max_retries=max_retries,
            timeout_s=timeout_s,
        )
        self._llm = llm

    def pre_validate(self, *, input: WritingInput, state: GraphState) -> None:
        if input.plan.plan_id != input.analysis.plan_id:
            raise ContractViolationError("plan_id_mismatch")

    def post_validate(self, *, output: TechnicalArticleDraft, input: WritingInput, state: GraphState) -> None:
        if output.plan_id != input.plan.plan_id:
            raise ContractViolationError("plan_id_mismatch_output")
        md = output.markdown
        required_sections = [
            "# ",
            "## 摘要",
            "## 研究背景",
            "## 核心技术分析",
            "## 产业落地",
            "## 趋势预判",
            "## 参考文献",
            "## 附录",
        ]
        if any(s not in md for s in required_sections):
            raise ContractViolationError("missing_required_sections")

        for fr in output.figure_requests:
            if fr.paragraph_anchor not in md:
                raise ContractViolationError("figure_anchor_missing_in_markdown")

        analysis_urls = {c.url for c in input.analysis.citations}
        if any(c.url not in analysis_urls for c in output.references):
            raise ContractViolationError("references_must_come_from_analysis_citations")

        if "<script" in md.lower():
            raise ContractViolationError("unsafe_html_detected")

    def _invoke(self, *, input: WritingInput, state: GraphState) -> TechnicalArticleDraft:
        sys_content = system_context(state=state, node=self.node) + "\n\n" + writing_system_prompt()
        sys = SystemMessage(content=sys_content)
        structured = self._llm.with_structured_output(TechnicalArticleDraft)

        prompt = writing_user_prompt(
            plan_json=input.plan.model_dump(mode="json"),
            analysis_json=input.analysis.model_dump(mode="json"),
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
        out.generated_at = datetime.now(tz=timezone.utc)
        out.markdown = enforce_markdown_no_html(out.markdown)
        return out


def extract_fig_anchors(markdown: str) -> list[tuple[str, str]]:
    pattern = r"\[\[FIG:(?P<figure_id>[a-zA-Z0-9_\-]+):(?P<anchor>[a-zA-Z0-9_\-]{3,120})\]\]"
    return [(m.group("figure_id"), m.group("anchor")) for m in re.finditer(pattern, markdown)]
