from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from pydantic import BaseModel

from agents.prompting import enforce_markdown_no_html
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
        sys = SystemMessage(
            content=(
                "你是技术文章作者。必须基于分析报告撰写专业技术文章，结构固定："
                "摘要、研究背景、核心技术分析、产业落地、趋势预判、参考文献、附录。"
                "每个章节必须对应分析报告的核心内容，禁止出现无来源观点。"
                "你需要在需要配图的段落插入锚点标记，格式为：[[FIG:<figure_id>:<anchor>]]。"
                "同时在 figure_requests 中列出图需求，paragraph_anchor 字段必须等于 Markdown 里的 <anchor>。"
                "输出必须是 TechnicalArticleDraft 结构化对象。"
            )
        )
        structured = self._llm.with_structured_output(TechnicalArticleDraft)

        prompt = (
            f"研究计划：{input.plan.model_dump(mode='json')}\n\n"
            f"分析报告：{input.analysis.model_dump(mode='json')}\n\n"
            "请输出 TechnicalArticleDraft。要求：\n"
            "- markdown 使用中文，标题层级规范；\n"
            "- 引用必须来自 analysis.citations（引用信息放入 references）；\n"
            "- 文章中每个需要配图的段落插入 [[FIG:...]] 锚点；\n"
            "- 参考文献章节必须列出 references（至少 8 条）。"
        )
        out = structured.invoke([sys, {"role": "user", "content": prompt}])
        out.plan_id = input.plan.plan_id
        out.generated_at = datetime.now(tz=timezone.utc)
        out.markdown = enforce_markdown_no_html(out.markdown)
        return out


def extract_fig_anchors(markdown: str) -> list[tuple[str, str]]:
    pattern = r"\[\[FIG:(?P<figure_id>[a-zA-Z0-9_\-]+):(?P<anchor>[a-zA-Z0-9_\-]{3,120})\]\]"
    return [(m.group("figure_id"), m.group("anchor")) for m in re.finditer(pattern, markdown)]
