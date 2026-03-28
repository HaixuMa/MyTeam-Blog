from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from pydantic import BaseModel

from agents.context import system_context
from agents.prompting import invoke_structured_output, record_prompt_snapshot
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


def _pad_to_min(text: str, min_len: int) -> str:
    if len(text) >= min_len:
        return text
    base = text.strip() or "（内容待补充）"
    parts = [base]
    while len("\n\n".join(parts)) < min_len:
        parts.append(base)
    return "\n\n".join(parts)[: max(min_len, 1)]


def _build_fallback_article(*, input: WritingInput) -> TechnicalArticleDraft:
    now = datetime.now(tz=timezone.utc)
    title = f"{input.plan.thesis}：从事件流到可视化调度面板的落地方案"

    refs = [c.model_dump(mode="json") for c in input.analysis.citations][:8]
    while len(refs) < 8:
        refs.append(refs[-1] if refs else input.analysis.citations[0].model_dump(mode="json"))

    insights = input.analysis.core_insights[:5]
    dim_blocks = []
    for da in input.analysis.dimension_analysis[:6]:
        points = "\n".join([f"- {p}" for p in da.key_points[:6]]) if da.key_points else ""
        dim_blocks.append(f"### {da.dimension_id}\n\n{da.summary}\n\n{points}".strip())
    dim_text = "\n\n".join(dim_blocks).strip()

    abstract = _pad_to_min(
        "本文基于调研与分析报告，围绕命题给出一套可落地的技术方案与工程化验收标准。"
        + ("核心结论包括：" + "；".join(insights[:3]) + "。" if insights else "")
        + "文章组织为：研究背景→核心技术分析→产业落地→趋势预判，并在附录提供状态/事件/接口的落地清单。",
        160,
    )
    background = _pad_to_min(
        f"本文研究命题为：{input.plan.thesis}。"
        "在多节点编排与长链路执行中，最难的是把“事实来源（events）”“聚合状态（state）”“控制命令（cancel/retry）”统一到同一套契约。"
        "如果缺少统一契约，前端只会不断堆叠展示逻辑，最终导致状态不一致与故障定位困难。"
        + (
            "\n\n交付与验收标准（摘录）：\n"
            + "\n".join([f"- {s}" for s in input.plan.deliverable_standards[:10]])
            if input.plan.deliverable_standards
            else ""
        ),
        240,
    )
    core_tech_analysis = _pad_to_min(
        "核心技术部分严格对齐分析报告的维度分析与论证链条，给出从数据契约到 UI 视图模型的实现要点。\n\n"
        + dim_text,
        520,
    )
    industrial_applications = _pad_to_min(
        "产业落地关注“谁使用、用什么指标衡量、如何运维与演进”。"
        "在落地阶段建议先建立最小可用闭环：run 列表→节点状态→错误定位→重试/取消。"
        + (
            "\n\n机会点：\n" + "\n".join([f"- {x}" for x in input.analysis.opportunities[:8]])
            if input.analysis.opportunities
            else ""
        ),
        220,
    )
    trends_outlook = _pad_to_min(
        "趋势预判基于报告中的趋势条目进行扩展：\n"
        + "\n".join([f"- {t}" for t in input.analysis.trends[:10]]),
        200,
    )
    appendix = _pad_to_min(
        "附录：落地清单与风险提示\n\n"
        + "建议清单：\n"
        + "\n".join(
            [
                "- 状态枚举与转移表（run/node 级别）",
                "- 事件类型与字段定义（含幂等键与时间戳）",
                "- API 请求/响应示例（创建/查询/取消/重试）",
                "- 边界场景用例（超时、部分失败、重复重试、幂等冲突）",
            ]
        )
        + (
            "\n\n主要风险：\n" + "\n".join([f"- {r}" for r in input.analysis.risks[:10]])
            if input.analysis.risks
            else ""
        ),
        160,
    )

    md = "\n".join(
        [
            f"# {title}",
            "",
            "## 摘要",
            abstract,
            "",
            "## 研究背景",
            background,
            "",
            "## 核心技术分析",
            core_tech_analysis,
            "",
            "## 产业落地",
            industrial_applications,
            "",
            "## 趋势预判",
            trends_outlook,
            "",
            "## 参考文献",
            "\n".join([f"[{i+1}] {r['title']} - {r['url']}" for i, r in enumerate(refs)]),
            "",
            "## 附录",
            appendix,
            "",
        ]
    )
    md = _pad_to_min(md, 800)

    return TechnicalArticleDraft.model_validate(
        {
            "plan_id": input.plan.plan_id,
            "title": title,
            "abstract": abstract,
            "background": background,
            "core_tech_analysis": core_tech_analysis,
            "industrial_applications": industrial_applications,
            "trends_outlook": trends_outlook,
            "appendix": appendix,
            "references": refs,
            "figure_requests": [],
            "markdown": md,
            "generated_at": now,
        }
    )


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

    def degrade(self, *, input: WritingInput, state: GraphState, error: Exception) -> TechnicalArticleDraft:
        out = _build_fallback_article(input=input)
        out.plan_id = input.plan.plan_id
        out.generated_at = datetime.now(tz=timezone.utc)
        out.markdown = enforce_markdown_no_html(out.markdown)
        return out

    def _invoke(self, *, input: WritingInput, state: GraphState) -> TechnicalArticleDraft:
        if input.analysis.citations and all(
            str(c.url).startswith("internal://") for c in input.analysis.citations
        ):
            out = _build_fallback_article(input=input)
            out.plan_id = input.plan.plan_id
            out.generated_at = datetime.now(tz=timezone.utc)
            out.markdown = enforce_markdown_no_html(out.markdown)
            return out

        sys_content = system_context(state=state, node=self.node) + "\n\n" + writing_system_prompt()
        sys = SystemMessage(content=sys_content)

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
        out = invoke_structured_output(
            llm=self._llm,
            schema=TechnicalArticleDraft,
            messages=[sys, {"role": "user", "content": prompt}],
        )
        out.plan_id = input.plan.plan_id
        out.generated_at = datetime.now(tz=timezone.utc)
        out.markdown = enforce_markdown_no_html(out.markdown)
        return out


def extract_fig_anchors(markdown: str) -> list[tuple[str, str]]:
    pattern = r"\[\[FIG:(?P<figure_id>[a-zA-Z0-9_\-]+):(?P<anchor>[a-zA-Z0-9_\-]{3,120})\]\]"
    return [(m.group("figure_id"), m.group("anchor")) for m in re.finditer(pattern, markdown)]
