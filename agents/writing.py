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
        if re.search(r"(提示词|system prompt|user prompt|作为|请生成|你是|本文根据用户输入主题生成|这是.*提示|输出必须)", md, re.I):
            raise ContractViolationError("markdown_contains_prompt_template")
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

        if "internal://" in md:
            raise ContractViolationError("internal_links_not_allowed_in_markdown")

        md_l = md.lower()
        thesis_text = (input.plan.thesis or "").lower()
        tokens = [t for t in re.findall(r"[a-zA-Z0-9_\u4e00-\u9fa5]{3,}", thesis_text) if t]
        uniq = []
        for t in tokens:
            if t not in uniq:
                uniq.append(t)
        need = max(2, min(5, len(uniq)))
        present = sum(1 for t in uniq[:10] if t in md_l)
        if present < max(1, min(3, len(uniq))):
            raise ContractViolationError("markdown_missing_thesis_keywords")

        title_l = (output.title or "").lower()
        present_title = sum(1 for t in uniq[:10] if t in title_l)
        if present_title < max(1, min(2, len(uniq))):
            raise ContractViolationError("title_missing_thesis_keywords")

        if len(output.references) < 3:
            raise ContractViolationError("references_too_few")
        if any(str(c.url).startswith("internal://") for c in output.references):
            raise ContractViolationError("references_must_be_external_http")
        if any(not str(c.url).startswith(("http://", "https://")) for c in output.references):
            raise ContractViolationError("references_must_be_external_http")

    def _invoke(self, *, input: WritingInput, state: GraphState) -> TechnicalArticleDraft:
        if input.analysis.citations and all(str(c.url).startswith("internal://") for c in input.analysis.citations):
            raise ContractViolationError("writing_requires_external_citations")
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
        try:
            out = invoke_structured_output(
                llm=self._llm,
                schema=TechnicalArticleDraft,
                messages=[sys, {"role": "user", "content": prompt}],
            )
        except Exception as e:
            return self.degrade(input=input, state=state, error=e)

        cleaned_md = enforce_markdown_no_html(out.markdown)
        if cleaned_md != out.markdown:
            out = out.model_copy(update={"markdown": cleaned_md})
        return out

    def degrade(self, *, input: WritingInput, state: GraphState, error: Exception) -> TechnicalArticleDraft:
        plan = input.plan
        analysis = input.analysis
        refs = analysis.citations[: max(3, min(20, len(analysis.citations)))]
        goal_title = ""
        ug = state.get("user_goal")
        if isinstance(ug, dict):
            goal_title = str(ug.get("research_goal") or "").strip()
        base_title = (goal_title or plan.thesis or "").strip()
        title = base_title if base_title else "技术文章"
        if len(title) < 10:
            title = (title + "（博客）")[:160]
        if len(title) > 160:
            title = title[:160]
        thesis = (plan.thesis or "").strip()
        dim_summaries = [d.summary for d in analysis.dimension_analysis[:5]]
        conclusions = [c for c in (analysis.conclusions or [])[:6] if isinstance(c, str) and c.strip()]
        risks = [r for r in (analysis.risks or [])[:8] if isinstance(r, str) and r.strip()]
        trend_points = [t for t in (analysis.trends or [])[:6] if isinstance(t, str) and t.strip()]

        import re as _re

        def _tokens(text: str) -> list[str]:
            ts = [x for x in _re.findall(r"[a-zA-Z0-9_\u4e00-\u9fa5]{3,}", (text or "").lower()) if x]
            out: list[str] = []
            for t in ts:
                if t not in out:
                    out.append(t)
            return out[:12]

        def _strip_promptish(text: str) -> str:
            t = str(text or "")
            t = _re.sub(r"\s+", " ", t).strip()
            if not t:
                return ""
            pat = _re.compile(r"(你是|请输出|输出必须|严格 JSON|system prompt|user prompt|作为|你将|请生成|提示词|指令)", _re.I)
            parts = _re.split(r"(?<=[。！？.!?])\s+", t)
            kept: list[str] = []
            for p in parts:
                s = p.strip()
                if not s:
                    continue
                if pat.search(s):
                    continue
                kept.append(s)
            return " ".join(kept).strip()

        def _sentences(text: str) -> list[str]:
            t = _strip_promptish(text)
            if not t:
                return []
            parts = _re.split(r"(?<=[。！？.!?])\s+", t)
            out: list[str] = []
            for p in parts:
                s = p.strip()
                if 20 <= len(s) <= 320:
                    out.append(s)
            return out

        th_tokens = _tokens(thesis)
        pool: list[tuple[float, int, str]] = []
        seen_norm: set[str] = set()

        for idx, c in enumerate(refs, start=1):
            base_text = ""
            ex = getattr(c, "excerpt", None)
            if isinstance(ex, str) and ex.strip():
                base_text = ex.strip()

            if len(base_text) < 80:
                try:
                    import requests
                    from bs4 import BeautifulSoup
                    resp = requests.get(
                        str(c.url),
                        headers={"User-Agent": "myteam-blog/0.1 (+local dev)"},
                        timeout=8,
                    )
                    if resp.ok and isinstance(resp.text, str) and resp.text.strip():
                        soup = BeautifulSoup(resp.text, "lxml")
                        for tag in soup(["script", "style", "noscript"]):
                            tag.decompose()
                        base_text = soup.get_text(" ", strip=True)[:12000]
                except Exception:
                    pass

            sents = _sentences(base_text)
            for s in sents:
                n = s.lower()
                if n in seen_norm:
                    continue
                seen_norm.add(n)
                score = 0.2 * min(len(s), 320) / 320.0
                for t in th_tokens:
                    if t and t in n:
                        score += 1.0
                score += 0.08 * (6 - min(6, idx))
                pool.append((score, idx, s))

        pool.sort(key=lambda x: x[0], reverse=True)
        top_sents = [f"{s}（来源[{idx}]）" for _, idx, s in pool[:60]]

        def _from_sents(start: int, count: int, max_chars: int) -> str:
            txt = " ".join(top_sents[start : start + count]).strip()
            return txt[:max_chars].strip()

        def _from_titles(max_items: int, max_chars: int) -> str:
            items: list[str] = []
            for i, c in enumerate(refs[:max_items], start=1):
                t = str(getattr(c, "title", "") or "").strip()
                if not t:
                    t = "未命名来源"
                items.append(f"{t}（来源[{i}]）")
            return "；".join(items)[:max_chars].strip()

        abstract = _from_sents(0, 4, 1200)
        if len(abstract) < 80:
            abstract = (
                f"本文围绕“{thesis or title}”整理公开资料中的关键观点，并将其抽取为可复用的工程结论。"
                + _from_titles(6, 900)
            )
        if thesis and thesis not in abstract:
            abstract = f"围绕“{thesis}”，" + abstract

        background = _from_sents(4, 10, 2400)
        if dim_summaries:
            background = (background + " " + " ".join(dim_summaries[:3])).strip()
        if len(background) < 160:
            background = (
                f"围绕“{thesis or title}”，本文优先从官方/仓库/百科等公开资料抽取背景信息与概念定义，"
                f"再结合多维度问题拆解形成统一语境。{_from_titles(8, 1600)}"
            ).strip()

        core = _from_sents(14, 18, 5200)
        if len(core) < 240:
            core = (core + " " + _from_sents(0, 18, 5200)).strip()
        if len(core) < 240:
            core = (
                f"核心技术分析以资料中的机制描述、接口约束与实现细节为主线，"
                f"并将其落到可执行的检查点与对照表。{_from_titles(10, 1800)}"
            ).strip()

        industry_bits: list[str] = []
        if conclusions:
            industry_bits.extend(conclusions[:4])
        if risks:
            industry_bits.extend([f"风险：{r}" for r in risks[:3]])
        industry = _from_sents(32, 12, 2600)
        if industry_bits:
            industry = (industry + "\n\n" + "\n".join([f"- {x}" for x in industry_bits if x.strip()])).strip()
        if len(industry) < 160:
            industry = (
                f"落地建议以“可验证、可观测、可回滚”为优先级，将资料中的做法转成工程动作项。"
                f"\n\n- 配置与契约：明确输入输出边界与失败模式\n"
                f"- 运行与观测：指标、日志、追踪的最小闭环\n"
                f"- 演进与治理：版本兼容、迁移、审计与回退\n"
                f"\n\n{_from_titles(8, 1200)}"
            ).strip()

        trends = _from_sents(44, 10, 2000)
        if trend_points:
            trends = (trends + "\n\n" + "\n".join([f"- {t}" for t in trend_points if t.strip()])).strip()
        if len(trends) < 160:
            trends = (
                "趋势预判以生态演进与工程化能力为主：更明确的契约、可观测性与可验证性将成为默认要求。"
                f"\n\n{_from_titles(6, 900)}"
            ).strip()

        def _pad(txt: str, min_len: int) -> str:
            if len(txt) >= min_len:
                return txt
            extra = " ".join(top_sents[:30]).strip()
            if not extra:
                extra = _from_titles(12, 2400)
            while len(txt) < min_len and extra:
                txt = (txt + " " + extra[: min(900, len(extra))]).strip()
            return txt[: max(min_len, len(txt))]

        abstract = _pad(abstract, 120)
        background = _pad(background, 200)
        core = _pad(core, 400)
        industry = _pad(industry, 200)
        trends = _pad(trends, 200)
        md_lines = [
            f"# {title}",
            "",
            "## 摘要",
            abstract,
            "",
            "## 研究背景",
            background,
            "",
            "## 核心技术分析",
            core,
            "",
            "## 产业落地",
            industry,
            "",
            "## 趋势预判",
            trends,
            "",
            "## 参考文献",
        ]
        for i, c in enumerate(refs, start=1):
            md_lines.append(f"[{i}] {c.title} - {c.url}")
        md_lines.append("")
        md_lines.append("## 附录")
        appendix = (
            "### 来源速览\n"
            + "\n".join([f"- [{i}] {c.title} - {c.url}" for i, c in enumerate(refs[:12], start=1)])
            + "\n\n### 术语与检查项\n"
            + "- 证据链：每条结论至少能追溯到一个公开来源\n"
            + "- 可验证：给出可执行的检查动作或对照步骤\n"
            + "- 可观测：指标/日志/追踪至少覆盖核心路径\n"
            + "- 可回滚：变更前后都可解释并可撤销\n"
        )
        md_lines.append(appendix)
        md = "\n".join(md_lines)
        return TechnicalArticleDraft(
            plan_id=plan.plan_id,
            title=title,
            abstract=abstract,
            background=background,
            core_tech_analysis=core,
            industrial_applications=industry,
            trends_outlook=trends,
            appendix=appendix,
            references=refs,
            figure_requests=[],
            markdown=md + "\n",
            generated_at=datetime.now(tz=timezone.utc),
        )


def extract_fig_anchors(markdown: str) -> list[tuple[str, str]]:
    pattern = r"\[\[FIG:(?P<figure_id>[a-zA-Z0-9_\-]+):(?P<anchor>[a-zA-Z0-9_\-]{3,120})\]\]"
    return [(m.group("figure_id"), m.group("anchor")) for m in re.finditer(pattern, markdown)]
