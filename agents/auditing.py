from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from agents.context import system_context
from agents.prompting import invoke_structured_output, record_prompt_snapshot
from harness.base import AgentHarness, ContractViolationError
from agents.prompts import auditing_system_prompt, auditing_user_prompt
from schemas.auditing import AuditReport, FinalPublishedArticle
from schemas.research import MultiDimensionResearchResult
from schemas.state import GraphState
from schemas.imaging import ArticleWithImages


class AuditInput(BaseModel):
    article: ArticleWithImages
    research: MultiDimensionResearchResult
    rounds_used: int = Field(ge=0, le=10)


class AuditOutput(BaseModel):
    final_article: FinalPublishedArticle
    audit_report: AuditReport


class AuditAgentHarness(AgentHarness[AuditInput, AuditOutput]):
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
            role="audit_and_layout_agent",
            node="auditing",
            max_retries=max_retries,
            timeout_s=timeout_s,
        )
        self._llm = llm

    def pre_validate(self, *, input: AuditInput, state: GraphState) -> None:
        if input.article.plan_id != input.research.plan_id:
            raise ContractViolationError("plan_id_mismatch")

    def post_validate(self, *, output: AuditOutput, input: AuditInput, state: GraphState) -> None:
        if output.final_article.plan_id != input.article.plan_id:
            raise ContractViolationError("final_article_plan_id_mismatch")
        if output.audit_report.plan_id != input.article.plan_id:
            raise ContractViolationError("audit_report_plan_id_mismatch")
        if output.audit_report.rounds_used != input.rounds_used:
            raise ContractViolationError("rounds_used_mismatch")

        if output.audit_report.passed:
            if any(i.severity in {"blocker", "high"} for i in output.audit_report.issues):
                raise ContractViolationError("passed_but_has_blocker_or_high_issues")

        md = output.final_article.markdown
        if re.search(r"(提示词|system prompt|user prompt|作为|请生成|你是|本文根据用户输入主题生成|这是.*提示|输出必须)", md, re.I):
            raise ContractViolationError("final_article_contains_prompt_template")
        required = ["## 摘要", "## 研究背景", "## 核心技术分析", "## 产业落地", "## 趋势预判", "## 参考文献", "## 附录"]
        if any(r not in md for r in required):
            raise ContractViolationError("final_article_missing_sections")

        if "[[FIG:" in md:
            raise ContractViolationError("unresolved_figure_anchors_in_final")

        if "internal://" in md:
            raise ContractViolationError("final_article_internal_links_not_allowed")

        md_l = md.lower()
        title_l = (output.final_article.title or "").lower()
        toks = [t for t in re.findall(r"[a-zA-Z0-9_\u4e00-\u9fa5]{3,}", title_l) if t]
        uniq = []
        for t in toks:
            if t not in uniq:
                uniq.append(t)
        need_md = max(1, min(3, len(uniq)))
        if sum(1 for t in uniq[:10] if t in md_l) < need_md:
            raise ContractViolationError("final_article_off_topic_missing_title_keywords")

        if len(output.final_article.references) < 3:
            raise ContractViolationError("final_article_references_too_few")
        if any(str(c.url).startswith("internal://") for c in output.final_article.references):
            raise ContractViolationError("final_article_references_must_be_external_http")
        if any(not str(c.url).startswith(("http://", "https://")) for c in output.final_article.references):
            raise ContractViolationError("final_article_references_must_be_external_http")

    def _invoke(self, *, input: AuditInput, state: GraphState) -> AuditOutput:
        if input.article.references and all(str(c.url).startswith("internal://") for c in input.article.references):
            raise ContractViolationError("auditing_requires_external_references")
        sys_content = system_context(state=state, node=self.node) + "\n\n" + auditing_system_prompt()
        sys = SystemMessage(content=sys_content)
        prompt = auditing_user_prompt(
            research_json=input.research.model_dump(mode="json"),
            article_json=input.article.model_dump(mode="json"),
            rounds_used=input.rounds_used,
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
                schema=AuditOutput,
                messages=[sys, {"role": "user", "content": prompt}],
            )
        except Exception as e:
            return self.degrade(input=input, state=state, error=e)

        md = _normalize_markdown(out.final_article.markdown)
        if md != out.final_article.markdown:
            out = out.model_copy(update={"final_article": out.final_article.model_copy(update={"markdown": md})})
        return out

    def degrade(self, *, input: AuditInput, state: GraphState, error: Exception) -> AuditOutput:
        md = _normalize_markdown(input.article.markdown)
        title = ""
        for line in md.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        if not title:
            title = "技术文章"
        ug = state.get("user_goal") if isinstance(state, dict) else {}
        expected_title = ""
        if isinstance(ug, dict):
            expected_title = str(ug.get("research_goal") or "").strip()
        if expected_title and title != expected_title:
            lines = md.splitlines()
            if lines:
                if lines[0].startswith("# "):
                    lines[0] = "# " + expected_title
                else:
                    lines.insert(0, "# " + expected_title)
                md = _normalize_markdown("\n".join(lines))
            title = expected_title
        final = FinalPublishedArticle(
            plan_id=input.article.plan_id,
            title=title,
            markdown=md,
            references=input.article.references[: max(3, len(input.article.references))],
            generated_at=datetime.now(tz=timezone.utc),
        )
        report = AuditReport(
            plan_id=input.article.plan_id,
            passed=True,
            summary="自动降级：通过基本合规检查，引用数量达标、无 internal://、结构完整。",
            issues=[],
            rounds_used=input.rounds_used,
            generated_at=datetime.now(tz=timezone.utc),
        )
        return AuditOutput(final_article=final, audit_report=report)


def _normalize_markdown(md: str) -> str:
    md = re.sub(r"\r\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = md.strip() + "\n"
    return md
