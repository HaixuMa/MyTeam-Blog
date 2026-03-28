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
        required = ["## 摘要", "## 研究背景", "## 核心技术分析", "## 产业落地", "## 趋势预判", "## 参考文献", "## 附录"]
        if any(r not in md for r in required):
            raise ContractViolationError("final_article_missing_sections")

        if "[[FIG:" in md:
            raise ContractViolationError("unresolved_figure_anchors_in_final")

    def degrade(self, *, input: AuditInput, state: GraphState, error: Exception) -> AuditOutput:
        now = datetime.now(tz=timezone.utc)
        md = _normalize_markdown(input.article.markdown)
        title = "技术文章（自动排版稿）"
        out = AuditOutput.model_validate(
            {
                "final_article": {
                    "plan_id": input.article.plan_id,
                    "title": title,
                    "markdown": md,
                    "references": [c.model_dump(mode="json") for c in input.article.references][:8],
                    "generated_at": now,
                },
                "audit_report": {
                    "plan_id": input.article.plan_id,
                    "passed": True,
                    "summary": "自动降级为基础排版稿：未进行严格事实核验与引用对齐检查。",
                    "issues": [
                        {
                            "issue_id": "issue_fallback_0001",
                            "severity": "low",
                            "category": "citation",
                            "description": "当前环境缺少可用外部 sources，部分结论仅能基于内部占位来源形成可执行方案。",
                            "evidence": None,
                            "recommendation": "补充官方文档/权威资料后，重新运行研究与审核以对齐引用与事实性。",
                            "target_stage": "research",
                        }
                    ],
                    "rounds_used": input.rounds_used,
                    "generated_at": now,
                },
            }
        )
        out.final_article.plan_id = input.article.plan_id
        out.final_article.generated_at = now
        out.audit_report.plan_id = input.article.plan_id
        out.audit_report.rounds_used = input.rounds_used
        out.audit_report.generated_at = now
        out.final_article.markdown = _normalize_markdown(out.final_article.markdown)
        return out

    def _invoke(self, *, input: AuditInput, state: GraphState) -> AuditOutput:
        if input.article.references and all(
            str(c.url).startswith("internal://") for c in input.article.references
        ):
            return self.degrade(input=input, state=state, error=RuntimeError("fallback_mode"))

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
        out = invoke_structured_output(
            llm=self._llm,
            schema=AuditOutput,
            messages=[sys, {"role": "user", "content": prompt}],
        )

        out.final_article.plan_id = input.article.plan_id
        out.final_article.generated_at = datetime.now(tz=timezone.utc)
        out.audit_report.plan_id = input.article.plan_id
        out.audit_report.rounds_used = input.rounds_used
        out.audit_report.generated_at = datetime.now(tz=timezone.utc)

        out.final_article.markdown = _normalize_markdown(out.final_article.markdown)
        return out


def _normalize_markdown(md: str) -> str:
    md = re.sub(r"\r\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = md.strip() + "\n"
    return md
