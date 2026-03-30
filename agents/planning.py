from __future__ import annotations

import logging
import re
import uuid

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from agents.context import system_context
from agents.prompting import invoke_structured_output, record_prompt_snapshot, sanitize_user_text
from agents.prompts import planning_system_prompt, planning_user_prompt
from harness.base import AgentHarness, ContractViolationError
from schemas.common import ClarificationQuestion
from schemas.planning import ResearchExecutionPlan, ResearchMilestone, UserResearchGoal
from schemas.state import GraphState


class _PlanOrClarify(BaseModel):
    mode: str = Field(description="plan 或 clarify")
    plan: ResearchExecutionPlan | None = None
    clarification: ResearchExecutionPlan | None = None


class PlanningAgentHarness(AgentHarness[UserResearchGoal, ResearchExecutionPlan]):
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
            role="deep_planning_agent",
            node="planning",
            max_retries=max_retries,
            timeout_s=timeout_s,
        )
        self._llm = llm

    def pre_validate(self, *, input: UserResearchGoal, state: GraphState) -> None:
        if len(input.research_goal.strip()) < 6:
            raise ContractViolationError("research_goal_too_short")
        if len(input.research_goal) > 2000:
            raise ContractViolationError("research_goal_too_long")

    def post_validate(self, *, output: ResearchExecutionPlan, input: UserResearchGoal, state: GraphState) -> None:
        if output.clarification_needed:
            if not output.clarification_questions:
                raise ContractViolationError("clarification_needed_but_no_questions")
            return
        goal_l = (input.research_goal or "").lower()
        thesis_l = (output.thesis or "").lower()
        toks = [t for t in re.findall(r"[a-zA-Z0-9_\u4e00-\u9fa5]{3,}", goal_l) if t]
        uniq = []
        for t in toks:
            if t not in uniq:
                uniq.append(t)
        need = max(1, min(3, len(uniq)))
        present = sum(1 for t in uniq[:10] if t in thesis_l)
        if present < need:
            raise ContractViolationError("planning_off_topic_missing_goal_keywords")

        if len(output.dimensions) < 4 or len(output.dimensions) > 6:
            raise ContractViolationError("dimensions_must_be_4_to_6")

        for d in output.dimensions:
            if len(d.acceptance_criteria) < 2:
                raise ContractViolationError(f"dimension_acceptance_criteria_too_few: {d.dimension_id}")
            if any(len(x.strip()) < 6 for x in d.acceptance_criteria):
                raise ContractViolationError(f"dimension_acceptance_criteria_too_vague: {d.dimension_id}")

        if len(output.milestones) < 2:
            raise ContractViolationError("milestones_too_few")

        if len(output.deliverable_standards) < 3:
            raise ContractViolationError("deliverable_standards_too_few")

    def _invoke(self, *, input: UserResearchGoal, state: GraphState) -> ResearchExecutionPlan:
        cleaned_goal = sanitize_user_text(input.research_goal)
        clarifications = input.clarifications

        sys_content = system_context(state=state, node=self.node) + "\n\n" + planning_system_prompt()
        sys = SystemMessage(content=sys_content)

        plan_id = f"plan_{uuid.uuid4().hex[:12]}"
        record_prompt_snapshot(
            state=state,
            node=self.node,
            role=self.role,
            system_prompt=sys_content,
            user_prompt=planning_user_prompt(
                cleaned_goal=cleaned_goal,
                user_requirements=input.user_requirements,
                deadline=input.deadline,
                output_language=input.output_language,
                clarifications=clarifications,
                plan_id=plan_id,
            ),
        )
        plan_obj = _build_deterministic_plan(
            plan_id=plan_id,
            cleaned_goal=cleaned_goal,
            output_language=input.output_language,
        )
        try:
            plan = ResearchExecutionPlan.model_validate(plan_obj)
        except Exception as e:
            msg = str(e).strip().replace("\n", " ")[:800]
            raise ContractViolationError(f"planning_deterministic_plan_validate_failed: {msg}") from e
        return plan


def _build_deterministic_plan(*, plan_id: str, cleaned_goal: str, output_language: str) -> dict:
    thesis = cleaned_goal or ("User-provided research topic" if output_language != "zh" else "用户提供的研究主题")

    dimensions = [
        {
            "dimension_id": "d1",
            "name": "主题关键机制与边界",
            "objectives": ["明确核心概念与作用", "界定状态/数据的边界与前置条件"],
            "key_questions": [
                "主题的核心机制与流程是什么？",
                "哪些内容需要持久化或长期保留？",
                "在多步骤/多轮流程中如何支持恢复与回放？",
            ],
            "acceptance_criteria": [
                "给出机制与流程描述。",
                "明确边界与不应持久化内容的判定规则（若适用）。",
            ],
            "required_source_types": ["official_doc", "repo_release", "webpage"],
            "priority": 1,
        },
        {
            "dimension_id": "d2",
            "name": "集成与配置模式",
            "objectives": ["给出最小可用配置", "总结在项目中的封装与初始化模式"],
            "key_questions": [
                "如何初始化与绑定到业务流程？",
                "配置项有哪些常见坑与推荐做法？",
                "不同环境如何区分位置与生命周期？",
            ],
            "acceptance_criteria": [
                "提供可运行的配置步骤与参数说明（不含敏感信息）。",
                "列出至少 3 条可验证的集成注意事项与对应处理方式。",
            ],
            "required_source_types": ["official_doc", "repo_release", "blog"],
            "priority": 2,
        },
        {
            "dimension_id": "d3",
            "name": "一致性与事务/约束策略",
            "objectives": ["分析并发访问下的影响", "制定一致性与失败恢复策略"],
            "key_questions": [
                "并发读写有哪些风险？",
                "对吞吐与延迟的影响是什么？",
                "如何设计边界以避免冲突与脏读？",
            ],
            "acceptance_criteria": [
                "给出并发场景下的风险清单与对应缓解措施。",
                "明确建议的事务/隔离策略与适用条件。",
            ],
            "required_source_types": ["official_doc", "standard", "paper"],
            "priority": 1,
        },
        {
            "dimension_id": "d4",
            "name": "性能、空间与生命周期管理",
            "objectives": ["控制体积与写入开销", "制定清理与备份策略"],
            "key_questions": [
                "随时间增长的空间问题如何评估与治理？",
                "如何进行裁剪、压缩、清理与定期备份？",
                "哪些指标可以用于监控性能与稳定性？",
            ],
            "acceptance_criteria": [
                "给出至少 3 条性能/空间优化建议并说明验证方式。",
                "给出可执行的清理与备份策略（频率与风险控制）。",
            ],
            "required_source_types": ["official_doc", "webpage", "dataset"],
            "priority": 3,
        },
        {
            "dimension_id": "d5",
            "name": "迁移、版本兼容与运维落地",
            "objectives": ["制定 schema 变更与迁移策略", "形成运维与故障排查手册要点"],
            "key_questions": [
                "数据/结构如何演进与兼容旧数据？",
                "如何进行迁移、回滚与完整性校验？",
                "故障场景（损坏/磁盘满/权限）如何诊断与恢复？",
            ],
            "acceptance_criteria": [
                "给出迁移/回滚的流程化步骤与校验点。",
                "列出至少 3 个常见故障场景与可执行处置步骤。",
            ],
            "required_source_types": ["official_doc", "repo_release", "webpage"],
            "priority": 4,
        },
    ]

    deliverable_standards = [
        "全文中文输出，结构清晰，术语一致。",
        "所有关键结论均给出可点击的 http/https 引用链接。",
        "参考文献不少于 3 条，优先官方文档/标准/仓库发布说明。",
        "不允许出现 internal:// 引用或无法访问的占位链接。",
    ]
    milestones = [
        {
            "name": "资料收集与证据链",
            "description": "围绕主题收集权威资料并形成可引用的证据链条，记录每条结论对应引用。",
            "success_criteria": ["收集到不少于 12 个候选来源链接", "每个维度至少有 2 个可用来源"],
            "due_offset_days": 0,
        },
        {
            "name": "综合分析与最佳实践提炼",
            "description": "对来源进行交叉验证，提炼可操作的最佳实践要点，并明确适用条件与风险。",
            "success_criteria": ["形成 5 个维度的最佳实践要点", "每条要点均有对应引用支持"],
            "due_offset_days": 1,
        },
        {
            "name": "成稿与审计（引用合规）",
            "description": "完成文章写作并进行引用合规审计，确保所有引用为 http/https 且数量达标，避免内部链接与空泛断言。",
            "success_criteria": ["最终稿包含不少于 3 条参考文献", "审计通过：无 internal:// 且引用可访问"],
            "due_offset_days": 2,
        },
    ]
    source_policy = (
        "只允许引用 http/https 链接；优先官方文档、标准/规范、项目仓库发布说明与权威论文。"
        "严禁 internal:// 或任何不可公开访问的链接。"
        "每个关键结论必须附带 citations，并能追溯到原始来源。"
    )
    info_source_requirements = [
        "主题相关的官方文档与实现说明（http/https）。",
        "项目仓库的发布说明/源码引用（http/https）。",
        "标准/规范或权威论文用于交叉验证（http/https）。",
    ]
    risks = [
        "并发写入或互斥导致的性能抖动与延迟尖峰。",
        "数据与状态增长导致的空间与维护成本上升。",
        "版本升级带来的兼容性与迁移风险。",
    ]

    return {
        "plan_id": plan_id,
        "thesis": thesis,
        "dimensions": dimensions,
        "deliverable_standards": deliverable_standards,
        "milestones": milestones,
        "source_policy": source_policy,
        "info_source_requirements": info_source_requirements,
        "risks": risks,
        "clarification_needed": False,
        "clarification_questions": [],
    }

