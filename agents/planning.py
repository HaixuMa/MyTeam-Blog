from __future__ import annotations

import logging
import uuid

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from agents.context import system_context
from agents.prompting import invoke_structured_output, record_prompt_snapshot, sanitize_user_text
from agents.prompts import planning_system_prompt, planning_user_prompt
from harness.base import AgentHarness, ContractViolationError
from schemas.common import ClarificationQuestion
from schemas.planning import ResearchDimensionPlan, ResearchExecutionPlan, ResearchMilestone, UserResearchGoal
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
        if len(input.research_goal.strip()) < 10:
            raise ContractViolationError("research_goal_too_short")
        if len(input.research_goal) > 2000:
            raise ContractViolationError("research_goal_too_long")

    def post_validate(self, *, output: ResearchExecutionPlan, input: UserResearchGoal, state: GraphState) -> None:
        if output.clarification_needed:
            if not output.clarification_questions:
                raise ContractViolationError("clarification_needed_but_no_questions")
            return

        if len(output.dimensions) < 5 or len(output.dimensions) > 8:
            raise ContractViolationError("dimensions_must_be_5_to_8")

        for d in output.dimensions:
            if len(d.acceptance_criteria) < 2:
                raise ContractViolationError(f"dimension_acceptance_criteria_too_few: {d.dimension_id}")
            if any(len(x.strip()) < 6 for x in d.acceptance_criteria):
                raise ContractViolationError(f"dimension_acceptance_criteria_too_vague: {d.dimension_id}")

        if len(output.milestones) < 3:
            raise ContractViolationError("milestones_too_few")

        if len(output.deliverable_standards) < 3:
            raise ContractViolationError("deliverable_standards_too_few")

    def _invoke(self, *, input: UserResearchGoal, state: GraphState) -> ResearchExecutionPlan:
        cleaned_goal = sanitize_user_text(input.research_goal)
        clarifications = input.clarifications

        sys_content = system_context(state=state, node=self.node) + "\n\n" + planning_system_prompt()
        sys = SystemMessage(content=sys_content)

        plan_id = f"plan_{uuid.uuid4().hex[:12]}"
        prompt = planning_user_prompt(
            cleaned_goal=cleaned_goal,
            user_requirements=input.user_requirements,
            deadline=input.deadline,
            output_language=input.output_language,
            clarifications=clarifications,
            plan_id=plan_id,
        )
        record_prompt_snapshot(
            state=state,
            node=self.node,
            role=self.role,
            system_prompt=sys_content,
            user_prompt=prompt,
        )
        try:
            plan = invoke_structured_output(
                llm=self._llm,
                schema=ResearchExecutionPlan,
                messages=[sys, {"role": "user", "content": prompt}],
            )
        except Exception:
            plan = _fallback_plan(cleaned_goal=cleaned_goal, plan_id=plan_id, clarifications=clarifications)
        if plan.plan_id != plan_id:
            plan.plan_id = plan_id
        plan.clarification_needed = False
        plan.clarification_questions = []
        return plan


def _fallback_plan(*, cleaned_goal: str, plan_id: str, clarifications: dict[str, str]) -> ResearchExecutionPlan:
    thesis = cleaned_goal.strip()[:600]
    dims: list[ResearchDimensionPlan] = [
        ResearchDimensionPlan(
            dimension_id="d1",
            name="需求与范围界定",
            objectives=["明确面板目标与边界", "定义任务/节点/状态的最小契约"],
            key_questions=["面板必须展示哪些节点与字段？", "任务状态的生命周期与转移规则是什么？"],
            acceptance_criteria=["输出一份字段字典与状态机图", "给出至少 6 个边界场景并写出期望行为"],
            required_source_types=["official_doc", "webpage"],
            priority=5,
        ),
        ResearchDimensionPlan(
            dimension_id="d2",
            name="前端交互与状态管理",
            objectives=["确定 UI 信息架构与交互", "确定状态同步与渲染策略"],
            key_questions=["轮询/推送/混合方案如何选？", "任务列表与事件流如何增量渲染？"],
            acceptance_criteria=["完成可运行的 UI 交互原型说明", "提出性能指标与降级方案（至少 3 条）"],
            required_source_types=["official_doc", "blog"],
            priority=4,
        ),
        ResearchDimensionPlan(
            dimension_id="d3",
            name="后端 API 与调度运行时",
            objectives=["定义 API 形状与错误模型", "定义取消/重试/幂等的语义"],
            key_questions=["run 的创建/查询/取消/重试接口如何设计？", "并发与互斥（run_lock）如何保证？"],
            acceptance_criteria=["给出 API 列表与请求/响应示例", "明确取消/重试的幂等策略与状态转移规则"],
            required_source_types=["official_doc", "webpage"],
            priority=5,
        ),
        ResearchDimensionPlan(
            dimension_id="d4",
            name="可观测性与故障恢复",
            objectives=["统一事件模型与日志字段", "设计失败可恢复路径"],
            key_questions=["需要哪些 event 类型与字段？", "失败后如何定位到具体 prompt/节点/异常？"],
            acceptance_criteria=["事件模型可覆盖全链路并可追溯", "定义重试/降级/终止条件并与 UI 对齐"],
            required_source_types=["blog", "webpage"],
            priority=4,
        ),
        ResearchDimensionPlan(
            dimension_id="d5",
            name="安全与合规（最小）",
            objectives=["避免泄露密钥与敏感数据", "限制外部请求与资源消耗"],
            key_questions=["前端输出是否包含敏感字段？", "外部工具调用的白名单与限流策略是什么？"],
            acceptance_criteria=["明确敏感字段脱敏/隐藏策略", "定义限流与超时策略并给出默认值"],
            required_source_types=["official_doc", "other"],
            priority=3,
        ),
    ]

    questions: list[ClarificationQuestion] = []

    return ResearchExecutionPlan(
        plan_id=plan_id,
        thesis=thesis,
        dimensions=dims,
        deliverable_standards=[
            "所有输出必须可追溯到节点事件与 prompt 快照",
            "关键结论必须给出可验证的验收标准与示例",
            "失败场景必须有明确的降级/重试/终止策略",
        ],
        milestones=[
            ResearchMilestone(
                name="计划确认",
                description="产出研究计划与接口/状态机草案，确保团队对范围一致。",
                success_criteria=["计划字段齐全并可执行", "接口与状态机覆盖核心流程"],
                due_offset_days=0,
            ),
            ResearchMilestone(
                name="原型与事件模型落地",
                description="完成 UI 原型与事件模型定义，确保前后端对齐。",
                success_criteria=["UI 可展示 run/节点状态", "事件模型可支持错误定位"],
                due_offset_days=2,
            ),
            ResearchMilestone(
                name="端到端联调与验收",
                description="把计划、研究、写作等全链路事件在面板中串起来并完成验收项。",
                success_criteria=["端到端流程可跑通", "取消/重试/失败展示符合预期"],
                due_offset_days=5,
            ),
        ],
        source_policy="优先使用官方文档与可验证的技术资料；禁止编造来源；所有结论需给出可验证依据或明确假设。",
        info_source_requirements=["接口/协议需引用官方或权威资料", "关键行为需有可复现步骤或示例", "错误/异常需给出定位路径与最小复现"],
        risks=["模型输出不稳定导致结构化解析失败", "实时推送方案的服务端资源占用", "前端渲染性能与事件量增长"],
        clarification_needed=False,
        clarification_questions=questions,
    )

