from __future__ import annotations

import logging
import uuid

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from agents.context import system_context
from agents.prompting import record_prompt_snapshot, sanitize_user_text
from agents.prompts import planning_system_prompt, planning_user_prompt
from harness.base import AgentHarness, ContractViolationError
from schemas.planning import ResearchExecutionPlan, UserResearchGoal
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

        structured = self._llm.with_structured_output(ResearchExecutionPlan)

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
        plan = structured.invoke([sys, {"role": "user", "content": prompt}])
        if plan.plan_id != plan_id:
            plan.plan_id = plan_id
        return plan

