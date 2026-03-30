from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field

from schemas.common import ClarificationQuestion, StrictModel


class UserResearchGoal(StrictModel):
    research_goal: str = Field(min_length=10, max_length=2000)
    user_requirements: list[str] = Field(default_factory=list, max_length=20)
    deadline: date | None = None
    output_language: Literal["zh", "en"] = "zh"
    max_sources_per_dimension: int = Field(default=8, ge=3, le=20)
    allowed_tools: list[
        Literal["tavily", "arxiv", "wikipedia", "web_loader", "image_generation"]
    ] = Field(default_factory=lambda: ["tavily", "arxiv", "wikipedia", "web_loader", "image_generation"])
    clarifications: dict[str, str] = Field(default_factory=dict, max_length=30)


class ResearchDimensionPlan(StrictModel):
    dimension_id: str = Field(min_length=2, max_length=40)
    name: str = Field(min_length=3, max_length=120)
    objectives: list[str] = Field(min_length=1, max_length=5)
    key_questions: list[str] = Field(min_length=2, max_length=8)
    acceptance_criteria: list[str] = Field(min_length=2, max_length=6)
    required_source_types: list[str] = Field(min_length=2, max_length=6)
    priority: int = Field(ge=1, le=5)


class ResearchMilestone(StrictModel):
    name: str = Field(min_length=3, max_length=120)
    description: str = Field(min_length=10, max_length=600)
    success_criteria: list[str] = Field(min_length=1, max_length=10)
    due_offset_days: int | None = Field(default=None, ge=0, le=365)


class ResearchExecutionPlan(StrictModel):
    plan_id: str = Field(min_length=6, max_length=80)
    thesis: str = Field(min_length=10, max_length=600)
    dimensions: list[ResearchDimensionPlan] = Field(min_length=4, max_length=6)
    deliverable_standards: list[str] = Field(min_length=3, max_length=12)
    milestones: list[ResearchMilestone] = Field(min_length=2, max_length=8)
    source_policy: str = Field(min_length=10, max_length=1200)
    info_source_requirements: list[str] = Field(min_length=3, max_length=12)
    risks: list[str] = Field(default_factory=list, max_length=12)
    clarification_needed: bool = False
    clarification_questions: list[ClarificationQuestion] = Field(default_factory=list, max_length=8)
