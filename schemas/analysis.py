from __future__ import annotations

from datetime import datetime

from pydantic import Field

from schemas.common import Citation, StrictModel


class ArgumentStep(StrictModel):
    step_id: str = Field(min_length=4, max_length=40)
    statement: str = Field(min_length=10, max_length=1200)
    supported_by_finding_ids: list[str] = Field(default_factory=list, max_length=30)


class DimensionAnalysis(StrictModel):
    dimension_id: str = Field(min_length=2, max_length=40)
    summary: str = Field(min_length=20, max_length=2000)
    key_points: list[str] = Field(min_length=3, max_length=12)
    supported_by_finding_ids: list[str] = Field(min_length=3, max_length=30)
    open_questions: list[str] = Field(default_factory=list, max_length=12)


class DeepResearchAnalysisReport(StrictModel):
    plan_id: str = Field(min_length=6, max_length=80)
    thesis: str = Field(min_length=10, max_length=600)
    core_insights: list[str] = Field(min_length=5, max_length=12)
    dimension_analysis: list[DimensionAnalysis] = Field(min_length=5, max_length=8)
    argument_map: list[ArgumentStep] = Field(min_length=8, max_length=40)
    conclusions: list[str] = Field(min_length=3, max_length=10)
    risks: list[str] = Field(default_factory=list, max_length=12)
    opportunities: list[str] = Field(default_factory=list, max_length=12)
    trends: list[str] = Field(min_length=3, max_length=10)
    citations: list[Citation] = Field(min_length=8, max_length=80)
    generated_at: datetime

