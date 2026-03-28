from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from schemas.common import Citation, ResearchFinding, StrictModel


class ValidationResult(StrictModel):
    passed: bool
    checks: list[str] = Field(default_factory=list, max_length=30)
    failed_checks: list[str] = Field(default_factory=list, max_length=30)
    notes: str | None = Field(default=None, max_length=1200)


class DimensionResearchResult(StrictModel):
    dimension_id: str = Field(min_length=2, max_length=40)
    agent_type: Literal["academic", "tech", "industry", "competitor"]
    findings: list[ResearchFinding] = Field(default_factory=list, max_length=60)
    sources: list[Citation] = Field(default_factory=list, max_length=60)
    notes: str = Field(min_length=0, max_length=2000)
    validation: ValidationResult


class ConflictRecord(StrictModel):
    conflict_id: str = Field(min_length=6, max_length=80)
    claim_a: str = Field(min_length=10, max_length=1200)
    claim_b: str = Field(min_length=10, max_length=1200)
    related_finding_ids: list[str] = Field(min_length=2, max_length=10)
    resolution_status: Literal["unresolved", "needs_followup", "resolved"]
    followup_suggestion: str | None = Field(default=None, max_length=800)


class MultiDimensionResearchResult(StrictModel):
    plan_id: str = Field(min_length=6, max_length=80)
    thesis: str = Field(min_length=10, max_length=600)
    dimension_results: list[DimensionResearchResult] = Field(min_length=4, max_length=60)
    conflicts: list[ConflictRecord] = Field(default_factory=list, max_length=50)
    deduped: bool
    generated_at: datetime

