from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from schemas.common import Citation, StrictModel


class AuditIssue(StrictModel):
    issue_id: str = Field(min_length=6, max_length=80)
    severity: Literal["low", "medium", "high", "blocker"]
    category: Literal["factual", "logical", "style", "citation", "format"]
    description: str = Field(min_length=10, max_length=1200)
    evidence: str | None = Field(default=None, max_length=2000)
    recommendation: str = Field(min_length=10, max_length=1200)
    target_stage: Literal["research", "analysis", "writing", "imaging"]


class AuditReport(StrictModel):
    plan_id: str = Field(min_length=6, max_length=80)
    passed: bool
    summary: str = Field(min_length=20, max_length=2000)
    issues: list[AuditIssue] = Field(default_factory=list, max_length=80)
    rounds_used: int = Field(ge=0, le=10)
    generated_at: datetime


class FinalPublishedArticle(StrictModel):
    plan_id: str = Field(min_length=6, max_length=80)
    title: str = Field(min_length=10, max_length=160)
    markdown: str = Field(min_length=800, max_length=90000)
    references: list[Citation] = Field(min_length=8, max_length=200)
    generated_at: datetime

