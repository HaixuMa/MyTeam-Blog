from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)


class Citation(StrictModel):
    source_type: Literal[
        "paper",
        "patent",
        "official_doc",
        "blog",
        "wikipedia",
        "webpage",
        "dataset",
        "other",
    ]
    title: str = Field(min_length=3, max_length=400)
    url: str = Field(min_length=5, max_length=2000)
    published_date: date | None = None
    authors: list[str] = Field(default_factory=list, max_length=30)
    organization: str | None = Field(default=None, max_length=200)
    accessed_at: datetime
    excerpt: str | None = Field(default=None, max_length=2000)
    reliability_score: float = Field(ge=0.0, le=1.0)


class ResearchFinding(StrictModel):
    finding_id: str = Field(min_length=6, max_length=80)
    dimension_id: str = Field(min_length=2, max_length=80)
    claim: str = Field(min_length=10, max_length=1200)
    evidence: str = Field(min_length=10, max_length=2000)
    citations: list[Citation] = Field(default_factory=list, min_length=1, max_length=10)
    confidence: float = Field(ge=0.0, le=1.0)
    conflicts_with_finding_ids: list[str] = Field(default_factory=list, max_length=20)
    tags: list[str] = Field(default_factory=list, max_length=20)


class ClarificationQuestion(StrictModel):
    question_id: str = Field(min_length=3, max_length=40)
    question: str = Field(min_length=6, max_length=500)
    rationale: str = Field(min_length=6, max_length=800)
    expected_answer_format: str = Field(min_length=3, max_length=200)
    options: list[str] | None = Field(default=None, max_length=12)


class ExecutionEvent(StrictModel):
    timestamp: datetime
    trace_id: str
    node: str
    role: str
    status: Literal["start", "success", "retry", "degraded", "failed", "skipped", "halted"]
    duration_ms: int | None = Field(default=None, ge=0)
    attempt: int = Field(ge=1)
    message: str = Field(min_length=1, max_length=2000)
    input_summary: str | None = Field(default=None, max_length=4000)
    output_summary: str | None = Field(default=None, max_length=4000)
    error_type: str | None = Field(default=None, max_length=200)
    error_message: str | None = Field(default=None, max_length=2000)

