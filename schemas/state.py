from __future__ import annotations

from datetime import datetime
from typing import Literal, NotRequired, TypedDict


Stage = Literal[
    "start",
    "planning",
    "research_academic",
    "research_tech",
    "research_industry",
    "research_competitor",
    "research_aggregate",
    "analysis",
    "writing",
    "imaging",
    "auditing",
    "publish",
    "halted",
    "failed",
]


class GraphState(TypedDict):
    trace_id: str
    created_at: str
    stage: Stage

    user_goal: dict
    plan: NotRequired[dict]

    research_partials: NotRequired[list[dict]]
    research_result: NotRequired[dict]

    analysis_report: NotRequired[dict]
    article_draft: NotRequired[dict]
    article_with_images: NotRequired[dict]
    final_article: NotRequired[dict]
    audit_report: NotRequired[dict]

    execution_events: list[dict]
    retries: dict[str, int]
    audit_rounds_used: int

    halted_reason: NotRequired[str]
    fatal_error: NotRequired[str]


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

