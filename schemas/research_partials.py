from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from schemas.research import DimensionResearchResult
from schemas.common import StrictModel


class ResearchPartialBatch(StrictModel):
    plan_id: str = Field(min_length=6, max_length=80)
    agent_type: Literal["academic", "tech", "industry", "competitor"]
    results: list[DimensionResearchResult] = Field(min_length=1, max_length=20)
    generated_at: datetime

