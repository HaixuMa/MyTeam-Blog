from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from schemas.common import Citation, StrictModel


class FigureRequest(StrictModel):
    figure_id: str = Field(min_length=4, max_length=40)
    section: Literal[
        "abstract",
        "background",
        "core_tech_analysis",
        "industrial_applications",
        "trends_outlook",
        "appendix",
    ]
    paragraph_anchor: str = Field(
        min_length=3,
        max_length=120,
        description="用于在 Markdown 中定位插图位置的锚点字符串（由写作 Agent 在文中生成对应标记）。",
    )
    figure_type: Literal[
        "architecture_diagram",
        "flowchart",
        "concept_diagram",
        "data_chart",
        "comparison_matrix",
    ]
    purpose: str = Field(min_length=10, max_length=300)
    must_include: list[str] = Field(min_length=2, max_length=12)
    style_guidelines: list[str] = Field(min_length=1, max_length=8)
    prompt_seed: str = Field(min_length=10, max_length=600)


class TechnicalArticleDraft(StrictModel):
    plan_id: str = Field(min_length=6, max_length=80)
    title: str = Field(min_length=10, max_length=160)
    abstract: str = Field(min_length=120, max_length=2000)
    background: str = Field(min_length=200, max_length=6000)
    core_tech_analysis: str = Field(min_length=400, max_length=12000)
    industrial_applications: str = Field(min_length=200, max_length=8000)
    trends_outlook: str = Field(min_length=200, max_length=6000)
    appendix: str = Field(min_length=0, max_length=12000)
    references: list[Citation] = Field(min_length=3, max_length=120)
    figure_requests: list[FigureRequest] = Field(default_factory=list, max_length=10)
    markdown: str = Field(min_length=800, max_length=60000)
    generated_at: datetime
