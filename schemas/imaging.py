from __future__ import annotations

from datetime import datetime

from pydantic import Field

from schemas.common import Citation, StrictModel


class GeneratedImage(StrictModel):
    figure_id: str = Field(min_length=4, max_length=40)
    file_path: str = Field(min_length=1, max_length=500)
    prompt: str = Field(min_length=10, max_length=4000)
    model: str = Field(min_length=1, max_length=120)
    width: int = Field(ge=128, le=4096)
    height: int = Field(ge=128, le=4096)
    generated_at: datetime
    skipped: bool
    skip_reason: str | None = Field(default=None, max_length=600)


class ArticleWithImages(StrictModel):
    plan_id: str = Field(min_length=6, max_length=80)
    markdown: str = Field(min_length=800, max_length=80000)
    images: list[GeneratedImage] = Field(default_factory=list, max_length=20)
    references: list[Citation] = Field(min_length=8, max_length=120)
    generated_at: datetime

