from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from harness.base import AgentHarness, ContractViolationError, RecoverableHarnessError
from schemas.imaging import ArticleWithImages, GeneratedImage
from schemas.state import GraphState
from schemas.writing import FigureRequest, TechnicalArticleDraft
from tools.image_tools import ImageToolbox


class _PromptOut(BaseModel):
    prompt: str = Field(min_length=10, max_length=2000)
    safety_notes: str = Field(default="", max_length=800)


class ImagingAgentHarness(AgentHarness[TechnicalArticleDraft, ArticleWithImages]):
    def __init__(
        self,
        *,
        logger: logging.Logger,
        llm: BaseChatModel,
        toolbox: ImageToolbox,
        max_retries: int,
        timeout_s: float,
        max_image_retries: int,
    ) -> None:
        super().__init__(
            logger=logger,
            role="imaging_agent",
            node="imaging",
            max_retries=max_retries,
            timeout_s=timeout_s,
        )
        self._llm = llm
        self._toolbox = toolbox
        self._max_image_retries = max_image_retries

    def pre_validate(self, *, input: TechnicalArticleDraft, state: GraphState) -> None:
        if not input.markdown.strip():
            raise ContractViolationError("empty_markdown")

    def post_validate(self, *, output: ArticleWithImages, input: TechnicalArticleDraft, state: GraphState) -> None:
        if output.plan_id != input.plan_id:
            raise ContractViolationError("plan_id_mismatch")
        for fr in input.figure_requests:
            anchor_token = f"[[FIG:{fr.figure_id}:{fr.paragraph_anchor}]]"
            if anchor_token in output.markdown:
                raise ContractViolationError("unreplaced_figure_anchor")

    def _invoke(self, *, input: TechnicalArticleDraft, state: GraphState) -> ArticleWithImages:
        trace_id = state["trace_id"]

        sys = SystemMessage(
            content=(
                "你是技术文章配图设计师。你需要把 figure_request 转换为高质量专业文生图 prompt。"
                "prompt 必须清晰描述图类型、元素、布局、风格，并避免侵权与敏感内容。"
                "输出必须是结构化对象，仅包含 prompt 与 safety_notes。"
            )
        )
        structured = self._llm.with_structured_output(_PromptOut)

        md = input.markdown
        images: list[GeneratedImage] = []

        for fr in input.figure_requests:
            prompt_in = (
                f"figure_id: {fr.figure_id}\n"
                f"figure_type: {fr.figure_type}\n"
                f"purpose: {fr.purpose}\n"
                f"must_include: {fr.must_include}\n"
                f"style_guidelines: {fr.style_guidelines}\n"
                f"seed: {fr.prompt_seed}\n"
            )
            p = structured.invoke([sys, {"role": "user", "content": prompt_in}])

            anchor_token = f"[[FIG:{fr.figure_id}:{fr.paragraph_anchor}]]"
            success = False
            last_err: Exception | None = None
            for attempt in range(1, self._max_image_retries + 1):
                try:
                    gen = self._toolbox.generate_image(
                        trace_id=trace_id,
                        prompt=p.prompt,
                        filename_stem=f"{fr.figure_id}_{attempt}",
                    )
                    images.append(
                        GeneratedImage(
                            figure_id=fr.figure_id,
                            file_path=gen.file_path,
                            prompt=p.prompt,
                            model=gen.model,
                            width=gen.width,
                            height=gen.height,
                            generated_at=gen.generated_at,
                            skipped=gen.skipped,
                            skip_reason=gen.skip_reason,
                        )
                    )
                    md = _embed_figure(
                        markdown=md,
                        figure_request=fr,
                        anchor_token=anchor_token,
                        image_path=gen.file_path,
                        skipped=gen.skipped,
                        skip_reason=gen.skip_reason,
                    )
                    success = True
                    break
                except RecoverableHarnessError as e:
                    last_err = e
                    continue
                except Exception as e:
                    last_err = e
                    break

            if not success:
                images.append(
                    GeneratedImage(
                        figure_id=fr.figure_id,
                        file_path="",
                        prompt=p.prompt,
                        model="skipped_error",
                        width=1024,
                        height=1024,
                        generated_at=datetime.now(tz=timezone.utc),
                        skipped=True,
                        skip_reason=str(last_err)[:600] if last_err else "unknown",
                    )
                )
                md = _embed_figure(
                    markdown=md,
                    figure_request=fr,
                    anchor_token=anchor_token,
                    image_path="",
                    skipped=True,
                    skip_reason=str(last_err)[:600] if last_err else "unknown",
                )

        return ArticleWithImages(
            plan_id=input.plan_id,
            markdown=md,
            images=images,
            references=input.references,
            generated_at=datetime.now(tz=timezone.utc),
        )


def _embed_figure(
    *,
    markdown: str,
    figure_request: FigureRequest,
    anchor_token: str,
    image_path: str,
    skipped: bool,
    skip_reason: str | None,
) -> str:
    if anchor_token not in markdown:
        return markdown
    if skipped:
        replacement = f"\n\n> 配图跳过（{figure_request.figure_id}）：{skip_reason or 'unknown'}\n\n"
        return markdown.replace(anchor_token, replacement)
    alt = f"{figure_request.figure_type}:{figure_request.figure_id}"
    replacement = f"\n\n![{alt}]({image_path})\n\n"
    return markdown.replace(anchor_token, replacement)

