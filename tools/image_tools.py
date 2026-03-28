from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

from harness.base import PermissionDeniedError, RecoverableHarnessError
from tools.rate_limit import TokenBucket


@dataclass(frozen=True)
class ImageGenerationResult:
    file_path: str
    model: str
    width: int
    height: int
    generated_at: datetime
    skipped: bool
    skip_reason: str | None


class ImageToolbox:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        allowed_tools: list[str],
        rate_limit_per_minute: int,
        openai_api_key_present: bool,
        data_dir: Path,
        request_timeout_s: float,
    ) -> None:
        self._logger = logger
        self._allowed = set(allowed_tools)
        self._bucket = TokenBucket.per_minute(limit=rate_limit_per_minute)
        self._openai_key_present = openai_api_key_present
        self._data_dir = data_dir
        self._timeout_s = request_timeout_s

    def _require(self, tool_name: str) -> None:
        if tool_name not in self._allowed:
            raise PermissionDeniedError(f"tool_not_allowed: {tool_name}")
        if not self._bucket.consume(tokens=1):
            raise RecoverableHarnessError("tool_rate_limited")

    def generate_image(
        self,
        *,
        trace_id: str,
        prompt: str,
        filename_stem: str,
        size: str = "1024x1024",
    ) -> ImageGenerationResult:
        self._require("image_generation")

        now = datetime.now(tz=timezone.utc)
        out_dir = (self._data_dir / "images" / trace_id).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        if not self._openai_key_present:
            return ImageGenerationResult(
                file_path=str((out_dir / f"{filename_stem}.skipped.txt").resolve()),
                model="skipped_no_openai_key",
                width=1024,
                height=1024,
                generated_at=now,
                skipped=True,
                skip_reason="OPENAI_API_KEY missing, image generation skipped",
            )

        try:
            from langchain_openai import DallEAPIWrapper
        except Exception as e:
            raise RecoverableHarnessError(f"dalle_import_failed: {e}") from e

        dalle = DallEAPIWrapper()
        try:
            image_url = str(dalle.run(prompt))
        except Exception as e:
            raise RecoverableHarnessError(f"dalle_invoke_failed: {e}") from e

        if image_url.startswith("http"):
            png_path = (out_dir / f"{filename_stem}.png").resolve()
            try:
                resp = requests.get(image_url, timeout=self._timeout_s)
                resp.raise_for_status()
                png_path.write_bytes(resp.content)
                return ImageGenerationResult(
                    file_path=str(png_path),
                    model="dall-e",
                    width=int(size.split("x")[0]),
                    height=int(size.split("x")[1]),
                    generated_at=now,
                    skipped=False,
                    skip_reason=None,
                )
            except Exception:
                return ImageGenerationResult(
                    file_path=image_url,
                    model="dall-e",
                    width=int(size.split("x")[0]),
                    height=int(size.split("x")[1]),
                    generated_at=now,
                    skipped=False,
                    skip_reason="download_failed; embedded as remote URL",
                )

        return ImageGenerationResult(
            file_path=image_url,
            model="dall-e",
            width=int(size.split("x")[0]),
            height=int(size.split("x")[1]),
            generated_at=now,
            skipped=False,
            skip_reason="unknown_url_format",
        )

