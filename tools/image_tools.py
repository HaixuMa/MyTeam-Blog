from __future__ import annotations

import logging
import time
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
        api_key: str | None,
        image_provider: str = "openai",
        image_model: str = "dall-e",
        image_base_url: str | None = None,
        image_poll_interval_s: float = 5.0,
        image_max_poll_seconds: int = 300,
        data_dir: Path,
        request_timeout_s: float,
    ) -> None:
        self._logger = logger
        self._allowed = set(allowed_tools)
        self._bucket = TokenBucket.per_minute(limit=rate_limit_per_minute)
        self._api_key = api_key
        self._image_provider = image_provider.strip().lower()
        self._image_model = image_model.strip()
        self._image_base_url = (image_base_url or "").strip() or None
        self._poll_interval_s = float(image_poll_interval_s)
        self._max_poll_seconds = int(image_max_poll_seconds)
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

        if not self._api_key:
            return ImageGenerationResult(
                file_path=str((out_dir / f"{filename_stem}.skipped.txt").resolve()),
                model="skipped_no_openai_key",
                width=1024,
                height=1024,
                generated_at=now,
                skipped=True,
                skip_reason="API key missing, image generation skipped",
            )

        if self._image_provider == "modelscope":
            return self._generate_image_modelscope(
                trace_id=trace_id,
                prompt=prompt,
                filename_stem=filename_stem,
                size=size,
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

    def _generate_image_modelscope(
        self,
        *,
        trace_id: str,
        prompt: str,
        filename_stem: str,
        size: str,
    ) -> ImageGenerationResult:
        if not self._image_base_url:
            raise RecoverableHarnessError("IMAGE_BASE_URL missing for modelscope image provider")

        now = datetime.now(tz=timezone.utc)
        out_dir = (self._data_dir / "images" / trace_id).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        base = self._image_base_url.rstrip("/")
        common_headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(
                f"{base}/v1/images/generations",
                headers={**common_headers, "X-ModelScope-Async-Mode": "true"},
                json={"model": self._image_model, "prompt": prompt},
                timeout=self._timeout_s,
            )
            resp.raise_for_status()
        except Exception as e:
            raise RecoverableHarnessError(f"modelscope_image_submit_failed: {e}") from e

        data = resp.json()
        task_id = str(data.get("task_id") or "")
        if not task_id:
            raise RecoverableHarnessError("modelscope_image_missing_task_id")

        deadline = time.time() + max(1, self._max_poll_seconds)
        last_status: str | None = None
        while True:
            if time.time() >= deadline:
                raise RecoverableHarnessError(
                    f"modelscope_image_task_timeout: task_id={task_id} last_status={last_status}"
                )

            try:
                r = requests.get(
                    f"{base}/v1/tasks/{task_id}",
                    headers={**common_headers, "X-ModelScope-Task-Type": "image_generation"},
                    timeout=self._timeout_s,
                )
                r.raise_for_status()
                payload = r.json()
            except Exception as e:
                raise RecoverableHarnessError(f"modelscope_image_poll_failed: {e}") from e

            last_status = str(payload.get("task_status") or "")
            if last_status == "SUCCEED":
                out_images = payload.get("output_images") or []
                if not isinstance(out_images, list) or not out_images:
                    raise RecoverableHarnessError("modelscope_image_no_output_images")
                url = str(out_images[0])
                png_path = (out_dir / f"{filename_stem}.png").resolve()
                try:
                    img_resp = requests.get(url, timeout=self._timeout_s)
                    img_resp.raise_for_status()
                    png_path.write_bytes(img_resp.content)
                    return ImageGenerationResult(
                        file_path=str(png_path),
                        model=self._image_model,
                        width=int(size.split("x")[0]),
                        height=int(size.split("x")[1]),
                        generated_at=now,
                        skipped=False,
                        skip_reason=None,
                    )
                except Exception:
                    return ImageGenerationResult(
                        file_path=url,
                        model=self._image_model,
                        width=int(size.split("x")[0]),
                        height=int(size.split("x")[1]),
                        generated_at=now,
                        skipped=False,
                        skip_reason="download_failed; embedded as remote URL",
                    )

            if last_status == "FAILED":
                raise RecoverableHarnessError(f"modelscope_image_task_failed: task_id={task_id}")

            time.sleep(max(0.5, self._poll_interval_s))

