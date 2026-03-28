from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "trace_id"):
            base["trace_id"] = getattr(record, "trace_id")
        if hasattr(record, "role"):
            base["role"] = getattr(record, "role")
        if hasattr(record, "node"):
            base["node"] = getattr(record, "node")
        if hasattr(record, "status"):
            base["status"] = getattr(record, "status")
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)


def configure_logging(level: str) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)


@dataclass(frozen=True)
class Span:
    name: str
    trace_id: str
    role: str
    node: str
    start_s: float


def span_start(name: str, *, trace_id: str, role: str, node: str) -> Span:
    return Span(name=name, trace_id=trace_id, role=role, node=node, start_s=time.perf_counter())


def span_end(span: Span) -> int:
    return int((time.perf_counter() - span.start_s) * 1000)


def log_event(
    logger: logging.Logger,
    *,
    level: int,
    message: str,
    trace_id: str,
    role: str,
    node: str,
    status: str,
    extra: Mapping[str, object] | None = None,
    exc_info: bool = False,
) -> None:
    payload: dict[str, object] = {"trace_id": trace_id, "role": role, "node": node, "status": status}
    if extra:
        payload.update(dict(extra))
    logger.log(level, message, extra=payload, exc_info=exc_info)

