from __future__ import annotations

import concurrent.futures
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Callable, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from logging_utils import log_event, span_end, span_start
from schemas.common import ExecutionEvent
from schemas.state import GraphState

InT = TypeVar("InT", bound=BaseModel)
OutT = TypeVar("OutT", bound=BaseModel)


class HarnessError(RuntimeError):
    pass


class RecoverableHarnessError(HarnessError):
    pass


class ContractViolationError(HarnessError):
    pass


class PermissionDeniedError(HarnessError):
    pass


class AgentHarness(ABC, Generic[InT, OutT]):
    """
    Harness Engineering 落地核心：
    - 强类型输入/输出契约：InT / OutT (Pydantic v2)
    - 前置校验：pre_validate
    - 后置校验：post_validate
    - 异常处理：超时、可恢复错误重试、降级输出、错误上报
    - 可观测性：结构化日志 + ExecutionEvent 事件流写入 GraphState
    """

    role: str
    node: str
    max_retries: int
    timeout_s: float

    def __init__(
        self,
        *,
        logger: logging.Logger,
        role: str,
        node: str,
        max_retries: int,
        timeout_s: float,
    ) -> None:
        self._logger = logger
        self.role = role
        self.node = node
        self.max_retries = max_retries
        self.timeout_s = timeout_s

    @abstractmethod
    def pre_validate(self, *, input: InT, state: GraphState) -> None:
        raise NotImplementedError

    @abstractmethod
    def post_validate(self, *, output: OutT, input: InT, state: GraphState) -> None:
        raise NotImplementedError

    @abstractmethod
    def _invoke(self, *, input: InT, state: GraphState) -> OutT:
        raise NotImplementedError

    def degrade(self, *, input: InT, state: GraphState, error: Exception) -> OutT:
        raise error

    def _record_event(
        self,
        *,
        state: GraphState,
        status: str,
        attempt: int,
        message: str,
        duration_ms: int | None = None,
        input_summary: str | None = None,
        output_summary: str | None = None,
        error: Exception | None = None,
    ) -> None:
        evt = ExecutionEvent(
            timestamp=datetime.now(tz=timezone.utc),
            trace_id=state["trace_id"],
            node=self.node,
            role=self.role,
            status=status,
            duration_ms=duration_ms,
            attempt=attempt,
            message=message,
            input_summary=input_summary,
            output_summary=output_summary,
            error_type=type(error).__name__ if error else None,
            error_message=str(error)[:2000] if error else None,
        )
        state["execution_events"].append(evt.model_dump(mode="json"))

    def run(self, *, input: InT, state: GraphState) -> OutT:
        trace_id = state["trace_id"]
        self._record_event(
            state=state,
            status="start",
            attempt=1,
            message="agent_start",
            input_summary=_safe_model_summary(input),
        )
        log_event(
            self._logger,
            level=logging.INFO,
            message="agent_start",
            trace_id=trace_id,
            role=self.role,
            node=self.node,
            status="start",
        )

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            span = span_start("agent_run", trace_id=trace_id, role=self.role, node=self.node)
            try:
                self.pre_validate(input=input, state=state)
                output = _invoke_with_timeout(
                    fn=lambda: self._invoke(input=input, state=state),
                    timeout_s=self.timeout_s,
                )
                self.post_validate(output=output, input=input, state=state)

                duration_ms = span_end(span)
                self._record_event(
                    state=state,
                    status="success",
                    attempt=attempt,
                    message="agent_success",
                    duration_ms=duration_ms,
                    output_summary=_safe_model_summary(output),
                )
                log_event(
                    self._logger,
                    level=logging.INFO,
                    message="agent_success",
                    trace_id=trace_id,
                    role=self.role,
                    node=self.node,
                    status="success",
                    extra={"duration_ms": duration_ms, "attempt": attempt},
                )
                return output
            except (ValidationError, ContractViolationError, RecoverableHarnessError) as e:
                last_error = e
                duration_ms = span_end(span)
                self._record_event(
                    state=state,
                    status="retry",
                    attempt=attempt,
                    message="agent_retry",
                    duration_ms=duration_ms,
                    error=e,
                )
                log_event(
                    self._logger,
                    level=logging.WARNING,
                    message="agent_retry",
                    trace_id=trace_id,
                    role=self.role,
                    node=self.node,
                    status="retry",
                    extra={"duration_ms": duration_ms, "attempt": attempt, "error": str(e)[:500]},
                )
                continue
            except PermissionDeniedError as e:
                last_error = e
                duration_ms = span_end(span)
                self._record_event(
                    state=state,
                    status="failed",
                    attempt=attempt,
                    message="permission_denied",
                    duration_ms=duration_ms,
                    error=e,
                )
                log_event(
                    self._logger,
                    level=logging.ERROR,
                    message="permission_denied",
                    trace_id=trace_id,
                    role=self.role,
                    node=self.node,
                    status="failed",
                    extra={"duration_ms": duration_ms, "attempt": attempt},
                    exc_info=True,
                )
                raise
            except TimeoutError as e:
                last_error = e
                duration_ms = span_end(span)
                self._record_event(
                    state=state,
                    status="retry",
                    attempt=attempt,
                    message="agent_timeout",
                    duration_ms=duration_ms,
                    error=e,
                )
                log_event(
                    self._logger,
                    level=logging.WARNING,
                    message="agent_timeout",
                    trace_id=trace_id,
                    role=self.role,
                    node=self.node,
                    status="retry",
                    extra={"duration_ms": duration_ms, "attempt": attempt},
                )
                continue
            except Exception as e:
                last_error = e
                duration_ms = span_end(span)
                self._record_event(
                    state=state,
                    status="failed",
                    attempt=attempt,
                    message="agent_failed",
                    duration_ms=duration_ms,
                    error=e,
                )
                log_event(
                    self._logger,
                    level=logging.ERROR,
                    message="agent_failed",
                    trace_id=trace_id,
                    role=self.role,
                    node=self.node,
                    status="failed",
                    extra={"duration_ms": duration_ms, "attempt": attempt},
                    exc_info=True,
                )
                break

        if last_error is None:
            raise HarnessError("unknown_harness_failure")

        degraded = self.degrade(input=input, state=state, error=last_error)
        self._record_event(
            state=state,
            status="degraded",
            attempt=self.max_retries,
            message="agent_degraded",
            output_summary=_safe_model_summary(degraded),
            error=last_error,
        )
        log_event(
            self._logger,
            level=logging.WARNING,
            message="agent_degraded",
            trace_id=trace_id,
            role=self.role,
            node=self.node,
            status="degraded",
            extra={"error": str(last_error)[:500]},
        )
        return degraded


T = TypeVar("T")


def _invoke_with_timeout(*, fn: Callable[[], T], timeout_s: float) -> T:
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError as e:
        fut.cancel()
        ex.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError("agent_invoke_timeout") from e
    finally:
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


def _safe_model_summary(model: BaseModel) -> str:
    dumped = model.model_dump(mode="json")
    text = str(dumped)
    if len(text) <= 4000:
        return text
    return text[:3900] + "...(truncated)"
