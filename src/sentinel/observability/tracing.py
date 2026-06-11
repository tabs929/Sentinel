"""Tracer — builds a span tree for one turn and emits structured events.

A *trace* is the full tree for a single turn:

    turn (root span)
    ├── chat <model>            (LLM call span)
    ├── execute_tool <name>     (tool call span)
    │   └── retry <name> #2     (retry attempt span)
    ├── chat <model>            (LLM call span)
    └── execute_tool <name>     (tool call span)

Spans are accumulated in memory during the turn and persisted in one shot by
the state store. Every span start/end also emits an :class:`AgentEvent` to any
registered listeners (e.g. the CLI renderer), so the loop only has to talk to
the tracer.

Attribute names follow the OpenTelemetry GenAI semantic conventions.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sentinel.models import (
    AgentEvent,
    EventType,
    Span,
    SpanKind,
    SpanStatus,
    Trace,
)

Listener = Callable[[AgentEvent], None]


class GenAI:
    """OpenTelemetry GenAI semantic-convention attribute keys."""

    SYSTEM = "gen_ai.system"
    OPERATION_NAME = "gen_ai.operation.name"
    REQUEST_MODEL = "gen_ai.request.model"
    REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    RESPONSE_MODEL = "gen_ai.response.model"
    RESPONSE_FINISH_REASON = "gen_ai.response.finish_reasons"
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    # Sentinel-specific extensions.
    USAGE_COST_USD = "gen_ai.usage.cost_usd"
    TOOL_NAME = "gen_ai.tool.name"
    TOOL_INPUT = "gen_ai.tool.input"
    TOOL_ATTEMPTS = "sentinel.tool.attempts"
    TOOL_UNAVAILABLE = "sentinel.tool.unavailable"
    RETRY_ATTEMPT = "sentinel.retry.attempt"
    RETRY_WAIT_S = "sentinel.retry.wait_seconds"


_KIND_TO_START_EVENT = {
    SpanKind.TURN: EventType.TURN_STARTED,
    SpanKind.LLM: EventType.LLM_CALL_STARTED,
    SpanKind.TOOL: EventType.TOOL_CALL_STARTED,
    SpanKind.GUARDRAIL: EventType.GUARDRAIL_TRIGGERED,
}
_KIND_TO_END_EVENT = {
    SpanKind.TURN: EventType.TURN_ENDED,
    SpanKind.LLM: EventType.LLM_CALL_ENDED,
    SpanKind.TOOL: EventType.TOOL_CALL_ENDED,
}


class Tracer:
    def __init__(self, trace: Trace, listeners: list[Listener] | None = None) -> None:
        self.trace = trace
        self._listeners: list[Listener] = list(listeners or [])
        # Monotonic clock keyed by span id for accurate durations.
        self._starts: dict[str, float] = {}

    # --- listeners -------------------------------------------------------------

    def add_listener(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def emit(self, event: AgentEvent) -> None:
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:  # noqa: BLE001 - a bad listener must not break the agent
                pass

    # --- spans -----------------------------------------------------------------

    def start_span(
        self,
        name: str,
        kind: SpanKind,
        *,
        parent_id: str | None = None,
        attributes: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> Span:
        span = Span(
            trace_id=self.trace.id,
            parent_id=parent_id,
            name=name,
            kind=kind,
            attributes=attributes or {},
            start_time=datetime.now(UTC),
        )
        self.trace.spans.append(span)
        self._starts[span.id] = time.perf_counter()

        if kind == SpanKind.TURN and self.trace.root_span_id is None:
            self.trace.root_span_id = span.id

        event_type = _KIND_TO_START_EVENT.get(kind)
        if event_type is not None:
            self.emit(
                AgentEvent(
                    type=event_type,
                    span_id=span.id,
                    parent_id=parent_id,
                    message=message,
                    data=dict(span.attributes),
                )
            )
        return span

    def end_span(
        self,
        span: Span,
        *,
        status: SpanStatus = SpanStatus.OK,
        attributes: dict[str, Any] | None = None,
        error: str | None = None,
        message: str | None = None,
    ) -> Span:
        if attributes:
            span.attributes.update(attributes)
        span.status = status
        span.error = error
        span.end_time = datetime.now(UTC)
        started = self._starts.pop(span.id, None)
        if started is not None:
            span.duration_ms = (time.perf_counter() - started) * 1000
        else:
            span.duration_ms = max(
                0.0, (span.end_time - span.start_time).total_seconds() * 1000
            )

        event_type = _KIND_TO_END_EVENT.get(span.kind)
        if event_type is not None:
            data = dict(span.attributes)
            data["status"] = status.value
            data["duration_ms"] = span.duration_ms
            if error:
                data["error"] = error
            self.emit(
                AgentEvent(
                    type=event_type,
                    span_id=span.id,
                    parent_id=span.parent_id,
                    message=message,
                    data=data,
                )
            )
        return span

    def record_retry(
        self,
        parent_span: Span,
        *,
        tool_name: str,
        attempt: int,
        error: str,
        wait_seconds: float,
    ) -> Span:
        """Record a single retry attempt as a zero-duration child span."""
        now = datetime.now(UTC)
        span = Span(
            trace_id=self.trace.id,
            parent_id=parent_span.id,
            name=f"retry {tool_name} #{attempt}",
            kind=SpanKind.RETRY,
            status=SpanStatus.ERROR,
            start_time=now,
            end_time=now,
            error=error,
            attributes={
                GenAI.TOOL_NAME: tool_name,
                GenAI.RETRY_ATTEMPT: attempt,
                GenAI.RETRY_WAIT_S: wait_seconds,
            },
        )
        self.trace.spans.append(span)
        self.emit(
            AgentEvent(
                type=EventType.TOOL_RETRY,
                span_id=span.id,
                parent_id=parent_span.id,
                message=f"Retrying {tool_name} (attempt {attempt}) after error: {error}",
                data={"attempt": attempt, "wait_seconds": wait_seconds, "error": error},
            )
        )
        return span

    def record_guardrail(
        self,
        *,
        parent_id: str | None,
        name: str,
        triggered: list[str],
        reason: str | None,
        blocked: bool,
    ) -> Span:
        now = datetime.now(UTC)
        span = Span(
            trace_id=self.trace.id,
            parent_id=parent_id,
            name=f"guardrail {name}",
            kind=SpanKind.GUARDRAIL,
            status=SpanStatus.ERROR if blocked else SpanStatus.OK,
            start_time=now,
            end_time=now,
            error=reason if blocked else None,
            attributes={
                "sentinel.guardrail.name": name,
                "sentinel.guardrail.blocked": blocked,
                "sentinel.guardrail.triggered": triggered,
            },
        )
        self.trace.spans.append(span)
        self.emit(
            AgentEvent(
                type=EventType.GUARDRAIL_TRIGGERED,
                span_id=span.id,
                parent_id=parent_id,
                message=reason,
                data={"name": name, "blocked": blocked, "triggered": triggered},
            )
        )
        return span

    def message(self, text: str, **data: Any) -> None:
        """Emit a free-form informational event (not a span)."""
        self.emit(AgentEvent(type=EventType.MESSAGE, message=text, data=data))
