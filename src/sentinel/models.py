"""Shared Pydantic models — the contracts every layer plugs into.

These models intentionally live in one module so the tools layer, agent loop,
state store, observability layer, and API all speak the same language.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class ToolSpec(BaseModel):
    """The advertised interface of a tool, in Anthropic tool-schema shape."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolResult(BaseModel):
    """The *only* thing a tool ever returns. Tools never raise."""

    tool_name: str
    ok: bool
    # Human/LLM-readable rendering fed back into the conversation.
    content: str
    # Structured payload for programmatic consumers (may be None on failure).
    data: dict[str, Any] | None = None
    error: str | None = None
    # True when the tool could not run because it is not configured
    # (e.g. missing API key). This is a graceful state, not a crash.
    unavailable: bool = False
    latency_ms: float = 0.0
    # Number of attempts made (1 = no retries).
    attempts: int = 1

    @classmethod
    def success(
        cls,
        tool_name: str,
        content: str,
        data: dict[str, Any] | None = None,
        *,
        latency_ms: float = 0.0,
        attempts: int = 1,
    ) -> ToolResult:
        return cls(
            tool_name=tool_name,
            ok=True,
            content=content,
            data=data,
            latency_ms=latency_ms,
            attempts=attempts,
        )

    @classmethod
    def failure(
        cls,
        tool_name: str,
        error: str,
        *,
        latency_ms: float = 0.0,
        attempts: int = 1,
    ) -> ToolResult:
        return cls(
            tool_name=tool_name,
            ok=False,
            content=f"Tool '{tool_name}' failed: {error}",
            error=error,
            latency_ms=latency_ms,
            attempts=attempts,
        )

    @classmethod
    def tool_unavailable(cls, tool_name: str, reason: str) -> ToolResult:
        return cls(
            tool_name=tool_name,
            ok=False,
            unavailable=True,
            error=reason,
            content=(
                f"Tool '{tool_name}' is unavailable: {reason}. "
                f"It was not invoked."
            ),
        )


# ---------------------------------------------------------------------------
# Conversation / sessions
# ---------------------------------------------------------------------------


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class TurnStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"  # rejected by a guardrail


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


class ToolInvocation(BaseModel):
    """A record of one tool call within a turn (denormalized for the UI)."""

    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    ok: bool
    unavailable: bool = False
    error: str | None = None
    latency_ms: float = 0.0
    attempts: int = 1


class Turn(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("turn"))
    session_id: str
    index: int
    user_message: str
    assistant_message: str | None = None
    status: TurnStatus = TurnStatus.RUNNING
    usage: Usage = Field(default_factory=Usage)
    tool_invocations: list[ToolInvocation] = Field(default_factory=list)
    trace_id: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class Session(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("sess"))
    title: str | None = None
    status: SessionStatus = SessionStatus.ACTIVE
    # Full Anthropic-shaped message history, persisted so the session can be
    # resumed across process restarts.
    messages: list[dict[str, Any]] = Field(default_factory=list)
    total_usage: Usage = Field(default_factory=Usage)
    turn_count: int = 0
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Observability — spans & events (OpenTelemetry GenAI flavored)
# ---------------------------------------------------------------------------


class SpanKind(str, Enum):
    TURN = "turn"  # root span for a turn
    LLM = "llm"  # one Claude call
    TOOL = "tool"  # one tool invocation
    RETRY = "retry"  # a single retry attempt (child of a TOOL span)
    GUARDRAIL = "guardrail"


class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


class Span(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("span"))
    trace_id: str
    parent_id: str | None = None
    name: str
    kind: SpanKind
    status: SpanStatus = SpanStatus.UNSET
    start_time: datetime = Field(default_factory=_utcnow)
    end_time: datetime | None = None
    duration_ms: float = 0.0
    # OpenTelemetry GenAI-style attributes, e.g. gen_ai.usage.input_tokens.
    attributes: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class Trace(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("trace"))
    session_id: str
    turn_id: str | None = None
    turn_index: int | None = None
    root_span_id: str | None = None
    spans: list[Span] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


class EventType(str, Enum):
    TURN_STARTED = "turn.started"
    TURN_ENDED = "turn.ended"
    LLM_CALL_STARTED = "llm.call.started"
    LLM_CALL_ENDED = "llm.call.ended"
    TOOL_CALL_STARTED = "tool.call.started"
    TOOL_CALL_ENDED = "tool.call.ended"
    TOOL_RETRY = "tool.retry"
    GUARDRAIL_TRIGGERED = "guardrail.triggered"
    MESSAGE = "message"


class AgentEvent(BaseModel):
    """A structured event emitted by the agent loop and consumed by listeners
    (the tracer, the CLI renderer, etc.)."""

    type: EventType
    timestamp: datetime = Field(default_factory=_utcnow)
    span_id: str | None = None
    parent_id: str | None = None
    message: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


class GuardrailDecision(BaseModel):
    allowed: bool
    # Names of triggered rules, for observability.
    triggered: list[str] = Field(default_factory=list)
    reason: str | None = None
    # Possibly-sanitized text (e.g. with leaked secrets redacted).
    sanitized: str | None = None
