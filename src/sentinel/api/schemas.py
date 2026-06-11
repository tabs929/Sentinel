"""Request/response schemas for the API layer."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from sentinel.models import (
    SessionStatus,
    Span,
    ToolInvocation,
    TurnStatus,
    Usage,
)


class CreateSessionRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class SendMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)


class SessionSummary(BaseModel):
    id: str
    title: str | None
    status: SessionStatus
    turn_count: int
    total_usage: Usage
    created_at: datetime
    updated_at: datetime


class SessionDetail(SessionSummary):
    messages: list[dict] = Field(default_factory=list)


class TurnView(BaseModel):
    id: str
    session_id: str
    index: int
    user_message: str
    assistant_message: str | None
    status: TurnStatus
    usage: Usage
    tool_invocations: list[ToolInvocation]
    trace_id: str | None
    error: str | None
    created_at: datetime


class SendMessageResponse(BaseModel):
    session_id: str
    turn: TurnView
    final_text: str
    blocked: bool
    failed_tools: list[str]
    unavailable_tools: list[str]


class TraceSummary(BaseModel):
    id: str
    session_id: str
    turn_id: str | None
    turn_index: int | None
    root_span_id: str | None
    created_at: datetime


class TraceDetail(TraceSummary):
    spans: list[Span]


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
