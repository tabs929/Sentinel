"""FastAPI app.

All endpoints are async, validated by Pydantic, and return structured JSON.
Unhandled errors are converted to structured JSON, never stack traces.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from sentinel.api.schemas import (
    CreateSessionRequest,
    SendMessageRequest,
    SendMessageResponse,
    SessionDetail,
    SessionSummary,
    TraceDetail,
    TraceSummary,
    TurnView,
)
from sentinel.models import Session, Trace, Turn
from sentinel.service import SentinelService, SessionNotFoundError


class ApiError(Exception):
    def __init__(self, status_code: int, error: str, detail: str | None = None) -> None:
        self.status_code = status_code
        self.error = error
        self.detail = detail


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = await SentinelService.create()
    try:
        yield
    finally:
        await app.state.service.close()


app = FastAPI(
    title="Sentinel API",
    version="0.1.0",
    description="A reliable, observable multi-tool AI agent.",
    lifespan=lifespan,
)


def get_service(request: Request) -> SentinelService:
    return request.app.state.service


# --- error handlers --------------------------------------------------------


@app.exception_handler(ApiError)
async def _api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.error, "detail": exc.detail},
    )


@app.exception_handler(RequestValidationError)
async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def _unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_error", "detail": str(exc)},
    )


# --- serializers -----------------------------------------------------------


def _session_summary(s: Session) -> SessionSummary:
    return SessionSummary(
        id=s.id,
        title=s.title,
        status=s.status,
        turn_count=s.turn_count,
        total_usage=s.total_usage,
        created_at=s.created_at,
        updated_at=s.updated_at,
    )


def _turn_view(t: Turn) -> TurnView:
    return TurnView(**t.model_dump())


def _trace_summary(t: Trace) -> TraceSummary:
    return TraceSummary(
        id=t.id,
        session_id=t.session_id,
        turn_id=t.turn_id,
        turn_index=t.turn_index,
        root_span_id=t.root_span_id,
        created_at=t.created_at,
    )


# --- routes ----------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/sessions", response_model=SessionSummary, status_code=201)
async def create_session(req: CreateSessionRequest, request: Request) -> SessionSummary:
    service = get_service(request)
    session = await service.start_session(title=req.title)
    return _session_summary(session)


@app.get("/sessions", response_model=list[SessionSummary])
async def list_sessions(request: Request) -> list[SessionSummary]:
    service = get_service(request)
    return [_session_summary(s) for s in await service.list_sessions()]


@app.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str, request: Request) -> SessionDetail:
    service = get_service(request)
    session = await service.get_session(session_id)
    if session is None:
        raise ApiError(404, "session_not_found", f"No session '{session_id}'.")
    return SessionDetail(**_session_summary(session).model_dump(), messages=session.messages)


@app.get("/sessions/{session_id}/turns", response_model=list[TurnView])
async def get_session_turns(session_id: str, request: Request) -> list[TurnView]:
    service = get_service(request)
    if await service.get_session(session_id) is None:
        raise ApiError(404, "session_not_found", f"No session '{session_id}'.")
    return [_turn_view(t) for t in await service.get_turns(session_id)]


@app.post(
    "/sessions/{session_id}/messages",
    response_model=SendMessageResponse,
)
async def send_message(
    session_id: str, req: SendMessageRequest, request: Request
) -> SendMessageResponse:
    service = get_service(request)
    try:
        result = await service.send_message(session_id, req.message)
    except SessionNotFoundError as exc:
        raise ApiError(404, "session_not_found", f"No session '{exc}'.") from exc
    except RuntimeError as exc:
        # e.g. ANTHROPIC_API_KEY not configured.
        raise ApiError(503, "llm_unavailable", str(exc)) from exc
    return SendMessageResponse(
        session_id=session_id,
        turn=_turn_view(result.turn),
        final_text=result.final_text,
        blocked=result.blocked,
        failed_tools=result.failed_tools,
        unavailable_tools=result.unavailable_tools,
    )


@app.get("/traces", response_model=list[TraceSummary])
async def list_traces(request: Request) -> list[TraceSummary]:
    service = get_service(request)
    return [_trace_summary(t) for t in await service.list_traces()]


@app.get("/traces/{trace_id}", response_model=TraceDetail)
async def get_trace(trace_id: str, request: Request) -> TraceDetail:
    service = get_service(request)
    trace = await service.get_trace(trace_id)
    if trace is None:
        raise ApiError(404, "trace_not_found", f"No trace '{trace_id}'.")
    return TraceDetail(**_trace_summary(trace).model_dump(), spans=trace.spans)


@app.get(
    "/sessions/{session_id}/turns/{turn_index}/trace",
    response_model=TraceDetail,
)
async def get_turn_trace(
    session_id: str, turn_index: int, request: Request
) -> TraceDetail:
    service = get_service(request)
    trace = await service.get_trace_for_turn(session_id, turn_index)
    if trace is None:
        raise ApiError(
            404,
            "trace_not_found",
            f"No trace for session '{session_id}' turn {turn_index}.",
        )
    return TraceDetail(**_trace_summary(trace).model_dump(), spans=trace.spans)
