"""Async persistence for sessions, turns, traces and spans.

A :class:`Store` wraps a single SQLite connection. Sessions survive process
restarts: the full message history is serialized on the session row, so an
agent can be rehydrated by id with complete prior context.
"""

from __future__ import annotations

import json
from datetime import datetime

import aiosqlite

from sentinel.config import Settings, get_settings
from sentinel.models import (
    Session,
    SessionStatus,
    Span,
    SpanKind,
    SpanStatus,
    ToolInvocation,
    Trace,
    Turn,
    TurnStatus,
    Usage,
)
from sentinel.state.db import connect, init_db


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class Store:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self.conn = conn

    # --- lifecycle -------------------------------------------------------------

    @classmethod
    async def open(cls, settings: Settings | None = None) -> Store:
        settings = settings or get_settings()
        conn = await connect(settings.db_path)
        await init_db(conn)
        return cls(conn)

    async def close(self) -> None:
        await self.conn.close()

    # --- sessions --------------------------------------------------------------

    async def save_session(self, session: Session) -> None:
        await self.conn.execute(
            """
            INSERT INTO sessions (id, title, status, messages_json,
                total_input_tokens, total_output_tokens, total_cost_usd,
                turn_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                status=excluded.status,
                messages_json=excluded.messages_json,
                total_input_tokens=excluded.total_input_tokens,
                total_output_tokens=excluded.total_output_tokens,
                total_cost_usd=excluded.total_cost_usd,
                turn_count=excluded.turn_count,
                updated_at=excluded.updated_at
            """,
            (
                session.id,
                session.title,
                session.status.value,
                json.dumps(session.messages),
                session.total_usage.input_tokens,
                session.total_usage.output_tokens,
                session.total_usage.cost_usd,
                session.turn_count,
                _dt(session.created_at),
                _dt(session.updated_at),
            ),
        )
        await self.conn.commit()

    async def get_session(self, session_id: str) -> Session | None:
        cur = await self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        row = await cur.fetchone()
        return self._row_to_session(row) if row else None

    async def list_sessions(self) -> list[Session]:
        cur = await self.conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC"
        )
        rows = await cur.fetchall()
        return [self._row_to_session(r) for r in rows]

    @staticmethod
    def _row_to_session(row: aiosqlite.Row) -> Session:
        return Session(
            id=row["id"],
            title=row["title"],
            status=SessionStatus(row["status"]),
            messages=json.loads(row["messages_json"]),
            total_usage=Usage(
                input_tokens=row["total_input_tokens"],
                output_tokens=row["total_output_tokens"],
                cost_usd=row["total_cost_usd"],
            ),
            turn_count=row["turn_count"],
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
        )

    # --- turns -----------------------------------------------------------------

    async def save_turn(self, turn: Turn) -> None:
        await self.conn.execute(
            """
            INSERT INTO turns (id, session_id, idx, user_message, assistant_message,
                status, input_tokens, output_tokens, cost_usd,
                tool_invocations_json, trace_id, error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                assistant_message=excluded.assistant_message,
                status=excluded.status,
                input_tokens=excluded.input_tokens,
                output_tokens=excluded.output_tokens,
                cost_usd=excluded.cost_usd,
                tool_invocations_json=excluded.tool_invocations_json,
                trace_id=excluded.trace_id,
                error=excluded.error
            """,
            (
                turn.id,
                turn.session_id,
                turn.index,
                turn.user_message,
                turn.assistant_message,
                turn.status.value,
                turn.usage.input_tokens,
                turn.usage.output_tokens,
                turn.usage.cost_usd,
                json.dumps([inv.model_dump() for inv in turn.tool_invocations]),
                turn.trace_id,
                turn.error,
                _dt(turn.created_at),
            ),
        )
        await self.conn.commit()

    async def get_turns(self, session_id: str) -> list[Turn]:
        cur = await self.conn.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY idx ASC",
            (session_id,),
        )
        rows = await cur.fetchall()
        return [self._row_to_turn(r) for r in rows]

    async def get_turn(self, session_id: str, index: int) -> Turn | None:
        cur = await self.conn.execute(
            "SELECT * FROM turns WHERE session_id = ? AND idx = ?",
            (session_id, index),
        )
        row = await cur.fetchone()
        return self._row_to_turn(row) if row else None

    @staticmethod
    def _row_to_turn(row: aiosqlite.Row) -> Turn:
        return Turn(
            id=row["id"],
            session_id=row["session_id"],
            index=row["idx"],
            user_message=row["user_message"],
            assistant_message=row["assistant_message"],
            status=TurnStatus(row["status"]),
            usage=Usage(
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                cost_usd=row["cost_usd"],
            ),
            tool_invocations=[
                ToolInvocation(**inv) for inv in json.loads(row["tool_invocations_json"])
            ],
            trace_id=row["trace_id"],
            error=row["error"],
            created_at=_parse_dt(row["created_at"]),
        )

    # --- traces & spans --------------------------------------------------------

    async def save_trace(self, trace: Trace) -> None:
        await self.conn.execute(
            """
            INSERT INTO traces (id, session_id, turn_id, turn_index, root_span_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                turn_id=excluded.turn_id,
                turn_index=excluded.turn_index,
                root_span_id=excluded.root_span_id
            """,
            (
                trace.id,
                trace.session_id,
                trace.turn_id,
                trace.turn_index,
                trace.root_span_id,
                _dt(trace.created_at),
            ),
        )
        for span in trace.spans:
            await self._save_span(span)
        await self.conn.commit()

    async def _save_span(self, span: Span) -> None:
        await self.conn.execute(
            """
            INSERT INTO spans (id, trace_id, parent_id, name, kind, status,
                start_time, end_time, duration_ms, attributes_json, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status=excluded.status,
                end_time=excluded.end_time,
                duration_ms=excluded.duration_ms,
                attributes_json=excluded.attributes_json,
                error=excluded.error
            """,
            (
                span.id,
                span.trace_id,
                span.parent_id,
                span.name,
                span.kind.value,
                span.status.value,
                _dt(span.start_time),
                _dt(span.end_time),
                span.duration_ms,
                json.dumps(span.attributes),
                span.error,
            ),
        )

    async def list_traces(self) -> list[Trace]:
        cur = await self.conn.execute(
            "SELECT * FROM traces ORDER BY created_at DESC"
        )
        rows = await cur.fetchall()
        return [
            Trace(
                id=r["id"],
                session_id=r["session_id"],
                turn_id=r["turn_id"],
                turn_index=r["turn_index"],
                root_span_id=r["root_span_id"],
                created_at=_parse_dt(r["created_at"]),
            )
            for r in rows
        ]

    async def get_trace(self, trace_id: str) -> Trace | None:
        cur = await self.conn.execute("SELECT * FROM traces WHERE id = ?", (trace_id,))
        row = await cur.fetchone()
        if not row:
            return None
        spans = await self._get_spans(trace_id)
        return Trace(
            id=row["id"],
            session_id=row["session_id"],
            turn_id=row["turn_id"],
            turn_index=row["turn_index"],
            root_span_id=row["root_span_id"],
            spans=spans,
            created_at=_parse_dt(row["created_at"]),
        )

    async def get_trace_for_turn(self, session_id: str, turn_index: int) -> Trace | None:
        cur = await self.conn.execute(
            "SELECT id FROM traces WHERE session_id = ? AND turn_index = ?",
            (session_id, turn_index),
        )
        row = await cur.fetchone()
        return await self.get_trace(row["id"]) if row else None

    async def _get_spans(self, trace_id: str) -> list[Span]:
        cur = await self.conn.execute(
            "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time ASC",
            (trace_id,),
        )
        rows = await cur.fetchall()
        return [
            Span(
                id=r["id"],
                trace_id=r["trace_id"],
                parent_id=r["parent_id"],
                name=r["name"],
                kind=SpanKind(r["kind"]),
                status=SpanStatus(r["status"]),
                start_time=_parse_dt(r["start_time"]),
                end_time=_parse_dt(r["end_time"]),
                duration_ms=r["duration_ms"],
                attributes=json.loads(r["attributes_json"]),
                error=r["error"],
            )
            for r in rows
        ]
