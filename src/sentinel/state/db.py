"""SQLite schema and connection helpers (async via aiosqlite)."""

from __future__ import annotations

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id                  TEXT PRIMARY KEY,
    title               TEXT,
    status              TEXT NOT NULL,
    messages_json       TEXT NOT NULL,
    total_input_tokens  INTEGER NOT NULL DEFAULT 0,
    total_output_tokens INTEGER NOT NULL DEFAULT 0,
    total_cost_usd      REAL NOT NULL DEFAULT 0,
    turn_count          INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS turns (
    id                  TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL REFERENCES sessions(id),
    idx                 INTEGER NOT NULL,
    user_message        TEXT NOT NULL,
    assistant_message   TEXT,
    status              TEXT NOT NULL,
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    cost_usd            REAL NOT NULL DEFAULT 0,
    tool_invocations_json TEXT NOT NULL DEFAULT '[]',
    trace_id            TEXT,
    error               TEXT,
    created_at          TEXT NOT NULL,
    UNIQUE(session_id, idx)
);

CREATE TABLE IF NOT EXISTS traces (
    id                  TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL,
    turn_id             TEXT,
    turn_index          INTEGER,
    root_span_id        TEXT,
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS spans (
    id                  TEXT PRIMARY KEY,
    trace_id            TEXT NOT NULL REFERENCES traces(id),
    parent_id           TEXT,
    name                TEXT NOT NULL,
    kind                TEXT NOT NULL,
    status              TEXT NOT NULL,
    start_time          TEXT NOT NULL,
    end_time            TEXT,
    duration_ms         REAL NOT NULL DEFAULT 0,
    attributes_json     TEXT NOT NULL DEFAULT '{}',
    error               TEXT
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id);
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
"""


async def connect(db_path: str) -> aiosqlite.Connection:
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON;")
    await conn.execute("PRAGMA journal_mode = WAL;")
    return conn


async def init_db(conn: aiosqlite.Connection) -> None:
    await conn.executescript(SCHEMA)
    await conn.commit()
