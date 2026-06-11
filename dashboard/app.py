"""Sentinel observability dashboard (Streamlit).

Reads the SQLite database directly (read-only) and renders four views:

1. Sessions          - every session with cost, turn count, status.
2. Turn inspector    - the tool-call sequence for a chosen turn.
3. Trace viewer      - the span tree for a turn as a waterfall diagram.
4. Cost & reliability- spend over time, tool success rates, retries, latency.

Run with:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
import sqlite3

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DB_PATH = os.environ.get("SENTINEL_DB_PATH", "sentinel.db")

st.set_page_config(page_title="Sentinel Observability", layout="wide")


@st.cache_resource
def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _df(conn: sqlite3.Connection, query: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(query, conn, params=params)


def _table_exists(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("SELECT 1 FROM sessions LIMIT 1")
        return True
    except sqlite3.Error:
        return False


def sessions_view(conn: sqlite3.Connection) -> None:
    st.header("Sessions")
    df = _df(
        conn,
        """
        SELECT id, title, status, turn_count,
               total_input_tokens, total_output_tokens, total_cost_usd,
               created_at, updated_at
        FROM sessions ORDER BY created_at DESC
        """,
    )
    if df.empty:
        st.info("No sessions yet. Run the demo or send a message via the API/CLI.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessions", len(df))
    c2.metric("Total turns", int(df["turn_count"].sum()))
    c3.metric("Total cost", f"${df['total_cost_usd'].sum():.4f}")
    c4.metric("Total tokens", int(df["total_input_tokens"].sum() + df["total_output_tokens"].sum()))

    st.dataframe(
        df.rename(
            columns={
                "total_cost_usd": "cost_usd",
                "total_input_tokens": "in_tokens",
                "total_output_tokens": "out_tokens",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def turn_inspector(conn: sqlite3.Connection) -> None:
    st.header("Turn inspector")
    sessions = _df(conn, "SELECT id, title FROM sessions ORDER BY created_at DESC")
    if sessions.empty:
        st.info("No sessions yet.")
        return

    labels = {
        f"{r.id} — {r.title or '(untitled)'}": r.id for r in sessions.itertuples()
    }
    chosen = st.selectbox("Session", list(labels.keys()))
    session_id = labels[chosen]

    turns = _df(
        conn,
        "SELECT * FROM turns WHERE session_id = ? ORDER BY idx",
        (session_id,),
    )
    if turns.empty:
        st.info("This session has no turns yet.")
        return

    turn_idx = st.selectbox("Turn", turns["idx"].tolist())
    turn = turns[turns["idx"] == turn_idx].iloc[0]

    st.subheader("Conversation")
    st.markdown(f"**User:** {turn['user_message']}")
    st.markdown(f"**Assistant:** {turn['assistant_message'] or '(none)'}")

    cols = st.columns(4)
    cols[0].metric("Status", turn["status"])
    cols[1].metric("Cost", f"${turn['cost_usd']:.5f}")
    cols[2].metric("In tokens", int(turn["input_tokens"]))
    cols[3].metric("Out tokens", int(turn["output_tokens"]))
    if turn["error"]:
        st.error(f"Turn error: {turn['error']}")

    st.subheader("Tool call sequence")
    invocations = json.loads(turn["tool_invocations_json"] or "[]")
    if not invocations:
        st.write("No tools were called in this turn.")
    else:
        inv_df = pd.DataFrame(invocations)
        inv_df["outcome"] = inv_df.apply(
            lambda r: "unavailable" if r.get("unavailable") else ("ok" if r["ok"] else "failed"),
            axis=1,
        )
        st.dataframe(
            inv_df[["tool_name", "outcome", "attempts", "latency_ms", "error"]],
            use_container_width=True,
            hide_index=True,
        )


def trace_viewer(conn: sqlite3.Connection) -> None:
    st.header("Trace viewer")
    traces = _df(
        conn,
        """
        SELECT t.id, t.session_id, t.turn_index, t.created_at
        FROM traces t ORDER BY t.created_at DESC
        """,
    )
    if traces.empty:
        st.info("No traces yet.")
        return

    labels = {
        f"{r.created_at} — session {r.session_id} turn {r.turn_index}": r.id
        for r in traces.itertuples()
    }
    chosen = st.selectbox("Trace", list(labels.keys()))
    trace_id = labels[chosen]

    spans = _df(
        conn,
        "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time",
        (trace_id,),
    )
    if spans.empty:
        st.info("No spans on this trace.")
        return

    spans["start_dt"] = pd.to_datetime(spans["start_time"])
    spans["end_dt"] = pd.to_datetime(spans["end_time"]).fillna(spans["start_dt"])
    t0 = spans["start_dt"].min()
    spans["start_ms"] = (spans["start_dt"] - t0).dt.total_seconds() * 1000
    spans["finish_ms"] = spans["start_ms"] + spans["duration_ms"].clip(lower=0.5)
    spans["row"] = [f"{i:02d} {n}" for i, n in zip(spans.index, spans["name"], strict=False)]

    color_map = {
        "turn": "#3b82f6",
        "llm": "#a855f7",
        "tool": "#eab308",
        "retry": "#ef4444",
        "guardrail": "#dc2626",
    }
    fig = go.Figure()
    for _, s in spans.iterrows():
        fig.add_trace(
            go.Bar(
                x=[s["finish_ms"] - s["start_ms"]],
                y=[s["row"]],
                base=[s["start_ms"]],
                orientation="h",
                marker_color=color_map.get(s["kind"], "#9ca3af"),
                name=s["kind"],
                hovertext=(
                    f"{s['name']}<br>kind={s['kind']}<br>status={s['status']}"
                    f"<br>{s['duration_ms']:.1f} ms"
                    + (f"<br>error: {s['error']}" if s["error"] else "")
                ),
                hoverinfo="text",
                showlegend=False,
            )
        )
    fig.update_layout(
        height=60 + 28 * len(spans),
        xaxis_title="milliseconds since turn start",
        yaxis=dict(autorange="reversed"),
        barmode="overlay",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Spans")
    st.dataframe(
        spans[["name", "kind", "status", "duration_ms", "error"]],
        use_container_width=True,
        hide_index=True,
    )


def metrics_view(conn: sqlite3.Connection) -> None:
    st.header("Cost & reliability metrics")

    turns = _df(conn, "SELECT * FROM turns ORDER BY created_at")
    spans = _df(conn, "SELECT * FROM spans")

    if turns.empty:
        st.info("No data yet.")
        return

    # --- spend over time ---
    turns["created_dt"] = pd.to_datetime(turns["created_at"])
    turns["cum_cost"] = turns["cost_usd"].cumsum()
    st.subheader("Cumulative spend over time")
    fig = px.line(turns, x="created_dt", y="cum_cost", markers=True,
                  labels={"created_dt": "time", "cum_cost": "cumulative $"})
    st.plotly_chart(fig, use_container_width=True)

    tool_spans = spans[spans["kind"] == "tool"].copy()
    retry_spans = spans[spans["kind"] == "retry"].copy()
    if tool_spans.empty:
        st.info("No tool spans recorded yet.")
        return

    def _attempts(row: str) -> int:
        try:
            return int(json.loads(row).get("sentinel.tool.attempts", 1))
        except Exception:
            return 1

    def _tool_name(row: str) -> str:
        try:
            return json.loads(row).get("gen_ai.tool.name", "unknown")
        except Exception:
            return "unknown"

    tool_spans["tool"] = tool_spans["attributes_json"].apply(_tool_name)
    tool_spans["attempts"] = tool_spans["attributes_json"].apply(_attempts)
    tool_spans["success"] = tool_spans["status"] == "ok"

    agg = (
        tool_spans.groupby("tool")
        .agg(
            calls=("id", "count"),
            success_rate=("success", "mean"),
            avg_attempts=("attempts", "mean"),
            p50_latency_ms=("duration_ms", lambda s: s.quantile(0.5)),
            p99_latency_ms=("duration_ms", lambda s: s.quantile(0.99)),
        )
        .reset_index()
    )
    agg["success_rate"] = (agg["success_rate"] * 100).round(1)
    agg["avg_attempts"] = agg["avg_attempts"].round(2)
    agg["p50_latency_ms"] = agg["p50_latency_ms"].round(1)
    agg["p99_latency_ms"] = agg["p99_latency_ms"].round(1)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Tool success rate (%)")
        st.plotly_chart(
            px.bar(agg, x="tool", y="success_rate", range_y=[0, 100], text="success_rate"),
            use_container_width=True,
        )
    with c2:
        st.subheader("Latency p50 / p99 (ms)")
        lat = agg.melt(id_vars="tool", value_vars=["p50_latency_ms", "p99_latency_ms"],
                       var_name="percentile", value_name="ms")
        st.plotly_chart(px.bar(lat, x="tool", y="ms", color="percentile", barmode="group"),
                        use_container_width=True)

    st.subheader("Per-tool reliability table")
    st.dataframe(agg, use_container_width=True, hide_index=True)

    st.caption(
        f"Total retries recorded: {len(retry_spans)} across {len(tool_spans)} tool calls."
    )


def main() -> None:
    st.title("🛡️ Sentinel Observability")
    st.caption(f"Reading: `{DB_PATH}`")

    try:
        conn = _connect(DB_PATH)
    except sqlite3.Error as exc:
        st.error(f"Could not open database '{DB_PATH}': {exc}")
        return

    if not _table_exists(conn):
        st.warning("Database has no Sentinel tables yet. Run the agent first.")
        return

    view = st.sidebar.radio(
        "View",
        ["Sessions", "Turn inspector", "Trace viewer", "Cost & reliability"],
    )
    if view == "Sessions":
        sessions_view(conn)
    elif view == "Turn inspector":
        turn_inspector(conn)
    elif view == "Trace viewer":
        trace_viewer(conn)
    else:
        metrics_view(conn)


if __name__ == "__main__":
    main()
