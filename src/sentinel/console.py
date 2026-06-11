"""Rich console rendering of agent events and traces.

Used by the CLI and the demo script to print a live, turn-by-turn view of
what the agent is doing.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from sentinel.models import AgentEvent, EventType, Span, SpanKind, SpanStatus, Trace
from sentinel.observability.tracing import GenAI, Listener

_EVENT_STYLE = {
    EventType.TURN_STARTED: ("bold cyan", "turn"),
    EventType.TURN_ENDED: ("bold cyan", "turn done"),
    EventType.LLM_CALL_STARTED: ("magenta", "llm"),
    EventType.LLM_CALL_ENDED: ("magenta", "llm done"),
    EventType.TOOL_CALL_STARTED: ("yellow", "tool"),
    EventType.TOOL_CALL_ENDED: ("yellow", "tool done"),
    EventType.TOOL_RETRY: ("red", "retry"),
    EventType.GUARDRAIL_TRIGGERED: ("bold red", "guardrail"),
    EventType.MESSAGE: ("dim", "note"),
}


def build_rich_listener(console: Console | None = None) -> Listener:
    console = console or Console()

    def listener(event: AgentEvent) -> None:
        style, label = _EVENT_STYLE.get(event.type, ("white", event.type.value))
        detail = ""
        if event.type == EventType.LLM_CALL_ENDED:
            d = event.data
            detail = (
                f"in={d.get(GenAI.USAGE_INPUT_TOKENS, 0)} "
                f"out={d.get(GenAI.USAGE_OUTPUT_TOKENS, 0)} "
                f"${d.get(GenAI.USAGE_COST_USD, 0):.5f} "
                f"{d.get('duration_ms', 0):.0f}ms"
            )
        elif event.type == EventType.TOOL_CALL_STARTED:
            detail = event.message or ""
        elif event.type == EventType.TOOL_CALL_ENDED:
            d = event.data
            ok = d.get("status") == "ok"
            mark = "[green]ok[/green]" if ok else "[red]FAIL[/red]"
            detail = (
                f"{mark} attempts={d.get(GenAI.TOOL_ATTEMPTS, 1)} "
                f"{d.get('sentinel.tool.latency_ms', 0):.0f}ms"
            )
            if d.get("error"):
                detail += f" :: {d['error']}"
        else:
            detail = event.message or ""

        console.print(f"  [{style}]{label:<10}[/{style}] {detail}")

    return listener


def render_trace_tree(trace: Trace, console: Console | None = None) -> None:
    """Render a trace as an indented tree with per-span latency."""
    console = console or Console()
    by_id = {s.id: s for s in trace.spans}
    children: dict[str | None, list[Span]] = {}
    for s in trace.spans:
        children.setdefault(s.parent_id, []).append(s)

    root_id = trace.root_span_id
    root_span = by_id.get(root_id) if root_id else None
    label = _span_label(root_span) if root_span else f"trace {trace.id}"
    tree = Tree(label)

    def add(node: Tree, parent_id: str | None) -> None:
        for span in sorted(children.get(parent_id, []), key=lambda s: s.start_time):
            child = node.add(_span_label(span))
            add(child, span.id)

    if root_span is not None:
        add(tree, root_span.id)
    else:
        add(tree, None)
    console.print(tree)


def _span_label(span: Span) -> str:
    color = {
        SpanKind.TURN: "cyan",
        SpanKind.LLM: "magenta",
        SpanKind.TOOL: "yellow",
        SpanKind.RETRY: "red",
        SpanKind.GUARDRAIL: "bold red",
    }.get(span.kind, "white")
    status_mark = "[green]●[/green]" if span.status == SpanStatus.OK else "[red]●[/red]"
    extra = ""
    if span.kind == SpanKind.LLM:
        a = span.attributes
        extra = (
            f" (in={a.get(GenAI.USAGE_INPUT_TOKENS, 0)}, "
            f"out={a.get(GenAI.USAGE_OUTPUT_TOKENS, 0)}, "
            f"${a.get(GenAI.USAGE_COST_USD, 0):.5f})"
        )
    return (
        f"{status_mark} [{color}]{span.name}[/{color}] "
        f"[dim]{span.duration_ms:.0f}ms[/dim]{extra}"
    )


def render_waterfall(trace: Trace, console: Console | None = None, width: int = 40) -> None:
    """Render a simple ASCII waterfall of spans by start offset and duration."""
    console = console or Console()
    spans = [s for s in trace.spans if s.kind != SpanKind.RETRY]
    if not spans:
        console.print("[dim]no spans[/dim]")
        return
    t0 = min(s.start_time for s in spans)
    total = max((s.end_time or s.start_time) - t0 for s in spans).total_seconds() * 1000
    total = max(total, 1.0)

    table = Table(show_header=True, header_style="bold")
    table.add_column("span")
    table.add_column("timeline", width=width + 2)
    table.add_column("ms", justify="right")
    for s in sorted(spans, key=lambda s: s.start_time):
        offset = (s.start_time - t0).total_seconds() * 1000
        start_col = int((offset / total) * width)
        bar_len = max(1, int((s.duration_ms / total) * width))
        bar = " " * start_col + "█" * min(bar_len, width - start_col)
        color = "green" if s.status == SpanStatus.OK else "red"
        table.add_row(s.name, f"[{color}]{bar}[/]", f"{s.duration_ms:.0f}")
    console.print(table)
