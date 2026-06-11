"""Scripted, end-to-end demo of Sentinel.

Runs a multi-turn conversation that exercises every tool, prints a live
turn-by-turn trace, and finishes with a cost / latency / reliability summary.

* If ``ANTHROPIC_API_KEY`` is set, it uses the real Claude agent.
* Otherwise it falls back to a deterministic *scripted* LLM so the full
  pipeline (tools, retries, tracing, graceful degradation) can be demonstrated
  completely offline. Tools without credentials degrade gracefully and the
  agent reports what it could not do — which is exactly what we want to show.

Run:  python demo/demo.py      (from the repo root, inside the venv)
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sentinel.config import get_settings
from sentinel.console import build_rich_listener, render_trace_tree, render_waterfall
from sentinel.llm.client import LLMClient, LLMResponse
from sentinel.service import SentinelService

console = Console()


class ScriptedLLM(LLMClient):
    """A deterministic LLM. Each turn, it pops a list of 'steps' from a queue.

    A step is either a list of tool calls (-> tool_use) or a final string
    (-> end_turn). This lets the demo run with zero network/LLM dependency.
    """

    def __init__(self) -> None:
        self._queue: list[list[Any]] = []
        self._current: list[Any] = []

    def enqueue_turn(self, steps: list[Any]) -> None:
        self._queue.append(list(steps))

    async def create(self, **kwargs: Any) -> LLMResponse:
        if not self._current:
            if self._queue:
                self._current = self._queue.pop(0)
            else:
                self._current = ["All done."]
        step = self._current.pop(0)
        if isinstance(step, str):
            return LLMResponse(
                content=[{"type": "text", "text": step}],
                stop_reason="end_turn",
                model="scripted-model",
                input_tokens=120,
                output_tokens=60,
            )
        # step is a list of (id, tool_name, input) tuples.
        content = [
            {"type": "tool_use", "id": bid, "name": name, "input": inp}
            for bid, name, inp in step
        ]
        return LLMResponse(
            content=content,
            stop_reason="tool_use",
            model="scripted-model",
            input_tokens=140,
            output_tokens=40,
        )


def _build_scripted_llm() -> ScriptedLLM:
    llm = ScriptedLLM()
    # Turn 1: weather in two cities (no key needed) -> final answer.
    llm.enqueue_turn(
        [
            [
                ("w1", "get_weather", {"location": "San Francisco"}),
                ("w2", "get_weather", {"location": "Tokyo"}),
            ],
            "Here is the current weather for San Francisco and Tokyo.",
        ]
    )
    # Turn 2: GitHub lookup (likely unavailable -> graceful degradation).
    llm.enqueue_turn(
        [
            [
                (
                    "g1",
                    "github",
                    {"action": "get_repo", "owner": "anthropics", "repo": "anthropic-sdk-python"},
                )
            ],
            "I tried to fetch the GitHub repo but the tool reported a problem; "
            "see the note above for what failed.",
        ]
    )
    # Turn 3: multiple tools at once (Linear + Slack), demonstrating partial
    # completion when several tools fail.
    llm.enqueue_turn(
        [
            [
                ("l1", "linear", {"action": "list_issues", "limit": 3}),
                ("s1", "slack", {"action": "post_message", "channel": "#general", "text": "hi"}),
            ],
            "I attempted both the Linear lookup and the Slack post and have "
            "reported the outcome of each.",
        ]
    )
    return llm


async def run_demo() -> None:
    settings = get_settings()
    live = bool(settings.anthropic_api_key)

    # Use a throwaway database for the demo so it never clobbers real data.
    tmp_db = os.path.join(tempfile.gettempdir(), "sentinel_demo.db")
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    settings.db_path = tmp_db

    llm = None if live else _build_scripted_llm()
    service = await SentinelService.create(settings=settings, llm=llm)

    mode = "[green]LIVE (Claude)[/green]" if live else "[yellow]SCRIPTED (offline)[/yellow]"
    console.print(
        Panel.fit(
            f"Sentinel demo — mode: {mode}\n"
            f"Tools: {service.registry.availability()}",
            title="🛡️  Sentinel",
        )
    )

    prompts = [
        "What's the weather in San Francisco and Tokyo right now?",
        "Look up the anthropics/anthropic-sdk-python repo on GitHub.",
        "List my recent Linear issues and post a hello message to #general on Slack.",
    ]

    session = await service.start_session(title="Sentinel demo")
    listener = build_rich_listener(console)
    results = []

    for i, prompt in enumerate(prompts, start=1):
        console.rule(f"[bold]Turn {i}")
        console.print(f"[bold cyan]User:[/bold cyan] {prompt}\n")
        result = await service.send_message(session.id, prompt, listeners=[listener])
        results.append(result)
        console.print(f"\n[bold green]Assistant:[/bold green] {result.final_text}")
        if result.failed_tools or result.unavailable_tools:
            console.print(
                f"[dim]degradation -> failed: {result.failed_tools or '—'}, "
                f"unavailable: {result.unavailable_tools or '—'}[/dim]"
            )
        console.print()
        render_trace_tree(result.trace, console)

    # --- final summary --------------------------------------------------------
    console.rule("[bold]Cost & latency summary")
    refreshed = await service.get_session(session.id)
    table = Table(title="Per-turn breakdown")
    table.add_column("turn", justify="right")
    table.add_column("tools called")
    table.add_column("in tok", justify="right")
    table.add_column("out tok", justify="right")
    table.add_column("cost $", justify="right")
    table.add_column("turn ms", justify="right")
    for i, r in enumerate(results, start=1):
        root = next((s for s in r.trace.spans if s.kind.value == "turn"), None)
        ms = root.duration_ms if root else 0.0
        tool_names = ", ".join(inv.tool_name for inv in r.turn.tool_invocations) or "—"
        table.add_row(
            str(i),
            tool_names,
            str(r.turn.usage.input_tokens),
            str(r.turn.usage.output_tokens),
            f"{r.turn.usage.cost_usd:.5f}",
            f"{ms:.0f}",
        )
    console.print(table)

    console.print(
        Panel.fit(
            f"Total turns: {refreshed.turn_count}\n"
            f"Total tokens: {refreshed.total_usage.input_tokens} in / "
            f"{refreshed.total_usage.output_tokens} out\n"
            f"Total cost: ${refreshed.total_usage.cost_usd:.5f}\n"
            f"Session status: {refreshed.status.value}",
            title="Session totals",
        )
    )

    console.rule("[bold]Last turn waterfall")
    render_waterfall(results[-1].trace, console)

    console.print(
        f"\n[dim]Traces persisted to {tmp_db}. "
        f"Explore them with:  SENTINEL_DB_PATH={tmp_db} streamlit run dashboard/app.py[/dim]"
    )
    await service.close()


if __name__ == "__main__":
    asyncio.run(run_demo())
