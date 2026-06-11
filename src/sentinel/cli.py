"""Sentinel CLI (typer)."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

from sentinel.console import build_rich_listener, render_trace_tree, render_waterfall
from sentinel.service import SentinelService

app = typer.Typer(help="Sentinel: a reliable, observable multi-tool AI agent.")
console = Console()


@app.command()
def tools() -> None:
    """List the available tools and whether each is configured."""

    async def _run() -> None:
        service = await SentinelService.create()
        try:
            for name, available in service.registry.availability().items():
                mark = "[green]available[/green]" if available else "[yellow]unavailable[/yellow]"
                console.print(f"  {name:<12} {mark}")
        finally:
            await service.close()

    asyncio.run(_run())


@app.command()
def chat(
    message: str = typer.Argument(..., help="Message to send."),
    session_id: str = typer.Option(None, "--session", "-s", help="Existing session id."),
    show_trace: bool = typer.Option(True, help="Print the span tree after the turn."),
) -> None:
    """Send a single message, creating a session if none is given."""

    async def _run() -> None:
        service = await SentinelService.create()
        try:
            if session_id:
                sid = session_id
            else:
                session = await service.start_session(title=message[:60])
                sid = session.id
                console.print(f"[dim]Started session {sid}[/dim]")

            listener = build_rich_listener(console)
            result = await service.send_message(sid, message, listeners=[listener])

            console.rule("[bold]Assistant")
            console.print(result.final_text)
            console.print(
                f"\n[dim]tokens in/out: {result.turn.usage.input_tokens}/"
                f"{result.turn.usage.output_tokens}  "
                f"cost: ${result.turn.usage.cost_usd:.5f}[/dim]"
            )
            if show_trace and result.trace.spans:
                console.rule("[bold]Trace")
                render_trace_tree(result.trace, console)
                render_waterfall(result.trace, console)
        finally:
            await service.close()

    asyncio.run(_run())


@app.command()
def sessions() -> None:
    """List all sessions."""

    async def _run() -> None:
        service = await SentinelService.create()
        try:
            for s in await service.list_sessions():
                console.print(
                    f"  {s.id}  [bold]{s.status.value}[/bold]  "
                    f"turns={s.turn_count}  ${s.total_usage.cost_usd:.5f}  "
                    f"{s.title or ''}"
                )
        finally:
            await service.close()

    asyncio.run(_run())


@app.command()
def trace(trace_id: str = typer.Argument(..., help="Trace id to render.")) -> None:
    """Render a stored trace as a span tree + waterfall."""

    async def _run() -> None:
        service = await SentinelService.create()
        try:
            t = await service.get_trace(trace_id)
            if t is None:
                console.print(f"[red]No trace '{trace_id}'.[/red]")
                raise typer.Exit(1)
            render_trace_tree(t, console)
            render_waterfall(t, console)
        finally:
            await service.close()

    asyncio.run(_run())


if __name__ == "__main__":
    app()
