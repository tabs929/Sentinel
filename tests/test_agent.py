"""Agent loop: graceful degradation, tracing, cost accounting."""

from __future__ import annotations

from sentinel.agent.loop import Agent
from sentinel.models import Session, SpanKind, TurnStatus
from tests.conftest import FakeLLM, text_response, tool_use_response


async def test_completes_turn_when_one_of_three_tools_fails(settings, registry):
    # First LLM turn asks for all three tools; second turn ends.
    llm = FakeLLM(
        [
            tool_use_response(
                [
                    ("a", "echo", {"text": "ok"}),
                    ("b", "always_fail", {}),
                    ("c", "unavailable_tool", {}),
                ]
            ),
            text_response("Here is what I found; some tools failed."),
        ]
    )
    agent = Agent(llm, registry, settings=settings)
    session = Session()

    result = await agent.run_turn(session, "do three things")

    assert result.turn.status == TurnStatus.COMPLETED
    assert "always_fail" in result.failed_tools
    assert "unavailable_tool" in result.unavailable_tools
    assert result.final_text  # never silently empty
    # All three tool invocations recorded on the turn.
    assert len(result.turn.tool_invocations) == 3


async def test_trace_has_expected_span_tree(settings, registry):
    llm = FakeLLM(
        [
            tool_use_response([("a", "echo", {"text": "hi"})]),
            text_response("done"),
        ]
    )
    agent = Agent(llm, registry, settings=settings)
    session = Session()

    result = await agent.run_turn(session, "hello")
    spans = result.trace.spans

    kinds = [s.kind for s in spans]
    assert kinds.count(SpanKind.TURN) == 1
    assert kinds.count(SpanKind.LLM) == 2  # one call per loop step
    assert kinds.count(SpanKind.TOOL) == 1

    root = next(s for s in spans if s.kind == SpanKind.TURN)
    assert result.trace.root_span_id == root.id
    # Every non-root span hangs under the root.
    for s in spans:
        if s.id != root.id:
            assert s.parent_id is not None


async def test_retry_produces_child_retry_spans(settings, registry):
    llm = FakeLLM(
        [
            tool_use_response([("a", "always_fail", {})]),
            text_response("Sorry, the tool failed after retries."),
        ]
    )
    agent = Agent(llm, registry, settings=settings)
    session = Session()

    result = await agent.run_turn(session, "use the failing tool")

    retry_spans = [s for s in result.trace.spans if s.kind == SpanKind.RETRY]
    tool_span = next(s for s in result.trace.spans if s.kind == SpanKind.TOOL)
    # 3 attempts => 2 retry spans, all children of the tool span.
    assert len(retry_spans) == 2
    assert all(s.parent_id == tool_span.id for s in retry_spans)


async def test_cost_tracking_accurate_across_multi_turn(settings, registry):
    # Turn 1: 2 LLM calls (10/5 then 8/4). Turn 2: 1 LLM call (12/6).
    llm = FakeLLM(
        [
            tool_use_response([("a", "echo", {"text": "x"})], input_tokens=10, output_tokens=5),
            text_response("first done", input_tokens=8, output_tokens=4),
            text_response("second done", input_tokens=12, output_tokens=6),
        ]
    )
    agent = Agent(llm, registry, settings=settings)
    session = Session()

    r1 = await agent.run_turn(session, "first")
    assert r1.turn.usage.input_tokens == 18
    assert r1.turn.usage.output_tokens == 9

    r2 = await agent.run_turn(session, "second")
    assert r2.turn.usage.input_tokens == 12
    assert r2.turn.usage.output_tokens == 6

    # Session totals accumulate across turns.
    assert session.turn_count == 2
    assert session.total_usage.input_tokens == 30
    assert session.total_usage.output_tokens == 15
    assert session.total_usage.cost_usd > 0
    # Total cost equals the sum of per-turn costs.
    per_turn_sum = r1.turn.usage.cost_usd + r2.turn.usage.cost_usd
    assert abs(session.total_usage.cost_usd - per_turn_sum) < 1e-9
