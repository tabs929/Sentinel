"""State survives a simulated process restart; traces persist."""

from __future__ import annotations

from sentinel.agent.loop import Agent
from sentinel.models import Session
from sentinel.state.store import Store
from tests.conftest import FakeLLM, text_response, tool_use_response


async def test_session_survives_restart_and_resumes(settings, registry):
    # --- "process 1": run a turn and persist everything ---
    store1 = await Store.open(settings)
    llm1 = FakeLLM(
        [
            tool_use_response([("a", "echo", {"text": "hi"})]),
            text_response("first answer"),
        ]
    )
    agent1 = Agent(llm1, registry, store=store1, settings=settings)
    session = Session(title="resumable")
    await store1.save_session(session)
    r1 = await agent1.run_turn(session, "hello there")
    session_id = session.id
    trace_id = r1.trace.id
    await store1.close()

    # --- "process 2": fresh store from the same db file ---
    store2 = await Store.open(settings)
    loaded = await store2.get_session(session_id)
    assert loaded is not None
    assert loaded.turn_count == 1
    assert loaded.title == "resumable"
    # Full prior context is present (user + tool_use + tool_result + assistant).
    assert len(loaded.messages) >= 2
    assert loaded.messages[0]["role"] == "user"

    # The agent resumes with full context and continues the conversation.
    llm2 = FakeLLM([text_response("second answer with context")])
    agent2 = Agent(llm2, registry, store=store2, settings=settings)
    r2 = await agent2.run_turn(loaded, "and now?")
    assert r2.turn.index == 1
    assert loaded.turn_count == 2

    # The persisted turn and trace are retrievable.
    turns = await store2.get_turns(session_id)
    assert len(turns) == 2
    trace = await store2.get_trace(trace_id)
    assert trace is not None
    assert len(trace.spans) > 0
    await store2.close()


async def test_trace_for_turn_lookup(settings, registry):
    store = await Store.open(settings)
    llm = FakeLLM([text_response("answer")])
    agent = Agent(llm, registry, store=store, settings=settings)
    session = Session()
    await store.save_session(session)
    await agent.run_turn(session, "ping")

    trace = await store.get_trace_for_turn(session.id, 0)
    assert trace is not None
    assert trace.turn_index == 0
    await store.close()
