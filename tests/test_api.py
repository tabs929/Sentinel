"""API layer: endpoints work and errors are structured JSON, not stack traces."""

from __future__ import annotations

import httpx
import pytest

from sentinel.api.app import app
from sentinel.service import SentinelService
from sentinel.state.store import Store
from tests.conftest import FakeLLM, text_response, tool_use_response


@pytest.fixture
async def client(settings, registry):
    store = await Store.open(settings)
    llm = FakeLLM(
        [
            tool_use_response([("a", "echo", {"text": "hi"})]),
            text_response("done via api"),
        ]
    )
    app.state.service = SentinelService(store, settings=settings, llm=llm, registry=registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await store.close()


async def test_full_flow(client):
    # create session
    resp = await client.post("/sessions", json={"title": "api test"})
    assert resp.status_code == 201
    session_id = resp.json()["id"]

    # send a message
    resp = await client.post(f"/sessions/{session_id}/messages", json={"message": "hi"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["final_text"] == "done via api"
    assert body["turn"]["status"] == "completed"

    # session detail includes history
    resp = await client.get(f"/sessions/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["turn_count"] == 1

    # traces listed and fetchable
    resp = await client.get("/traces")
    assert resp.status_code == 200
    traces = resp.json()
    assert len(traces) == 1
    trace_id = traces[0]["id"]

    resp = await client.get(f"/traces/{trace_id}")
    assert resp.status_code == 200
    assert len(resp.json()["spans"]) > 0

    # trace for a specific turn
    resp = await client.get(f"/sessions/{session_id}/turns/0/trace")
    assert resp.status_code == 200
    assert resp.json()["turn_index"] == 0


async def test_unknown_session_returns_structured_404(client):
    resp = await client.get("/sessions/does-not-exist")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"] == "session_not_found"
    assert "detail" in body


async def test_validation_error_is_structured(client):
    resp = await client.post("/sessions/x/messages", json={"message": ""})
    assert resp.status_code == 422
    assert resp.json()["error"] == "validation_error"
