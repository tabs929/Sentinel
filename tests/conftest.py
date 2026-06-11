"""Shared test fixtures and fakes (no network, no real LLM)."""

from __future__ import annotations

from typing import Any

import pytest

from sentinel.config import Settings
from sentinel.llm.client import LLMClient, LLMResponse
from sentinel.tools.base import BaseTool, ToolError, ToolRegistry

# --- fake LLM --------------------------------------------------------------


class FakeLLM(LLMClient):
    """Returns scripted responses in order; records the calls it received."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        if not self._responses:
            # Default to a benign end_turn if the script runs out.
            return LLMResponse(
                content=[{"type": "text", "text": "(no more scripted responses)"}],
                stop_reason="end_turn",
                model="fake-model",
                input_tokens=1,
                output_tokens=1,
            )
        return self._responses.pop(0)


def text_response(text: str, *, input_tokens: int = 10, output_tokens: int = 5) -> LLMResponse:
    return LLMResponse(
        content=[{"type": "text", "text": text}],
        stop_reason="end_turn",
        model="fake-model",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def tool_use_response(
    calls: list[tuple[str, str, dict]], *, input_tokens: int = 10, output_tokens: int = 5
) -> LLMResponse:
    """calls: list of (block_id, tool_name, input)."""
    content = [
        {"type": "tool_use", "id": bid, "name": name, "input": inp}
        for bid, name, inp in calls
    ]
    return LLMResponse(
        content=content,
        stop_reason="tool_use",
        model="fake-model",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


# --- fake tools ------------------------------------------------------------


class EchoTool(BaseTool):
    name = "echo"
    description = "Echoes its input back."

    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"text": {"type": "string"}}}

    async def _call(self, text: str = "", **_: Any) -> tuple[str, dict[str, Any]]:
        return f"echo: {text}", {"text": text}


class AlwaysFailTool(BaseTool):
    name = "always_fail"
    description = "Always raises a retryable error."

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self.attempts = 0

    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def _call(self, **_: Any) -> tuple[str, dict[str, Any]]:
        self.attempts += 1
        raise ToolError("simulated transient failure")


class UnavailableTool(BaseTool):
    name = "unavailable_tool"
    description = "A tool that is never configured."

    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    @property
    def is_available(self) -> bool:
        return False

    async def _call(self, **_: Any) -> tuple[str, dict[str, Any]]:  # pragma: no cover
        return "should not run", {}


# --- fixtures --------------------------------------------------------------


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        anthropic_api_key="test-key-not-used",
        db_path=str(tmp_path / "test.db"),
        max_steps=6,
        tool_max_attempts=3,
        max_input_chars=200,
    )


@pytest.fixture
def registry(settings) -> ToolRegistry:
    return ToolRegistry(
        [
            EchoTool(settings),
            AlwaysFailTool(settings),
            UnavailableTool(settings),
        ]
    )
