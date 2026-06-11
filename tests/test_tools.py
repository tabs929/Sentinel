"""Tools always return a ToolResult and never raise."""

from __future__ import annotations

import pytest

from sentinel.tools.base import ToolError
from tests.conftest import AlwaysFailTool, EchoTool, UnavailableTool


async def test_success_result(settings):
    result = await EchoTool(settings).run({"text": "hi"})
    assert result.ok is True
    assert result.unavailable is False
    assert "echo: hi" in result.content
    assert result.attempts == 1


async def test_failure_returns_graceful_result_never_raises(settings):
    tool = AlwaysFailTool(settings)
    # Must not raise, even though _call always raises.
    result = await tool.run({})
    assert result.ok is False
    assert result.unavailable is False
    assert result.error is not None
    assert "simulated transient failure" in result.error


async def test_unavailable_tool_does_not_run(settings):
    tool = UnavailableTool(settings)
    result = await tool.run({})
    assert result.unavailable is True
    assert result.ok is False
    assert "unavailable" in result.content.lower()


async def test_retry_fires_correct_number_of_times(settings):
    # tool_max_attempts = 3 in the settings fixture.
    tool = AlwaysFailTool(settings)
    retries: list[int] = []

    def on_retry(attempt: int, exc: BaseException, wait: float) -> None:
        retries.append(attempt)

    result = await tool.run({}, on_retry=on_retry)

    assert tool.attempts == 3  # 1 initial + 2 retries
    assert result.attempts == 3
    assert len(retries) == 2  # on_retry fires before each retry sleep
    assert isinstance(ToolError("x"), Exception)


async def test_weather_tool_is_always_available(settings):
    from sentinel.tools.weather import WeatherTool

    assert WeatherTool(settings).is_available is True


@pytest.mark.parametrize("tool_name", ["github", "linear", "slack"])
async def test_keyed_tools_unavailable_without_keys(settings, tool_name):
    from sentinel.tools import build_default_registry

    reg = build_default_registry(settings)
    tool = reg.get(tool_name)
    assert tool is not None
    assert tool.is_available is False
    result = await tool.run({"action": "x"})
    assert result.unavailable is True
