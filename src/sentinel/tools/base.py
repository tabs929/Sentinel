"""BaseTool, the retry wrapper, and the ToolRegistry.

Reliability contract enforced here, so the agent never has to think about it:

* ``run()`` ALWAYS returns a :class:`ToolResult`; it never raises.
* A missing API key yields a graceful "unavailable" result.
* Transient failures are retried with exponential backoff via ``tenacity``.
* Each retry is surfaced through the ``on_retry`` callback so the
  observability layer can record a child span.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from sentinel.config import Settings, get_settings
from sentinel.models import ToolResult, ToolSpec

# Callback invoked before each retry sleep: (attempt_number, error, sleep_seconds).
RetryCallback = Callable[[int, BaseException, float], None]


class ToolError(Exception):
    """Raised internally by a tool's ``_call`` to signal a retryable failure.

    Tools raise this (or any exception) inside ``_call``; the public ``run``
    wrapper converts it into a ToolResult.
    """


class BaseTool(ABC):
    """Abstract base class for all tools."""

    #: Unique tool name exposed to the LLM (snake_case).
    name: str = ""
    #: Natural-language description shown to the LLM.
    description: str = ""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # --- Interface to implement -------------------------------------------------

    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON schema describing the tool's arguments."""

    @abstractmethod
    async def _call(self, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        """Perform the actual work.

        Returns ``(content, data)`` where ``content`` is the LLM-facing text
        and ``data`` is the structured payload. May raise on failure; the
        wrapper handles retries and error formatting.
        """

    @property
    def is_available(self) -> bool:
        """Whether the tool is configured to run. Override when a key is needed."""
        return True

    def unavailable_reason(self) -> str:
        return "tool not configured"

    # --- Exceptions that should trigger a retry --------------------------------

    #: Override to broaden/narrow which exceptions are retried.
    retryable_exceptions: tuple[type[BaseException], ...] = (
        ToolError,
        ConnectionError,
        TimeoutError,
    )

    # --- Public, never-raising entrypoint --------------------------------------

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema(),
        )

    async def run(
        self,
        tool_input: dict[str, Any] | None = None,
        *,
        on_retry: RetryCallback | None = None,
    ) -> ToolResult:
        """Execute the tool. ALWAYS returns a ToolResult; never raises."""
        tool_input = tool_input or {}

        if not self.is_available:
            return ToolResult.tool_unavailable(self.name, self.unavailable_reason())

        max_attempts = max(1, self.settings.tool_max_attempts)
        start = time.perf_counter()
        attempts_made = 0

        def _before_sleep(state: RetryCallState) -> None:
            # tenacity calls this *after* a failed attempt, before sleeping.
            if on_retry is None or state.outcome is None:
                return
            exc = state.outcome.exception()
            if exc is None:
                return
            sleep_for = getattr(state.next_action, "sleep", 0.0) or 0.0
            on_retry(state.attempt_number, exc, float(sleep_for))

        try:
            retryer = AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=0.2, max=4.0),
                retry=retry_if_exception_type(self.retryable_exceptions),
                before_sleep=_before_sleep,
                reraise=True,
            )
            async for attempt in retryer:
                with attempt:
                    attempts_made = attempt.retry_state.attempt_number
                    content, data = await self._call(**tool_input)
            latency_ms = (time.perf_counter() - start) * 1000
            return ToolResult.success(
                self.name,
                content=content,
                data=data,
                latency_ms=latency_ms,
                attempts=attempts_made,
            )
        except Exception as exc:  # noqa: BLE001 - the whole point is to never escape
            latency_ms = (time.perf_counter() - start) * 1000
            return ToolResult.failure(
                self.name,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=latency_ms,
                attempts=attempts_made or 1,
            )


class ToolRegistry:
    """Holds the available tools and exposes them to the agent and the LLM."""

    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        if not tool.name:
            raise ValueError("Tool must define a non-empty name")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def all(self) -> list[BaseTool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def anthropic_tools(self) -> list[dict[str, Any]]:
        """Tool specs in the shape the Anthropic Messages API expects."""
        return [tool.spec().to_anthropic() for tool in self._tools.values()]

    def availability(self) -> dict[str, bool]:
        return {name: tool.is_available for name, tool in self._tools.items()}


def build_default_registry(settings: Settings | None = None) -> ToolRegistry:
    """Construct a registry with all four built-in tools."""
    from sentinel.tools.github import GitHubTool
    from sentinel.tools.linear import LinearTool
    from sentinel.tools.slack import SlackTool
    from sentinel.tools.weather import WeatherTool

    settings = settings or get_settings()
    return ToolRegistry(
        [
            WeatherTool(settings),
            GitHubTool(settings),
            LinearTool(settings),
            SlackTool(settings),
        ]
    )
