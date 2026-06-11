"""LLM client: a thin, testable wrapper over the Anthropic Messages API.

The agent loop depends only on the :class:`LLMClient` interface, so tests can
inject a fake client without any network access.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from sentinel.config import Settings, get_settings


class LLMResponse(BaseModel):
    """A normalized response from a single LLM call."""

    #: Anthropic-shaped content blocks (text / tool_use), ready to append to
    #: the conversation history.
    content: list[dict[str, Any]] = Field(default_factory=list)
    stop_reason: str | None = None
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def tool_use_blocks(self) -> list[dict[str, Any]]:
        return [b for b in self.content if b.get("type") == "tool_use"]

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text").strip()


class LLMClient(ABC):
    """Interface the agent loop depends on."""

    @abstractmethod
    async def create(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
        model: str,
        max_tokens: int,
    ) -> LLMResponse:
        ...


class AnthropicClient(LLMClient):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set; cannot call the LLM. "
                "Set it in your environment or .env file."
            )
        # Imported lazily so the rest of the package works without the SDK.
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=self.settings.anthropic_api_key)

    async def create(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
        model: str,
        max_tokens: int,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        resp = await self._client.messages.create(**kwargs)
        content = [block.model_dump() for block in resp.content]
        return LLMResponse(
            content=content,
            stop_reason=resp.stop_reason,
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
