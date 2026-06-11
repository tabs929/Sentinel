"""Central configuration loaded from environment / .env.

Every secret is optional. The application never crashes because a key is
missing; tools that need a missing key simply report themselves as
``unavailable``.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # --- LLM ---
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    model: str = Field(default="claude-sonnet-4-20250514", alias="SENTINEL_MODEL")
    max_tokens: int = Field(default=2048, alias="SENTINEL_MAX_TOKENS")
    max_steps: int = Field(default=8, alias="SENTINEL_MAX_STEPS")

    # --- Storage ---
    db_path: str = Field(default="sentinel.db", alias="SENTINEL_DB_PATH")

    # --- Guardrails ---
    max_input_chars: int = Field(default=8000, alias="SENTINEL_MAX_INPUT_CHARS")

    # --- Tool credentials ---
    github_token: str | None = Field(default=None, alias="GITHUB_TOKEN")
    linear_api_key: str | None = Field(default=None, alias="LINEAR_API_KEY")
    slack_bot_token: str | None = Field(default=None, alias="SLACK_BOT_TOKEN")

    # --- Tool retry/timeout ---
    tool_max_attempts: int = Field(default=3, alias="SENTINEL_TOOL_MAX_ATTEMPTS")
    tool_timeout_seconds: float = Field(default=15.0, alias="SENTINEL_TOOL_TIMEOUT_SECONDS")

    def secret_values(self) -> list[str]:
        """All non-empty secret strings, used by output guardrails to detect leaks."""
        candidates = [
            self.anthropic_api_key,
            self.github_token,
            self.linear_api_key,
            self.slack_bot_token,
        ]
        return [c for c in candidates if c]


# Pricing per 1M tokens (USD). Used for cost accounting; falls back to the
# default entry for unknown models.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.0},
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    "_default": {"input": 3.0, "output": 15.0},
}


def cost_for(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute the USD cost of an LLM call given token counts."""
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["_default"])
    return (input_tokens / 1_000_000) * pricing["input"] + (
        output_tokens / 1_000_000
    ) * pricing["output"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
