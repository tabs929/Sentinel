"""Slack tool — post messages / list channels via the Slack Web API.

Requires ``SLACK_BOT_TOKEN``. Without it the tool reports itself unavailable.
"""

from __future__ import annotations

from typing import Any

import httpx

from sentinel.tools.base import BaseTool, ToolError

API_ROOT = "https://slack.com/api"


class SlackTool(BaseTool):
    name = "slack"
    description = (
        "Interact with Slack. Action 'list_channels' lists public channels; "
        "action 'post_message' posts a message to a channel (args: 'channel', "
        "'text'). Requires a Slack bot token."
    )

    retryable_exceptions = (ToolError, httpx.HTTPError, ConnectionError, TimeoutError)

    @property
    def is_available(self) -> bool:
        return bool(self.settings.slack_bot_token)

    def unavailable_reason(self) -> str:
        return "SLACK_BOT_TOKEN is not set"

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list_channels", "post_message"],
                    "description": "Which operation to perform.",
                },
                "channel": {
                    "type": "string",
                    "description": "Channel id or name (required for post_message).",
                },
                "text": {
                    "type": "string",
                    "description": "Message text (required for post_message).",
                },
            },
            "required": ["action"],
        }

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.slack_bot_token}",
            "Content-Type": "application/json; charset=utf-8",
        }

    async def _call(
        self,
        action: str | None = None,
        channel: str | None = None,
        text: str | None = None,
        **_: Any,
    ) -> tuple[str, dict[str, Any]]:
        timeout = self.settings.tool_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout, headers=self._headers()) as client:
            if action == "list_channels":
                resp = await client.get(
                    f"{API_ROOT}/conversations.list",
                    params={"limit": 50, "exclude_archived": "true", "types": "public_channel"},
                )
                payload = self._payload(resp)
                channels = [
                    {"id": c["id"], "name": c["name"]} for c in payload.get("channels", [])
                ]
                names = ", ".join(f"#{c['name']}" for c in channels) or "(none)"
                return f"Slack channels: {names}.", {"channels": channels}

            if action == "post_message":
                if not channel or not text:
                    raise ToolError("'channel' and 'text' are required for post_message")
                resp = await client.post(
                    f"{API_ROOT}/chat.postMessage",
                    json={"channel": channel, "text": text},
                )
                payload = self._payload(resp)
                return (
                    f"Posted message to {channel} (ts={payload.get('ts')}).",
                    {"channel": channel, "ts": payload.get("ts")},
                )

            raise ToolError(f"unknown action '{action}'")

    @staticmethod
    def _payload(resp: httpx.Response) -> dict[str, Any]:
        if resp.status_code >= 500:
            raise ToolError(f"Slack server error ({resp.status_code})")
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("ok", False):
            err = payload.get("error", "unknown_error")
            # Rate limiting is retryable; everything else is a hard error.
            if err in ("ratelimited", "service_unavailable"):
                raise ToolError(f"Slack transient error: {err}")
            raise RuntimeError(f"Slack API error: {err}")
        return payload
