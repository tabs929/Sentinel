"""Linear tool — query issues via the Linear GraphQL API.

Requires ``LINEAR_API_KEY``. Without it the tool reports itself unavailable.
"""

from __future__ import annotations

from typing import Any

import httpx

from sentinel.tools.base import BaseTool, ToolError

API_URL = "https://api.linear.app/graphql"

_LIST_ISSUES_QUERY = """
query Issues($first: Int!) {
  issues(first: $first, orderBy: updatedAt) {
    nodes {
      identifier
      title
      state { name }
      priorityLabel
      url
    }
  }
}
"""


class LinearTool(BaseTool):
    name = "linear"
    description = (
        "Query issues from Linear. Action 'list_issues' returns the most "
        "recently updated issues with their state and priority. Requires a "
        "Linear API key."
    )

    retryable_exceptions = (ToolError, httpx.HTTPError, ConnectionError, TimeoutError)

    @property
    def is_available(self) -> bool:
        return bool(self.settings.linear_api_key)

    def unavailable_reason(self) -> str:
        return "LINEAR_API_KEY is not set"

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list_issues"],
                    "description": "Which operation to perform.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max issues to return (default 5).",
                },
            },
            "required": ["action"],
        }

    async def _call(
        self,
        action: str | None = None,
        limit: int = 5,
        **_: Any,
    ) -> tuple[str, dict[str, Any]]:
        if action != "list_issues":
            raise ToolError(f"unknown action '{action}'")

        first = max(1, min(int(limit or 5), 25))
        headers = {
            "Authorization": self.settings.linear_api_key or "",
            "Content-Type": "application/json",
        }
        timeout = self.settings.tool_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            resp = await client.post(
                API_URL,
                json={"query": _LIST_ISSUES_QUERY, "variables": {"first": first}},
            )
            if resp.status_code in (401, 403):
                raise RuntimeError(f"Linear auth error ({resp.status_code})")
            if resp.status_code >= 500:
                raise ToolError(f"Linear server error ({resp.status_code})")
            resp.raise_for_status()
            payload = resp.json()

        if payload.get("errors"):
            raise ToolError(f"Linear API error: {payload['errors']}")

        nodes = payload.get("data", {}).get("issues", {}).get("nodes", [])
        lines = [
            f"{n['identifier']} [{n['state']['name']}] {n['title']} "
            f"(priority: {n.get('priorityLabel') or 'none'})"
            for n in nodes
        ]
        content = (
            "Recent Linear issues:\n" + "\n".join(lines)
            if lines
            else "No Linear issues found."
        )
        data = {"issues": nodes}
        return content, data
