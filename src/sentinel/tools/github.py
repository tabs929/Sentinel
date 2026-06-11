"""GitHub tool — read-only repository lookups via the GitHub REST API.

Requires ``GITHUB_TOKEN``. Without it the tool reports itself unavailable.
"""

from __future__ import annotations

from typing import Any

import httpx

from sentinel.tools.base import BaseTool, ToolError

API_ROOT = "https://api.github.com"


class GitHubTool(BaseTool):
    name = "github"
    description = (
        "Look up public information about a GitHub repository. Supports two "
        "actions: 'get_repo' (stars, description, language, open issues) and "
        "'list_issues' (most recent open issues). Requires a GitHub token."
    )

    retryable_exceptions = (ToolError, httpx.HTTPError, ConnectionError, TimeoutError)

    @property
    def is_available(self) -> bool:
        return bool(self.settings.github_token)

    def unavailable_reason(self) -> str:
        return "GITHUB_TOKEN is not set"

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get_repo", "list_issues"],
                    "description": "Which operation to perform.",
                },
                "owner": {"type": "string", "description": "Repository owner / org."},
                "repo": {"type": "string", "description": "Repository name."},
                "limit": {
                    "type": "integer",
                    "description": "Max issues to return for list_issues (default 5).",
                },
            },
            "required": ["action", "owner", "repo"],
        }

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _call(
        self,
        action: str | None = None,
        owner: str | None = None,
        repo: str | None = None,
        limit: int = 5,
        **_: Any,
    ) -> tuple[str, dict[str, Any]]:
        if not owner or not repo:
            raise ToolError("'owner' and 'repo' are required")
        if action not in ("get_repo", "list_issues"):
            raise ToolError(f"unknown action '{action}'")

        timeout = self.settings.tool_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout, headers=self._headers()) as client:
            if action == "get_repo":
                resp = await client.get(f"{API_ROOT}/repos/{owner}/{repo}")
                self._raise_for_api(resp)
                r = resp.json()
                content = (
                    f"{r['full_name']}: {r.get('description') or 'no description'}. "
                    f"\u2b50 {r['stargazers_count']} stars, "
                    f"{r['open_issues_count']} open issues, "
                    f"language {r.get('language') or 'n/a'}."
                )
                data = {
                    "full_name": r["full_name"],
                    "description": r.get("description"),
                    "stars": r["stargazers_count"],
                    "forks": r["forks_count"],
                    "open_issues": r["open_issues_count"],
                    "language": r.get("language"),
                    "url": r["html_url"],
                }
                return content, data

            # list_issues
            resp = await client.get(
                f"{API_ROOT}/repos/{owner}/{repo}/issues",
                params={"state": "open", "per_page": max(1, min(int(limit or 5), 20))},
            )
            self._raise_for_api(resp)
            issues = [i for i in resp.json() if "pull_request" not in i]
            lines = [f"#{i['number']} {i['title']}" for i in issues]
            content = (
                f"Open issues in {owner}/{repo}:\n" + "\n".join(lines)
                if lines
                else f"No open issues in {owner}/{repo}."
            )
            data = {
                "owner": owner,
                "repo": repo,
                "issues": [
                    {"number": i["number"], "title": i["title"], "url": i["html_url"]}
                    for i in issues
                ],
            }
            return content, data

    @staticmethod
    def _raise_for_api(resp: httpx.Response) -> None:
        if resp.status_code == 404:
            raise ToolError("repository not found (404)")
        if resp.status_code in (401, 403):
            # Auth/ratelimit errors are not worth retrying.
            raise RuntimeError(f"GitHub auth/ratelimit error ({resp.status_code})")
        if resp.status_code >= 500:
            raise ToolError(f"GitHub server error ({resp.status_code})")
        resp.raise_for_status()
