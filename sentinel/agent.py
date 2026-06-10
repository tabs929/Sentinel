from __future__ import annotations

import json
import traceback
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
except ImportError:  # pragma: no cover - tested implicitly through fallback behavior
    trace = None
    OTLPSpanExporter = None
    Resource = None
    TracerProvider = None
    BatchSpanProcessor = None


class TracerProtocol(Protocol):
    def start_as_current_span(self, name: str): ...


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 3
    backoff_seconds: float = 0.2


@dataclass
class TurnResult:
    user_message: str
    response: str
    tool_outputs: dict[str, Any]
    degraded: bool
    failures: list[str]


class HttpClient:
    def __init__(self, retry_policy: RetryPolicy | None = None) -> None:
        self.retry_policy = retry_policy or RetryPolicy()

    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any] | None = None,
        timeout: float = 15,
    ) -> Any:
        encoded_body = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(url=url, method=method, headers=headers, data=encoded_body)
        for attempt in range(1, self.retry_policy.attempts + 1):
            try:
                with urlopen(request, timeout=timeout) as response:
                    payload = response.read().decode("utf-8")
                    return json.loads(payload) if payload else {}
            except HTTPError as exc:
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.retry_policy.attempts:
                    time.sleep(self.retry_policy.backoff_seconds * attempt)
                    continue
                raise
            except URLError as exc:
                if attempt < self.retry_policy.attempts:
                    time.sleep(self.retry_policy.backoff_seconds * attempt)
                    continue
                raise


class GitHubClient:
    def __init__(self, token: str, http_client: HttpClient | None = None) -> None:
        self.token = token
        self.http = http_client or HttpClient()

    def get_issue(self, repo: str, number: int) -> dict[str, Any]:
        authorization_header = "Bearer " + self.token
        return self.http.request(
            "GET",
            f"https://api.github.com/repos/{repo}/issues/{number}",
            headers={
                "Authorization": authorization_header,
                "Accept": "application/vnd.github+json",
                "User-Agent": "sentinel-agent",
            },
        )


class LinearClient:
    def __init__(self, token: str, http_client: HttpClient | None = None) -> None:
        self.token = token
        self.http = http_client or HttpClient()

    def search_issues(self, query: str) -> dict[str, Any]:
        return self.http.request(
            "POST",
            "https://api.linear.app/graphql",
            headers={
                "Authorization": self.token,
                "Content-Type": "application/json",
            },
            body={
                "query": "query SearchIssues($q: String!) { issueSearch(query: $q) { nodes { id title url } } }",
                "variables": {"q": query},
            },
        )


class SlackClient:
    def __init__(self, token: str, http_client: HttpClient | None = None) -> None:
        self.token = token
        self.http = http_client or HttpClient()

    def post_message(self, channel: str, text: str) -> dict[str, Any]:
        authorization_header = "Bearer " + self.token
        return self.http.request(
            "POST",
            "https://slack.com/api/chat.postMessage",
            headers={
                "Authorization": authorization_header,
                "Content-Type": "application/json; charset=utf-8",
            },
            body={"channel": channel, "text": text},
        )


def configure_otel_tracing(service_name: str = "sentinel-agent", otlp_endpoint: str | None = None) -> TracerProtocol:
    if trace is None:
        return _NoopTracer()

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if otlp_endpoint and OTLPSpanExporter is not None and BatchSpanProcessor is not None:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)


class _NoopSpan:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def record_exception(self, exc: Exception) -> None:
        pass


class _NoopTracer:
    def start_as_current_span(self, name: str):
        return _NoopSpan()


class MultiTurnAgent:
    """Production-focused multi-turn API orchestrator with tracing and graceful degradation."""

    def __init__(
        self,
        *,
        llm_call: Callable[[str, list[dict[str, str]], dict[str, Any]], str],
        github_client: GitHubClient,
        linear_client: LinearClient,
        slack_client: SlackClient,
        tracer: TracerProtocol | None = None,
    ) -> None:
        self.llm_call = llm_call
        self.github = github_client
        self.linear = linear_client
        self.slack = slack_client
        self.tracer = tracer or configure_otel_tracing()
        self.history: list[dict[str, str]] = []

    def run_turn(self, user_message: str, *, github_repo: str, issue_number: int, slack_channel: str) -> TurnResult:
        tool_outputs: dict[str, Any] = {}
        failures: list[str] = []

        with self._span("agent.turn", {"turn.input": user_message}):
            tool_outputs["github"] = self._invoke_tool(
                "github.get_issue",
                lambda: self.github.get_issue(github_repo, issue_number),
                failures,
            )
            tool_outputs["linear"] = self._invoke_tool(
                "linear.search_issues",
                lambda: self.linear.search_issues(user_message),
                failures,
            )

            slack_text = self._render_slack_payload(user_message, tool_outputs, failures)
            tool_outputs["slack"] = self._invoke_tool(
                "slack.post_message",
                lambda: self.slack.post_message(slack_channel, slack_text),
                failures,
            )

            with self._span("llm.call", {"llm.input_length": len(user_message)}):
                prompt_context = {
                    "github": tool_outputs["github"],
                    "linear": tool_outputs["linear"],
                    "failures": failures,
                }
                response = self.llm_call(user_message, self.history, prompt_context)

            self.history.append({"role": "user", "content": user_message})
            self.history.append({"role": "assistant", "content": response})

            return TurnResult(
                user_message=user_message,
                response=response,
                tool_outputs=tool_outputs,
                degraded=bool(failures),
                failures=failures,
            )

    def _render_slack_payload(self, message: str, outputs: dict[str, Any], failures: list[str]) -> str:
        if failures:
            return f"Sentinel degraded response for: {message}. Failures: {', '.join(failures)}"
        github_output = outputs.get("github")
        github_title = github_output.get("title", "unknown") if isinstance(github_output, dict) else "unknown"
        return f"Sentinel summary for '{message}' (GitHub issue: {github_title})"

    def _invoke_tool(self, span_name: str, func: Callable[[], Any], failures: list[str]) -> Any:
        with self._span(f"tool.{span_name}"):
            try:
                return func()
            except Exception as exc:  # noqa: BLE001 - graceful degradation path
                failures.append(span_name)
                traceback_text = traceback.format_exc()
                with self._span(
                    "tool.failure",
                    {"tool.name": span_name, "tool.error": str(exc), "tool.traceback": traceback_text},
                ):
                    pass
                return {
                    "status": "degraded",
                    "tool": span_name,
                    "error": str(exc),
                    "traceback": traceback_text,
                }

    def _span(self, name: str, attrs: dict[str, Any] | None = None):
        span_cm = self.tracer.start_as_current_span(name)

        class _SpanContext:
            def __enter__(self):
                self.span = span_cm.__enter__()
                if attrs and hasattr(self.span, "set_attribute"):
                    for key, value in attrs.items():
                        self.span.set_attribute(key, value)
                return self.span

            def __exit__(self, exc_type, exc, tb):
                if exc and hasattr(self.span, "record_exception"):
                    self.span.record_exception(exc)
                return span_cm.__exit__(exc_type, exc, tb)

        return _SpanContext()
