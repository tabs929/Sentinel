import unittest
from unittest.mock import patch
from urllib.error import URLError

from sentinel.agent import HttpClient, MultiTurnAgent, RetryPolicy


class MockSpan:
    def __init__(self, name, sink):
        self.name = name
        self.sink = sink
        self.attributes = {}

    def __enter__(self):
        self.sink.append((self.name, self.attributes))
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exc):
        self.attributes["exception"] = str(exc)


class MockTracer:
    def __init__(self):
        self.spans = []

    def start_as_current_span(self, name):
        return MockSpan(name, self.spans)


class MockGitHubClient:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail

    def get_issue(self, repo, number):
        if self.should_fail:
            raise RuntimeError("github unavailable")
        return {"repo": repo, "number": number, "title": "Issue title"}


class MockLinearClient:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail

    def search_issues(self, query):
        if self.should_fail:
            raise RuntimeError("linear unavailable")
        return {"results": [{"id": "LIN-1", "title": query}]}


class MockSlackClient:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.messages = []

    def post_message(self, channel, text):
        if self.should_fail:
            raise RuntimeError("slack unavailable")
        self.messages.append((channel, text))
        return {"ok": True, "channel": channel}


class AgentTests(unittest.TestCase):
    def test_http_client_retries_url_errors(self):
        attempts = {"count": 0}
        policy = RetryPolicy(attempts=3, backoff_seconds=0)

        def fake_urlopen(request, timeout):
            attempts["count"] += 1
            if attempts["count"] < policy.attempts:
                raise URLError("temporary")

            class MockResponse:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def read(self):
                    return b'{"ok": true}'

            return MockResponse()

        client = HttpClient(policy)
        with patch("sentinel.agent.urlopen", side_effect=fake_urlopen):
            result = client.request("GET", "https://example.com", headers={})

        self.assertEqual(result["ok"], True)
        self.assertEqual(attempts["count"], 3)

    def test_http_client_raises_after_exhausted_retries(self):
        attempts = {"count": 0}

        def always_fail(request, timeout):
            attempts["count"] += 1
            raise URLError("still failing")

        client = HttpClient(RetryPolicy(attempts=3, backoff_seconds=0))
        with patch("sentinel.agent.urlopen", side_effect=always_fail):
            with self.assertRaises(URLError):
                client.request("GET", "https://example.com", headers={})

        self.assertEqual(attempts["count"], 3)

    def test_agent_gracefully_degrades_and_traces_failures(self):
        tracer = MockTracer()
        agent = MultiTurnAgent(
            llm_call=lambda user_message, history, context: "fallback response",
            github_client=MockGitHubClient(should_fail=True),
            linear_client=MockLinearClient(),
            slack_client=MockSlackClient(),
            tracer=tracer,
        )

        result = agent.run_turn(
            "status update",
            github_repo="tabs929/Sentinel",
            issue_number=1,
            slack_channel="#ops",
        )

        self.assertTrue(result.degraded)
        self.assertIn("github.get_issue", result.failures)
        self.assertEqual(result.response, "fallback response")
        span_names = [name for name, _ in tracer.spans]
        self.assertIn("agent.turn", span_names)
        self.assertIn("tool.github.get_issue", span_names)
        self.assertIn("tool.failure", span_names)
        self.assertIn("llm.call", span_names)

    def test_agent_gracefully_degrades_on_linear_and_slack_failures(self):
        tracer = MockTracer()
        agent = MultiTurnAgent(
            llm_call=lambda user_message, history, context: "degraded response",
            github_client=MockGitHubClient(),
            linear_client=MockLinearClient(should_fail=True),
            slack_client=MockSlackClient(should_fail=True),
            tracer=tracer,
        )

        result = agent.run_turn(
            "status update",
            github_repo="tabs929/Sentinel",
            issue_number=1,
            slack_channel="#ops",
        )

        self.assertTrue(result.degraded)
        self.assertIn("linear.search_issues", result.failures)
        self.assertIn("slack.post_message", result.failures)


if __name__ == "__main__":
    unittest.main()
