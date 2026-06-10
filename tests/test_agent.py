import unittest
from unittest.mock import patch
from urllib.error import URLError

from sentinel.agent import HttpClient, MultiTurnAgent, RetryPolicy


class FakeSpan:
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


class FakeTracer:
    def __init__(self):
        self.spans = []

    def start_as_current_span(self, name):
        return FakeSpan(name, self.spans)


class FakeGitHubClient:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail

    def get_issue(self, repo, number):
        if self.should_fail:
            raise RuntimeError("github unavailable")
        return {"repo": repo, "number": number, "title": "Issue title"}


class FakeLinearClient:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail

    def search_issues(self, query):
        if self.should_fail:
            raise RuntimeError("linear unavailable")
        return {"results": [{"id": "LIN-1", "title": query}]}


class FakeSlackClient:
    def __init__(self):
        self.messages = []

    def post_message(self, channel, text):
        self.messages.append((channel, text))
        return {"ok": True, "channel": channel}


class AgentTests(unittest.TestCase):
    def test_http_client_retries_transient_errors(self):
        attempts = {"count": 0}

        def fake_urlopen(request, timeout):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise URLError("temporary")

            class _Response:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def read(self):
                    return b'{"ok": true}'

            return _Response()

        client = HttpClient(RetryPolicy(attempts=3, backoff_seconds=0))
        with patch("sentinel.agent.urlopen", side_effect=fake_urlopen):
            result = client.request("GET", "https://example.com", headers={})

        self.assertEqual(result["ok"], True)
        self.assertEqual(attempts["count"], 3)

    def test_agent_gracefully_degrades_and_traces_failures(self):
        tracer = FakeTracer()
        agent = MultiTurnAgent(
            llm_call=lambda user_message, history, context: "fallback response",
            github_client=FakeGitHubClient(should_fail=True),
            linear_client=FakeLinearClient(),
            slack_client=FakeSlackClient(),
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


if __name__ == "__main__":
    unittest.main()
