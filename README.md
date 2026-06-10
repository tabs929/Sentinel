# Sentinel

Production-oriented multi-turn orchestration agent for GitHub, Linear, and Slack APIs.

## Features
- Multi-turn agent state with per-turn tool orchestration
- Real API client wrappers for GitHub, Linear, and Slack
- Built-in retry logic for transient HTTP failures
- Graceful degradation when tool calls fail
- OpenTelemetry span coverage for turn execution, tool calls, failures, and LLM calls

## Install
```bash
pip install -r requirements.txt
```

## Quick start
```python
from sentinel import MultiTurnAgent, configure_otel_tracing
from sentinel.agent import GitHubClient, LinearClient, SlackClient

tracer = configure_otel_tracing(
    service_name="sentinel-agent",
    otlp_endpoint="http://localhost:4318/v1/traces",  # optional
)

agent = MultiTurnAgent(
    llm_call=lambda msg, history, context: "LLM response",
    github_client=GitHubClient(token="ghp_xxx"),
    linear_client=LinearClient(token="lin_api_xxx"),
    slack_client=SlackClient(token="xoxb-xxx"),
    tracer=tracer,
)

result = agent.run_turn(
    "Summarize issue status",
    github_repo="tabs929/Sentinel",
    issue_number=1,
    slack_channel="#ops",
)
print(result.response, result.degraded)
```
