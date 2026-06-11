"""Tools layer: every tool returns a ToolResult and never raises."""

from sentinel.tools.base import BaseTool, ToolError, ToolRegistry, build_default_registry
from sentinel.tools.github import GitHubTool
from sentinel.tools.linear import LinearTool
from sentinel.tools.slack import SlackTool
from sentinel.tools.weather import WeatherTool

__all__ = [
    "BaseTool",
    "ToolError",
    "ToolRegistry",
    "build_default_registry",
    "WeatherTool",
    "GitHubTool",
    "LinearTool",
    "SlackTool",
]
