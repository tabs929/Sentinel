"""Observability layer: structured events + OpenTelemetry-style GenAI spans."""

from sentinel.observability.tracing import GenAI, Tracer

__all__ = ["Tracer", "GenAI"]
