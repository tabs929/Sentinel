"""Input and output guardrails.

Input guardrails run *before* the user's message reaches the LLM. They block
obvious prompt-injection attempts and enforce a length limit.

Output guardrails run *after* the LLM produces its final answer. They ensure
no tool credential leaks into the response and that the response has the
expected shape (a non-empty string).
"""

from __future__ import annotations

import re

from sentinel.config import Settings, get_settings
from sentinel.models import GuardrailDecision

_RE = re.compile

# Patterns that strongly indicate a prompt-injection / jailbreak attempt.
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "ignore_instructions",
        _RE(
            r"\bignore\s+(all\s+|the\s+)?(previous|prior|above|earlier)\s+"
            r"\w*\s*(instructions|rules|prompt|directives)\b",
            re.I,
        ),
    ),
    (
        "disregard_instructions",
        _RE(r"\bdisregard\s+.{0,25}?(instructions|rules|prompt|guidelines|directives)\b", re.I),
    ),
    (
        "reveal_system_prompt",
        _RE(
            r"\b(reveal|show|print|repeat|output)\b.{0,30}"
            r"\b(system\s+prompt|your\s+instructions|initial\s+prompt)\b",
            re.I,
        ),
    ),
    (
        "exfiltrate_secrets",
        _RE(
            r"\b(print|reveal|show|leak|give\s+me|what\s+is)\b.{0,40}"
            r"\b(api[_\s-]?key|token|secret|password|credential)s?\b",
            re.I,
        ),
    ),
    (
        "override_role",
        _RE(
            r"\byou\s+are\s+now\b.{0,40}\b(dan|developer\s+mode|unrestricted|jailbroken)\b",
            re.I,
        ),
    ),
    (
        "pretend_no_rules",
        _RE(r"\bpretend\b.{0,30}\b(no|without)\b.{0,20}\b(rules|restrictions|guidelines)\b", re.I),
    ),
]


class InputGuardrail:
    name = "input"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def check(self, text: str) -> GuardrailDecision:
        triggered: list[str] = []
        reasons: list[str] = []

        if not text or not text.strip():
            return GuardrailDecision(
                allowed=False,
                triggered=["empty_input"],
                reason="Message is empty.",
            )

        max_chars = self.settings.max_input_chars
        if len(text) > max_chars:
            triggered.append("length_limit")
            reasons.append(
                f"Message is {len(text)} chars, exceeding the {max_chars}-char limit."
            )

        for rule_name, pattern in _INJECTION_PATTERNS:
            if pattern.search(text):
                triggered.append(rule_name)
                reasons.append(f"Matched prompt-injection rule '{rule_name}'.")

        allowed = len(triggered) == 0
        return GuardrailDecision(
            allowed=allowed,
            triggered=triggered,
            reason=" ".join(reasons) if reasons else None,
        )


class OutputGuardrail:
    name = "output"

    def __init__(self, secrets: list[str] | None = None) -> None:
        # Only treat sufficiently long secrets as leak-worthy, to avoid
        # redacting incidental short strings.
        self._secrets = [s for s in (secrets or []) if s and len(s) >= 8]

    def check(self, text: str) -> GuardrailDecision:
        triggered: list[str] = []
        reasons: list[str] = []

        if not isinstance(text, str) or not text.strip():
            return GuardrailDecision(
                allowed=False,
                triggered=["empty_output"],
                reason="The agent produced an empty response.",
                sanitized="",
            )

        sanitized = text
        for secret in self._secrets:
            if secret in sanitized:
                sanitized = sanitized.replace(secret, "[REDACTED_CREDENTIAL]")
                triggered.append("credential_leak")
                reasons.append("Redacted a tool credential from the response.")

        # Output is still delivered (sanitized); the leak is flagged for traces.
        return GuardrailDecision(
            allowed=True,
            triggered=triggered,
            reason=" ".join(reasons) if reasons else None,
            sanitized=sanitized,
        )
