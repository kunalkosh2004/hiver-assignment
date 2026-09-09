"""Base abstractions for the resilient LLM provider layer.

The layer is intentionally non-blocking: if a provider package is not installed
or no API key is set, the provider is marked `available=False` and the router
skips it. Deterministic baseline runs never depend on this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class StructuredResponse:
    """A typed, structured result produced by an LLM provider.

    `parsed` holds the validated output dict (conforming to the request's
    JSON schema). `raw` preserves the provider's original response for audit.
    """
    parsed: dict[str, Any]
    raw: Any = None
    provider: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    """Common interface all providers implement."""

    name: str

    def available(self) -> bool: ...  # noqa: E704 (protocol stub)

    def is_primary(self) -> bool: ...  # noqa: E704

    def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any],
        *,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> StructuredResponse:
        """Return a JSON-validated, schema-compliant structured response.

        Raises LLMError (retryable) or LLMUnavailableError (permanent) so the
        router can decide to fall back.
        """
        ...


class LLMError(Exception):
    """Retryable transient provider error."""


class LLMUnavailableError(Exception):
    """Permanent failure (no key, bad model, schema violation)."""