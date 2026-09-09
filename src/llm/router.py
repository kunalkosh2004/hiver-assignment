"""Router with automatic primary-first + fallback behavior.

Default primary = Gemini, fallback = OpenAI (per spec). On a *retryable* error
the router retries the primary (bounded, backoff) then fails over to the
fallback provider. Unavailable providers (no key / no package) are skipped
silently. The router itself performs no network I/O on construction, so it is
safe to import in offline/CI contexts.
"""
from __future__ import annotations

import time
from typing import Any

from .base import LLMError, LLMUnavailableError, StructuredResponse
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider


class LLMRouter:
    def __init__(
        self,
        *,
        primary: Any = None,
        fallback: Any = None,
        max_attempts_primary: int = 1,
        fallback_enabled: bool = True,
        retry_base_sleep: float = 0.5,
    ) -> None:
        self.primary = primary or GeminiProvider()
        self.fallback = fallback or OpenAIProvider()
        self.max_attempts_primary = max(1, max_attempts_primary)
        self.fallback_enabled = fallback_enabled
        self.retry_base_sleep = retry_base_sleep
        self.last_attempted: list[str] = []

    def available_providers(self) -> list[str]:
        out = []
        for p in (self.primary, self.fallback):
            try:
                if p.available():
                    out.append(p.name)
            except Exception:  # noqa: BLE001
                continue
        return out

    def enabled(self) -> bool:
        return bool(self.available_providers())

    def _avail(self, p: Any) -> bool:
        try:
            return bool(p and p.available())
        except Exception:  # noqa: BLE001
            return False

    def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any],
        *,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> StructuredResponse:
        """Primary-first with bounded retries, then fallback."""
        self.last_attempted = []
        errors: list[str] = []

        # 1) primary with bounded retry
        if self._avail(self.primary):
            for attempt in range(self.max_attempts_primary):
                self.last_attempted.append(self.primary.name)
                try:
                    return self.primary.complete_structured(
                        system_prompt, user_prompt, response_schema,
                        temperature=temperature, timeout=timeout)
                except LLMUnavailableError as exc:
                    errors.append(str(exc))
                    break  # permanent — stop retrying primary
                except LLMError as exc:
                    errors.append(str(exc))
                    if attempt < self.max_attempts_primary - 1:
                        time.sleep(self.retry_base_sleep * (attempt + 1))

        # 2) fallback
        if self.fallback_enabled and self._avail(self.fallback):
            self.last_attempted.append(self.fallback.name)
            try:
                return self.fallback.complete_structured(
                    system_prompt, user_prompt, response_schema,
                    temperature=temperature, timeout=timeout)
            except LLMUnavailableError as exc:
                errors.append(str(exc))
            except LLMError as exc:
                errors.append(str(exc))

        if not errors:
            errors.append("no provider available (set OPENAI_API_KEY/GEMINI_API_KEY)")
        raise LLMError(f"llm: no response. Errors: {'; '.join(errors)}")