"""OpenAI Chat Completions provider with structured output.

Lazy-imports `openai`; if the package is missing or no API key is set, the
provider reports `available() == False` and the router will skip it. This keeps
the deterministic baselines fully runnable without any LLM dependency.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any

from .base import LLMError, LLMUnavailableError, StructuredResponse

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_RETRIES = 2


class OpenAIProvider:
    name = "openai"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: float = 30.0,
    ) -> None:
        self.model = model or os.getenv("LLM_FALLBACK_MODEL", "").strip() or DEFAULT_MODEL
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None

    def available(self) -> bool:
        if not self._api_key:
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def is_primary(self) -> bool:
        return False

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(
                api_key=self._api_key, timeout=self.timeout, max_retries=0)
        return self._client

    @staticmethod
    def _backoff_sleep(attempt: int, max_retries: int) -> float:
        base = min(2.0 ** attempt, 8.0)
        return base * (0.5 + random.random() * 0.5)  # bounded jitter

    def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any],
        *,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> StructuredResponse:
        if not self.available():
            raise LLMUnavailableError("openai: no API key or package missing")

        client = self._get_client()
        timeout = timeout or self.timeout
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    timeout=timeout,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content
                parsed = self._validate(content, response_schema)
                return StructuredResponse(parsed=parsed, raw=resp, provider=self.name)
            except (LLMUnavailableError, LLMError, json.JSONDecodeError, Exception) as exc:  # noqa: BLE001
                last_exc = exc
                if isinstance(exc, LLMUnavailableError):
                    raise
                if attempt < self.max_retries:
                    time.sleep(self._backoff_sleep(attempt, self.max_retries))
        raise LLMError(f"openai: failed after {self.max_retries + 1} attempts: {last_exc}")

    def _validate(self, content: str, schema: dict) -> dict:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMUnavailableError(f"openai: non-JSON response: {exc}")
        # Minimal structural validation (membership checks are fractal elsewhere).
        required = schema.get("required", [])
        missing = [k for k in required if k not in parsed]
        if missing:
            raise LLMUnavailableError(f"openai: schema violation, missing {missing}")
        return parsed