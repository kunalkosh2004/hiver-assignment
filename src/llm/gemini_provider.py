"""Google Gemini (generativeai) provider with structured output + fallback.

Lazy-imports `google.generativeai`; if missing or no key, `available()==False`
and the router skips it. Deterministic baselines never depend on this module.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any

from .base import LLMError, LLMUnavailableError, StructuredResponse

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_MAX_RETRIES = 2


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        self.max_retries = max_retries
        self.timeout = timeout
        self._model_obj = None

    def available(self) -> bool:
        if not self._api_key:
            return False
        try:
            import google.generativeai  # noqa: F401
        except ImportError:
            return False
        return True

    def is_primary(self) -> bool:
        return True

    def _get_model(self):
        if self._model_obj is None:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            genai_safety = [
                {"category": c, "threshold": "BLOCK_NONE"}
                for c in (
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                )
            ]
            self._model_obj = genai.GenerativeModel(
                self.model,
                generation_config={"temperature": 0.0, "response_mime_type": "application/json"},
                safety_settings=genai_safety,
            )
        return self._model_obj

    @staticmethod
    def _backoff_sleep(attempt: int) -> float:
        base = min(2.0 ** attempt, 8.0)
        return base * (0.5 + random.random() * 0.5)

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
            raise LLMUnavailableError("gemini: no API key or package missing")
        model = self._get_model()
        timeout = timeout or self.timeout
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = model.generate_content(
                    f"{system_prompt}\n\n{user_prompt}",
                    request_options={"timeout": timeout},
                )
                text = getattr(resp, "text", "") or ""
                parsed = self._validate(text, response_schema)
                return StructuredResponse(parsed=parsed, raw=resp, provider=self.name)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self._backoff_sleep(attempt))
        raise LLMError(f"gemini: failed after {self.max_retries + 1} attempts: {last_exc}")

    def _validate(self, text: str, schema: dict) -> dict:
        # Gemini often wraps JSON in code fences.
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMUnavailableError(f"gemini: non-JSON response: {exc}")
        required = schema.get("required", [])
        missing = [k for k in required if k not in parsed]
        if missing:
            raise LLMUnavailableError(f"gemini: schema violation, missing {missing}")
        return parsed