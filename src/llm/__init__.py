"""LLM provider layer — resilient OpenAI + Gemini with automatic fallback.

The layer is strictly optional: if no API key(s) are set or the provider
packages are not installed, providers report `available() == False` and the
router simply reports the LLM as disabled. Deterministic Phase-4 baselines run
without ever importing this module.
"""
from __future__ import annotations

from .base import LLMError, LLMUnavailableError, LLMProvider, StructuredResponse
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider
from .router import LLMRouter

__all__ = [
    "LLMError",
    "LLMUnavailableError",
    "LLMProvider",
    "StructuredResponse",
    "GeminiProvider",
    "OpenAIProvider",
    "LLMRouter",
    "make_router",
]


def make_router(**kwargs):
    """Factory with spec defaults: primary = Gemini, fallback = OpenAI."""
    return LLMRouter(**kwargs)