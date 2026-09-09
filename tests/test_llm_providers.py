"""Mocked unit tests for the LLM provider layer (no API credits, no network).

Run with either:
    python -m pytest tests/test_llm_providers.py
or:
    python -m unittest tests.test_llm_providers
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm import (  # noqa: E402
    GeminiProvider,
    LLMError,
    LLMUnavailableError,
    LLMRouter,
    OpenAIProvider,
    StructuredResponse,
)
from src.llm.analysis import classify_requests  # noqa: E402
from src.llm.schemas import INTENT_SCHEMA  # noqa: E402

SCHEMA = INTENT_SCHEMA


# --------------------------------------------------------------------------- #
# OpenAI provider
# --------------------------------------------------------------------------- #
class TestOpenAIProvider(unittest.TestCase):
    def test_missing_key_unavailable(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            p = OpenAIProvider(api_key="")
            assert p.available() is False

    def test_key_set_package_ok_is_available(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "x"}, clear=True), \
                mock.patch.dict(sys.modules, {"openai": mock.MagicMock()}):
            p = OpenAIProvider()
            assert p.available() is True

    def test_missing_package_unavailable(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "x"}, clear=True):
            # force ImportError on import
            real_import = __import__
            def fake_import(name, *a, **k):
                if name == "openai":
                    raise ImportError("no openai")
                return real_import(name, *a, **k)
            with mock.patch("builtins.__import__", side_effect=fake_import):
                p = OpenAIProvider()
                assert p.available() is False

    def test_parses_json_response(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "x"}, clear=True), \
                mock.patch.dict(sys.modules, {"openai": mock.MagicMock()}):
            p = OpenAIProvider()

            fake_resp = mock.MagicMock()
            fake_resp.choices[0].message.content = json.dumps({
                "primary_intent": "refund_request",
                "secondary_intents": [],
                "message_type": "support_request",
                "context_dependency": "self_contained",
                "label_confidence": 0.9,
            })
            client = p._get_client()
            client.chat.completions.create.return_value = fake_resp
            out = p.complete_structured("sys", "user", SCHEMA)
            assert out.provider == "openai"
            assert out.parsed["primary_intent"] == "refund_request"

    def test_non_json_raises_unavailable(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "x"}, clear=True), \
                mock.patch.dict(sys.modules, {"openai": mock.MagicMock()}):
            p = OpenAIProvider()
            client = p._get_client()
            client.chat.completions.create.return_value.choices[0].message.content = "not json"
            try:
                p.complete_structured("sys", "user", SCHEMA)
                assert False, "should raise"
            except LLMUnavailableError:
                pass
            except LLMError:
                pass
            except Exception as exc:
                assert isinstance(exc, (LLMUnavailableError, LLMError))

    def test_retries_on_transient_error(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "x"}, clear=True), \
                mock.patch.dict(sys.modules, {"openai": mock.MagicMock()}):
            p = OpenAIProvider(max_retries=2)
            client = p._get_client()

            def _resp():
                r = mock.MagicMock()
                r.choices[0].message.content = json.dumps({
                    "primary_intent": "none",
                    "secondary_intents": [],
                    "message_type": "acknowledgement",
                    "context_dependency": "self_contained",
                    "label_confidence": 0.5,
                })
                return r

            client.chat.completions.create.side_effect = [
                Exception("rate limit"),
                Exception("rate limit"),
                _resp(),
            ]
            out = p.complete_structured("s", "u", SCHEMA)
            assert out.parsed["primary_intent"] == "none"
            assert client.chat.completions.create.call_count == 3


# --------------------------------------------------------------------------- #
# Gemini provider
# --------------------------------------------------------------------------- #
class TestGeminiProvider(unittest.TestCase):
    def test_missing_key_unavailable(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            p = GeminiProvider(api_key="")
            assert p.available() is False

    def test_parses_json(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "x"}, clear=True), \
                mock.patch.dict(sys.modules, {"google.generativeai": mock.MagicMock()}):
            p = GeminiProvider()
            m = mock.MagicMock()
            m.generate_content.return_value.text = json.dumps({
                "primary_intent": "delivery_delay",
                "secondary_intents": [],
                "message_type": "complaint",
                "context_dependency": "context_required",
                "label_confidence": 0.85,
            })
            with mock.patch.object(p, "_get_model", return_value=m):
                out = p.complete_structured("sys", "user", SCHEMA)
            assert out.parsed["primary_intent"] == "delivery_delay"

    def test_strips_code_fence_json(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "x"}, clear=True), \
                mock.patch.dict(sys.modules, {"google.generativeai": mock.MagicMock()}):
            p = GeminiProvider()
            m = mock.MagicMock()
            m.generate_content.return_value.text = '```json\n{"primary_intent": "none", ' \
                '"secondary_intents": [], "message_type": "unknown", ' \
                '"context_dependency": "self_contained", "label_confidence": 0.3}\n```'
            with mock.patch.object(p, "_get_model", return_value=m):
                out = p.complete_structured("sys", "user", SCHEMA)
            assert out.parsed["primary_intent"] == "none"


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
def _fake_ok(name, parsed):
    class _P:
        def __init__(self):
            self.name = name
            self.calls = 0
        def available(self):
            return True
        def is_primary(self):
            return name == "gemini"
        def complete_structured(self, *a, **k):
            self.calls += 1
            return StructuredResponse(parsed=parsed, provider=self.name)
    return _P()


class TestRouter(unittest.TestCase):
    def test_primary_first_when_ok(self):
        p = _fake_ok("gemini", {"primary_intent": "none"})
        f = _fake_ok("openai", {"primary_intent": "x"})
        r = LLMRouter(primary=p, fallback=f)
        out = r.complete_structured("s", "u", SCHEMA)
        assert out.provider == "gemini"
        assert p.calls == 1 and f.calls == 0

    def test_maintains_schema_keys_passthrough(self):
        p = _fake_ok("gemini", {"primary_intent": "none"})
        r = LLMRouter(primary=p)
        out = r.complete_structured("s", "u", SCHEMA)
        assert out.parsed["primary_intent"] == "none"

    def test_fallback_used_when_primary_retryable_fails(self):
        class _Fail:
            name = "gemini"
            def available(self):
                return True
            def is_primary(self):
                return True
            def complete_structured(self, *a, **k):
                raise LLMError("boom")
        class _Ok:
            name = "openai"
            calls = 0
            def available(self):
                return True
            def is_primary(self):
                return False
            def complete_structured(self, *a, **k):
                self.calls += 1
                return StructuredResponse(parsed={"primary_intent": "charge_issue"}, provider="openai")
        pri = _Fail()
        fb = _Ok()
        r = LLMRouter(primary=pri, fallback=fb, max_attempts_primary=2)
        out = r.complete_structured("s", "u", SCHEMA)
        assert out.provider == "openai"
        assert fb.calls == 1
        assert r.last_attempted == ["gemini", "gemini", "openai"]

    def test_both_unavailable_raises(self):
        class _No:
            name = "n"
            def available(self):
                return False
            def complete_structured(self, *a, **k):
                raise AssertionError("should not call")
        r = LLMRouter(primary=_No(), fallback=_No())
        assert r.enabled() is False
        try:
            r.complete_structured("s", "u", SCHEMA)
            assert False, "should have raised"
        except LLMError:
            pass


# --------------------------------------------------------------------------- #
# Analysis job
# --------------------------------------------------------------------------- #
class TestAnalysis(unittest.TestCase):
    def test_disabled_router_returns_summary(self, tmp_path=None):
        import tempfile
        d = Path(tempfile.mkdtemp())
        class _No:
            name = "none"
            def available(self):
                return False
        r = LLMRouter(primary=_No(), fallback=_No())
        res = classify_requests(r, [{"prompt": "x", "id": 1}], out_dir=d)
        assert res["enabled"] is False
        assert res["n"] == 0

    def test_classifies_with_enabled_router(self):
        import tempfile
        from datetime import datetime
        d = Path(tempfile.mkdtemp())
        p = _fake_ok("gemini", {"primary_intent": "refund_request",
                                "secondary_intents": [],
                                "message_type": "support_request",
                                "context_dependency": "self_contained",
                                "label_confidence": 0.8})
        r = LLMRouter(primary=p)
        res = classify_requests(r, [{"prompt": "msg", "id": 1},
                                    {"prompt": "msg2", "id": 2}],
                                out_dir=d, max_calls_per_run=10, sleep_between=0)
        assert res["enabled"] is True
        assert res["n_ok"] == 2
        mf = d / "llm_classification_manifest.json"
        assert mf.exists()
        data = json.loads(mf.read_text())
        assert "NOT GROUND TRUTH" in data["marker"]

    def test_max_calls_respected(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        p = _fake_ok("gemini", {"primary_intent": "none"})
        r = LLMRouter(primary=p)
        items = [{"prompt": f"m{i}", "id": i} for i in range(20)]
        res = classify_requests(r, items, out_dir=d, max_calls_per_run=5, sleep_between=0)
        assert res["n_calls"] == 5
        assert res["n_ok"] == 5


if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)