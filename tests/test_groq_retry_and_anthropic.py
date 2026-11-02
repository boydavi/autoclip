"""Groq 429/5xx retry + per-chunk response cache, and the Anthropic request shape."""

from __future__ import annotations

import asyncio
import json
import sys
import types

import pytest
from autoclip.pipeline import transcribe
from autoclip.pipeline.transcribe import TranscriptionError
from autoclip.providers import anthropic_provider
from autoclip.providers.anthropic_provider import AnthropicProvider, accepts_sampling_params
from autoclip.providers.base import DetectionConfig


class FakeApiError(Exception):
    def __init__(self, status: int | None, retry_after: str | None = None) -> None:
        super().__init__(f"status {status}")
        self.status_code = status
        headers = {"retry-after": retry_after} if retry_after is not None else {}
        self.response = types.SimpleNamespace(headers=headers)


class FakeConnectionError(Exception):
    pass


FakeConnectionError.__name__ = "APIConnectionError"


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch):
    """Replace ``openai.OpenAI`` with a client that plays back scripted outcomes."""
    state: dict = {"script": [], "calls": 0, "client_kwargs": None, "sleeps": []}

    class Transcriptions:
        def create(self, **kwargs):
            state["calls"] += 1
            outcome = state["script"].pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    class Client:
        def __init__(self, **kwargs) -> None:
            state["client_kwargs"] = kwargs
            self.audio = types.SimpleNamespace(transcriptions=Transcriptions())

    module = types.ModuleType("openai")
    module.OpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setattr(transcribe.time, "sleep", lambda s: state["sleeps"].append(s))
    return state


@pytest.fixture
def chunk(tmp_path):
    path = tmp_path / "chunk.mp3"
    path.write_bytes(b"audio")
    return path


OK = {"words": [{"word": "hi", "start": 0.0, "end": 0.2}], "segments": []}


class TestGroqRetry:
    def test_client_does_not_retry_on_its_own(self, fake_openai, chunk) -> None:
        fake_openai["script"] = [OK]

        transcribe._call_groq_api(chunk, "k", "", "m")

        assert fake_openai["client_kwargs"]["max_retries"] == 0

    def test_429_waits_for_retry_after_then_succeeds(self, fake_openai, chunk) -> None:
        fake_openai["script"] = [FakeApiError(429, "42"), OK]

        assert transcribe._call_groq_api(chunk, "k", "", "m") == OK
        assert fake_openai["sleeps"] == [42.0]
        assert fake_openai["calls"] == 2

    def test_429_without_header_backs_off_exponentially(self, fake_openai, chunk) -> None:
        fake_openai["script"] = [FakeApiError(429), FakeApiError(429), OK]

        transcribe._call_groq_api(chunk, "k", "", "m")

        assert fake_openai["sleeps"] == [5.0, 10.0]

    def test_5xx_and_connection_errors_are_retried(self, fake_openai, chunk) -> None:
        fake_openai["script"] = [FakeApiError(503), FakeConnectionError("reset"), OK]

        assert transcribe._call_groq_api(chunk, "k", "", "m") == OK
        assert fake_openai["calls"] == 3

    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    def test_client_errors_fail_immediately(self, fake_openai, chunk, status) -> None:
        fake_openai["script"] = [FakeApiError(status)]

        with pytest.raises(TranscriptionError, match="failed"):
            transcribe._call_groq_api(chunk, "k", "", "m")
        assert fake_openai["calls"] == 1
        assert fake_openai["sleeps"] == []

    def test_unrelated_exceptions_are_not_retried(self, fake_openai, chunk) -> None:
        fake_openai["script"] = [ValueError("bug")]

        with pytest.raises(TranscriptionError):
            transcribe._call_groq_api(chunk, "k", "", "m")
        assert fake_openai["calls"] == 1

    def test_gives_up_after_the_attempt_limit(self, fake_openai, chunk) -> None:
        fake_openai["script"] = [FakeApiError(503)] * transcribe._GROQ_MAX_ATTEMPTS

        with pytest.raises(TranscriptionError, match="failed"):
            transcribe._call_groq_api(chunk, "k", "", "m")
        assert fake_openai["calls"] == transcribe._GROQ_MAX_ATTEMPTS
        assert len(fake_openai["sleeps"]) == transcribe._GROQ_MAX_ATTEMPTS - 1

    def test_an_exhausted_hourly_quota_fails_fast_with_the_wait(self, fake_openai, chunk) -> None:
        fake_openai["script"] = [FakeApiError(429, "2400")]

        with pytest.raises(TranscriptionError, match="40 min"):
            transcribe._call_groq_api(chunk, "k", "", "m")
        assert fake_openai["sleeps"] == []


class TestGroqChunkCache:
    def test_second_call_reuses_the_saved_response(self, monkeypatch, chunk) -> None:
        calls: list[int] = []
        monkeypatch.setattr(
            transcribe, "_call_groq_api", lambda *a: calls.append(1) or dict(OK)
        )

        first = transcribe._cached_groq_call(chunk, "k", "en", "m")
        second = transcribe._cached_groq_call(chunk, "k", "en", "m")

        assert first == second == OK
        assert calls == [1]

    def test_a_different_model_or_language_is_not_reused(self, monkeypatch, chunk) -> None:
        calls: list[int] = []
        monkeypatch.setattr(
            transcribe, "_call_groq_api", lambda *a: calls.append(1) or dict(OK)
        )

        transcribe._cached_groq_call(chunk, "k", "en", "m")
        transcribe._cached_groq_call(chunk, "k", "en", "other-model")
        transcribe._cached_groq_call(chunk, "k", "fr", "other-model")

        assert len(calls) == 3

    def test_a_changed_audio_file_is_not_reused(self, monkeypatch, chunk) -> None:
        calls: list[int] = []
        monkeypatch.setattr(
            transcribe, "_call_groq_api", lambda *a: calls.append(1) or dict(OK)
        )
        transcribe._cached_groq_call(chunk, "k", "en", "m")

        chunk.write_bytes(b"different audio")
        transcribe._cached_groq_call(chunk, "k", "en", "m")

        assert len(calls) == 2

    def test_a_corrupt_cache_file_is_ignored(self, monkeypatch, chunk) -> None:
        chunk.with_name(chunk.name + ".groq.json").write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(transcribe, "_call_groq_api", lambda *a: dict(OK))

        assert transcribe._cached_groq_call(chunk, "k", "en", "m") == OK

    def test_failed_calls_are_not_cached(self, monkeypatch, chunk) -> None:
        def boom(*a):
            raise TranscriptionError("rate limited")

        monkeypatch.setattr(transcribe, "_call_groq_api", boom)
        with pytest.raises(TranscriptionError):
            transcribe._cached_groq_call(chunk, "k", "en", "m")

        assert not chunk.with_name(chunk.name + ".groq.json").exists()

    def test_resume_after_a_mid_file_failure_only_pays_for_the_rest(
        self, monkeypatch, tmp_path
    ) -> None:
        chunks = []
        for i in range(3):
            path = tmp_path / f"c{i}.mp3"
            path.write_bytes(b"x" * (i + 1))
            chunks.append((path, i * 1200.0))
        monkeypatch.setattr(transcribe, "_compress_for_groq", lambda p, w: p)
        monkeypatch.setattr(transcribe, "_split_audio", lambda p, w: chunks)
        paid: list[str] = []

        def api(path, key, language, model):
            if path.name == "c2.mp3" and "fail" in paid:
                paid.remove("fail")
                raise TranscriptionError("429")
            paid.append(path.name)
            return {"words": [{"word": "w", "start": 8.0, "end": 8.5}], "segments": []}

        monkeypatch.setattr(transcribe, "_call_groq_api", api)
        paid.append("fail")
        from autoclip.config import WhisperSettings

        settings = WhisperSettings(provider="groq")
        with pytest.raises(TranscriptionError):
            transcribe.transcribe_groq(chunks[0][0], settings, api_key="k")
        assert paid == ["c0.mp3", "c1.mp3"]

        result = transcribe.transcribe_groq(chunks[0][0], settings, api_key="k")

        assert paid == ["c0.mp3", "c1.mp3", "c2.mp3"]  # c0/c1 came from the cache
        assert len(result.words) == 3


class TestSamplingParamRule:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-3-5-haiku-20241022",
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-5",
            "claude-opus-4-1",
        ],
    )
    def test_legacy_models_accept_temperature(self, model: str) -> None:
        assert accepts_sampling_params(model)

    @pytest.mark.parametrize(
        "model",
        [
            "claude-haiku-5-5",
            "claude-sonnet-5",
            "claude-sonnet-5-5",
            "claude-opus-5",
            "claude-opus-5-5",
            "claude-opus-4-8",
        ],
    )
    def test_current_models_do_not(self, model: str) -> None:
        assert not accepts_sampling_params(model)


class FakeMessages:
    def __init__(self, text: str) -> None:
        self.text = text
        self.kwargs: dict = {}

    async def create(self, **kwargs):
        self.kwargs = kwargs
        content = [
            types.SimpleNamespace(type="thinking", thinking="", signature="s"),
            types.SimpleNamespace(type="text", text=self.text),
        ]
        return types.SimpleNamespace(content=content)


def make_provider(model: str, monkeypatch: pytest.MonkeyPatch, text: str) -> tuple:
    messages = FakeMessages(text)
    provider = AnthropicProvider(model, api_key="k")
    monkeypatch.setattr(
        provider, "_client", lambda: types.SimpleNamespace(messages=messages)
    )
    return provider, messages


class TestAnthropicRequestShape:
    CFG = DetectionConfig()

    def test_current_model_request_has_no_prefill_and_no_sampling_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider, messages = make_provider("claude-haiku-5-5", monkeypatch, '{"clips": []}')

        asyncio.run(provider._complete("sys", "user text", self.CFG))

        sent = messages.kwargs
        assert sent["messages"] == [{"role": "user", "content": "user text"}]
        assert "temperature" not in sent
        assert sent["max_tokens"] >= 8192
        assert sent["model"] == "claude-haiku-5-5"

    def test_legacy_model_still_gets_temperature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider, messages = make_provider("claude-haiku-4-5-20251001", monkeypatch, "{}")

        asyncio.run(provider._complete("sys", "u", self.CFG))

        assert messages.kwargs["temperature"] == self.CFG.temperature
        assert messages.kwargs["messages"][-1]["role"] == "user"

    def test_returns_only_text_blocks_ignoring_thinking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider, _ = make_provider("claude-haiku-5-5", monkeypatch, '{"clips": []}')

        assert asyncio.run(provider._complete("s", "u", self.CFG)) == '{"clips": []}'

    def test_prose_and_fences_around_the_json_still_parse(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from autoclip.providers.base import extract_json_object

        reply = 'Here you go:\n```json\n{"clips": [{"start": 1}]}\n```'
        provider, _ = make_provider("claude-haiku-5-5", monkeypatch, reply)

        raw = asyncio.run(provider._complete("s", "u", self.CFG))

        assert extract_json_object(raw) == {"clips": [{"start": 1}]}

    def test_suggested_models_list_the_current_haiku(self) -> None:
        assert "claude-haiku-5-5" in anthropic_provider.SUGGESTED_MODELS
        json.dumps(anthropic_provider.SUGGESTED_MODELS)
