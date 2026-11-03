"""Groq as the single provider: one key for clip selection and transcription."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from autoclip import config
from autoclip.config import Settings
from autoclip.providers import PROVIDERS, GroqProvider, build_provider
from autoclip.providers.base import DetectionConfig


def test_groq_is_a_registered_provider_with_groq_defaults() -> None:
    provider = GroqProvider(api_key="k")

    assert PROVIDERS["groq"] is GroqProvider
    assert provider.name == "groq"
    assert provider.model == "openai/gpt-oss-120b"
    assert provider.base_url == "https://api.groq.com/openai/v1"


def test_default_settings_point_groq_at_gpt_oss() -> None:
    entry = Settings().providers["groq"]

    assert entry.model == "openai/gpt-oss-120b"
    assert entry.base_url == "https://api.groq.com/openai/v1"


def test_build_provider_uses_the_one_groq_key(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    def fake_secret(name: str, settings=None):
        asked.append(name)
        return "gsk-test"

    monkeypatch.setattr("autoclip.providers.get_secret", fake_secret)

    provider = build_provider("groq", Settings())

    assert asked == ["groq"]
    assert provider.api_key == "gsk-test"


def test_env_key_serves_both_llm_and_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOCLIP_GROQ_KEY", "gsk-env")

    assert config.get_secret("groq") == "gsk-env"
    assert build_provider("groq", Settings()).api_key == "gsk-env"


def test_missing_key_hint_names_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.providers.base import ProviderError

    with pytest.raises(ProviderError) as info:
        GroqProvider()._client()

    assert "set-secret groq" in (info.value.hint or "")


def test_chat_request_reaches_groq_and_reply_content_is_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai

    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "c1", "object": "chat.completion", "created": 0, "model": "openai/gpt-oss-120b",
            "choices": [{
                "index": 0, "finish_reason": "stop",
                # gpt-oss on Groq returns its chain of thought beside the answer.
                "message": {"role": "assistant", "reasoning": "thinking...",
                            "content": '{"clips": []}'},
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        })

    real = openai.AsyncOpenAI

    class Patched(real):
        def __init__(self, **kw):
            kw["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            super().__init__(**kw)

    monkeypatch.setattr(openai, "AsyncOpenAI", Patched)

    text = asyncio.run(GroqProvider(api_key="gsk-x")._complete("sys", "user", DetectionConfig()))

    assert text == '{"clips": []}'
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer gsk-x"
    assert seen["body"]["model"] == "openai/gpt-oss-120b"
    assert seen["body"]["messages"][-1]["role"] == "user"


def test_strict_json_rejection_is_retried_in_plain_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {
                "message": "Failed to validate JSON. Please adjust your prompt.",
                "type": "invalid_request_error", "code": "json_validate_failed",
                "failed_generation": "```json\n{\"clips\": []}\n```",
            }})
        return httpx.Response(200, json={
            "id": "c1", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": '{"clips": []}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    real = openai.AsyncOpenAI

    class Patched(real):
        def __init__(self, **kw):
            kw["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            super().__init__(**kw)

    monkeypatch.setattr(openai, "AsyncOpenAI", Patched)

    text = asyncio.run(GroqProvider(api_key="gsk-x")._complete("sys", "user", DetectionConfig()))

    assert text == '{"clips": []}'
    assert len(bodies) == 2
    assert "response_format" in bodies[0] and "response_format" not in bodies[1]
