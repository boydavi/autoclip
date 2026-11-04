"""Groq usage tracking: tokens counted per call, stored per job, priced as an estimate."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from autoclip import db
from autoclip.api.schemas import JobOut
from autoclip.db import store
from autoclip.db.models import Job, Source, new_id
from autoclip.providers import GroqProvider
from autoclip.providers.base import DetectionConfig
from autoclip.usage import Usage, llm_cost_usd, whisper_cost_usd


def test_chat_cost_uses_the_listed_price_and_never_guesses_an_unknown_model() -> None:
    # 1M input + 1M output of gpt-oss-120b = 0.15 + 0.60
    assert llm_cost_usd("openai/gpt-oss-120b", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert llm_cost_usd("some/new-model", 5_000_000, 5_000_000) == 0.0


def test_whisper_cost_is_per_hour_of_audio() -> None:
    assert whisper_cost_usd("whisper-large-v3-turbo", 3600) == pytest.approx(0.04)
    assert whisper_cost_usd("whisper-large-v3-turbo", 5580) == pytest.approx(0.062, abs=0.001)   # 93 minutes
    assert whisper_cost_usd("mystery", 3600) == 0.0


def test_usage_adds_up_and_tolerates_missing_counts() -> None:
    usage = Usage()
    usage.add(1000, 200)
    usage.add(None, None)
    usage.add(500, 100)
    assert (usage.calls, usage.input_tokens, usage.output_tokens) == (3, 1500, 300)


def _fake_groq(monkeypatch: pytest.MonkeyPatch, usage: dict | None) -> None:
    import openai

    def handler(request: httpx.Request) -> httpx.Response:
        body = {
            "id": "c1", "object": "chat.completion", "created": 0, "model": "openai/gpt-oss-120b",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": '{"clips": []}'}}],
        }
        if usage is not None:
            body["usage"] = usage
        return httpx.Response(200, json=body)

    real = openai.AsyncOpenAI

    class Patched(real):
        def __init__(self, **kw):
            kw["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            super().__init__(**kw)

    monkeypatch.setattr(openai, "AsyncOpenAI", Patched)


def test_the_provider_counts_the_tokens_each_call_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_groq(monkeypatch, {"prompt_tokens": 8900, "completion_tokens": 700, "total_tokens": 9600})
    provider = GroqProvider(api_key="gsk-x")
    for _ in range(2):
        asyncio.run(provider._complete("s", "u", DetectionConfig()))
    assert (provider.usage.calls, provider.usage.input_tokens, provider.usage.output_tokens) == (2, 17800, 1400)


def test_a_reply_without_usage_still_works_and_counts_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_groq(monkeypatch, None)
    provider = GroqProvider(api_key="gsk-x")
    assert asyncio.run(provider._complete("s", "u", DetectionConfig())) == '{"clips": []}'
    assert provider.usage.input_tokens == 0


def test_two_providers_do_not_share_counts() -> None:
    a, b = GroqProvider(api_key="k"), GroqProvider(api_key="k")
    a.usage.add(10, 5)
    assert b.usage.calls == 0


@pytest.fixture
def job(initialised_db: int) -> Job:
    source = store.create_source(Source(id=new_id(), type="upload", path="x.mp4", title="x", duration_s=100.0))
    return store.create_job(Job(id=new_id(), source_id=source.id))


def test_a_new_job_starts_at_zero(job: Job) -> None:
    fresh = store.get_job(job.id)
    assert (fresh.llm_calls, fresh.llm_input_tokens, fresh.transcribe_seconds, fresh.est_cost_usd) == (0, 0, 0.0, 0.0)


def test_usage_accumulates_across_stages_and_retries(job: Job) -> None:
    store.add_job_usage(job.id, transcribe_seconds=5580, cost_usd=0.062)
    store.add_job_usage(job.id, calls=9, input_tokens=80_000, output_tokens=9_000, cost_usd=0.0174)
    store.add_job_usage(job.id, calls=1, input_tokens=9_000, output_tokens=1_000, cost_usd=0.0020)   # a retried window
    got = store.get_job(job.id)
    assert (got.llm_calls, got.llm_input_tokens, got.llm_output_tokens) == (10, 89_000, 10_000)
    assert got.transcribe_seconds == 5580
    assert got.est_cost_usd == pytest.approx(0.0814)


def test_the_api_exposes_the_job_usage(job: Job) -> None:
    store.add_job_usage(job.id, calls=3, input_tokens=1234, output_tokens=56, transcribe_seconds=60.0, cost_usd=0.1234567)
    out = JobOut.of(store.get_job(job.id))
    assert (out.llm_calls, out.llm_input_tokens, out.llm_output_tokens, out.transcribe_seconds) == (3, 1234, 56, 60.0)
    assert out.est_cost_usd == 0.12346
    assert json.loads(out.model_dump_json())["llm_input_tokens"] == 1234


def test_the_migration_adds_the_usage_columns(initialised_db: int) -> None:
    with db.connection() as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
    assert {"llm_calls", "llm_input_tokens", "llm_output_tokens", "transcribe_seconds", "est_cost_usd"} <= cols
