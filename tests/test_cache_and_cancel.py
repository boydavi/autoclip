"""Transcript-cache compatibility and highlight-stage cancellation."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from autoclip.config import WhisperSettings
from autoclip.pipeline import highlights
from autoclip.pipeline.runner import transcript_matches_settings
from autoclip.pipeline.transcript import Transcript, Word
from autoclip.providers import DetectionConfig
from autoclip.providers.base import ClipCandidates, LLMProvider, ProviderStatus


def row(**overrides) -> SimpleNamespace:
    base = {
        "source": "whisper",
        "model": "small",
        "language": "en",
        "has_diarization": False,
    }
    return SimpleNamespace(**{**base, **overrides})


class TestTranscriptCacheCompatibility:
    def test_same_settings_reuse(self) -> None:
        assert transcript_matches_settings(row(), WhisperSettings(model="small"))

    def test_different_model_does_not_reuse(self) -> None:
        assert not transcript_matches_settings(row(), WhisperSettings(model="large-v3"))

    def test_different_language_does_not_reuse(self) -> None:
        assert not transcript_matches_settings(row(), WhisperSettings(model="small", language="fr"))

    def test_auto_language_reuses_any(self) -> None:
        assert transcript_matches_settings(row(language="fr"), WhisperSettings(model="small"))

    def test_diarization_requested_needs_a_diarized_transcript(self) -> None:
        wanted = WhisperSettings(model="small", diarization=True)

        assert not transcript_matches_settings(row(), wanted)
        assert transcript_matches_settings(row(has_diarization=True), wanted)

    def test_groq_and_local_do_not_share(self) -> None:
        groq = WhisperSettings(provider="groq", groq_model="whisper-large-v3-turbo")

        assert not transcript_matches_settings(row(), groq)
        assert transcript_matches_settings(
            row(source="groq", model="groq/whisper-large-v3-turbo"), groq
        )

    def test_youtube_captions_always_reuse(self) -> None:
        captions = row(source="youtube", model="youtube-captions")

        assert transcript_matches_settings(captions, WhisperSettings(model="large-v3"))


class CountingProvider(LLMProvider):
    name = "counting"
    requires_key = False

    def __init__(self) -> None:
        super().__init__("counting-model")
        self.calls = 0

    async def _complete(self, system, user, config) -> str:
        raise AssertionError("unused")

    async def health_check(self) -> ProviderStatus:
        return ProviderStatus(name=self.name, available=True)

    async def detect_highlights(self, window, config) -> ClipCandidates:
        self.calls += 1
        return ClipCandidates.model_validate({"clips": []})


def long_transcript(words: int = 4000) -> Transcript:
    return Transcript(
        words=[Word(text=f"w{i}.", start=i * 0.5, end=i * 0.5 + 0.4) for i in range(words)]
    )


class TestHighlightCancellation:
    def test_cancelled_before_start_makes_no_provider_calls(self) -> None:
        provider = CountingProvider()

        clips = asyncio.run(
            highlights.detect(
                long_transcript(), provider, DetectionConfig(), job_id="j", is_cancelled=lambda: True
            )
        )

        assert clips == []
        assert provider.calls == 0

    def test_cancelling_midway_skips_the_remaining_windows(self) -> None:
        provider = CountingProvider()
        transcript = long_transcript()
        total_windows = len(highlights.build_windows(transcript))
        assert total_windows > 3

        clips = asyncio.run(
            highlights.detect(
                transcript,
                provider,
                DetectionConfig(),
                job_id="j",
                concurrency=1,
                is_cancelled=lambda: provider.calls >= 2,
            )
        )

        assert clips == []
        assert provider.calls == 2

    def test_backoff_sleep_wakes_early_when_cancelled(self) -> None:
        async def run() -> float:
            loop = asyncio.get_running_loop()
            started = loop.time()
            await highlights._sleep_unless_cancelled(30.0, lambda: True)
            return loop.time() - started

        assert asyncio.run(run()) < 1.0

    def test_not_cancelled_still_detects_normally(self) -> None:
        provider = CountingProvider()

        clips = asyncio.run(
            highlights.detect(
                long_transcript(), provider, DetectionConfig(), job_id="j", is_cancelled=lambda: False
            )
        )

        assert clips == []
        assert provider.calls == len(highlights.build_windows(long_transcript()))


def test_detect_without_cancel_callback_is_unchanged() -> None:
    provider = CountingProvider()

    asyncio.run(highlights.detect(long_transcript(), provider, DetectionConfig(), job_id="j"))

    assert provider.calls > 0
