"""Groq cloud transcription (API mocked) and cost-estimate arithmetic."""

from __future__ import annotations

import pytest
from autoclip import config
from autoclip.api import estimate
from autoclip.config import WhisperSettings
from autoclip.pipeline import transcribe
from autoclip.pipeline.highlights import OVERLAP_S, WINDOW_S
from autoclip.pipeline.transcribe import TranscriptionError
from autoclip.pipeline.transcript import Word


def groq_word(text: str, start: float, end: float) -> dict:
    return {"word": f" {text} ", "start": start, "end": end}


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"\0" * 1024)
    return path


@pytest.fixture
def stub_upload(monkeypatch: pytest.MonkeyPatch, audio):
    """Skip real compression/splitting; hand back whatever chunks a test sets."""
    state: dict = {"chunks": [(audio, 0.0)], "responses": {}, "calls": []}

    monkeypatch.setattr(transcribe, "_compress_for_groq", lambda path, workdir: path)
    monkeypatch.setattr(transcribe, "_split_audio", lambda path, workdir: state["chunks"])

    def fake_call(file_path, api_key, language, model):
        state["calls"].append(
            {"file": file_path, "key": api_key, "language": language, "model": model}
        )
        return state["responses"][file_path]

    monkeypatch.setattr(transcribe, "_call_groq_api", fake_call)
    return state


class TestGroqTranscription:
    def test_single_chunk_maps_words_and_segments(self, audio, stub_upload) -> None:
        stub_upload["responses"][audio] = {
            "words": [groq_word("hello", 0.0, 0.4), groq_word("world", 0.5, 0.9)],
            "segments": [{"text": " hello world ", "start": 0.0, "end": 1.0}],
        }

        result = transcribe.transcribe_groq(audio, WhisperSettings(provider="groq"), api_key="k")

        assert [w.text for w in result.words] == ["hello", "world"]
        assert result.source == "groq"
        assert result.model == "groq/whisper-large-v3-turbo"
        segment = result.segments[0]
        assert (segment.text, segment.first_word, segment.last_word) == ("hello world", 0, 1)

    def test_passes_key_language_and_model_to_the_api(self, audio, stub_upload) -> None:
        stub_upload["responses"][audio] = {"words": [groq_word("hi", 0.0, 0.2)]}
        settings = WhisperSettings(provider="groq", language="fr", groq_model="whisper-large-v3")

        result = transcribe.transcribe_groq(audio, settings, api_key="secret-key")

        call = stub_upload["calls"][0]
        assert (call["key"], call["language"], call["model"]) == (
            "secret-key",
            "fr",
            "whisper-large-v3",
        )
        assert result.language == "fr"

    def test_language_defaults_to_en_when_auto(self, audio, stub_upload) -> None:
        stub_upload["responses"][audio] = {"words": [groq_word("hi", 0.0, 0.2)]}

        result = transcribe.transcribe_groq(audio, WhisperSettings(provider="groq"), api_key="k")

        assert result.language == "en"
        assert stub_upload["calls"][0]["language"] == ""

    def test_later_chunks_are_offset_and_overlap_words_dropped(
        self, audio, tmp_path, stub_upload
    ) -> None:
        second = tmp_path / "chunk2.mp3"
        second.write_bytes(b"\0")
        stub_upload["chunks"] = [(audio, 0.0), (second, 1200.0)]
        stub_upload["responses"][audio] = {
            "words": [groq_word("first", 10.0, 10.5), groq_word("tail", 1197.0, 1198.0)],
            "segments": [],
        }
        stub_upload["responses"][second] = {
            "words": [
                # Inside the 5 s overlap: already present in chunk one, so dropped.
                groq_word("tail", 2.0, 3.0),
                groq_word("next", 6.0, 6.5),
            ],
            "segments": [{"text": "next", "start": 6.0, "end": 7.0}],
        }

        result = transcribe.transcribe_groq(audio, WhisperSettings(provider="groq"), api_key="k")

        assert [w.text for w in result.words] == ["first", "tail", "next"]
        assert result.words[-1].start == pytest.approx(1206.0)
        starts = [w.start for w in result.words]
        assert starts == sorted(starts)
        assert result.segments[0].start == pytest.approx(1206.0)
        assert result.segments[0].first_word == 2

    def test_progress_reports_per_chunk_and_completes(self, audio, tmp_path, stub_upload) -> None:
        second = tmp_path / "chunk2.mp3"
        second.write_bytes(b"\0")
        stub_upload["chunks"] = [(audio, 0.0), (second, 1200.0)]
        for chunk in (audio, second):
            stub_upload["responses"][chunk] = {"words": [groq_word("w", 8.0, 8.5)]}
        seen: list[float] = []

        transcribe.transcribe_groq(
            audio, WhisperSettings(provider="groq"), api_key="k", on_progress=seen.append
        )

        assert seen == [0.0, 0.5, 1.0]

    def test_no_speech_is_an_error(self, audio, stub_upload) -> None:
        stub_upload["responses"][audio] = {"words": [], "segments": []}

        with pytest.raises(TranscriptionError, match="no speech"):
            transcribe.transcribe_groq(audio, WhisperSettings(provider="groq"), api_key="k")


class TestGroqDispatch:
    def test_missing_key_fails_before_any_upload(
        self, audio, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(config, "get_secret", lambda *a, **k: None)
        monkeypatch.setattr(
            transcribe,
            "transcribe_groq",
            lambda *a, **k: pytest.fail("must not upload without a key"),
        )

        with pytest.raises(TranscriptionError, match="API key not set"):
            transcribe.transcribe(audio, WhisperSettings(provider="groq"))

    def test_key_is_forwarded_to_the_groq_path(
        self, audio, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(config, "get_secret", lambda *a, **k: "stored-key")
        captured: dict = {}

        def fake_groq(path, settings, *, api_key, **kwargs):
            captured["key"] = api_key
            return "transcript"

        monkeypatch.setattr(transcribe, "transcribe_groq", fake_groq)

        assert transcribe.transcribe(audio, WhisperSettings(provider="groq")) == "transcript"
        assert captured["key"] == "stored-key"


class TestFindWordRange:
    WORDS = [Word(text=str(i), start=float(i), end=i + 0.8) for i in range(10)]

    def test_covers_words_overlapping_the_span(self) -> None:
        assert transcribe._find_word_range(self.WORDS, 2.0, 4.5) == (2, 4)

    def test_span_with_no_words_falls_back_to_the_last_word(self) -> None:
        assert transcribe._find_word_range(self.WORDS, 100.0, 101.0) == (9, 9)


class TestWindowCount:
    ADVANCE = WINDOW_S - OVERLAP_S

    @pytest.mark.parametrize("duration", [0, 1, 60, WINDOW_S])
    def test_short_media_is_one_window(self, duration: float) -> None:
        assert estimate._window_count(duration) == 1

    def test_one_second_past_a_full_advance_adds_a_window(self) -> None:
        assert estimate._window_count(OVERLAP_S + self.ADVANCE) == 1
        assert estimate._window_count(OVERLAP_S + self.ADVANCE + 1) == 2

    def test_long_media_scales_linearly(self) -> None:
        assert estimate._window_count(3600) == 9


class TestLlmPricing:
    #: Standard-tier $/MTok (input, output) from platform.claude.com/docs/en/about-claude/pricing.
    OFFICIAL = {
        "claude-haiku-5-5": (0.10, 0.50),
        "claude-sonnet-5-5": (2.00, 10.00),
        "claude-sonnet-5": (2.00, 10.00),
        "claude-opus-5-5": (4.00, 20.00),
    }

    @staticmethod
    def _expected(inp: float, out: float) -> float:
        input_tokens = (
            estimate._SYSTEM_TOKENS
            + estimate._WORDS_PER_S * WINDOW_S * estimate._TOKENS_PER_WORD
        )
        return (input_tokens * inp + estimate._OUTPUT_TOKENS * out) / 1_000_000

    @pytest.mark.parametrize("model", sorted(OFFICIAL))
    def test_current_claude_models_match_the_official_rates(self, model: str) -> None:
        assert estimate._llm_per_window("anthropic", model) == pytest.approx(
            self._expected(*self.OFFICIAL[model])
        )

    def test_pricier_models_cost_more_per_window(self) -> None:
        haiku = estimate._llm_per_window("anthropic", "claude-haiku-5-5")
        sonnet = estimate._llm_per_window("anthropic", "claude-sonnet-5-5")
        opus = estimate._llm_per_window("anthropic", "claude-opus-5-5")

        assert haiku < sonnet < opus

    def test_haiku_120_minute_source_costs_about_two_cents(self) -> None:
        windows = estimate._window_count(7200)

        assert windows == 17
        assert 17 * estimate._llm_per_window("anthropic", "claude-haiku-5-5") == pytest.approx(
            0.0186, abs=0.0008
        )

    def test_window_tokens_match_what_groq_really_billed(self) -> None:
        # Measured: 6.0K input per window on a 10-minute talk, 8.9K on a fast 93-minute podcast.
        tokens = estimate._SYSTEM_TOKENS + estimate._WORDS_PER_S * WINDOW_S * estimate._TOKENS_PER_WORD
        assert 6000 <= tokens <= 9500
        assert 600 <= estimate._OUTPUT_TOKENS <= 900     # measured ~730 incl. reasoning

    def test_legacy_opus_keeps_its_old_price(self) -> None:
        assert estimate._llm_per_window("anthropic", "claude-opus-4") == pytest.approx(
            self._expected(15.00, 75.00)
        )

    @pytest.mark.parametrize(
        ("model", "rates"),
        [("openai/gpt-oss-120b", (0.15, 0.60)), ("openai/gpt-oss-20b", (0.075, 0.30))],
    )
    def test_groq_gpt_oss_models_match_the_published_rates(self, model, rates) -> None:
        assert estimate._llm_per_window("groq", model) == pytest.approx(self._expected(*rates))

    def test_local_models_are_free(self) -> None:
        assert estimate._llm_per_window("ollama", "llama3") == 0.0

    def test_unknown_provider_uses_a_conservative_fallback(self) -> None:
        assert estimate._llm_per_window("mystery", "m") == estimate._llm_per_window(
            "anthropic", "sonnet"
        )

    def test_legacy_sonnet_window_cost_matches_the_published_rates(self) -> None:
        input_tokens = (
            estimate._SYSTEM_TOKENS
            + estimate._WORDS_PER_S * WINDOW_S * estimate._TOKENS_PER_WORD
        )
        expected = (input_tokens * 3.00 + estimate._OUTPUT_TOKENS * 15.00) / 1_000_000

        assert estimate._llm_per_window("anthropic", "claude-sonnet-4") == pytest.approx(expected)


class TestWhisperCost:
    def test_local_transcription_is_free(self) -> None:
        assert estimate._whisper_cost("local", "whatever", 3600) == 0.0

    def test_groq_turbo_bills_per_hour_of_audio(self) -> None:
        assert estimate._whisper_cost("groq", "whisper-large-v3-turbo", 3600) == pytest.approx(0.04)

    def test_groq_large_v3_costs_more(self) -> None:
        assert estimate._whisper_cost("groq", "whisper-large-v3", 3600) == pytest.approx(0.111)

    def test_unknown_groq_model_uses_the_turbo_rate(self) -> None:
        assert estimate._whisper_cost("groq", "future-model", 3600) == pytest.approx(0.04)
