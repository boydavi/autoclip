"""Per-job choice of the LLM and the transcription provider (the saved settings may say otherwise)."""

import pytest
from pydantic import ValidationError

from autoclip.api import jobs as jobs_api
from autoclip.api.schemas import JobSettingsIn
from autoclip.config import Settings


def _saved() -> Settings:
    s = Settings()
    s.active_provider = "anthropic"
    s.whisper.provider = "local"
    return s


def test_a_job_can_pick_groq_for_both_the_llm_and_the_transcription():
    merged = jobs_api._apply_overrides(_saved(), JobSettingsIn(provider="groq", whisper_provider="groq"))
    assert merged.active_provider == "groq" and merged.whisper.provider == "groq"


def test_omitted_providers_keep_the_saved_settings_and_the_saved_object_is_untouched():
    saved = _saved()
    merged = jobs_api._apply_overrides(saved, JobSettingsIn())
    assert merged.active_provider == "anthropic" and merged.whisper.provider == "local"
    jobs_api._apply_overrides(saved, JobSettingsIn(provider="groq", whisper_provider="groq"))
    assert saved.active_provider == "anthropic" and saved.whisper.provider == "local"


def test_an_unknown_transcription_provider_is_rejected_instead_of_silently_ignored():
    with pytest.raises(ValidationError):
        JobSettingsIn(whisper_provider="mystery")
