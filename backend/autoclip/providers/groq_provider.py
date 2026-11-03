"""Groq provider — clip selection on Groq's hosted open models.

Groq speaks the OpenAI Chat Completions dialect, so this reuses
:class:`OpenAIProvider` with Groq's endpoint and defaults baked in. It exists as
its own provider so one Groq key (``autoclip config set-secret groq``) serves
both clip selection here and Whisper transcription.
"""

from __future__ import annotations

from .openai_provider import OpenAIProvider

DEFAULT_MODEL = "openai/gpt-oss-120b"
BASE_URL = "https://api.groq.com/openai/v1"

#: Self-serve production models. The Llama models are Enterprise-only and
#: Qwen is a preview that Groq may withdraw, so neither is suggested.
SUGGESTED_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]


class GroqProvider(OpenAIProvider):
    name = "groq"
    requires_key = True
    suggested_models = SUGGESTED_MODELS

    def __init__(self, model: str = "", *, api_key: str | None = None, base_url: str | None = None):
        super().__init__(model or DEFAULT_MODEL, api_key=api_key, base_url=base_url or BASE_URL)
