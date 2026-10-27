"""Google Gemini provider."""

from __future__ import annotations

import asyncio
import logging

from .base import DetectionConfig, LLMProvider, ProviderError, ProviderStatus

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.0-flash"

SUGGESTED_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-pro",
    "gemini-1.5-flash",
]


class GeminiProvider(LLMProvider):
    name = "gemini"
    requires_key = True

    def __init__(self, model: str = "", *, api_key: str | None = None, base_url: str | None = None):
        super().__init__(model or DEFAULT_MODEL, api_key=api_key, base_url=base_url)

    def _client(self):
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - google-genai is a core dep
            raise ProviderError(
                "The google-genai package is not installed.", provider=self.name
            ) from exc

        if not self.api_key:
            raise ProviderError(
                "No Gemini API key is set.",
                provider=self.name,
                hint="Add one with `autoclip config set-secret gemini`.",
            )
        return genai.Client(api_key=self.api_key)

    async def _complete(self, system: str, user: str, config: DetectionConfig) -> str:
        from google.genai import types

        client = self._client()
        request_config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=config.temperature,
            # Gemini enforces JSON natively, which removes most parse failures.
            response_mime_type="application/json",
            max_output_tokens=8192,
        )

        for attempt in range(TRANSIENT_ATTEMPTS):
            try:
                # The SDK's async surface has moved between releases; running the
                # sync call in a thread works across all of them and keeps the event
                # loop free.
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=self.model,
                    contents=user,
                    config=request_config,
                )
            except Exception as exc:
                if attempt + 1 < TRANSIENT_ATTEMPTS and _is_transient(exc):
                    wait = TRANSIENT_BACKOFF_S * (2**attempt)
                    log.warning("Gemini transient error (%s); retry %d in %.0fs.",
                                str(exc)[:80], attempt + 1, wait)
                    await asyncio.sleep(wait)
                    continue
                raise _translate(exc, self.name, self.model) from exc
            return getattr(response, "text", "") or ""
        raise AssertionError("unreachable")

    async def health_check(self) -> ProviderStatus:
        if not self.api_key:
            return ProviderStatus(
                name=self.name, available=False, detail="No API key set", models=SUGGESTED_MODELS
            )
        try:
            client = self._client()
            models = await asyncio.to_thread(lambda: list(client.models.list()))
            names = sorted(
                m.name.removeprefix("models/") for m in models if getattr(m, "name", None)
            )[:50]
        except Exception as exc:
            return ProviderStatus(name=self.name, available=False, detail=str(exc)[:200])
        return ProviderStatus(
            name=self.name, available=True, detail=self.model, models=names or SUGGESTED_MODELS
        )


#: Rate limits and overload are routine on hosted Flash models; one hiccup must
#: not cost a whole transcript window.
TRANSIENT_ATTEMPTS = 5
TRANSIENT_BACKOFF_S = 5.0
_TRANSIENT_MARKERS = ("429", "503", "500", "unavailable", "resource_exhausted", "overloaded",
                      "deadline", "timed out", "timeout", "connection reset", "internal")


def _is_transient(exc: Exception) -> bool:
    lowered = str(exc).lower()
    if "api key" in lowered or "401" in lowered or "403" in lowered or "404" in lowered:
        return False
    return any(marker in lowered for marker in _TRANSIENT_MARKERS)


def _translate(exc: Exception, provider: str, model: str) -> ProviderError:
    message = str(exc)
    lowered = message.lower()

    if "api key" in lowered or "401" in lowered or "permission" in lowered:
        return ProviderError(
            "Google rejected the API key.",
            provider=provider,
            hint="Re-add it with `autoclip config set-secret gemini`.",
        )
    if "429" in lowered or "quota" in lowered or "resource_exhausted" in lowered:
        return ProviderError(
            "Gemini rate limit or quota reached.",
            provider=provider,
            hint="Wait and retry, or switch providers.",
        )
    if "404" in lowered or "not found" in lowered:
        return ProviderError(
            f"Gemini does not recognise the model '{model}'.",
            provider=provider,
            hint=f"Try one of: {', '.join(SUGGESTED_MODELS)}",
        )
    if "safety" in lowered or "blocked" in lowered:
        return ProviderError(
            "Gemini blocked the response under its safety filters.",
            provider=provider,
            hint="This can happen on transcripts with strong language. Try another provider.",
        )
    return ProviderError(f"Gemini request failed: {message}", provider=provider)
