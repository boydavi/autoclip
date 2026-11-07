"""LLM provider abstraction.

Highlight detection is the one place AutoClip asks a language model to make a
judgement call, so the interface is deliberately narrow: a window of transcript
goes in, a validated list of clip candidates comes out.

Two design choices carry most of the weight:

**Word indices, not seconds.** The model returns ``start_word_index`` /
``end_word_index``. Timing is then looked up from measured word timestamps, so a
model that is bad at arithmetic — which they all are — cannot produce a clip
that starts at the wrong moment.

**Validate, then retry with the error.** Small local models produce malformed
JSON often enough that a single retry carrying the actual validation message
turns most failures into successes. That loop lives here so every provider
inherits it.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from ..usage import Usage

log = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


class ProviderError(RuntimeError):
    """A provider could not produce a usable response."""

    def __init__(self, message: str, *, provider: str = "", hint: str = "") -> None:
        super().__init__(message)
        self.provider = provider
        self.hint = hint

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base}\n\n{self.hint}" if self.hint else base


# --------------------------------------------------------------------------
# Contracts
# --------------------------------------------------------------------------


#: Music moods; match the subfolders of the music library (energetic/, calm/, dramatic/).
MOODS = ("energetic", "calm", "dramatic")


class ClipCandidate(BaseModel):
    """One proposed clip, as returned by the model."""

    start_word_index: int = Field(ge=0)
    end_word_index: int = Field(ge=0)
    title: str = ""
    hook: str = ""
    score: int = Field(default=50, ge=0, le=100)
    reason: str = ""
    #: Post text for the platform caption box, and its hashtags (``#tag`` each).
    description: str = ""
    hashtags: list[str] = Field(default_factory=list)
    #: Up to three viewer-facing titles for the platform's title box (``title`` is a folder label).
    post_titles: list[str] = Field(default_factory=list)
    #: Emotional register of what is said; picks the music mood downstream.
    mood: str = ""

    @field_validator("mood", mode="before")
    @classmethod
    def _coerce_mood(cls, value: Any) -> str:
        """Only the three moods the music library is organised by; anything else is unknown."""
        text = "" if value is None else str(value).strip().lower()
        return text if text in MOODS else ""

    @field_validator("title", "hook", "reason", "description", mode="before")
    @classmethod
    def _coerce_to_string(cls, value: Any) -> str:
        # Models occasionally return null or a number where text was asked for.
        return "" if value is None else str(value)

    @field_validator("post_titles", mode="before")
    @classmethod
    def _coerce_post_titles(cls, value: Any) -> list[str]:
        """Accept a list or a numbered/newline separated string; keep up to three clean, distinct titles."""
        if value is None:
            return []
        items = value.splitlines() if isinstance(value, str) else list(value)
        titles: list[str] = []
        for item in items:
            text = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", str(item)).strip().strip('"\u201c\u201d').strip()
            if text and len(text) <= 100 and text.lower() not in {t.lower() for t in titles}:
                titles.append(text)
        return titles[:3]

    @field_validator("hashtags", mode="before")
    @classmethod
    def _coerce_hashtags(cls, value: Any) -> list[str]:
        """Accept a list or a space/comma separated string; normalise to ``#tag``."""
        if value is None:
            return []
        items = re.split(r"[\s,]+", value) if isinstance(value, str) else list(value)
        tags: list[str] = []
        for item in items:
            word = re.sub(r"[^\w]", "", str(item))
            tag = f"#{word}" if word else ""
            if tag and tag.lower() not in {t.lower() for t in tags}:
                tags.append(tag)
        return tags[:8]

    @field_validator("score", mode="before")
    @classmethod
    def _coerce_score(cls, value: Any) -> int:
        """Accept floats and 0-1 fractions, which models emit despite the schema."""
        if value is None:
            return 50
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 50
        if 0.0 < number <= 1.0:
            number *= 100
        return max(0, min(100, round(number)))


class ClipCandidates(BaseModel):
    clips: list[ClipCandidate] = Field(default_factory=list)


@dataclass
class TranscriptWindow:
    """A slice of transcript presented to the model.

    ``first_word`` lets a provider work on a window while still reporting
    indices in the full transcript's coordinate space.
    """

    text: str
    first_word: int
    last_word: int
    duration_s: float = 0.0
    speakers: list[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return self.last_word - self.first_word + 1


@dataclass
class DetectionConfig:
    min_duration_s: float = 20.0
    max_duration_s: float = 90.0
    max_clips: int = 10
    language: str = ""
    #: Prompt file stem in ``autoclip/prompts/``. Versioned so contributors can
    #: iterate on prompts without touching code.
    prompt_version: str = "highlight_v1"
    temperature: float = 0.3


@dataclass
class ProviderStatus:
    name: str
    available: bool
    detail: str = ""
    models: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Base provider
# --------------------------------------------------------------------------


class LLMProvider(ABC):
    """Base class for highlight-detection providers."""

    #: Stable identifier, matching the key used in settings.
    name: str = ""
    #: Whether the provider needs an API key.
    requires_key: bool = True

    def __init__(self, model: str, *, api_key: str | None = None, base_url: str | None = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    @property
    def usage(self) -> Usage:
        """Tokens spent by this provider instance so far (providers that report usage fill it in)."""
        if not hasattr(self, "_usage"):
            self._usage = Usage()
        return self._usage

    # -- to implement ------------------------------------------------------

    @abstractmethod
    async def _complete(self, system: str, user: str, config: DetectionConfig) -> str:
        """Send one prompt and return the raw response text."""

    @abstractmethod
    async def health_check(self) -> ProviderStatus:
        """Report whether this provider is usable right now."""

    # -- shared behaviour --------------------------------------------------

    async def complete_json(self, system: str, user: str, config: DetectionConfig) -> dict[str, Any]:
        """One prompt, one JSON object back (fences and chatter tolerated)."""
        return extract_json_object(await self._complete(system, user, config))

    async def detect_highlights(
        self, window: TranscriptWindow, config: DetectionConfig
    ) -> ClipCandidates:
        """Ask the model for clip candidates in ``window``, validating the reply.

        One retry is attempted on a schema violation, feeding the validation
        error back so the model can correct itself.
        """
        system = load_prompt(config.prompt_version)
        user = render_window_prompt(window, config)

        raw = await self._complete(system, user, config)
        try:
            return self._parse(raw, window)
        except (ValidationError, ValueError) as first_error:
            log.warning("%s returned invalid JSON; retrying with feedback.", self.name)
            repair = (
                f"{user}\n\n"
                "Your previous response did not match the required schema.\n"
                f"Validation error:\n{first_error}\n\n"
                "Respond again with ONLY the corrected JSON object. No prose, no "
                "markdown fences."
            )
            raw = await self._complete(system, repair, config)
            try:
                return self._parse(raw, window)
            except (ValidationError, ValueError) as second_error:
                raise ProviderError(
                    f"{self.name} returned malformed clip data twice.",
                    provider=self.name,
                    hint=(
                        f"Last validation error: {second_error}\n\n"
                        "Smaller local models struggle with strict JSON. Try a larger "
                        "model, or switch to a hosted provider."
                    ),
                ) from second_error

    def _parse(self, raw: str, window: TranscriptWindow) -> ClipCandidates:
        """Validate a raw response and clamp indices into the window."""
        payload = extract_json_object(raw)
        candidates = ClipCandidates.model_validate(payload)

        cleaned: list[ClipCandidate] = []
        for candidate in candidates.clips:
            start = max(window.first_word, min(candidate.start_word_index, window.last_word))
            end = max(window.first_word, min(candidate.end_word_index, window.last_word))
            if end <= start:
                # A zero-or-negative-length clip is a model slip, not a candidate.
                continue
            candidate.start_word_index = start
            candidate.end_word_index = end
            cleaned.append(candidate)

        return ClipCandidates(clips=cleaned)


# --------------------------------------------------------------------------
# Prompt handling
# --------------------------------------------------------------------------


def load_prompt(version: str) -> str:
    """Read a versioned prompt file from ``autoclip/prompts/``."""
    path = PROMPT_DIR / f"{version}.txt"
    if not path.exists():
        raise ProviderError(
            f"Prompt '{version}' not found at {path}.",
            hint="Prompt files live in backend/autoclip/prompts/ as versioned .txt files.",
        )
    return path.read_text(encoding="utf-8") + POST_CAPTION_PROMPT


#: Appended to every prompt style so the post caption is requested in one place.
POST_CAPTION_PROMPT = """

## Post caption

Every clip object also carries four extra fields. `post_titles`, `description` and
`hashtags` are written for the boxes a creator pastes into when posting:

`description` — one or two sentences, under 150 characters, in the speaker's
language. Lead with the hook, no spoilers of the payoff, no emoji spam.

`hashtags` — three to five, as a JSON array of strings like ["#startups", "#growth"].
Specific to the clip's topic; no generic tags such as #fyp.

`post_titles` — a JSON array of exactly three alternative titles for the platform's title
box. Unlike `title` (a folder label), these are written to make a viewer stop scrolling.
Each is under 70 characters, in the speaker's language, and takes a different angle: a
short statement lifted from what is said, a direct challenge to the viewer ("Stop waiting
for permission"), a pain or fear the viewer recognises, a contrast question ("Do you have
a people problem or a systems problem?"), or a setup left unfinished ("Before you cut
your price, read this:"). They must be true to the clip: use only what the speaker says,
invent no numbers, names or claims, and promise nothing the clip does not deliver. No
emoji, no hashtags.

`mood` — the emotional register of what is SAID, which chooses the background music.
Exactly one of "energetic", "calm", "dramatic". Judge the message, not the speaking
speed, and not how upbeat the speaker sounds. Decide in this order:
1. "dramatic" - only a genuine emotional or high-stakes moment: a real turning point, a
   painful or hard truth, a confession, a dangerous or life-changing event, tension about
   to break. The listener should feel weight. Explaining how something works is never
   dramatic, even when the claim is surprising or alarming (a science, biology or
   psychology explanation is not a drama), and neither is a rhetorical question or a
   mild example. The SUBJECT is not the tone: a clip about trauma, fear, loss, money or lack
   is not dramatic just because of its topic. Dramatic needs the moment itself to carry
   weight: the speaker's own high stakes, a story at its turning point, or a hard truth
   delivered with weight. A calm explanation of why a heavy topic works the way it does
   is calm.
2. "energetic" - the clip pushes the listener to act (a call to action, motivation, a
   challenge), rallies or empowers them (an emphatic declaration such as "you are the
   creator of your life", an invitation to change), celebrates a win or a success story,
   lands a joke, or ends on a quick punchy payoff. An upbeat explanation of an idea, a
   definition, a surprising fact, a study result or a demonstration, or advice delivered
   in a warm tone is NOT energetic.
3. "calm" - everything else: explanation, teaching, science or how-things-work content,
   reflection, storytelling without a punch, gentle advice, reassurance, a soft intimate
   message. When torn between calm and energetic, ask whether the speaker is rallying or
   empowering the listener (energetic) or informing and soothing them (calm).
Most clips are calm or energetic; dramatic is rare. Ask yourself: is the speaker moving
the listener's emotions (dramatic or energetic), or informing and soothing them (calm)?
"""


#: The mood rubric alone (everything from the ``mood`` field on), shared with the focused mood pass.
MOOD_RULES = POST_CAPTION_PROMPT[POST_CAPTION_PROMPT.index("`mood`"):].strip()


def render_window_prompt(window: TranscriptWindow, config: DetectionConfig) -> str:
    """Build the user message for one transcript window."""
    speaker_note = ""
    if window.speakers:
        speaker_note = (
            f"\nThis section has {len(window.speakers)} distinct speakers "
            f"({', '.join(window.speakers)}); speaker labels are shown inline.\n"
        )

    return (
        f"Transcript section, words {window.first_word} to {window.last_word}.\n"
        f"Each word is tagged with its index as [index]word.\n"
        f"{speaker_note}\n"
        f"Clip length must be between {config.min_duration_s:.0f} and "
        f"{config.max_duration_s:.0f} seconds.\n"
        f"Return at most {config.max_clips} clips.\n\n"
        f"---\n{window.text}\n---\n\n"
        "Respond with ONLY a JSON object matching the schema. No prose, no markdown fences."
    )


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(raw: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response.

    Handles the three things models do despite being told not to: wrap the JSON
    in markdown fences, prepend an explanatory sentence, and append a trailing
    note. Falls back to brace matching so a stray character outside the object
    doesn't cost a retry round-trip.
    """
    text = raw.strip()
    if not text:
        raise ValueError("The provider returned an empty response.")

    if match := _JSON_FENCE.search(text):
        text = match.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = _json_by_brace_matching(text)

    if isinstance(parsed, list):
        # Some models skip the wrapper object and return the array directly.
        return {"clips": parsed}
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}.")
    return parsed


def _json_by_brace_matching(text: str) -> Any:
    """Extract the first balanced ``{...}`` or ``[...]`` region and parse it."""
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            char = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])

    raise ValueError("No JSON object found in the response.")
