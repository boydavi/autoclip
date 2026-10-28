"""Highlight detection — transcript in, ranked clips out.

The transcript is cut into overlapping windows so that long videos fit inside
small context windows, each window is scored by the configured provider, and the
resulting candidates are merged, deduplicated, boundary-refined, and ranked.

Overlap matters: a clip straddling a window edge would otherwise be seen only in
halves by both windows and proposed by neither. The overlap costs tokens and
buys back the clips that live on the seams.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from ..db.models import Clip, new_id
from ..providers import ClipCandidate, DetectionConfig, LLMProvider, TranscriptWindow
from . import boundaries
from .prepare import Silence
from .transcript import Transcript

log = logging.getLogger(__name__)

#: Window length and overlap in seconds of speech.
WINDOW_S = 8 * 60
OVERLAP_S = 60

#: Two candidates covering this much of the same words are the same clip.
DEDUPE_IOU = 0.4

#: Minimum gap between the start times of any two kept clips (seconds).
#: Prevents the final selection from clustering in one hot stretch of a long video.
MIN_CLIP_GAP_S = 120.0

#: Hosted providers tolerate parallel windows; a local model is already
#: saturating the GPU, so extra concurrency only adds contention.
HOSTED_CONCURRENCY = 3
LOCAL_CONCURRENCY = 1

#: Rate-limit retry policy. On a rate-limit hit the window is retried up to
#: MAX_RETRIES times with exponential backoff (30 s, 60 s, 120 s …).
MAX_RETRIES = 3
RETRY_BASE_DELAY_S = 30.0

#: Substrings in an exception message that indicate a rate-limit response.
_RATE_LIMIT_MARKERS = (
    "rate limit", "ratelimit", "rate_limit",
    "429", "too many requests", "quota exceeded",
    "requests per minute", "rpm",
)


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _RATE_LIMIT_MARKERS)


class HighlightError(RuntimeError):
    """Highlight detection produced nothing usable."""


def build_windows(
    transcript: Transcript,
    *,
    window_s: float = WINDOW_S,
    overlap_s: float = OVERLAP_S,
) -> list[TranscriptWindow]:
    """Split a transcript into overlapping windows.

    Preconditions:
        overlap_s is less than window_s, otherwise windows would never advance.
    """
    if not transcript.words:
        return []
    if overlap_s >= window_s:
        raise ValueError("overlap_s must be smaller than window_s")

    windows: list[TranscriptWindow] = []
    total_words = len(transcript.words)
    cursor = 0

    while cursor < total_words:
        window_start_time = transcript.words[cursor].start
        window_end_time = window_start_time + window_s

        last = cursor
        while last + 1 < total_words and transcript.words[last + 1].end <= window_end_time:
            last += 1

        windows.append(_make_window(transcript, cursor, last))

        if last >= total_words - 1:
            break

        # Step forward by (window - overlap), measured in time then converted
        # back to a word index so overlap stays constant regardless of pace.
        next_time = transcript.words[last].end - overlap_s
        next_cursor = transcript.index_at_time(next_time)
        cursor = max(cursor + 1, next_cursor)

    return windows


def _make_window(transcript: Transcript, first: int, last: int) -> TranscriptWindow:
    start_s, end_s = transcript.time_range(first, last)
    words = transcript.slice(first, last)
    speakers = sorted({w.speaker for w in words if w.speaker})
    return TranscriptWindow(
        text=render_window_text(transcript, first, last),
        first_word=first,
        last_word=last,
        duration_s=end_s - start_s,
        speakers=speakers,
    )


def render_window_text(transcript: Transcript, first: int, last: int) -> str:
    """Render words as ``[index]word`` so the model can cite exact positions.

    Tagging every word is verbose, but it's the reason the model can't
    miscount — it never has to derive an index, only copy one.
    """
    parts: list[str] = []
    current_speaker: str | None = None

    for index in range(first, min(last + 1, len(transcript.words))):
        word = transcript.words[index]
        if word.speaker and word.speaker != current_speaker:
            current_speaker = word.speaker
            parts.append(f"\n<{word.speaker}>")
        parts.append(f"[{index}]{word.text}")

    return " ".join(parts).strip()


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


async def detect(
    transcript: Transcript,
    provider: LLMProvider,
    config: DetectionConfig,
    *,
    job_id: str,
    silences: list[Silence] | None = None,
    on_progress: Callable[[float], None] | None = None,
    concurrency: int | None = None,
) -> list[Clip]:
    """Run detection across the whole transcript and return ranked clips.

    ``concurrency`` overrides the built-in default (3 for hosted, 1 for local).
    Set it to 1 for free-tier keys that enforce a strict requests-per-minute
    limit — running windows serially avoids the rate-limit cascade that would
    otherwise kill a majority of windows and abort the job.
    """
    windows = build_windows(transcript)
    if not windows:
        raise HighlightError("The transcript is empty, so there is nothing to clip.")

    _concurrency = concurrency if concurrency is not None else (
        LOCAL_CONCURRENCY if provider.name == "ollama" else HOSTED_CONCURRENCY
    )
    log.info(
        "Detecting highlights across %d window(s) with %s (concurrency=%d).",
        len(windows), provider.name, _concurrency,
    )

    semaphore = asyncio.Semaphore(_concurrency)
    completed = 0
    failed_windows = 0
    lock = asyncio.Lock()

    async def run_window(window: TranscriptWindow) -> list[ClipCandidate]:
        nonlocal completed, failed_windows
        async with semaphore:
            failed = False
            candidates: list[ClipCandidate] = []
            last_exc: Exception | None = None

            for attempt in range(MAX_RETRIES):
                try:
                    result = await provider.detect_highlights(window, config)
                    candidates = result.clips
                    last_exc = None
                    break
                except Exception as exc:
                    if _is_rate_limit(exc) and attempt < MAX_RETRIES - 1:
                        delay = RETRY_BASE_DELAY_S * (2 ** attempt)
                        log.warning(
                            "Window %d-%d hit rate limit (attempt %d/%d); "
                            "retrying in %.0f s.",
                            window.first_word, window.last_word,
                            attempt + 1, MAX_RETRIES, delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    last_exc = exc
                    break

            if last_exc is not None:
                log.warning(
                    "Window %d-%d failed (%s); continuing with the remaining windows.",
                    window.first_word,
                    window.last_word,
                    last_exc,
                )
                failed = True

            async with lock:
                completed += 1
                if failed:
                    failed_windows += 1
                if on_progress:
                    on_progress(completed / len(windows))
            return candidates

    results = await asyncio.gather(*(run_window(w) for w in windows))
    candidates = [c for group in results for c in group]

    if failed_windows > 0:
        log.warning("%d of %d LLM windows failed.", failed_windows, len(windows))
        # Raise if a majority of windows failed — results would be too incomplete.
        threshold = max(1, len(windows) // 2)
        if failed_windows >= threshold:
            raise HighlightError(
                f"{failed_windows} of {len(windows)} LLM windows failed — results "
                "would be too incomplete to be useful. "
                "Check provider connectivity, API key, and try again."
            )

    if not candidates:
        raise HighlightError(
            "No clips were found. This can mean the video genuinely has no "
            "self-contained highlights, or that the model struggled with the "
            "transcript — try a larger model or a different provider."
        )

    log.info("Providers proposed %d raw candidates.", len(candidates))
    return build_clips(transcript, candidates, config, job_id=job_id, silences=silences or [])


def build_clips(
    transcript: Transcript,
    candidates: list[ClipCandidate],
    config: DetectionConfig,
    *,
    job_id: str,
    silences: list[Silence] | None = None,
) -> list[Clip]:
    """Deduplicate, refine, rank, and truncate raw candidates into clips."""
    deduped = dedupe(candidates)
    log.info("%d candidates remain after dedupe.", len(deduped))

    clips: list[Clip] = []
    for candidate in deduped:
        boundary = boundaries.refine(
            transcript,
            candidate.start_word_index,
            candidate.end_word_index,
            silences=silences or [],
            min_duration_s=config.min_duration_s,
            max_duration_s=config.max_duration_s,
        )
        if boundary is None:
            log.debug(
                "Dropped candidate %d-%d: no valid boundary within the duration range.",
                candidate.start_word_index,
                candidate.end_word_index,
            )
            continue

        clips.append(
            Clip(
                id=new_id(),
                job_id=job_id,
                start_s=boundary.start_s,
                end_s=boundary.end_s,
                start_word=boundary.start_word,
                end_word=boundary.end_word,
                title=candidate.title.strip()
                or transcript.text_between(
                    boundary.start_word, min(boundary.start_word + 8, boundary.end_word)
                ),
                hook=candidate.hook.strip(),
                score=candidate.score,
                reason=candidate.reason.strip(),
            )
        )

    # Refinement can move edges enough that two survivors now overlap.
    clips = _dedupe_clips(clips)
    clips.sort(key=lambda c: c.score, reverse=True)
    clips = _spread_clips(clips, max_clips=config.max_clips)

    for rank, clip in enumerate(clips, start=1):
        clip.rank = rank

    log.info("Produced %d final clips.", len(clips))
    return clips


def dedupe(
    candidates: list[ClipCandidate], *, iou_threshold: float = DEDUPE_IOU
) -> list[ClipCandidate]:
    """Greedily keep the highest-scoring candidate from each overlapping cluster.

    Overlapping windows mean the same moment is often proposed two or three
    times, sometimes with slightly different edges. Highest score wins.
    """
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    kept: list[ClipCandidate] = []

    for candidate in ordered:
        if any(
            _iou(
                candidate.start_word_index,
                candidate.end_word_index,
                existing.start_word_index,
                existing.end_word_index,
            )
            > iou_threshold
            for existing in kept
        ):
            continue
        kept.append(candidate)

    return kept


def _spread_clips(
    clips: list[Clip],
    *,
    max_clips: int,
    min_gap_s: float = MIN_CLIP_GAP_S,
) -> list[Clip]:
    """Pick up to ``max_clips`` from a score-sorted list with temporal diversity.

    Greedily accepts each clip (highest score first) unless its start time is
    within ``min_gap_s`` of an already-accepted clip.  Once the budget is
    exhausted, remaining candidates are appended without the gap constraint so
    we never return fewer clips than the video warrants.
    """
    kept: list[Clip] = []
    deferred: list[Clip] = []

    for clip in clips:
        too_close = any(abs(clip.start_s - k.start_s) < min_gap_s for k in kept)
        if not too_close and len(kept) < max_clips:
            kept.append(clip)
        else:
            deferred.append(clip)

    # If the gap constraint left us short, fill from deferred (still score-ordered).
    for clip in deferred:
        if len(kept) >= max_clips:
            break
        kept.append(clip)

    # Restore chronological order for the rank assignment that follows.
    kept.sort(key=lambda c: c.start_s)
    return kept


def _dedupe_clips(clips: list[Clip], *, iou_threshold: float = DEDUPE_IOU) -> list[Clip]:
    ordered = sorted(clips, key=lambda c: c.score, reverse=True)
    kept: list[Clip] = []
    for clip in ordered:
        if any(
            _iou(clip.start_word, clip.end_word, other.start_word, other.end_word) > iou_threshold
            for other in kept
        ):
            continue
        kept.append(clip)
    return kept


def _iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Intersection over union of two inclusive index ranges."""
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start) + 1)
    if intersection == 0:
        return 0.0
    union = (a_end - a_start + 1) + (b_end - b_start + 1) - intersection
    return intersection / union if union else 0.0
