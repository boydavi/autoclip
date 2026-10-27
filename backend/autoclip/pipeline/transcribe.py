"""Transcribe stage — audio to word-level timestamps.

faster-whisper does the transcription; WhisperX (optional) adds speaker labels.
Word-level timing is non-negotiable: clip boundaries, caption animation, and
trim-handle snapping all derive from it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from ..config import WhisperSettings
from ..system import GPUInfo, report
from .transcript import Segment, Transcript, Word

log = logging.getLogger(__name__)


class TranscriptionError(RuntimeError):
    """Transcription could not be completed."""


def resolve_compute(settings: WhisperSettings, gpu: GPUInfo | None = None) -> tuple[str, str]:
    """Return the ``(device, compute_type)`` to run Whisper with.

    An explicit ``settings.compute_type`` wins, but only if the device actually
    supports it. On CPU with no explicit override, ``int8`` is chosen: it is
    3-5x faster than float32 at comparable accuracy on modern CPUs.
    """
    gpu = gpu if gpu is not None else report().gpu
    device = gpu.device
    # On CPU, int8 is significantly faster and well-supported by CTranslate2.
    automatic = "int8" if device == "cpu" else gpu.compute_type

    override = settings.compute_type
    if not override:
        return device, automatic

    supported = gpu.supported_compute_types
    if supported and override not in supported:
        log.warning(
            "whisper.compute_type is set to %r, which %s does not support "
            "(supported: %s). Using %r instead.",
            override,
            device,
            ", ".join(sorted(supported)),
            automatic,
        )
        return device, automatic

    return device, override


def parse_youtube_captions(json3_path: Path) -> Transcript | None:
    """Parse a YouTube json3 automatic-caption file into a Transcript.

    Word timestamps are approximated by distributing each segment's duration
    linearly across its words — precise enough for clip boundaries without
    running Whisper.  Returns ``None`` if the file is missing or malformed.
    """
    try:
        data = json.loads(json3_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read YouTube captions %s: %s", json3_path, exc)
        return None

    events = data.get("events", [])
    words: list[Word] = []
    segments: list[Segment] = []

    for event in events:
        segs = event.get("segs")
        if not segs:
            continue  # structural / window-init events have no text

        start_s = event.get("tStartMs", 0) / 1000.0
        dur_s = event.get("dDurationMs", 0) / 1000.0
        end_s = start_s + dur_s

        text = "".join(s.get("utf8", "") for s in segs).strip()
        if not text:
            continue

        raw_words = text.split()
        if not raw_words:
            continue

        first_word = len(words)
        word_dur = dur_s / len(raw_words) if raw_words else 0.0
        for i, w in enumerate(raw_words):
            words.append(Word(
                text=w,
                start=start_s + i * word_dur,
                end=start_s + (i + 1) * word_dur,
            ))
        last_word = len(words) - 1
        segments.append(Segment(
            text=text,
            start=start_s,
            end=end_s,
            first_word=first_word,
            last_word=last_word,
        ))

    if not words:
        log.warning("YouTube captions %s contained no usable words.", json3_path)
        return None

    log.info("Parsed %d words from YouTube captions %s.", len(words), json3_path.name)
    return Transcript(
        words=words,
        segments=segments,
        language="en",
        model="youtube-captions",
        source="youtube",
    )


def _model_load_error(exc: Exception, device: str, compute_type: str) -> TranscriptionError:
    """Turn a Whisper model-load failure into something actionable.

    The two causes look identical from the outside and need opposite fixes, so
    they're distinguished rather than lumped under one guess.
    """
    message = str(exc)
    lowered = message.lower()

    if "compute type" in lowered or "not support" in lowered:
        supported = report().gpu.supported_compute_types
        options = ", ".join(sorted(supported)) if supported else "unknown"
        return TranscriptionError(
            f"This device does not support the '{compute_type}' compute type.\n"
            f"Supported on {device}: {options}.\n\n"
            "Clear whisper.compute_type in config.json to let AutoClip choose, "
            "or set it to one of the supported values."
        )

    if device == "cuda" and ("cudnn" in lowered or "library" in lowered or "dll" in lowered):
        return TranscriptionError(
            f"Could not load Whisper on the GPU: {message}\n\n"
            "This is a missing CUDA runtime library. Install it with "
            "`uv pip install 'autoclip[gpu]'`, then retry — the job resumes "
            "from this stage."
        )

    return TranscriptionError(f"Could not load the Whisper model: {message}")


# ---------------------------------------------------------------------------
# Groq cloud transcription
# ---------------------------------------------------------------------------

_GROQ_MAX_BYTES = 24 * 1024 * 1024   # 24 MB — safe margin under Groq's 25 MB limit
_GROQ_CHUNK_S = 1200                  # 20-minute chunks for oversized files
_GROQ_OVERLAP_S = 5                   # overlap between chunks to avoid cut words


def _compress_for_groq(audio: Path, workdir: Path) -> Path:
    """Re-encode audio to 32 kbps MP3 mono 16 kHz for Groq upload."""
    import subprocess
    out = workdir / "_groq_audio.mp3"
    if out.exists():
        return out
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio), "-ar", "16000", "-ac", "1",
         "-b:a", "32k", "-f", "mp3", str(out)],
        check=True, capture_output=True,
    )
    return out


def _split_audio(audio: Path, workdir: Path) -> list[tuple[Path, float]]:
    """Return (chunk_path, offset_s) pairs. Single pair when file fits in one upload."""
    if audio.stat().st_size <= _GROQ_MAX_BYTES:
        return [(audio, 0.0)]

    import subprocess
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(audio)],
        capture_output=True, text=True, check=True,
    )
    total_s = float(probe.stdout.strip() or 0)

    chunks: list[tuple[Path, float]] = []
    start = 0.0
    idx = 0
    while start < total_s:
        chunk_path = workdir / f"_groq_chunk_{idx:02d}.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio), "-ss", str(start),
             "-t", str(_GROQ_CHUNK_S + _GROQ_OVERLAP_S),
             "-ar", "16000", "-ac", "1", "-b:a", "32k", "-f", "mp3", str(chunk_path)],
            check=True, capture_output=True,
        )
        chunks.append((chunk_path, start))
        start += _GROQ_CHUNK_S
        idx += 1
    return chunks


def _call_groq_api(file_path: Path, api_key: str, language: str, model: str) -> dict:
    """POST one file to Groq's transcription endpoint; returns the verbose_json dict."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise TranscriptionError(
            "openai package is required for Groq transcription: pip install openai"
        ) from exc

    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
    with open(file_path, "rb") as fh:
        resp = client.audio.transcriptions.create(
            file=fh,
            model=model,
            language=language or None,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )
    return resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)


def _find_word_range(words: list[Word], start_s: float, end_s: float) -> tuple[int, int]:
    first: int | None = None
    last: int | None = None
    for i, w in enumerate(words):
        if w.end > start_s and w.start < end_s + 0.05:
            if first is None:
                first = i
            last = i
    if first is None:
        idx = max(0, len(words) - 1)
        return idx, idx
    return first, last  # type: ignore[return-value]


def transcribe_groq(
    audio: Path,
    settings: WhisperSettings,
    *,
    api_key: str,
    duration_s: float | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> Transcript:
    """Transcribe via Groq's hosted Whisper (~30-60 s for a 90-min file).

    Compresses the audio to 32 kbps MP3 first so it fits Groq's 25 MB limit.
    Files longer than ~2 hrs are split into 20-min chunks automatically.
    """
    workdir = audio.parent
    model = settings.groq_model
    language = settings.language

    log.info("Groq transcription: model=%s source=%s", model, audio.name)
    compressed = _compress_for_groq(audio, workdir)
    chunks = _split_audio(compressed, workdir)
    log.info("Groq upload: %d chunk(s), total %.1f MB",
             len(chunks), compressed.stat().st_size / 1024 / 1024)

    transcript = Transcript(language=language or "en", model=f"groq/{model}", source="groq")

    for i, (chunk_path, offset_s) in enumerate(chunks):
        if on_progress and len(chunks) > 1:
            on_progress(i / len(chunks))
        log.info("Groq chunk %d/%d (offset %.0f s)", i + 1, len(chunks), offset_s)
        data = _call_groq_api(chunk_path, api_key, language, model)

        for w in data.get("words") or []:
            w_start = float(w.get("start", 0))
            # Drop overlap words from non-first chunks to avoid duplicates.
            if offset_s > 0 and w_start < _GROQ_OVERLAP_S:
                continue
            transcript.words.append(Word(
                text=(w.get("word") or "").strip(),
                start=offset_s + w_start,
                end=offset_s + float(w.get("end", 0)),
            ))

        for seg in data.get("segments") or []:
            seg_start = float(seg.get("start", 0))
            if offset_s > 0 and seg_start < _GROQ_OVERLAP_S:
                continue
            abs_start = offset_s + seg_start
            abs_end = offset_s + float(seg.get("end", 0))
            first_w, last_w = _find_word_range(transcript.words, abs_start, abs_end)
            transcript.segments.append(Segment(
                text=(seg.get("text") or "").strip(),
                start=abs_start,
                end=abs_end,
                first_word=first_w,
                last_word=last_w,
            ))

    if not transcript.words:
        raise TranscriptionError(
            "Groq returned no speech. Verify the audio contains speech "
            "and that the Groq API key is valid (autoclip config set-secret groq)."
        )

    if on_progress:
        on_progress(1.0)
    log.info("Groq: %d words, %d segments.", len(transcript.words), len(transcript.segments))
    return transcript


def transcribe(
    audio: Path,
    settings: WhisperSettings | None = None,
    *,
    duration_s: float | None = None,
    on_progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Transcript:
    """Transcribe an audio file into a :class:`Transcript` with word timings.

    Preconditions:
        audio is a decodable audio file, ideally 16 kHz mono as produced by
        :func:`autoclip.pipeline.prepare.extract_audio`.
    """
    settings = settings or WhisperSettings()

    if settings.provider == "groq":
        from ..config import get_secret
        api_key = get_secret("groq")
        if not api_key:
            raise TranscriptionError(
                "Groq API key not set. Add it with: autoclip config set-secret groq"
            )
        return transcribe_groq(
            audio, settings,
            api_key=api_key,
            duration_s=duration_s,
            on_progress=on_progress,
        )

    # Must happen before faster-whisper pulls in CTranslate2, which resolves its
    # CUDA dependencies at import.
    from ..cuda import ensure_cuda_libraries

    ensure_cuda_libraries()

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - faster-whisper is a core dep
        raise TranscriptionError(
            "faster-whisper is not installed. Run `autoclip doctor` for details."
        ) from exc

    device, compute_type = resolve_compute(settings)
    log.info(
        "Transcribing with model=%s device=%s compute_type=%s",
        settings.model,
        device,
        compute_type,
    )

    try:
        model = WhisperModel(settings.model, device=device, compute_type=compute_type)
    except Exception as exc:
        raise _model_load_error(exc, device, compute_type) from exc

    segments_iter, info = model.transcribe(
        str(audio),
        language=settings.language or None,
        word_timestamps=True,
        vad_filter=True,
        # Trims long silences before decoding, which both speeds things up and
        # stops Whisper hallucinating text into empty audio.
        vad_parameters={"min_silence_duration_ms": 500},
    )

    total = duration_s or getattr(info, "duration", 0.0) or 0.0
    transcript = Transcript(
        language=getattr(info, "language", "") or settings.language,
        model=settings.model,
        source="whisper",
    )

    for segment in segments_iter:
        if cancelled is not None and cancelled():
            raise TranscriptionError("Transcription cancelled.")

        first_word = len(transcript.words)
        for word in getattr(segment, "words", None) or []:
            text = (word.word or "").strip()
            if not text:
                continue
            transcript.words.append(Word(text=text, start=float(word.start), end=float(word.end)))

        # A segment with no word timings still carries text worth keeping for
        # display, but it can't contribute to word-indexed clip boundaries.
        last_word = max(first_word, len(transcript.words) - 1)
        transcript.segments.append(
            Segment(
                text=(segment.text or "").strip(),
                start=float(segment.start),
                end=float(segment.end),
                first_word=first_word,
                last_word=last_word,
            )
        )

        if on_progress and total:
            on_progress(min(1.0, float(segment.end) / total))

    if not transcript.words:
        raise TranscriptionError(
            "No speech was detected in this audio. If the file really does contain "
            "speech, try a larger Whisper model or check that the correct audio track "
            "was extracted."
        )

    if on_progress:
        on_progress(1.0)

    log.info(
        "Transcribed %d words in %d segments.", len(transcript.words), len(transcript.segments)
    )
    return transcript


# --------------------------------------------------------------------------
# Diarization
# --------------------------------------------------------------------------


def diarize(
    audio: Path,
    transcript: Transcript,
    *,
    hf_token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> Transcript:
    """Label each word with a speaker, in place.

    Uses pyannote via WhisperX. Returns the transcript unchanged (with a logged
    warning) if diarization isn't available — speaker labels improve reframing
    but are never required for the pipeline to complete.

    Preconditions:
        transcript already has word-level timings.
    """
    if not transcript.words:
        return transcript

    pipeline_cls = _load_diarization_pipeline()
    if pipeline_cls is None:
        log.warning(
            "WhisperX is not installed; skipping diarization. "
            "Install it with `uv pip install 'autoclip[diarization]'`."
        )
        return transcript

    if not hf_token:
        log.warning(
            "Diarization needs a HuggingFace token. Set one with "
            "`autoclip config set-secret huggingface_token`, and accept the pyannote "
            "model licences on huggingface.co. Continuing without speaker labels."
        )
        return transcript

    device = report().gpu.device
    try:
        pipeline = pipeline_cls(use_auth_token=hf_token, device=device)
        diarization = pipeline(str(audio), min_speakers=min_speakers, max_speakers=max_speakers)
    except Exception as exc:
        log.warning("Diarization failed (%s); continuing without speaker labels.", exc)
        return transcript

    turns = _diarization_turns(diarization)
    if not turns:
        return transcript

    _assign_speakers(transcript, turns)
    log.info("Diarization labelled %d speakers.", len(transcript.speakers))
    return transcript


def _load_diarization_pipeline():
    """Import WhisperX's diarization pipeline across its API reshuffles."""
    try:
        from whisperx.diarize import DiarizationPipeline

        return DiarizationPipeline
    except ImportError:
        pass
    try:
        from whisperx import DiarizationPipeline  # type: ignore[attr-defined]

        return DiarizationPipeline
    except (ImportError, AttributeError):
        return None


def _diarization_turns(diarization) -> list[tuple[str, float, float]]:
    """Normalise WhisperX output into ``(speaker, start, end)`` tuples.

    Recent versions return a pandas DataFrame; older ones return a pyannote
    ``Annotation``. Both are handled so the extra can be upgraded independently.
    """
    turns: list[tuple[str, float, float]] = []

    if hasattr(diarization, "itertuples"):  # DataFrame
        for row in diarization.itertuples():
            speaker = getattr(row, "speaker", None)
            start = getattr(row, "start", None)
            end = getattr(row, "end", None)
            if speaker is not None and start is not None and end is not None:
                turns.append((str(speaker), float(start), float(end)))
        return turns

    if hasattr(diarization, "itertracks"):  # pyannote Annotation
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            turns.append((str(speaker), float(segment.start), float(segment.end)))

    return turns


def _assign_speakers(transcript: Transcript, turns: list[tuple[str, float, float]]) -> None:
    """Attach a speaker to every word by maximum temporal overlap.

    Overlap rather than midpoint containment: words straddling a turn boundary
    are common, and the speaker who covers more of the word is the better guess.
    """
    turns = sorted(turns, key=lambda t: t[1])

    for word in transcript.words:
        best_speaker: str | None = None
        best_overlap = 0.0
        for speaker, start, end in turns:
            if start > word.end:
                break
            overlap = min(word.end, end) - max(word.start, start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        word.speaker = best_speaker

    for segment in transcript.segments:
        segment.speaker = transcript.dominant_speaker(segment.first_word, segment.last_word)
