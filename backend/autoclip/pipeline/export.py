"""Export stage — crop path plus captions plus audio normalisation to an MP4.

The whole render is a single ffmpeg pass. Per-shot crop segments are trimmed out
of the source, cropped independently, scaled to a common output size, and
concatenated in the filtergraph — which is what allows a wide two-shot and a
tight single to sit in one clip without a mid-stream frame-size change.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..config import ExportSettings
from ..system import report
from . import captions as captions_module
from . import ffmpeg
from .captions import CaptionStyle
from .reframe.croppath import CropPath, segment_crop_filter
from .transcript import Word

log = logging.getLogger(__name__)

#: Audio formats accepted as background music tracks.
MUSIC_FORMATS: frozenset[str] = frozenset({".mp3", ".m4a", ".wav", ".ogg", ".flac", ".aac"})

#: Output dimensions per aspect ratio. 1080-wide is the platform sweet spot:
#: high enough to avoid re-encode mush, low enough to upload quickly.
RATIOS: dict[str, tuple[int, int]] = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
}

#: Loudness targets. -14 LUFS integrated with -1.5 dBTP is what every major
#: platform normalises toward, so hitting it ourselves avoids their processing.
LOUDNESS_TRUE_PEAK = -1.5
LOUDNESS_RANGE = 11.0

#: Backdrop for audio-only sources: a dark neutral that keeps white captions legible.
AUDIO_ONLY_BACKGROUND = "0x14171c"

AUDIO_BITRATE = "192k"
AUDIO_SAMPLE_RATE = 48_000


class ExportError(RuntimeError):
    """Rendering a clip failed."""


@dataclass
class ExportRequest:
    source: Path
    destination: Path
    start_s: float
    end_s: float
    crop_path: CropPath
    words: list[Word]
    style: CaptionStyle
    ratio: str = "9:16"
    burn_captions: bool = True
    music_track: Path | None = None
    #: False for audio-only sources: captions are burned onto a generated
    #: solid-colour background instead of cropped video.
    has_video: bool = True

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def ratio_dimensions(ratio: str) -> tuple[int, int]:
    if ratio not in RATIOS:
        raise ExportError(f"Unknown ratio {ratio!r}. Available: {', '.join(RATIOS)}")
    return RATIOS[ratio]


def slugify_title(title: str, *, max_length: int = 50) -> str:
    """Filename-safe slug for a clip title."""
    slug = re.sub(r"[^\w\s-]", "", title, flags=re.UNICODE).strip().lower()
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug[:max_length] or "clip"


def output_filename(title: str, ratio: str, *, rank: int | None = None) -> str:
    """``<slug>[-NN]_<ratio>.mp4``; pass the clip's rank so equal titles can't collide."""
    tag = f"-{rank:02d}" if rank is not None else ""
    return f"{slugify_title(title)}{tag}_{ratio.replace(':', 'x')}.mp4"


# --------------------------------------------------------------------------
# Filtergraph construction
# --------------------------------------------------------------------------


def build_video_filtergraph(
    request: ExportRequest,
    *,
    subtitle_name: str | None,
    fonts_name: str = "fonts",
    background_input: int | None = None,
) -> str:
    """Build the ``-filter_complex`` video chain for one clip.

    ``background_input`` is the ffmpeg input index of the generated background
    and is required when the request has no video.
    """
    out_w, out_h = ratio_dimensions(request.ratio)
    parts: list[str] = []
    labels: list[str] = []

    if not request.has_video:
        if background_input is None:
            raise ExportError("An audio-only clip needs a generated background input.")
        parts.append(f"[{background_input}:v]setsar=1,format=yuv420p[v0]")
        segments = []
    else:
        segments = request.crop_path.segments
        if not segments:
            raise ExportError("The crop path has no segments.")

    for index, segment in enumerate(segments):
        label = f"v{index}"
        labels.append(f"[{label}]")

        if len(segments) == 1:
            # No trim needed; the input is already seeked to the clip.
            source_label = "[0:v]"
            prefix = ""
        else:
            source_label = f"[s{index}]"
            prefix = (
                f"[0:v]trim=start={segment.start_s:.4f}:end={segment.end_s:.4f},"
                f"setpts=PTS-STARTPTS{source_label}"
            )
            parts.append(prefix)

        if segment.fit:
            parts.extend(_fit_chain(source_label, index, out_w, out_h))
            continue

        chain = [segment_crop_filter(segment)]
        if segment.zoom > 0:
            chain.append(_zoom_filter(segment.zoom, out_w, out_h))
        chain.append(f"scale={out_w}:{out_h}:flags=lanczos")
        chain.append("setsar=1,format=yuv420p")

        parts.append(f"{source_label}{','.join(chain)}[{label}]")

    if len(segments) <= 1:
        current = "[v0]"
    else:
        parts.append(f"{''.join(labels)}concat=n={len(segments)}:v=1:a=0[vcat]")
        current = "[vcat]"

    if request.burn_captions and subtitle_name is not None:
        # Bare relative names — ffmpeg runs with its cwd set to the render
        # workspace, so there is nothing here that needs escaping.
        parts.append(f"{current}ass=filename={subtitle_name}:fontsdir={fonts_name}[vout]")
    else:
        parts.append(f"{current}null[vout]")

    return ";".join(parts)


def _fit_chain(source_label: str, index: int, out_w: int, out_h: int) -> list[str]:
    """Fit the whole frame into the output over a blurred copy of itself.

    Used when the subjects are spread wider than any crop can hold. Plain black
    bars would be honest but look cheap on a phone; a blurred fill is what
    viewers are used to seeing and keeps the frame full-bleed.
    """
    return [
        f"{source_label}split=2[fitbg{index}][fitfg{index}]",
        (
            f"[fitbg{index}]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h},boxblur=24:2[bg{index}]"
        ),
        (
            f"[fitfg{index}]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:"
            f"flags=lanczos[fg{index}]"
        ),
        (f"[bg{index}][fg{index}]overlay=(W-w)/2:(H-h)/2,setsar=1,format=yuv420p[v{index}]"),
    ]


def _zoom_filter(zoom: float, out_w: int, out_h: int) -> str:
    """A slow push-in over the segment.

    Uses ``zoompan``, which is the only filter that can change apparent scale
    while holding a fixed output size. Off by default — at very small
    increments it can step visibly, and a locked frame beats a stuttering one.
    """
    end_zoom = 1.0 + zoom
    return f"zoompan=z='min(zoom+{zoom / 240:.6f},{end_zoom:.4f})':d=1:s={out_w}x{out_h}:fps=30"


def pick_music_track(music_dir: Path, *, rng_seed: int = 0) -> Path | None:
    """Return a track from *music_dir*, or None if the directory is empty/absent.

    Tracks are selected by index (seed % count) after a stable sort so the
    choice is reproducible: same seed + same directory = same track every time.
    """
    if not music_dir.is_dir():
        return None
    tracks = sorted(p for p in music_dir.iterdir() if p.suffix.lower() in MUSIC_FORMATS)
    if not tracks:
        return None
    return tracks[rng_seed % len(tracks)]


def resolve_music_track(
    settings: ExportSettings, music_dir: Path, *, rng_seed: int = 0
) -> Path | None:
    """The track to mix under a clip, or None when music is off or unavailable.

    A pinned ``music_track`` wins when the file exists (basename only, so a
    crafted name cannot escape *music_dir*); otherwise one is picked by seed.
    """
    if not settings.music_enabled:
        return None
    if settings.music_track:
        candidate = music_dir / Path(settings.music_track).name
        if candidate.suffix.lower() in MUSIC_FORMATS and candidate.is_file():
            return candidate
        log.warning(
            "Pinned music track %r not found in %s; falling back to random.",
            settings.music_track,
            music_dir,
        )
    return pick_music_track(music_dir, rng_seed=rng_seed)


def _loudnorm_filter(settings: ExportSettings, measured: dict[str, str] | None) -> str:
    """Build a loudnorm filter string, using measured stats when available (pass 2)."""
    base = f"loudnorm=I={settings.loudness_lufs}:TP={LOUDNESS_TRUE_PEAK}:LRA={LOUDNESS_RANGE}"
    if measured:
        try:
            return (
                base
                + f":measured_I={measured['input_i']}"
                + f":measured_TP={measured['input_tp']}"
                + f":measured_LRA={measured['input_lra']}"
                + f":measured_thresh={measured['input_thresh']}"
                + ":linear=true"
            )
        except KeyError:
            pass
    return base


def _run_for_stderr(
    command: list[str], *, timeout_s: float, cancelled: Callable[[], bool] | None
) -> str:
    """Run *command* and return its stderr, killing it on timeout or cancellation."""
    proc = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    waited = 0.0
    tick = 0.5
    try:
        while True:
            try:
                _, stderr = proc.communicate(timeout=tick)
                return stderr or ""
            except subprocess.TimeoutExpired:
                waited += tick
                if cancelled is not None and cancelled():
                    raise ffmpeg.Cancelled("Render cancelled.") from None
                if waited >= timeout_s:
                    raise
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()


def _loudnorm_analysis(
    source: Path,
    start_s: float,
    duration_s: float,
    settings: ExportSettings,
    *,
    music_track: Path | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, str]:
    """Run an audio-only analysis pass and return loudnorm measured stats.

    Polls ``cancelled`` while ffmpeg runs and raises :class:`ffmpeg.Cancelled`
    (the same signal a cancelled render raises) so cancelling is not held up by
    a long analysis.

    The stats are used by the main encode pass (pass 2) with linear=true, which
    applies a single gain correction rather than dynamic compression, avoiding
    pumping on clipped speech.

    Returns an empty dict on any failure so the caller falls back to single-pass.
    """
    ln = (
        f"loudnorm=I={settings.loudness_lufs}:TP={LOUDNESS_TRUE_PEAK}:LRA={LOUDNESS_RANGE}"
        ":print_format=json"
    )

    if music_track is not None:
        vol = settings.music_volume
        # [anlout] label is required: -filter_complex never auto-maps unlabeled outputs,
        # so without it ffmpeg exits with "no stream" and the JSON block never appears.
        audio_fg = (
            f"[1:a]volume={vol:.4f}[bgm];"
            f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=3[mixed];"
            f"[mixed]{ln}[anlout]"
        )
        args = [
            "-ss", f"{start_s:.4f}",
            "-t", f"{duration_s:.4f}",
            "-i", str(source),
            "-stream_loop", "-1",
            "-i", str(music_track),
            "-filter_complex", audio_fg,
            "-map", "[anlout]",
            "-vn", "-f", "null", os.devnull,
        ]
    else:
        args = [
            "-ss", f"{start_s:.4f}",
            "-t", f"{duration_s:.4f}",
            "-i", str(source),
            "-vn",
            "-af", ln,
            "-f", "null", os.devnull,
        ]

    command = [
        ffmpeg.ffmpeg_path(),
        "-hide_banner", "-nostdin", "-y",
        "-loglevel", "info",  # loudnorm prints stats at info level
        *args,
    ]

    # Generous timeout: 2× the clip duration plus 30 s for I/O overhead.
    # Without this, a corrupted or network-mounted source blocks the export
    # worker indefinitely because there is no cancellation path here.
    analysis_timeout = duration_s * 2 + 30

    try:
        stderr = _run_for_stderr(command, timeout_s=analysis_timeout, cancelled=cancelled)
        # loudnorm JSON block appears at the tail of stderr between { and }
        start_idx = stderr.rfind("{")
        end_idx = stderr.rfind("}") + 1
        if start_idx != -1 and end_idx > start_idx:
            return json.loads(stderr[start_idx:end_idx])
    except subprocess.TimeoutExpired:
        log.warning(
            "loudnorm analysis timed out after %.0fs; using single-pass normalisation.",
            analysis_timeout,
        )
        return {}
    except ffmpeg.Cancelled:
        raise
    except Exception:
        pass

    log.warning("loudnorm analysis did not produce parseable JSON; using single-pass normalisation.")
    return {}


def build_audio_filtergraph(settings: ExportSettings, measured: dict[str, str] | None = None) -> str:
    return _loudnorm_filter(settings, measured)


def build_music_audio_filtergraph(
    settings: ExportSettings, measured: dict[str, str] | None = None
) -> str:
    """Return the filter_complex audio chain that mixes input 1 (music) under input 0 (speech).

    Pass measured loudnorm stats for two-pass normalisation (linear=true), which
    avoids the pumping that single-pass dynamic loudnorm can introduce on mixed audio.
    """
    vol = settings.music_volume
    ln = _loudnorm_filter(settings, measured)
    return (
        f"[1:a]volume={vol:.4f}[bgm];"
        f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=3[mixed];"
        f"[mixed]{ln}[aout]"
    )


#: libx264 presets that Quick Sync lacks map to its fastest one.
_QSV_PRESETS = {"veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"}


def qsv_args(quality: int, preset: str = "medium") -> list[str]:
    """Intel Quick Sync H.264. ``-global_quality`` (ICQ) plays the role of CRF;
    QSV wants nv12 input, so the pixel format is set here rather than yuv420p."""
    return [
        "-c:v", "h264_qsv",
        "-preset", preset if preset in _QSV_PRESETS else "veryfast",
        "-global_quality", str(quality),
        "-profile:v", "high",
        "-pix_fmt", "nv12",
    ]


def _with_pix_fmt(args: list[str]) -> list[str]:
    return args if "-pix_fmt" in args else [*args, "-pix_fmt", "yuv420p"]


def encoder_args(settings: ExportSettings) -> list[str]:
    """Pick the video encoder and its quality settings.

    NVENC is materially faster where available. Its quality knob is ``-cq``
    rather than ``-crf``, and the two scales are close enough that reusing the
    configured value keeps output consistent between machines.
    """
    if settings.prefer_hardware_encoder and report().ffmpeg.nvenc_works:
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p5",
            "-rc",
            "vbr",
            "-cq",
            str(settings.crf),
            "-b:v",
            "0",
            "-profile:v",
            "high",
        ]
    if settings.prefer_hardware_encoder and getattr(report().ffmpeg, "qsv_works", False):
        return qsv_args(settings.crf, settings.preset)
    return [
        "-c:v",
        "libx264",
        "-preset",
        settings.preset,
        "-crf",
        str(settings.crf),
        "-profile:v",
        "high",
    ]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def export_clip(
    request: ExportRequest,
    *,
    work_dir: Path,
    settings: ExportSettings | None = None,
    on_progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Render one clip to an MP4 and return the output path.

    Preconditions:
        request.source exists; request.crop_path segments span the clip duration
        in clip-relative seconds.
    """
    settings = settings or ExportSettings()
    out_w, out_h = ratio_dimensions(request.ratio)

    # Each clip renders in its own workspace so concurrent exports can't collide
    # on the shared `captions.ass` name that the relative-path scheme requires.
    workspace = work_dir / request.destination.stem
    workspace.mkdir(parents=True, exist_ok=True)

    subtitle_name: str | None = None
    fonts_name = "fonts"
    render_cwd: Path | None = None

    if request.burn_captions and request.words:
        ass_path = captions_module.write_ass(
            workspace / "captions.ass",
            request.words,
            request.style,
            width=out_w,
            height=out_h,
            time_offset_s=request.start_s,
        )
        render_cwd, subtitle_name, fonts_name = ffmpeg.relative_filter_workspace(
            ass_path, captions_module.FONT_DIR
        )

    # The generated background is appended after the source (and music) inputs,
    # so existing [0:a] / [1:a] references keep their indices.
    background_input = 2 if request.music_track is not None else 1
    filtergraph = build_video_filtergraph(
        request,
        subtitle_name=subtitle_name,
        fonts_name=fonts_name,
        background_input=None if request.has_video else background_input,
    )
    background_args: list[str] = (
        []
        if request.has_video
        else [
            "-f",
            "lavfi",
            "-i",
            f"color=c={AUDIO_ONLY_BACKGROUND}:s={out_w}x{out_h}:r=30:d={request.duration_s:.4f}",
        ]
    )

    request.destination.parent.mkdir(parents=True, exist_ok=True)

    # Loudnorm two-pass: analyse the audio mix first so the encode pass can use
    # linear=true normalisation (single gain correction, no dynamic pumping).
    measured = _loudnorm_analysis(
        request.source,
        request.start_s,
        request.duration_s,
        settings,
        music_track=request.music_track,
        cancelled=cancelled,
    )

    # Input and output paths are ordinary argv arguments, so they stay absolute
    # and need no escaping regardless of the working directory.
    if request.music_track is not None:
        # Music is input 1 (-stream_loop -1 loops it for clips shorter than the
        # track; ffmpeg stops at the speech duration via amix duration=first).
        combined_filtergraph = filtergraph + ";" + build_music_audio_filtergraph(settings, measured)
        args = [
            "-ss",
            f"{request.start_s:.4f}",
            "-t",
            f"{request.duration_s:.4f}",
            "-i",
            str(request.source),
            "-stream_loop",
            "-1",
            "-i",
            str(request.music_track),
            *background_args,
            "-filter_complex",
            combined_filtergraph,
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            *_with_pix_fmt(encoder_args(settings)),
            "-c:a",
            "aac",
            "-b:a",
            AUDIO_BITRATE,
            "-ar",
            str(AUDIO_SAMPLE_RATE),
            "-movflags",
            "+faststart",
            str(request.destination),
        ]
    else:
        args = [
            # Seeking before -i decodes from the nearest keyframe and is far faster
            # than an output-side seek; modern ffmpeg keeps it frame-accurate.
            "-ss",
            f"{request.start_s:.4f}",
            "-t",
            f"{request.duration_s:.4f}",
            "-i",
            str(request.source),
            *background_args,
            "-filter_complex",
            filtergraph,
            "-map",
            "[vout]",
            "-map",
            "0:a?",
            "-af",
            build_audio_filtergraph(settings, measured),
            *_with_pix_fmt(encoder_args(settings)),
            "-c:a",
            "aac",
            "-b:a",
            AUDIO_BITRATE,
            "-ar",
            str(AUDIO_SAMPLE_RATE),
            "-movflags",
            "+faststart",
            str(request.destination),
        ]

    try:
        ffmpeg.run(
            args,
            total_duration_s=request.duration_s,
            on_progress=on_progress,
            cancelled=cancelled,
            cwd=render_cwd,
        )
    except ffmpeg.FFmpegError as exc:
        raise ExportError(f"Could not render {request.destination.name}.\n{exc}") from exc

    if not request.destination.exists() or request.destination.stat().st_size == 0:
        raise ExportError(f"ffmpeg reported success but {request.destination.name} is empty.")

    if settings.write_srt and request.words:
        captions_module.write_srt(
            request.destination.with_suffix(".srt"),
            request.words,
            time_offset_s=request.start_s,
        )

    log.info(
        "Exported %s (%.1fs, %s)",
        request.destination.name,
        request.duration_s,
        request.ratio,
    )
    return request.destination
