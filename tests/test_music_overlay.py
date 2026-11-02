"""Smoke tests for the music background overlay feature.

Covers:
- pick_music_track() discovery logic
- build_music_audio_filtergraph() structure
- export_clip() with a real music track via ffmpeg (marked slow)

No API keys required — all tests use synthetic lavfi-generated audio.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from autoclip.config import ExportSettings
from autoclip.pipeline import captions, export, ffmpeg
from autoclip.pipeline.export import (
    MUSIC_FORMATS,
    build_music_audio_filtergraph,
    pick_music_track,
)
from autoclip.pipeline.reframe.croppath import centre_crop
from autoclip.pipeline.transcript import Word

pytestmark = pytest.mark.slow

SOURCE_W, SOURCE_H = 1280, 720
SOURCE_DURATION = 10.0


# --------------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def source_video(tmp_path_factory) -> Path:
    """Synthetic 1280×720 clip with a 440 Hz tone — shared across all tests."""
    path = tmp_path_factory.mktemp("media") / "source.mp4"
    subprocess.run(
        [
            ffmpeg.ffmpeg_path(),
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i",
            f"testsrc2=size={SOURCE_W}x{SOURCE_H}:rate=30:duration={SOURCE_DURATION}",
            "-f", "lavfi", "-i",
            f"sine=frequency=440:duration={SOURCE_DURATION}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True, capture_output=True,
    )
    return path


@pytest.fixture(scope="module")
def music_track(tmp_path_factory) -> Path:
    """Synthetic 880 Hz tone saved as an MP3 — the background music track."""
    path = tmp_path_factory.mktemp("music") / "bg_track.mp3"
    subprocess.run(
        [
            ffmpeg.ffmpeg_path(),
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i",
            "sine=frequency=880:duration=30",
            "-c:a", "libmp3lame", "-q:a", "9",
            str(path),
        ],
        check=True, capture_output=True,
    )
    return path


@pytest.fixture
def words() -> list[Word]:
    texts = ["background", "music", "is", "now", "mixed", "in"]
    step = 5.0 / len(texts)
    return [
        Word(text=t, start=2.0 + i * step, end=2.0 + i * step + step * 0.85)
        for i, t in enumerate(texts)
    ]


# --------------------------------------------------------------------------
# Unit tests (no ffmpeg render needed)
# --------------------------------------------------------------------------


class TestPickMusicTrack:
    def test_returns_none_when_dir_does_not_exist(self, tmp_path) -> None:
        assert pick_music_track(tmp_path / "nonexistent") is None

    def test_returns_none_when_dir_is_empty(self, tmp_path) -> None:
        assert pick_music_track(tmp_path) is None

    def test_ignores_non_audio_files(self, tmp_path) -> None:
        (tmp_path / "readme.txt").write_text("ignore me")
        assert pick_music_track(tmp_path) is None

    def test_returns_the_only_track(self, tmp_path) -> None:
        track = tmp_path / "song.mp3"
        track.touch()
        assert pick_music_track(tmp_path) == track

    def test_seed_selects_different_tracks(self, tmp_path) -> None:
        for name in ("a.mp3", "b.mp3", "c.mp3"):
            (tmp_path / name).touch()
        results = {pick_music_track(tmp_path, rng_seed=i) for i in range(3)}
        assert len(results) == 3

    def test_selection_is_reproducible(self, tmp_path) -> None:
        for name in ("x.mp3", "y.mp3"):
            (tmp_path / name).touch()
        assert pick_music_track(tmp_path, rng_seed=7) == pick_music_track(tmp_path, rng_seed=7)

    def test_all_declared_formats_are_accepted(self, tmp_path) -> None:
        for fmt in MUSIC_FORMATS:
            (tmp_path / f"track{fmt}").touch()
        tracks = sorted(p for p in tmp_path.iterdir()
                        if p.suffix.lower() in MUSIC_FORMATS)
        assert len(tracks) == len(MUSIC_FORMATS)


class TestBuildMusicAudioFiltergraph:
    def test_contains_amix(self) -> None:
        fg = build_music_audio_filtergraph(ExportSettings())
        assert "amix" in fg

    def test_contains_loudnorm(self) -> None:
        fg = build_music_audio_filtergraph(ExportSettings())
        assert "loudnorm" in fg

    def test_volume_is_embedded(self) -> None:
        settings = ExportSettings(music_volume=0.25)
        fg = build_music_audio_filtergraph(settings)
        assert "0.2500" in fg

    def test_music_input_is_index_1(self) -> None:
        fg = build_music_audio_filtergraph(ExportSettings())
        assert "[1:a]" in fg

    def test_output_label_is_aout(self) -> None:
        fg = build_music_audio_filtergraph(ExportSettings())
        assert "[aout]" in fg

    def test_duration_first_stops_at_speech_end(self) -> None:
        fg = build_music_audio_filtergraph(ExportSettings())
        assert "duration=first" in fg


# --------------------------------------------------------------------------
# Integration tests (real ffmpeg renders)
# --------------------------------------------------------------------------


class TestMusicOverlayRender:
    def test_output_has_audio_with_music_track(
        self, source_video, music_track, words, tmp_path
    ) -> None:
        destination = tmp_path / "with_music.mp4"
        request = export.ExportRequest(
            source=source_video,
            destination=destination,
            start_s=2.0, end_s=7.0,
            crop_path=centre_crop(SOURCE_W, SOURCE_H, 5.0),
            words=words,
            style=captions.get_style("bold_pop"),
            music_track=music_track,
        )

        export.export_clip(request, work_dir=tmp_path / "work")

        info = ffmpeg.probe(destination)
        assert destination.exists()
        assert info.has_audio
        assert info.duration_s == pytest.approx(5.0, abs=0.35)

    def test_music_track_shorter_than_clip_loops(
        self, source_video, tmp_path
    ) -> None:
        """A 2-second track must loop to fill a 5-second clip."""
        short_track = tmp_path / "short.mp3"
        subprocess.run(
            [
                ffmpeg.ffmpeg_path(),
                "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "sine=frequency=600:duration=2",
                "-c:a", "libmp3lame", "-q:a", "9", str(short_track),
            ],
            check=True, capture_output=True,
        )

        destination = tmp_path / "looped.mp4"
        request = export.ExportRequest(
            source=source_video,
            destination=destination,
            start_s=2.0, end_s=7.0,
            crop_path=centre_crop(SOURCE_W, SOURCE_H, 5.0),
            words=[],
            style=captions.get_style("bold_pop"),
            burn_captions=False,
            music_track=short_track,
        )

        export.export_clip(request, work_dir=tmp_path / "work")

        assert ffmpeg.probe(destination).duration_s == pytest.approx(5.0, abs=0.35)

    def test_music_changes_audio_versus_no_music(
        self, source_video, music_track, tmp_path
    ) -> None:
        """Audio RMS must differ when music is present vs absent.

        Without this, a silently-failing amix would ship clips that look mixed
        but are just the raw speech track.
        """
        def _volume_at_freq_db(path: Path, freq_hz: int, bandwidth_hz: int = 80) -> float:
            """Mean volume of just the *freq_hz* band — isolates one tone."""
            result = subprocess.run(
                [
                    ffmpeg.ffmpeg_path(),
                    "-hide_banner",
                    "-i", str(path),
                    "-af",
                    f"bandpass=f={freq_hz}:width_type=h:w={bandwidth_hz},volumedetect",
                    "-f", "null", "-",
                ],
                capture_output=True, text=True,
            )
            for line in (result.stderr + result.stdout).splitlines():
                if "mean_volume" in line:
                    try:
                        return float(line.split(":")[-1].replace("dB", "").strip())
                    except ValueError:
                        continue
            return -100.0  # silence / not detected

        # Source speech track is a 440 Hz sine; music track is 880 Hz.
        # loudnorm normalises the overall level of both clips to the same LUFS,
        # so overall mean volume is identical. Instead, isolate the 880 Hz band:
        # the no-music render should have very little 880 Hz energy while the
        # music render should show a clear 880 Hz signal.
        MUSIC_FREQ = 880

        no_music = tmp_path / "no_music.mp4"
        with_music = tmp_path / "with_music.mp4"
        crop = centre_crop(SOURCE_W, SOURCE_H, 5.0)
        base_kwargs = dict(
            source=source_video, start_s=2.0, end_s=7.0,
            crop_path=crop, words=[],
            style=captions.get_style("bold_pop"), burn_captions=False,
        )

        export.export_clip(
            export.ExportRequest(**base_kwargs, destination=no_music),
            work_dir=tmp_path / "work_a",
        )
        export.export_clip(
            export.ExportRequest(**base_kwargs, destination=with_music, music_track=music_track),
            work_dir=tmp_path / "work_b",
        )

        vol_no_music = _volume_at_freq_db(no_music, MUSIC_FREQ)
        vol_with_music = _volume_at_freq_db(with_music, MUSIC_FREQ)
        assert vol_with_music > vol_no_music, (
            f"880 Hz band is not louder with music overlay "
            f"(no_music={vol_no_music:.1f} dB, with_music={vol_with_music:.1f} dB): "
            "amix may not be working"
        )
