# ruff: noqa: E501
"""The Clipper's own music mix: level-matched to the speech, ducked under it, and kept under a peak ceiling.

Measured on real material: the bundled tracks sit near -44 LUFS against speech at -14, so the old fixed
0.12 multiplier left the music about 49 dB under the voice (inaudible). The mix now measures both, ducks the
music while someone talks, and caps the peaks after loudnorm.
"""

import inspect
import re
import shutil
import subprocess

import pytest
from autoclip.config import ExportSettings
from autoclip.pipeline import export as ex
from autoclip.pipeline import ffmpeg

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
S = ExportSettings()


# ------------------------------------------------------------------ graph structure


def test_the_music_is_keyed_by_the_speech_so_it_ducks():
    graph = ex._ducked_music_graph(S)
    assert "asplit=2[voice][sc]" in graph and "[bgm][sc]sidechaincompress" in graph


def test_a_measured_gain_replaces_the_raw_multiplier():
    assert "volume=0.1200" in ex._ducked_music_graph(S)                  # no measurement: the old behaviour
    assert "volume=11.58dB" in ex._ducked_music_graph(S, 11.58)


def test_the_analysis_pass_and_the_final_encode_build_the_same_mix():
    assert "_music_mix_graph(settings, music_gain_db)" in inspect.getsource(ex._loudnorm_analysis)
    assert "_music_mix_graph(settings, music_gain_db)" in inspect.getsource(ex.build_music_audio_filtergraph)
    final = ex.build_music_audio_filtergraph(S, None, 5.0)
    assert final.startswith(ex._music_mix_graph(S, 5.0))                  # what is measured is what ships


def test_both_audio_chains_end_with_a_real_peak_ceiling():
    ceiling = "alimiter=limit=0.841:level=disabled"                       # -1.5 dBFS; level=disabled or it auto-levels
    assert ceiling in ex.build_audio_filtergraph(S) and ceiling in ex.build_music_audio_filtergraph(S)


# ------------------------------------------------------------------ the measured gain


def test_the_gain_puts_the_music_where_music_volume_says(monkeypatch):
    lufs = {"speech": -14.0, "music": -44.0}
    monkeypatch.setattr(ex, "_integrated_lufs", lambda args: lufs["music"] if "-stream_loop" in args else lufs["speech"])
    gain = ex._music_gain_db(S, "src.mp4", 0.0, 20.0, "track.mp3")
    assert gain == pytest.approx(30.0 - 18.42, abs=0.05)                   # 30 dB louder, then 0.12 = -18.4 dB


def test_an_untrustworthy_measurement_is_capped_or_dropped(monkeypatch):
    monkeypatch.setattr(ex, "_integrated_lufs", lambda args: -120.0 if "-stream_loop" in args else -14.0)
    assert ex._music_gain_db(S, "s", 0.0, 5.0, "t") == ex.MUSIC_GAIN_LIMITS_DB[1]      # never amplified into noise
    monkeypatch.setattr(ex, "_integrated_lufs", lambda args: None)
    assert ex._music_gain_db(S, "s", 0.0, 5.0, "t") is None                           # caller keeps the fixed gain


# ------------------------------------------------------------------ real ffmpeg


def _ff(*args):
    subprocess.run([ffmpeg.ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args], check=True)


def _mean_db(path, start, end):
    err = subprocess.run([ffmpeg.ffmpeg_path(), "-hide_banner", "-i", str(path), "-af",
                          f"atrim={start}:{end},volumedetect", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    return float(re.search(r"mean_volume:\s+(-?[\d.]+) dB", err).group(1))


@pytest.fixture
def signals(tmp_path):
    """Speech: a 440 Hz tone at -16 dBFS peak for 8 s, then silence. Music: an 880 Hz tone at -46 dBFS peak, as
    quiet as the bundled tracks. (ffmpeg's ``sine`` is 0.125 by default, hence the multipliers.)"""
    speech, music = tmp_path / "speech.wav", tmp_path / "music.wav"
    _ff("-f", "lavfi", "-i", "sine=frequency=440:duration=16", "-af",
        "volume='if(lt(t,8),1.28,0)':eval=frame", "-ar", "48000", str(speech))
    _ff("-f", "lavfi", "-i", "sine=frequency=880:duration=16", "-af", "volume=0.04", "-ar", "48000", str(music))
    return speech, music


@needs_ffmpeg
def test_integrated_loudness_is_measured_and_silence_is_not(signals, tmp_path):
    speech, _ = signals
    assert ex._integrated_lufs(["-t", "6", "-i", str(speech)]) == pytest.approx(-18.4, abs=2.0)
    silent = tmp_path / "silent.wav"
    _ff("-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "3", str(silent))
    assert ex._integrated_lufs(["-i", str(silent)]) is None


@needs_ffmpeg
def test_a_quiet_track_is_lifted_to_music_volume_below_the_speech_and_dips_under_talking(signals, tmp_path):
    speech, music = signals
    gain = ex._music_gain_db(S, speech, 0.0, 16.0, music)
    assert gain is not None and gain > 8          # the track is ~30 dB under the speech: lifted, minus the 18 dB
    graph = (f"[0:a]asplit=2[voice][sc];[1:a]volume={gain:.2f}dB,asplit=2[bgm][raw];"
             f"[bgm][sc]{ex.DUCK_FILTER}[ducked]")
    ducked, raw, voice = tmp_path / "d.wav", tmp_path / "r.wav", tmp_path / "v.wav"
    _ff("-t", "16", "-i", str(speech), "-stream_loop", "-1", "-i", str(music), "-filter_complex", graph,
        "-map", "[ducked]", "-t", "16", "-ac", "1", str(ducked), "-map", "[raw]", "-t", "16", "-ac", "1", str(raw),
        "-map", "[voice]", "-t", "16", "-ac", "1", str(voice))
    talking, pause = (1.0, 7.0), (10.0, 15.0)
    voice_db = _mean_db(voice, *talking)
    # level matching: in a pause the music sits about 18 dB under the voice (music_volume 0.12 = -18.4 dB)
    assert _mean_db(raw, *pause) - voice_db == pytest.approx(-18.4, abs=3.0)
    # ducking: it dips while someone talks and comes back in the pause
    assert _mean_db(ducked, *pause) - _mean_db(ducked, *talking) >= 4.0
    assert _mean_db(ducked, *pause) == pytest.approx(_mean_db(raw, *pause), abs=1.0)


@needs_ffmpeg
def test_a_hot_mix_stays_under_the_true_peak_ceiling(tmp_path):
    speech, music, out = tmp_path / "s.wav", tmp_path / "m.wav", tmp_path / "o.wav"
    _ff("-f", "lavfi", "-i", "sine=frequency=300:duration=8", "-af", "volume=7.6", "-ar", "48000", str(speech))
    _ff("-f", "lavfi", "-i", "sine=frequency=220:duration=8", "-af", "volume=7.2", "-ar", "48000", str(music))
    _ff("-i", str(speech), "-stream_loop", "-1", "-i", str(music), "-filter_complex",
        ex.build_music_audio_filtergraph(S, None, -6.0), "-map", "[aout]", "-t", "8", "-ar", "48000", str(out))
    err = subprocess.run([ffmpeg.ffmpeg_path(), "-hide_banner", "-i", str(out), "-af", "ebur128=peak=true", "-vn",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    assert float(re.findall(r"Peak:\s+(-?[\d.]+) dBFS", err)[-1]) <= -1.0
