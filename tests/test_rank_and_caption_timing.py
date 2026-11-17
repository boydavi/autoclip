"""Clip rank means best-first; YouTube captions without word timing share time by word length."""

import json
from types import SimpleNamespace

import pytest
from autoclip.pipeline import highlights
from autoclip.pipeline.transcribe import _word_spans, parse_youtube_captions


def _clip(score, start):
    return SimpleNamespace(score=score, start_s=start, rank=0)


def test_rank_one_is_the_best_clip_not_the_earliest():
    # the climax comes last in the video
    clips = [_clip(61, 10.0), _clip(95, 2500.0), _clip(80, 900.0)]
    highlights._assign_ranks(clips)
    assert [(c.rank, c.score) for c in clips] == [(1, 95), (2, 80), (3, 61)]


def test_equal_scores_rank_the_earlier_clip_first():
    clips = [_clip(70, 300.0), _clip(70, 100.0)]
    highlights._assign_ranks(clips)
    assert [c.start_s for c in clips] == [100.0, 300.0] and [c.rank for c in clips] == [1, 2]


def test_ranking_nothing_is_fine():
    clips: list = []
    highlights._assign_ranks(clips)
    assert clips == []


def test_building_the_final_clips_ranks_them_by_score():
    import inspect
    src = inspect.getsource(highlights.build_clips)
    assert "_assign_ranks(clips)" in src and "enumerate(clips, start=1)" not in src


# ------------------------------------------------------------------ caption timing


def test_word_time_is_shared_by_length_and_fills_the_stretch_exactly():
    spans = _word_spans(["I", "extraordinarily", "agree"], 1000, 4000)
    assert spans[0][0] == pytest.approx(1.0) and spans[-1][1] == pytest.approx(4.0)
    # no gaps or overlaps
    assert all(a[1] == pytest.approx(b[0]) for a, b in zip(spans, spans[1:], strict=False))
    durations = [e - s for s, e in spans]
    assert durations[1] > durations[2] > durations[0] > 0   # long words take longer
    assert sum(durations) == pytest.approx(3.0)


def test_one_word_takes_the_whole_stretch_and_an_empty_stretch_is_safe():
    assert _word_spans(["hello"], 500, 1500) == [(0.5, 1.5)]
    assert _word_spans(["a", "b"], 2000, 2000) == [(2.0, 2.0), (2.0, 2.0)]
    assert _word_spans([], 0, 1000) == []


def _caption_file(tmp_path, events):
    path = tmp_path / "c.en.json3"
    path.write_text(json.dumps({"events": events}), encoding="utf-8")
    return path


def test_a_caption_event_without_word_offsets_is_split_by_length(tmp_path):
    ts = parse_youtube_captions(_caption_file(tmp_path, [
        {"tStartMs": 1000, "dDurationMs": 3000, "segs": [{"utf8": "I extraordinarily agree"}]},
    ]))
    spans = {w.text: (w.start, w.end) for w in ts.words}
    assert spans["I"][0] == pytest.approx(1.0) and spans["agree"][1] == pytest.approx(4.0)
    long_word, short_word = spans["extraordinarily"], spans["I"]
    assert (long_word[1] - long_word[0]) > 2 * (short_word[1] - short_word[0])


def test_real_per_segment_offsets_are_still_respected(tmp_path):
    ts = parse_youtube_captions(_caption_file(tmp_path, [
        {"tStartMs": 1000, "dDurationMs": 3000, "segs": [
            {"utf8": "hello world", "tOffsetMs": 0}, {"utf8": " again now", "tOffsetMs": 1500}]},
    ]))
    by = {w.text: w for w in ts.words}
    # segment 1 runs 1.0-2.5 and segment 2 runs 2.5-4.0
    assert by["hello"].start == pytest.approx(1.0) and by["world"].end == pytest.approx(2.5)
    assert by["again"].start == pytest.approx(2.5) and by["now"].end == pytest.approx(4.0)
