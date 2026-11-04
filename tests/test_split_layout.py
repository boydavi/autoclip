"""Split screen: two people stacked, only for part of the clip, with smooth transitions."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from autoclip.pipeline import captions, export
from autoclip.pipeline.reframe.croppath import (
    SPLIT_MAX_SHARE,
    SPLIT_MIN_START_S,
    CropKeyframe,
    CropPath,
    CropSegment,
    SplitPane,
    Strategy,
    layout_of,
    plan_split_layout,
)

PANES = [SplitPane(cx=320, cy=300, face_w=80), SplitPane(cx=960, cy=300, face_w=80)]


def crop_seg(start: float, end: float) -> CropSegment:
    return CropSegment(start, end, 405, 720, [CropKeyframe(start, 400.0, 0.0)], Strategy.TRACK)


def wide_seg(start: float, end: float, panes=PANES) -> CropSegment:
    return CropSegment(
        start, end, 1280, 720, [CropKeyframe(start, 0.0, 0.0)], Strategy.WIDE,
        fit=True, split=list(panes),
    )


def layouts(segments):
    return [(round(s.start_s, 2), round(s.end_s, 2), layout_of(s)) for s in segments]


# ---------------------------------------------------------------- the policy


def test_a_two_person_stretch_after_the_hook_becomes_a_split():
    out = plan_split_layout([crop_seg(0, 4), wide_seg(4, 9)])
    assert layouts(out) == [(0, 4, "crop"), (4, 9, "split")]


def test_the_clip_never_opens_on_a_split():
    # the whole clip is two people: the first seconds keep the normal (fit) framing
    out = plan_split_layout([wide_seg(0, 10)])
    assert layout_of(out[0]) == "fit"
    assert out[0].end_s == SPLIT_MIN_START_S
    assert all(layout_of(s) != "split" for s in out if s.start_s < SPLIT_MIN_START_S)
    assert any(layout_of(s) == "split" for s in out)


def test_a_split_is_only_part_of_the_clip():
    out = plan_split_layout([wide_seg(0, 20)])
    split = sum(s.duration_s for s in out if layout_of(s) == "split")
    assert split <= SPLIT_MAX_SHARE * 20 + 1e-6
    assert layout_of(out[0]) != "split"
    assert layout_of(out[-1]) == "split"      # what is kept is the latest stretch


def test_segments_stay_contiguous_and_cover_the_clip():
    out = plan_split_layout([crop_seg(0, 3), wide_seg(3, 8), crop_seg(8, 10), wide_seg(10, 20)])
    assert out[0].start_s == 0 and out[-1].end_s == 20
    for a, b in zip(out, out[1:], strict=False):
        assert a.end_s == pytest.approx(b.start_s)


def test_a_stretch_that_is_too_short_stays_as_it_was():
    out = plan_split_layout([crop_seg(0, 5), wide_seg(5, 5.6), crop_seg(5.6, 10)])
    assert [layout_of(s) for s in out] == ["crop", "fit", "crop"]


@pytest.mark.parametrize("panes", [[], PANES[:1], PANES + PANES[:1]])
def test_only_exactly_two_people_split(panes):
    out = plan_split_layout([crop_seg(0, 3), wide_seg(3, 9, panes)])
    assert all(layout_of(s) != "split" for s in out)


def test_planning_does_not_change_the_input():
    original = [crop_seg(0, 3), wide_seg(0, 10)]
    plan_split_layout(original)
    assert [s.split_on for s in original] == [False, False]
    assert original[1].end_s == 10


def test_split_survives_the_crop_path_round_trip():
    path = CropPath(1280, 720, [crop_seg(0, 3), wide_seg(3, 9)])
    again = CropPath.from_dict(path.to_dict())
    assert again.segments[1].split == PANES
    assert again.segments[0].split == []


# ------------------------------------------------------------- the filtergraph


def request(segments, ratio="9:16", duration=9.0):
    return export.ExportRequest(
        source=Path("x.mp4"), destination=Path("y.mp4"), start_s=0.0, end_s=duration,
        crop_path=CropPath(1280, 720, segments), words=[], style=captions.get_style("bold_pop"),
        ratio=ratio, burn_captions=False,
    )


def test_split_graph_stacks_two_panes_and_crossfades_in_and_out():
    graph = export.build_video_filtergraph(
        request([crop_seg(0, 3), wide_seg(3, 7), crop_seg(7, 9)]), subtitle_name=None
    )
    assert "vstack=inputs=2" in graph
    assert graph.count("xfade=transition=fade") == 2
    # the first fade starts where the first shot really ends, the second after the split
    assert "offset=3.0000" in graph and "offset=7.0000" in graph


def test_a_boundary_into_a_split_runs_the_outgoing_shot_on_by_the_fade():
    graph = export.build_video_filtergraph(
        request([crop_seg(0, 3), wide_seg(3, 7), crop_seg(7, 9)]), subtitle_name=None
    )
    assert "trim=start=0.0000:end=3.2500" in graph        # 3 s + the 0.25 s fade
    assert "trim=start=3.0000:end=7.2500" in graph


def test_without_a_split_the_graph_is_unchanged():
    graph = export.build_video_filtergraph(
        request([crop_seg(0, 4), crop_seg(4, 9)]), subtitle_name=None
    )
    assert "xfade" not in graph and "vstack" not in graph
    assert "concat=n=2" in graph


def test_wide_ratio_never_splits():
    graph = export.build_video_filtergraph(
        request([crop_seg(0, 3), wide_seg(3, 9)], ratio="16:9"), subtitle_name=None
    )
    assert "vstack" not in graph


def test_split_windows_are_reported_for_the_captions():
    req = request([crop_seg(0, 4), wide_seg(4, 9)])          # 5 of 9 s: inside the share cap
    assert export.split_windows(req) == [(4.0, 9.0)]


def test_captions_in_a_split_are_centred_on_the_seam():
    from autoclip.pipeline.transcript import Word

    words = [Word("hello", 1.0, 1.4), Word("there", 1.5, 1.9), Word("again", 5.0, 5.4)]
    style = captions.get_style("clean_lower")
    subs = captions.build_ass(words, style, width=1080, height=1920, seam_windows=[(4.0, 9.0)])
    centred = [e.text.startswith("{\\an5\\pos(540,960)}") for e in subs.events]
    assert centred == [False, True]


# --------------------------------------------------------- a real render


@pytest.fixture(scope="module")
def source_video(tmp_path_factory) -> Path:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    path = tmp_path_factory.mktemp("split") / "bars.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         "smptehdbars=size=1280x720:rate=30:duration=9", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def render(source: Path, segments, out: Path) -> Path:
    graph = export.build_video_filtergraph(request(segments), subtitle_name=None)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
         "-filter_complex", graph, "-map", "[vout]", "-t", "9", str(out)],
        check=True,
    )
    return out


def seam_ratio(video: Path, t: float) -> float:
    """Brightness of the rows on the seam relative to the rows just above it."""
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(t), "-i", str(video),
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout
    frame = np.frombuffer(raw, dtype=np.uint8).reshape(1920, 1080).astype(float)
    return frame[958:962].mean() / max(frame[900:930].mean(), 1.0)


def probe(video: Path) -> tuple[int, int, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:format=duration", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    width, height = out[0].split(",")[:2]
    return int(width), int(height), float(out[1])


def test_real_render_split_appears_only_in_its_stretch(source_video, tmp_path):
    segments = [crop_seg(0, 2.5), wide_seg(2.5, 7), crop_seg(7, 9)]
    out = render(source_video, segments, tmp_path / "a.mp4")
    width, height, duration = probe(out)
    assert (width, height) == (1080, 1920)
    assert duration == pytest.approx(9.0, abs=0.12)           # fades must not shorten the clip
    assert seam_ratio(out, 1.0) > 0.7                         # normal framing before
    assert seam_ratio(out, 4.5) < 0.4                         # dark seam bar in the split
    assert seam_ratio(out, 8.0) > 0.7                         # normal framing after


def test_real_render_hard_cut_then_split_crossfade_renders(source_video, tmp_path):
    # a fit shot hard-cut to another, then crossfaded into a split: the graph must be valid
    segments = [wide_seg(0, 1.0, []), crop_seg(1.0, 3.0), wide_seg(3.0, 9.0)]
    out = render(source_video, segments, tmp_path / "c.mp4")
    assert probe(out)[2] == pytest.approx(9.0, abs=0.12)
    assert seam_ratio(out, 6.0) < 0.4


def test_real_render_two_person_clip_does_not_open_on_a_split(source_video, tmp_path):
    out = render(source_video, [wide_seg(0, 9)], tmp_path / "b.mp4")
    assert probe(out)[2] == pytest.approx(9.0, abs=0.12)
    assert seam_ratio(out, 0.8) > 0.7                         # opening seconds: not a split
    assert seam_ratio(out, 7.5) < 0.4                         # later: split


# ---------------------------------------------------- who is talking (listener dimmed)


def test_the_listeners_half_is_dimmed_during_each_turn():
    seg = wide_seg(4, 9)                                    # 5 of 9 s: inside the share cap
    seg.speaker_turns = [(4.0, 6.0, 0), (6.0, 9.0, 1)]
    graph = export.build_video_filtergraph(request([crop_seg(0, 4), seg]), subtitle_name=None)
    top_listener = "drawbox=x=0:y=0:w=1080:h=960"          # the right person talks: top half dims
    bottom_listener = "drawbox=x=0:y=960:w=1080:h=960"     # the left person talks: bottom half dims
    assert "enable='between(t,0.000,2.000)'" in graph and bottom_listener in graph
    assert "enable='between(t,2.000,5.000)'" in graph and top_listener in graph


def test_no_turns_means_no_dimming_and_they_survive_the_round_trip():
    graph = export.build_video_filtergraph(request([crop_seg(0, 3), wide_seg(3, 9)]), subtitle_name=None)
    assert "black@0.38" not in graph
    seg = wide_seg(3, 9)
    seg.speaker_turns = [(3.0, 6.0, 1)]
    again = CropPath.from_dict(CropPath(1280, 720, [seg]).to_dict())
    assert again.segments[0].speaker_turns == [(3.0, 6.0, 1)]


def test_real_render_dims_only_the_listener(source_video, tmp_path):
    seg = wide_seg(2.5, 9)
    seg.speaker_turns = [(2.5, 9.0, 0)]                    # the top person talks the whole stretch
    out = render(source_video, [crop_seg(0, 2.5), seg], tmp_path / "d.mp4")
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", "6", "-i", str(out), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    frame = np.frombuffer(raw, dtype=np.uint8).reshape(1920, 1080).astype(float)
    assert frame[1400:1500].mean() < 0.8 * frame[400:500].mean()    # bottom (listener) is darker


# ------------------------------------------------------------- emphasis colour on key words

BS = chr(92)


def _words():
    from autoclip.pipeline.transcript import Word

    return [Word("so", 0.0, 0.3), Word("feel", 0.3, 0.6), Word("the", 0.6, 0.8), Word("future", 0.8, 1.2)]


def test_key_words_get_the_emphasis_colour_and_grow_when_spoken():
    style = captions.get_style("bold_pop")
    subs = captions.build_ass(_words(), style, width=1080, height=1920, emphasis_spans=[(0.8, 1.2)])
    emph = BS + "c" + captions.ass_colour_override(style.emphasis_colour)
    last = subs.events[-1].text                         # "future" is the spoken word
    assert emph in last and f"{BS}fscx{style.scale_percent + style.emphasis_extra_scale}" in last
    before = subs.events[-2].text                       # while "the" is spoken, "future" is already marked
    assert emph in before and f"{BS}fscx{style.scale_percent}" in before


def test_without_emphasis_the_captions_are_unchanged():
    style = captions.get_style("bold_pop")
    plain = captions.build_ass(_words(), style, width=1080, height=1920)
    marked = captions.build_ass(_words(), style, width=1080, height=1920, emphasis_spans=[(50.0, 51.0)])
    assert [e.text for e in plain.events] == [e.text for e in marked.events]


def test_static_styles_colour_the_key_words_too():
    style = captions.get_style("clean_lower")
    subs = captions.build_ass(_words(), style, width=1080, height=1920, emphasis_spans=[(0.8, 1.2)])
    assert BS + "c" + captions.ass_colour_override(style.emphasis_colour) in subs.events[-1].text


# ------------------------------------------------- one caption line at a time


def _overlapping_words():
    from autoclip.pipeline.transcript import Word

    # "of" starts before "feelings" ends; "your" starts before "now." ends (real ASR timing)
    return [
        Word("the", 4.19, 4.49), Word("feelings", 4.49, 4.89), Word("of", 4.47, 5.21), Word("your", 5.21, 5.39),
        Word("emotions", 5.39, 5.99), Word("future", 6.0, 6.5), Word("now.", 6.5, 7.6), Word("your", 7.0, 7.3),
        Word("body", 7.3, 7.9), Word("is", 7.9, 8.1),
    ]


@pytest.mark.parametrize("style_key", ["bold_pop", "clean_lower", "karaoke_fill", "boxed"])
def test_never_two_caption_lines_at_once(style_key):
    style = captions.get_style(style_key)
    subs = captions.build_ass(_overlapping_words(), style, width=1080, height=1920)
    ordered = sorted(subs.events, key=lambda e: e.start)
    for a, b in zip(ordered, ordered[1:], strict=False):
        assert a.end <= b.start, (a.start, a.end, b.start)
    assert all(e.end > e.start for e in ordered)


def test_removing_overlaps_keeps_every_word_and_the_emphasis():
    style = captions.get_style("bold_pop")
    marked = captions.build_ass(_overlapping_words(), style, width=1080, height=1920, emphasis_spans=[(5.39, 5.99)])
    text = " ".join(e.text for e in marked.events)
    assert (BS + "c" + captions.ass_colour_override(style.emphasis_colour)) in text      # emphasis kept
    assert (BS + "c" + captions.ass_colour_override(style.accent)) in text               # spoken-word highlight kept
    for word in ("FEELINGS", "OF", "YOUR", "EMOTIONS", "FUTURE", "NOW.", "BODY"):
        assert word in text


# ------------------------------------------------ panes that follow the person


def test_the_eased_expression_holds_the_ends_and_moves_between_points():
    expr = export._eased([(0.0, 100.0), (2.0, 300.0)])
    assert "lerp(100.00,300.00" in expr and expr.endswith(",300.00)")
    assert export._eased([(1.0, 50.0), (3.0, 90.0), (5.0, 90.0)]).count("lerp(") == 2


def test_a_pane_path_survives_the_crop_path_round_trip():
    pane = SplitPane(cx=320, cy=300, face_w=80, path=[(0.0, 300.0, 300.0), (3.0, 420.0, 310.0)])
    seg = wide_seg(3, 9, [pane, PANES[1]])
    again = CropPath.from_dict(CropPath(1280, 720, [seg]).to_dict())
    assert [tuple(p) for p in again.segments[0].split[0].path] == [(0.0, 300.0, 300.0), (3.0, 420.0, 310.0)]
    assert again.segments[0].split[1].path == []


@pytest.fixture(scope="module")
def mover_video(tmp_path_factory) -> Path:
    """Grey frame; a white square walks left to right (x 200 -> 560) in the left half; one stays still on the right."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    path = tmp_path_factory.mktemp("mover") / "mover.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:size=1280x720:rate=30:duration=9",
         "-f", "lavfi", "-i", "color=c=white:size=40x40:rate=30:duration=9",
         "-filter_complex", "[0][1]overlay=x='180+360*min(t/8,1)':y=280:eof_action=pass,"
                            "drawbox=x=940:y=280:w=40:h=40:color=white:t=fill", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def _white_centroid_x(video: Path, t: float, top: bool = True) -> float:
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(t), "-i", str(video), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    frame = np.frombuffer(raw, dtype=np.uint8).reshape(1920, 1080)
    half = frame[:940] if top else frame[980:]
    _ys, xs = np.nonzero(half > 235)
    return float(xs.mean()) if len(xs) else -1.0


def test_real_render_a_following_pane_keeps_the_mover_centred(mover_video, tmp_path):
    moving = SplitPane(cx=312, cy=300, face_w=80, path=[(2.5, 312.5, 300.0), (8.0, 560.0, 300.0)])
    still = SplitPane(cx=960, cy=300, face_w=80)
    seg_follow = wide_seg(2.5, 9, [moving, still])
    seg_fixed = wide_seg(2.5, 9, [SplitPane(cx=312, cy=300, face_w=80), still])
    followed = render(mover_video, [crop_seg(0, 2.5), seg_follow], tmp_path / "f.mp4")
    fixed = render(mover_video, [crop_seg(0, 2.5), seg_fixed], tmp_path / "s.mp4")
    # the pane window is 336 px wide of the 1280 source -> the mover walks out of a fixed window
    drift_fixed = abs(_white_centroid_x(fixed, 8.0) - _white_centroid_x(fixed, 3.5))
    drift_follow = abs(_white_centroid_x(followed, 8.0) - _white_centroid_x(followed, 3.5))
    assert drift_fixed > 200                                   # a fixed window loses the person
    assert drift_follow < 0.35 * drift_fixed                   # a following window keeps them put
    assert abs(_white_centroid_x(followed, 8.0) - 540) < 140   # and near the middle of the pane
    # the person who does not move is unaffected by the other pane following
    assert abs(_white_centroid_x(followed, 8.0, top=False) - _white_centroid_x(fixed, 8.0, top=False)) < 2
