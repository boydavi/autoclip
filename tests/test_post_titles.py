"""Viewer-facing post titles (three options) and the CTA + link line in the post text."""

from __future__ import annotations

import inspect

import pytest
from autoclip import db
from autoclip.api import jobs as jobs_api
from autoclip.api.schemas import ClipOut, JobSettingsIn
from autoclip.config import Settings
from autoclip.db import models, store
from autoclip.db.models import Job, Source, new_id
from autoclip.pipeline import runner
from autoclip.postcaption import compose_post_text
from autoclip.providers.base import ClipCandidate, load_prompt


def _candidate(**extra) -> ClipCandidate:
    return ClipCandidate.model_validate({"start_word_index": 1, "end_word_index": 50, "title": "t", **extra})


# ------------------------------------------------------------------ the model's titles


def test_titles_default_to_empty():
    assert _candidate().post_titles == []
    assert _candidate(post_titles=None).post_titles == []


def test_numbering_quotes_and_duplicates_are_cleaned_and_three_are_kept():
    raw = '1. "Stop waiting for permission"\n2) Stop waiting for permission\n- Why does it stall?\n* Before you cut your price:\nA fifth'
    assert _candidate(post_titles=raw).post_titles == [
        "Stop waiting for permission", "Why does it stall?", "Before you cut your price:",
    ]


def test_a_list_works_and_overlong_titles_are_dropped():
    long_title = "x" * 101
    assert _candidate(post_titles=["Good one", long_title, "", "  Second  "]).post_titles == ["Good one", "Second"]


@pytest.mark.parametrize("version", ["highlight_v1", "highlight_comedy", "highlight_keynote"])
def test_every_prompt_style_asks_for_three_hook_titles_that_stay_true(version):
    text = load_prompt(version)
    assert "`post_titles`" in text and "exactly three" in text
    assert "invent no numbers, names or claims" in text
    assert "four extra fields" in text


# ------------------------------------------------------------------ the text layout


def test_the_post_text_layout_has_every_part_in_order():
    text = compose_post_text(
        titles=["Stop waiting", "Why does it stall?"], description="A short why.", cta="Follow for more.",
        link="https://example.com/start", hashtags="#a #b",
    )
    assert text == (
        "TITLE OPTIONS\n1. Stop waiting\n2. Why does it stall?\n\nA short why.\n\n"
        "Follow for more.\nhttps://example.com/start\n\n#a #b"
    )


def test_empty_parts_leave_no_blank_lines_or_headers():
    assert compose_post_text(description="Only this.") == "Only this."
    assert compose_post_text(titles=["T"], hashtags="#a") == "TITLE OPTIONS\n1. T\n\n#a"
    assert compose_post_text(link="https://example.com") == "https://example.com"
    assert compose_post_text() == ""


def test_the_clip_composes_its_own_post_text_with_the_job_cta_and_link():
    clip = models.Clip(
        id="c", job_id="j", start_s=0, end_s=30, description="Why it fails.", hashtags="#x",
        post_titles="One\nTwo",
    )
    assert clip.post_title_list == ["One", "Two"]
    assert clip.post_text("Follow.", "https://e.com") == "TITLE OPTIONS\n1. One\n2. Two\n\nWhy it fails.\n\nFollow.\nhttps://e.com\n\n#x"
    # the older caption (description + hashtags) is unchanged
    assert clip.post_caption == "Why it fails.\n\n#x"


def test_the_export_writes_the_full_post_text():
    assert "clip.post_text(self.settings.post.cta, self.settings.post.link, episode)" in inspect.getsource(runner)


# ------------------------------------------------------------------ settings and storage


def test_cta_and_link_default_to_nothing_and_can_be_overridden_per_job():
    assert (Settings().post.cta, Settings().post.link) == ("", "")
    merged = jobs_api._apply_overrides(Settings(), JobSettingsIn(cta="  Follow for more.  ", link="https://e.com"))
    assert (merged.post.cta, merged.post.link) == ("Follow for more.", "https://e.com")
    untouched = jobs_api._apply_overrides(Settings(), JobSettingsIn())
    assert (untouched.post.cta, untouched.post.link) == ("", "")


def test_titles_survive_the_database_and_reach_the_api(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOCLIP_HOME", str(tmp_path))
    db.init()
    source = store.create_source(Source(id=new_id(), type="upload", title="s", path=str(tmp_path / "s.mp4"), duration_s=100.0))
    job = store.create_job(Job(id=new_id(), source_id=source.id, provider="groq", settings={}))
    store.replace_clips(job.id, [models.Clip(id=new_id(), job_id=job.id, start_s=1, end_s=31, post_titles="A\nB\nC")])
    saved = store.list_clips(job.id)[0]
    assert saved.post_title_list == ["A", "B", "C"]
    assert ClipOut.of(saved).post_titles == ["A", "B", "C"]


def test_the_migration_adds_the_titles_column():
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    for migration in db.schema.MIGRATIONS:
        migration(conn)
    assert "post_titles" in {r["name"] for r in conn.execute("PRAGMA table_info(clips)")}


# ------------------------------------------------------------------ the "Full episode" line


@pytest.mark.parametrize(
    "url",
    [
        "https://youtu.be/5oY9TnaX430",
        "https://www.youtube.com/watch?v=5oY9TnaX430",
        "https://www.youtube.com/watch?list=PL1&v=5oY9TnaX430&t=5",
        "https://www.youtube.com/shorts/5oY9TnaX430",
        "https://www.youtube.com/live/5oY9TnaX430?si=abc",
    ],
)
def test_a_youtube_source_links_to_the_moment_the_clip_starts(url):
    from autoclip.postcaption import episode_line

    assert episode_line(url, 1579.99) == "Full episode: https://youtu.be/5oY9TnaX430?t=1579"


def test_other_sources_link_as_they_are_and_no_source_means_no_line():
    from autoclip.postcaption import episode_line

    assert episode_line("https://example.com/ep/12", 90) == "Full episode: https://example.com/ep/12"
    assert episode_line("", 90) == "" and episode_line(None, 90) == ""
    assert episode_line("https://youtu.be/5oY9TnaX430", -3) == "Full episode: https://youtu.be/5oY9TnaX430?t=0"


def test_the_episode_line_sits_with_the_cta_and_link():
    # the same literal is pinned in core/tests/test_post_text.py
    text = compose_post_text(
        titles=["Stop waiting"], description="A short why.", cta="Follow for more.", link="https://example.com/start",
        episode="Full episode: https://youtu.be/5oY9TnaX430?t=217", hashtags="#a",
    )
    assert text == (
        "TITLE OPTIONS\n1. Stop waiting\n\nA short why.\n\nFollow for more.\nhttps://example.com/start\n"
        "Full episode: https://youtu.be/5oY9TnaX430?t=217\n\n#a"
    )


def test_the_clip_and_the_settings_carry_the_episode_option():
    clip = models.Clip(id="c", job_id="j", start_s=217, end_s=240, description="D")
    assert clip.post_text(episode="Full episode: E") == "D\n\nFull episode: E"
    assert Settings().post.episode_link is False
    merged = jobs_api._apply_overrides(Settings(), JobSettingsIn(episode_link=True))
    assert merged.post.episode_link is True
    src = inspect.getsource(runner)
    assert "episode_line(self.source.url, clip.start_s) if self.settings.post.episode_link" in src
