"""Clip mood (energetic / calm / dramatic) from the model to the clip row and API."""

from __future__ import annotations

import pytest
from autoclip.db import models, schema
from autoclip.providers.base import (
    MOODS,
    POST_CAPTION_PROMPT,
    ClipCandidate,
    load_prompt,
)


def _candidate(**extra) -> ClipCandidate:
    return ClipCandidate.model_validate(
        {"start_word_index": 1, "end_word_index": 50, **extra}
    )


def test_moods_match_the_music_library_folders():
    assert MOODS == ("energetic", "calm", "dramatic")


def test_missing_mood_is_empty():
    assert _candidate().mood == ""


@pytest.mark.parametrize("raw", ["dramatic", "Dramatic", "  DRAMATIC  "])
def test_mood_is_normalised(raw):
    assert _candidate(mood=raw).mood == "dramatic"


@pytest.mark.parametrize("raw", [None, "", "neutral", "sad", 3, ["calm"]])
def test_unknown_mood_becomes_empty_not_an_error(raw):
    # A model slip must not drop the clip; downstream falls back to a heuristic.
    assert _candidate(mood=raw).mood == ""


@pytest.mark.parametrize("version", ["highlight_v1", "highlight_comedy", "highlight_keynote"])
def test_every_prompt_style_asks_for_the_mood(version):
    text = load_prompt(version)
    assert text.endswith(POST_CAPTION_PROMPT)
    assert "`mood`" in text
    for mood in MOODS:
        assert f'"{mood}"' in text
    # it must be about the message, not the speaking speed
    assert "not the speaking" in text


def test_clip_round_trips_through_the_database_and_api(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOCLIP_HOME", str(tmp_path))
    from autoclip import db
    from autoclip.api.schemas import ClipOut
    from autoclip.db import store
    from autoclip.db.models import Job, Source, new_id

    db.init()
    source = store.create_source(
        Source(id=new_id(), type="upload", title="s", path=str(tmp_path / "s.mp4"),
               duration_s=100.0)
    )
    job = store.create_job(Job(id=new_id(), source_id=source.id, provider="groq", settings={}))
    clip = models.Clip(id=new_id(), job_id=job.id, start_s=1, end_s=31, mood="dramatic")
    store.replace_clips(job.id, [clip])

    saved = store.list_clips(job.id)[0]
    assert saved.mood == "dramatic"
    assert ClipOut.of(saved).mood == "dramatic"


def test_build_clips_carries_the_mood():
    import inspect

    from autoclip.pipeline import highlights

    assert "mood=candidate.mood" in inspect.getsource(highlights.build_clips)


def test_migration_adds_the_mood_column():
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    for migration in schema.MIGRATIONS:
        migration(conn)
    assert "mood" in {r["name"] for r in conn.execute("PRAGMA table_info(clips)")}
