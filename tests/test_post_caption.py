"""Post caption (description + hashtags) from the model to the clip row and API."""

from __future__ import annotations

import pytest
from autoclip.db import models, schema
from autoclip.providers.base import (
    POST_CAPTION_PROMPT,
    ClipCandidate,
    extract_json_object,
    load_prompt,
)


def _candidate(**extra) -> ClipCandidate:
    return ClipCandidate.model_validate(
        {"start_word_index": 1, "end_word_index": 50, "title": "t", **extra}
    )


def test_missing_fields_default_to_empty():
    candidate = _candidate()
    assert candidate.description == ""
    assert candidate.hashtags == []


@pytest.mark.parametrize(
    "raw",
    [
        ["#startups", "growth"],
        "startups, #growth",
        "#startups #growth",
        ["startups", "#Startups", "growth", ""],
    ],
)
def test_hashtags_are_normalised(raw):
    assert _candidate(hashtags=raw).hashtags == ["#startups", "#growth"]


def test_hashtags_null_and_capped():
    assert _candidate(hashtags=None).hashtags == []
    assert len(_candidate(hashtags=[f"t{i}" for i in range(20)]).hashtags) == 8


def test_description_null_becomes_empty_string():
    assert _candidate(description=None).description == ""


def test_fenced_reply_with_caption_parses():
    payload = extract_json_object(
        '```json\n{"clips": [{"start_word_index": 1, "end_word_index": 9, '
        '"description": "Why it fails.", "hashtags": ["#a", "#b"]}]}\n```'
    )
    clip = ClipCandidate.model_validate(payload["clips"][0])
    assert (clip.description, clip.hashtags) == ("Why it fails.", ["#a", "#b"])


@pytest.mark.parametrize("version", ["highlight_v1", "highlight_comedy", "highlight_keynote"])
def test_every_prompt_style_asks_for_the_caption(version):
    text = load_prompt(version)
    assert text.endswith(POST_CAPTION_PROMPT)
    assert "`description`" in text and "`hashtags`" in text


def test_post_caption_joins_description_and_tags():
    clip = models.Clip(id="c", job_id="j", start_s=0, end_s=30)
    assert clip.post_caption == ""
    clip.description = "Why it fails."
    assert clip.post_caption == "Why it fails."
    clip.hashtags = "#a #b"
    assert clip.post_caption == "Why it fails.\n\n#a #b"


def test_clip_round_trips_through_the_database(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOCLIP_HOME", str(tmp_path))
    from autoclip import db
    from autoclip.db import store
    from autoclip.db.models import Job, Source, new_id

    db.init()
    source = store.create_source(
        Source(id=new_id(), type="upload", title="s", path=str(tmp_path / "s.mp4"),
               duration_s=100.0)
    )
    job = store.create_job(Job(id=new_id(), source_id=source.id, provider="groq", settings={}))
    clip = models.Clip(
        id=new_id(), job_id=job.id, start_s=1, end_s=31,
        description="Why it fails.", hashtags="#a #b",
    )
    store.replace_clips(job.id, [clip])

    saved = store.list_clips(job.id)[0]
    assert (saved.description, saved.hashtags) == ("Why it fails.", "#a #b")

    from autoclip.api.schemas import ClipOut

    out = ClipOut.of(saved)
    assert (out.description, out.hashtags) == ("Why it fails.", "#a #b")


def test_migration_adds_columns_to_an_existing_database():
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    for migration in schema.MIGRATIONS[:3]:
        migration(conn)
    schema.MIGRATIONS[3](conn)
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(clips)")}
    assert {"description", "hashtags"} <= columns
