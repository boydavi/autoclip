"""Job progress reporting never moves backwards."""

from __future__ import annotations

import pytest
from autoclip.config import Settings
from autoclip.db import store
from autoclip.db.models import Job, Source, new_id
from autoclip.pipeline import Stage
from autoclip.pipeline.runner import PipelineRunner


@pytest.fixture
def runner_and_events(initialised_db):
    source = store.create_source(Source(id=new_id(), type="upload", path="x.mp4", title="t"))
    job = store.create_job(Job(id=new_id(), source_id=source.id, provider="p", settings={}))
    events: list = []
    runner = PipelineRunner(job, source, settings=Settings(), on_progress=events.append)
    return runner, events, job


def test_a_dip_in_the_underlying_feed_is_not_reported(runner_and_events) -> None:
    runner, events, job = runner_and_events
    report = runner._stage_progress(Stage.EXPORT)

    # The exact shape seen on real footage: ffmpeg's clock dips before the end.
    for fraction in (0.5, 0.9556, 0.9771, 0.9672, 1.0):
        report(fraction)

    assert [e.stage_progress for e in events] == [0.5, 0.9556, 0.9771, 0.9771, 1.0]
    overall = [e.overall for e in events]
    assert overall == sorted(overall)
    assert store.get_job(job.id).progress == pytest.approx(overall[-1], abs=1e-4)


def test_progress_is_monotonic_across_stage_boundaries(runner_and_events) -> None:
    runner, events, _ = runner_and_events

    runner._stage_progress(Stage.PREPARE)(0.6)
    runner._finish_stage(Stage.PREPARE)
    runner._stage_progress(Stage.TRANSCRIBE)(0.0)
    runner._stage_progress(Stage.TRANSCRIBE)(0.4)
    runner._stage_progress(Stage.TRANSCRIBE)(0.2)
    runner._finish_stage(Stage.TRANSCRIBE)

    overall = [e.overall for e in events]
    assert overall == sorted(overall)
    assert overall[-1] == pytest.approx(0.40)


def test_progress_is_clamped_to_one(runner_and_events) -> None:
    runner, events, _ = runner_and_events

    runner._stage_progress(Stage.EXPORT)(5.0)

    assert events[-1].stage_progress == 1.0
    assert events[-1].overall <= 1.0
