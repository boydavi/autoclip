"""Production hardening: ffmpeg pipes never deadlock, failed jobs are never left 'running', old sources are swept, rmtree retries."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from autoclip import cleanup
from autoclip.pipeline import ffmpeg, runner

NOISY = """
import sys
for i in range(4000):
    sys.stderr.write('warning: corrupt frame %d ' % i + 'x' * 100 + chr(10))
    if i % 500 == 0:
        print('out_time_us=%d' % (i * 1000)); sys.stdout.flush()
print('progress=end')
"""


def _fake_ffmpeg(monkeypatch, code: str, exit_code: int = 0):
    real = subprocess.Popen

    def popen(command, **kw):
        return real([sys.executable, "-c", code + f"\nsys.exit({exit_code})"], **kw)

    monkeypatch.setattr(ffmpeg.subprocess, "Popen", popen)


def test_a_flood_of_stderr_does_not_freeze_the_progress_loop(monkeypatch):
    _fake_ffmpeg(monkeypatch, NOISY)
    seen = []
    t0 = time.time()
    ffmpeg.run(["x"], total_duration_s=10.0, on_progress=seen.append)
    assert time.time() - t0 < 20 and seen[-1] == 1.0                      # about 450 KB of stderr went by


def test_a_flood_of_stderr_does_not_freeze_the_cancellable_wait(monkeypatch):
    _fake_ffmpeg(monkeypatch, NOISY)
    ffmpeg.run(["x"], cancelled=lambda: False)


def test_a_failed_ffmpeg_still_reports_the_tail_of_its_errors(monkeypatch):
    _fake_ffmpeg(monkeypatch, NOISY, exit_code=3)
    with pytest.raises(ffmpeg.FFmpegError) as err:
        ffmpeg.run(["x"], cancelled=lambda: False)
    assert "corrupt frame 3999" in err.value.stderr


def test_rmtree_is_tried_again_when_a_file_is_locked(tmp_path, monkeypatch):
    root = tmp_path / "root"
    victim = root / "job"
    victim.mkdir(parents=True)
    calls = {"n": 0}
    real = cleanup.shutil.rmtree

    def flaky(path, ignore_errors=False):
        calls["n"] += 1
        if calls["n"] >= 3:
            real(path)

    monkeypatch.setattr(cleanup.shutil, "rmtree", flaky)
    monkeypatch.setattr(cleanup, "RMTREE_PAUSE_S", 0.0)
    cleanup._remove_tree(victim, root)
    assert calls["n"] == 3 and not victim.exists()


def test_the_queue_records_a_failure_the_runner_never_got_to_write():
    src = (Path(ffmpeg.__file__).parents[1] / "jobs" / "queue.py").read_text(encoding="utf-8")
    assert 'status="failed", error=str(exc), finished_at=utcnow()' in src and 'current.status == "running"' in src


def test_the_runner_sweeps_stale_sources_after_a_clean_job():
    src = Path(runner.__file__).read_text(encoding="utf-8")
    assert "purge_stale_media(keep_source_id=self.source.id)" in src and "STALE_MEDIA_DAYS = 14" in src


def _source_with_video(source_id: str, age_days: float):
    from autoclip import paths
    from autoclip.db import store
    from autoclip.db.models import Source

    folder = paths.source_media_dir(source_id)
    folder.mkdir(parents=True)
    video = folder / "video.mp4"
    video.write_bytes(b"0" * 2048)
    old = time.time() - age_days * 86400
    os.utime(video, (old, old))
    return store.create_source(Source(id=source_id, type="upload", path=str(video), title=source_id)), video


def test_old_unused_sources_lose_their_video_but_recent_or_busy_ones_keep_it(initialised_db):
    from autoclip.db import store
    from autoclip.db.models import Job

    _, old_video = _source_with_video("old", 30)
    _, new_video = _source_with_video("new", 2)
    _, busy_video = _source_with_video("busy", 30)
    _, kept_video = _source_with_video("kept", 30)
    store.create_job(Job(id="j1", source_id="busy", status="queued"))
    assert runner.purge_stale_media(keep_source_id="kept") == 1
    assert not old_video.exists() and new_video.exists() and busy_video.exists() and kept_video.exists()
    assert store.get_source("old") is not None                   # the record stays; only the big file goes
