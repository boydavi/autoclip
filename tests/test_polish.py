"""Small correctness fixes: filenames, SSE terminal events, key redaction, cancellable analysis."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time

import pytest
from autoclip.api.settings import redact_secrets
from autoclip.jobs.events import QUEUE_SIZE, Event, EventBroker
from autoclip.pipeline import export, ffmpeg


class TestRedactSecrets:
    def test_hides_a_masked_key_echoed_by_a_provider(self) -> None:
        message = (
            "Error code: 401 - {'error': {'message': 'Incorrect API key provided: "
            "gsk_AbCd**************wxyz. You can find your key.'}}"
        )

        redacted = redact_secrets(message)

        assert "gsk_AbCd" not in redacted
        assert "wxyz" not in redacted
        assert "Error code: 401" in redacted

    def test_hides_a_bare_key_shaped_token(self) -> None:
        assert "sk-proj-abc123def456" not in redact_secrets("bad key sk-proj-abc123def456 sent")

    def test_leaves_ordinary_messages_alone(self) -> None:
        text = "Not running at http://localhost:11434 (connection refused)"

        assert redact_secrets(text) == text


class TestOutputFilename:
    def test_equal_titles_get_distinct_names_when_ranked(self) -> None:
        first = export.output_filename("Big Idea", "9:16", rank=1)
        second = export.output_filename("Big Idea", "9:16", rank=2)

        assert first != second
        assert first == "big-idea-01_9x16.mp4"

    def test_unranked_name_is_unchanged(self) -> None:
        assert export.output_filename("Big Idea", "1:1") == "big-idea_1x1.mp4"


class TestTerminalEventsSurviveBackpressure:
    def _broker_with_full_queue(self) -> tuple[EventBroker, asyncio.Queue]:
        broker = EventBroker()
        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        for i in range(QUEUE_SIZE):
            queue.put_nowait(Event(type="progress", job_id="j", data={"n": i}))
        broker._subscribers["j"] = {queue}
        return broker, queue

    def test_done_event_is_delivered_when_the_queue_is_full(self) -> None:
        broker, queue = self._broker_with_full_queue()

        broker._publish_on_loop(Event(type="done", job_id="j"))

        drained = [queue.get_nowait() for _ in range(queue.qsize())]
        assert drained[-1].type == "done"
        assert len(drained) == QUEUE_SIZE

    def test_progress_event_is_still_dropped_when_full(self) -> None:
        broker, queue = self._broker_with_full_queue()

        broker._publish_on_loop(Event(type="progress", job_id="j", data={"n": "late"}))

        drained = [queue.get_nowait() for _ in range(queue.qsize())]
        assert all(event.data.get("n") != "late" for event in drained)


class TestCancellableSubprocess:
    SLEEP = [sys.executable, "-c", "import time; time.sleep(30)"]

    def test_cancel_kills_the_process_promptly(self) -> None:
        started = time.monotonic()

        with pytest.raises(ffmpeg.Cancelled):
            export._run_for_stderr(self.SLEEP, timeout_s=60, cancelled=lambda: True)

        assert time.monotonic() - started < 5

    def test_timeout_still_raises(self) -> None:
        with pytest.raises(subprocess.TimeoutExpired):
            export._run_for_stderr(self.SLEEP, timeout_s=1, cancelled=None)

    def test_returns_stderr_on_success(self) -> None:
        command = [sys.executable, "-c", "import sys; sys.stderr.write('measured')"]

        assert export._run_for_stderr(command, timeout_s=30, cancelled=None) == "measured"
