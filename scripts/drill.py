"""Live drill: Groq Whisper + Claude Haiku on a real media file (costs real money).

Keys are read from the environment and never printed:

    $env:AUTOCLIP_GROQ_KEY = "..."        # transcription
    $env:AUTOCLIP_ANTHROPIC_KEY = "..."   # clip selection
    .venv\\Scripts\\python.exe scripts\\drill.py path\\to\\talk.mp4 --max-minutes 10

It refuses sources longer than --max-minutes so a wrong file cannot run up a
bill, prints the estimated cost before spending anything, and reports a
pass/fail table with per-stage timings. Runs in a throwaway AUTOCLIP_HOME.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("media", type=Path)
    parser.add_argument("--max-minutes", type=float, default=15.0, help="spending guard")
    parser.add_argument(
        "--llm", choices=("groq", "anthropic"), default="groq",
        help="clip-selection provider (groq needs only AUTOCLIP_GROQ_KEY)",
    )
    parser.add_argument("--model", default=None, help="LLM model (default per provider)")
    parser.add_argument("--groq-model", default="whisper-large-v3-turbo")
    parser.add_argument("--max-clips", type=int, default=5)
    args = parser.parse_args()

    keys = ["AUTOCLIP_GROQ_KEY"] + (["AUTOCLIP_ANTHROPIC_KEY"] if args.llm == "anthropic" else [])
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        print("Set these in your own shell first (values are never printed): " + ", ".join(missing))
        return 2
    if not args.media.is_file():
        print(f"Not a file: {args.media}")
        return 2

    os.environ["AUTOCLIP_HOME"] = tempfile.mkdtemp(prefix="drill_home_")

    from autoclip import db
    from autoclip.api.estimate import _llm_per_window, _whisper_cost, _window_count
    from autoclip.config import Settings
    from autoclip.db import store
    from autoclip.db.models import Job, new_id
    from autoclip.pipeline import ingest
    from autoclip.pipeline.runner import PipelineRunner
    from autoclip.pipeline.transcript import Transcript

    db.init()
    source = store.create_source(ingest.ingest_file(args.media, title="drill"))
    minutes = source.duration_s / 60
    if minutes > args.max_minutes:
        print(f"REFUSED: source is {minutes:.1f} min, guard is {args.max_minutes:.1f} min "
              "(raise --max-minutes if intended).")
        return 2

    settings = Settings()
    model = args.model or ("openai/gpt-oss-120b" if args.llm == "groq" else "claude-haiku-5-5")
    settings.active_provider = args.llm
    settings.providers[args.llm].model = model
    settings.whisper.provider = "groq"
    settings.whisper.groq_model = args.groq_model
    settings.export.scan_only = True
    settings.clips.max_clips = args.max_clips

    windows = _window_count(source.duration_s)
    est = _whisper_cost("groq", args.groq_model, source.duration_s) + windows * _llm_per_window(
        args.llm, model
    )
    print(f"Source {minutes:.1f} min, {windows} LLM window(s). Estimated cost ~${est:.3f}.")

    capture = _Capture()
    logging.getLogger().addHandler(capture)
    logging.getLogger().setLevel(logging.WARNING)

    job = store.create_job(Job(id=new_id(), source_id=source.id, provider=args.llm, settings={}))
    marks: list[tuple[str, float]] = []
    start = time.perf_counter()
    error = None
    try:
        asyncio.run(
            PipelineRunner(
                job, source, settings=settings,
                on_progress=lambda e: marks.append((e.stage.value, time.perf_counter())),
            ).run()
        )
    except Exception as exc:  # noqa: BLE001 - the drill reports every failure
        error = exc
    total = time.perf_counter() - start

    spans: dict[str, tuple[float, float]] = {}
    prev = start
    for stage, t in marks:
        first, _ = spans.get(stage, (prev, t))
        spans[stage] = (first, t)
        prev = t

    final = store.get_job(job.id)
    clips = store.list_clips(job.id)
    transcript = None
    row = store.get_transcript_for_source(source.id)
    if row is not None and Path(row.json_path).exists():
        transcript = Transcript.load(Path(row.json_path))
    spoken_end = transcript.words[-1].end if transcript and transcript.words else 0.0
    cfg = settings.clips
    warnings = capture.lines

    checks = [
        ("job finished without exception", error is None, repr(error) if error else ""),
        ("job status is done", final is not None and final.status == "done",
         getattr(final, "status", "?")),
        ("transcript has words", bool(transcript and transcript.words),
         f"{len(transcript.words) if transcript else 0} words"),
        ("at least one clip proposed", len(clips) >= 1, f"{len(clips)} clip(s)"),
        ("clips inside the spoken range",
         bool(clips) and all(0 <= c.start_s < c.end_s <= spoken_end + 1 for c in clips),
         f"spoken end {spoken_end:.0f}s"),
        ("clip lengths within configured bounds",
         bool(clips) and all(
             cfg.min_duration_s - 1 <= c.end_s - c.start_s <= cfg.max_duration_s + 1 for c in clips
         ), f"{cfg.min_duration_s:.0f}-{cfg.max_duration_s:.0f}s"),
        ("no LLM window failed", not any("LLM windows failed" in w or "failed (" in w
                                         for w in warnings), ""),
    ]

    print("\n=== DRILL RESULT ===")
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    print(f"\nTotal {total:.1f}s  |  stages: " +
          ", ".join(f"{k} {v[1] - v[0]:.1f}s" for k, v in spans.items()))
    retries = [w for w in warnings if "retrying" in w]
    print(f"Groq/LLM retries seen: {len(retries)}")
    for w in warnings[:15]:
        print("  warn:", w[:200])
    for c in clips:
        print(f"  clip {c.start_s:7.1f}-{c.end_s:7.1f}  {getattr(c, 'title', '')[:70]}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
