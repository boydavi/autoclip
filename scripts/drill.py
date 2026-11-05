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
import json
import logging
import os
import subprocess
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


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format",
         str(path)],
        capture_output=True, text=True, check=False,
    ).stdout
    return json.loads(out) if out else {}


def _render_checks(clips) -> list[tuple[str, bool, str]]:
    from autoclip.db import store

    rows = []
    for clip in clips:
        done = [e for e in store.list_exports(clip.id) if e.status == "done"]
        rows.append((clip, done[0] if done else None))
    files = [(c, Path(e.path)) for c, e in rows if e is not None]
    info = [(c, p, _probe(p)) for c, p in files if p.exists()]

    def video(meta):
        return next((s for s in meta.get("streams", []) if s["codec_type"] == "video"), {})

    def has_audio(meta):
        return any(s["codec_type"] == "audio" for s in meta.get("streams", []))

    def duration_ok(clip, meta):
        return abs(float(meta["format"]["duration"]) - clip.duration_s) <= 1.5

    return [
        ("every clip exported", bool(rows) and len(files) == len(rows),
         f"{len(files)}/{len(rows)}"),
        ("export files exist and are non-empty",
         bool(files) and all(p.exists() and p.stat().st_size > 50_000 for _, p in files),
         ", ".join(f"{p.stat().st_size / 1e6:.1f}MB" for _, p in files if p.exists())),
        ("output is 1080x1920 (9:16)",
         bool(info) and all(
             (video(m).get("width"), video(m).get("height")) == (1080, 1920) for *_, m in info
         ), ""),
        ("output has an audio track", bool(info) and all(has_audio(m) for *_, m in info), ""),
        ("output duration matches the clip (+/-1.5s)",
         bool(info) and all(duration_ok(c, m) for c, _, m in info), ""),
        ("post caption .txt written beside each clip",
         bool(files) and all(
             p.with_suffix(".txt").exists() or not c.post_caption for c, p in files
         ) and any(p.with_suffix(".txt").exists() for _, p in files),
         ""),
    ]


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
    parser.add_argument(
        "--render", action="store_true",
        help="also export the clips (CPU heavy) and verify the output files",
    )
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
    settings.export.scan_only = not args.render
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
    if args.render:
        checks += _render_checks(clips)

    print("\n=== DRILL RESULT ===")
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    print(f"\nTotal {total:.1f}s  |  stages: " +
          ", ".join(f"{k} {v[1] - v[0]:.1f}s" for k, v in spans.items()))
    retries = [w for w in warnings if "retrying" in w]
    print(f"Groq/LLM retries seen: {len(retries)}")
    if final is not None:
        print(
            f"Groq usage: {final.llm_calls} LLM call(s), {final.llm_input_tokens:,} in / "
            f"{final.llm_output_tokens:,} out tokens, {final.transcribe_seconds / 60:.1f} min Whisper audio, "
            f"estimated ${final.est_cost_usd:.3f} (pre-run guess ${est:.3f}). The Groq console is exact."
        )
    for w in warnings[:15]:
        print("  warn:", w[:200])
    for c in clips:
        print(f"  clip {c.start_s:7.1f}-{c.end_s:7.1f}  [{c.mood or '-':9}] {c.title[:60]}")
        if c.description or c.hashtags:
            print(f"      {c.description[:110]}  {c.hashtags}")
        for n, title in enumerate(c.post_title_list, 1):
            print(f"      title option {n}: {title}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
