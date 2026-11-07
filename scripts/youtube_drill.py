"""YouTube drill: real download through Clipper, same-link de-dup, then a job (scan-only).

    .venv\\Scripts\\python.exe youtube_drill.py <youtube-url>

Needs AUTOCLIP_GROQ_KEY in the environment. Downloads at <=360p to keep it small
and prefers YouTube's own captions (falls back to Groq Whisper if there are none).
"""
# ruff: noqa: E501
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # titles can hold characters cp1252 cannot print

EXE = Path(sys.executable).with_name("autoclip.exe")
URL = sys.argv[1]
PORT = 8776
BASE = f"http://127.0.0.1:{PORT}/api"
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def req(method, path, **kw):
    return httpx.request(method, BASE + path, timeout=60, trust_env=False, **kw)


home = Path(tempfile.mkdtemp(prefix="ytdrill_"))
proc = subprocess.Popen([str(EXE), "serve", "--port", str(PORT), "--no-open"],
                        env={**os.environ, "AUTOCLIP_HOME": str(home)},
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(60):
        try:
            req("GET", "/sources")
            break
        except Exception:
            time.sleep(0.5)
    put = req("PUT", "/settings", json={
        "active_provider": "groq", "whisper": {"provider": "groq"},
        "export": {"scan_only": True},
        "ingest": {"ytdlp_format": "bv*[height<=360]+ba/b[height<=360]",
                   "prefer_youtube_captions": True},
    })
    assert put.status_code == 200, put.text

    # 1. download
    t0 = time.time()
    first = req("POST", "/sources/youtube", json={"url": URL})
    check("download request accepted (202)", first.status_code == 202, str(first.status_code))
    sid = first.json()["id"]
    src = {}
    while time.time() - t0 < 2400:
        src = req("GET", f"/sources/{sid}").json()
        if src["ingest_status"] in ("ready", "failed"):
            break
        time.sleep(3)
    took = time.time() - t0
    check("download finished ready", src.get("ingest_status") == "ready",
          f"{src.get('ingest_status')} in {took:.0f}s {(src.get('ingest_error') or '')[:150]}")
    if src.get("ingest_status") != "ready":
        raise SystemExit(1)
    size = Path(next(iter(req("GET", "/sources").json()))["filename"] or "x").name
    files = [p for p in (home).rglob("*") if p.suffix in (".mp4", ".mkv", ".webm") and p.stat().st_size > 1e6]
    mb = sum(p.stat().st_size for p in files) / 1e6
    check("source has duration, title and a video track",
          src["duration_s"] > 60 and bool(src["title"]) and src["has_video"] and src["has_audio"],
          f"{src['duration_s'] / 60:.0f} min, {src['width']}x{src['height']}, {mb:.0f} MB on disk")
    check("download respected the 360p cap", (src.get("height") or 9999) <= 360, f"{src.get('height')}p")

    # 2. same-link de-dup
    t1 = time.time()
    again = req("POST", "/sources/youtube", json={"url": URL})
    check("same link again reuses the download (201, same id)",
          again.status_code == 201 and again.json()["id"] == sid,
          f"HTTP {again.status_code} in {time.time() - t1:.0f}s")
    check("still exactly one source row", len(req("GET", "/sources").json()) == 1)

    # 3. job on the downloaded source
    t2 = time.time()
    job = req("POST", "/jobs", json={"source_id": sid, "settings": {"max_clips": 5}}).json()["id"]
    final = {}
    while time.time() - t2 < 1200:
        final = req("GET", f"/jobs/{job}").json()
        if final["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(1)
    jt = time.time() - t2
    check("job on the downloaded source is done", final["status"] == "done",
          f"{final['status']} in {jt:.0f}s {(final.get('error') or '')[:150]}")
    clips = req("GET", f"/jobs/{job}/clips").json()
    check("clips produced", len(clips) >= 1, f"{len(clips)} clip(s)")
    check("clips inside the video", all(0 <= c["start_s"] < c["end_s"] <= src["duration_s"] + 1 for c in clips))
    whisper_used = any((home / "work" / job).rglob("*.groq.json")) if (home / "work" / job).exists() else False
    print(f"      transcript route: {'Groq Whisper' if whisper_used else 'YouTube captions (no Whisper call)'}", flush=True)
    for c in clips:
        print(f"      clip {c['start_s']:7.1f}-{c['end_s']:7.1f}  {c['title'][:70]}", flush=True)
finally:
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    proc.wait()
    shutil.rmtree(home, ignore_errors=True)

print(f"\n=== {sum(results)}/{len(results)} passed ===")
sys.exit(0 if all(results) else 1)
