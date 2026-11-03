"""Repeat drill: same source twice -> second run must reuse the transcript (no Whisper call).

Needs AUTOCLIP_GROQ_KEY in the environment (~$0.01):

    .venv\\Scripts\\python.exe scripts\\repeat_drill.py path\\to\\10min.mp4
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

EXE = Path(sys.executable).with_name("autoclip.exe")
MEDIA = Path(sys.argv[1])
PORT = 8775
BASE = f"http://127.0.0.1:{PORT}/api"
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def req(method, path, **kw):
    return httpx.request(method, BASE + path, timeout=30, trust_env=False, **kw)


home = Path(tempfile.mkdtemp(prefix="rdrill_"))
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
    assert req("PUT", "/settings", json={"active_provider": "groq",
              "whisper": {"provider": "groq"}, "export": {"scan_only": True}}).status_code == 200
    with MEDIA.open("rb") as fh:
        src = req("POST", "/sources/upload", files={"file": (MEDIA.name, fh, "video/mp4")}).json()["id"]

    def run_job():
        job = req("POST", "/jobs", json={"source_id": src, "settings": {"max_clips": 3}}).json()["id"]
        t0, stages = time.time(), {}
        while True:
            j = req("GET", f"/jobs/{job}").json()
            stages.setdefault(j["current_stage"], time.time() - t0)
            if j["status"] in ("done", "failed", "cancelled"):
                return job, j, time.time() - t0, stages
            time.sleep(0.25)

    def whisper_cache_files(job):
        return list((home / "work" / job).rglob("*.groq.json")) if (home / "work" / job).exists() else []

    a, ja, ta, sa = run_job()
    b, jb, tb, sb = run_job()
    print(f"run A: {ja['status']} {ta:.1f}s   run B: {jb['status']} {tb:.1f}s", flush=True)
    check("run A finished", ja["status"] == "done", f"{ta:.1f}s")
    check("run B (same source) finished", jb["status"] == "done", f"{tb:.1f}s")
    check("run A called Groq Whisper (chunk cache file written)", len(whisper_cache_files(a)) >= 1,
          f"{len(whisper_cache_files(a))} file(s)")
    check("run B made NO Whisper call (no chunk cache file)", len(whisper_cache_files(b)) == 0,
          f"{len(whisper_cache_files(b))} file(s)")
    check("run B is faster than run A", tb < ta, f"{tb:.1f}s vs {ta:.1f}s")
    ca = req("GET", f"/jobs/{a}/clips").json()
    cb = req("GET", f"/jobs/{b}/clips").json()
    check("both runs produced clips", len(ca) >= 1 and len(cb) >= 1, f"A={len(ca)} B={len(cb)}")
    # same transcript -> same word indices should be reachable; allow LLM variation
    check("clips come from the same transcript (inside the same time range)",
          all(0 <= c["start_s"] < c["end_s"] <= 601 for c in cb), "")
    srcs = req("GET", "/sources").json()
    check("still exactly one source row", len(srcs) == 1, f"{len(srcs)}")
finally:
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    proc.wait()
    shutil.rmtree(home, ignore_errors=True)

print(f"\n=== {sum(results)}/{len(results)} passed ===")
sys.exit(0 if all(results) else 1)
