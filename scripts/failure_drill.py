"""Failure drills: bad key, no network, cancel mid-job, restart mid-job.

Each scenario gets its own throwaway AUTOCLIP_HOME and its own server process.
Scenarios 1-2 use a fake key / dead proxy and cost nothing; 3-4 use the real key
from the environment (never printed) on a 10-minute file (~$0.008 each).

    .venv\\Scripts\\python.exe scripts\\failure_drill.py path\\to\\10min.mp4
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
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""), flush=True)


class Server:
    def __init__(self, port: int, home: Path, env_extra: dict[str, str]):
        self.port, self.home = port, home
        env = {**os.environ, "AUTOCLIP_HOME": str(home), **env_extra}
        self.env = env
        self.proc: subprocess.Popen | None = None
        self.base = f"http://127.0.0.1:{port}/api"

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [str(EXE), "serve", "--port", str(self.port), "--no-open"],
            env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(60):
            try:
                httpx.get(f"{self.base}/sources", timeout=2, trust_env=False)
                return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("server did not start")

    def kill(self) -> None:
        if self.proc:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)],
                           capture_output=True)
            self.proc.wait()
            self.proc = None

    def req(self, method: str, path: str, **kw):
        return httpx.request(method, self.base + path, timeout=30, trust_env=False, **kw)


def setup(server: Server) -> str:
    r = server.req("PUT", "/settings", json={
        "active_provider": "groq", "whisper": {"provider": "groq"},
    })
    assert r.status_code == 200, r.text
    with MEDIA.open("rb") as fh:
        up = server.req("POST", "/sources/upload", files={"file": (MEDIA.name, fh, "video/mp4")})
    assert up.status_code == 201, up.text
    return up.json()["id"]


def submit(server: Server, source_id: str) -> str:
    r = server.req("POST", "/jobs", json={"source_id": source_id, "settings": {"max_clips": 2}})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def wait_for(server: Server, job_id: str, pred, timeout: float):
    start, last = time.time(), {}
    while time.time() - start < timeout:
        last = server.req("GET", f"/jobs/{job_id}").json()
        if pred(last):
            return last, time.time() - start
        time.sleep(0.5)
    return last, time.time() - start


def terminal(j: dict) -> bool:
    return j["status"] in ("done", "failed", "cancelled")


def scenario(name: str, port: int, env_extra: dict[str, str]):
    print(f"\n--- {name} ---", flush=True)
    home = Path(tempfile.mkdtemp(prefix="fdrill_"))
    server = Server(port, home, env_extra)
    server.start()
    return server, home


def cleanup(server: Server, home: Path) -> None:
    server.kill()
    shutil.rmtree(home, ignore_errors=True)


# ---------------------------------------------------------------- 1. bad key
server, home = scenario("1. bad key", 8771, {"AUTOCLIP_GROQ_KEY": "gsk_fake_key_for_drill"})
try:
    src = setup(server)
    job = submit(server, src)
    final, took = wait_for(server, job, terminal, 120)
    err = (final.get("error") or "")
    check("bad key: job fails (not hangs)", final["status"] == "failed", f"{final['status']} in {took:.0f}s")
    check("bad key: fails fast (< 60 s)", took < 60, f"{took:.0f}s")
    check("bad key: error names the key problem",
          any(w in err.lower() for w in ("api key", "401", "rejected", "invalid", "unauthor")),
          err.replace("\n", " ")[:140])
    # the worker must not be wedged: a second job is accepted and also fails cleanly
    job2 = submit(server, src)
    final2, _ = wait_for(server, job2, terminal, 120)
    check("bad key: queue not wedged (2nd job ends)", final2["status"] == "failed", final2["status"])
finally:
    cleanup(server, home)

# ------------------------------------------------------------- 2. no network
dead = "http://127.0.0.1:9"
server, home = scenario("2. no network", 8772, {
    "AUTOCLIP_GROQ_KEY": "gsk_fake_key_for_drill",
    "HTTPS_PROXY": dead, "HTTP_PROXY": dead, "ALL_PROXY": dead,
    "https_proxy": dead, "http_proxy": dead, "all_proxy": dead,
})
try:
    src = setup(server)
    job = submit(server, src)
    final, took = wait_for(server, job, terminal, 400)
    err = (final.get("error") or "")
    check("no network: job fails (not hangs)", final["status"] == "failed", f"{final['status']} in {took:.0f}s")
    check("no network: gives up in bounded time (< 300 s)", took < 300, f"{took:.0f}s")
    check("no network: error says connection problem",
          any(w in err.lower() for w in ("connect", "network", "reach", "proxy", "timed out")),
          err.replace("\n", " ")[:140])
finally:
    cleanup(server, home)

# ----------------------------------------------------------- 3. cancel mid-job
server, home = scenario("3. cancel mid-job", 8773, {})
try:
    src = setup(server)
    job = submit(server, src)
    j, _ = wait_for(server, job, lambda j: j["status"] == "running" and
                    j["current_stage"] in ("reframe", "captions", "export"), 240)
    check("cancel: job reached a late stage", j["current_stage"] in ("reframe", "captions", "export"),
          f"{j['status']}/{j['current_stage']}")
    server.req("POST", f"/jobs/{job}/cancel")
    final, took = wait_for(server, job, terminal, 60)
    check("cancel: ends as cancelled within 60 s", final["status"] == "cancelled", f"{final['status']} in {took:.0f}s")
    job2 = submit(server, src)
    final2, took2 = wait_for(server, job2, terminal, 400)
    check("cancel: next job runs to done", final2["status"] == "done", f"{final2['status']} in {took2:.0f}s")
finally:
    cleanup(server, home)

# ----------------------------------------------------- 4. restart mid-job
server, home = scenario("4. restart mid-job", 8774, {})
try:
    src = setup(server)
    job = submit(server, src)
    j, _ = wait_for(server, job, lambda j: j["status"] == "running" and
                    j["current_stage"] in ("reframe", "captions", "export"), 240)
    check("restart: job reached a late stage before the kill", j["current_stage"] in ("reframe", "captions", "export"),
          f"{j['status']}/{j['current_stage']}")
    server.kill()
    time.sleep(2)
    server.start()
    final, took = wait_for(server, job, terminal, 400)
    check("restart: interrupted job is resumed to done", final["status"] == "done",
          f"{final['status']} in {took:.0f}s {(final.get('error') or '')[:80]}")
    clips = server.req("GET", f"/jobs/{job}/clips").json()
    exports = [e for c in clips for e in c.get("exports", []) if e["status"] == "done"]
    check("restart: clips present", len(clips) >= 1, f"{len(clips)} clip(s)")
    check("restart: no duplicate exports (<= 1 per clip)",
          all(sum(1 for e in c.get("exports", []) if e["status"] == "done") <= 1 for c in clips),
          f"{len(exports)} export(s)")
finally:
    cleanup(server, home)

failed = [r for r in results if not r[1]]
print(f"\n=== {len(results) - len(failed)}/{len(results)} passed ===")
sys.exit(1 if failed else 0)
