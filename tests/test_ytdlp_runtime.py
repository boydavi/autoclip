"""yt-dlp needs a JavaScript runtime plus its solver to read all YouTube formats; use Node when it is the one installed."""

import shutil

from autoclip.pipeline import ingest


def _which(found: set):
    return lambda name: f"C:/bin/{name}.exe" if name in found else None


def test_node_is_used_when_it_is_the_only_runtime(monkeypatch):
    monkeypatch.setattr(shutil, "which", _which({"node"}))
    assert ingest._js_runtime_options() == {"js_runtimes": {"node": {}}}


def test_deno_is_left_to_yt_dlp_and_no_runtime_adds_nothing(monkeypatch):
    monkeypatch.setattr(shutil, "which", _which({"deno", "node"}))
    assert ingest._js_runtime_options() == {}                    # yt-dlp enables deno on its own
    monkeypatch.setattr(shutil, "which", _which(set()))
    assert ingest._js_runtime_options() == {}


def test_every_yt_dlp_options_block_gets_the_runtime():
    import inspect
    src = inspect.getsource(ingest)
    assert src.count("options.update(_js_runtime_options())") == 3       # captions, download, and the playlist/batch path
