"""Removing a job's or source's files from disk.

Deletion is deliberately conservative: only paths that resolve inside AutoClip's
own ``work``, ``exports`` and ``media`` folders are ever removed, so a source
that points at a file elsewhere on the machine is never touched.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from . import paths

log = logging.getLogger(__name__)


def _is_inside(path: Path, root: Path) -> bool:
    """True when *path* is *root* itself or lives under it."""
    try:
        resolved, resolved_root = path.resolve(), root.resolve()
    except OSError:
        return False
    return resolved == resolved_root or resolved_root in resolved.parents


def _remove_tree(path: Path, root: Path) -> None:
    if path.resolve() == root.resolve() or not _is_inside(path, root) or not path.exists():
        return
    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        log.warning("Could not fully remove %s (file in use?); it can be deleted by hand.", path)


def remove_job_files(job_id: str) -> None:
    """Delete a job's workspace (audio, transcript, crops, captions) and its renders."""
    _remove_tree(paths.job_work_dir(job_id), paths.work_dir())
    _remove_tree(paths.exports_dir() / job_id, paths.exports_dir())


def remove_source_files(source_id: str, source_path: str | None) -> None:
    """Delete a source's downloaded or uploaded media.

    Removes the per-source media folder, plus the folder the file actually lives
    in when it differs (async ingest names the folder by its own id). A file that
    sits directly in the media root is unlinked on its own; anything outside the
    media root is left alone.
    """
    media = paths.media_dir()
    _remove_tree(paths.source_media_dir(source_id), media)

    if not source_path:
        return
    file = Path(source_path)
    if file.parent.resolve() == media.resolve():
        if _is_inside(file, media):
            file.unlink(missing_ok=True)
        return
    _remove_tree(file.parent, media)
