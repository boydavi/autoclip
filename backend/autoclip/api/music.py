"""Music folder management endpoints.

Tracks are plain files in paths.music_dir().  No database rows — the
filesystem is the source of truth.  The router exposes list / upload / delete;
the frontend uses these to show the panel added in Settings.tsx.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from .. import paths
from ..pipeline.export import MUSIC_FORMATS

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/music", tags=["music"])

#: Hard ceiling for a single upload — 200 MB covers any realistic music file.
UPLOAD_MAX_BYTES = 200 * 1024 * 1024


class MusicTrackOut(BaseModel):
    filename: str
    size_bytes: int


def _list_tracks(music_dir: Path) -> list[MusicTrackOut]:
    if not music_dir.is_dir():
        return []
    tracks = sorted(
        p for p in music_dir.iterdir() if p.suffix.lower() in MUSIC_FORMATS
    )
    return [MusicTrackOut(filename=p.name, size_bytes=p.stat().st_size) for p in tracks]


def _safe_name(filename: str) -> str:
    """Strip path components so a caller cannot escape the music directory."""
    return Path(filename).name


@router.get("", response_model=list[MusicTrackOut])
async def list_tracks() -> list[MusicTrackOut]:
    return await asyncio.to_thread(_list_tracks, paths.music_dir())


@router.post("", response_model=MusicTrackOut, status_code=201)
async def upload_track(file: UploadFile = File(...)) -> MusicTrackOut:
    name = _safe_name(file.filename or "track")
    if Path(name).suffix.lower() not in MUSIC_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Accepted: {', '.join(sorted(MUSIC_FORMATS))}",
        )

    music_dir = paths.music_dir()
    music_dir.mkdir(parents=True, exist_ok=True)
    dest = music_dir / name

    written = 0
    chunk_size = 1024 * 1024  # 1 MB
    try:
        with dest.open("wb") as fh:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                written += len(chunk)
                if written > UPLOAD_MAX_BYTES:
                    fh.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds the 200 MB limit.",
                    )
                fh.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        dest.unlink(missing_ok=True)
        log.exception("Failed to write music track %s", name)
        raise HTTPException(status_code=500, detail="Upload failed.") from exc

    return MusicTrackOut(filename=name, size_bytes=written)


@router.delete("/{filename}", status_code=204)
async def delete_track(filename: str) -> Response:
    name = _safe_name(filename)
    path = paths.music_dir() / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Track not found.")
    if path.suffix.lower() not in MUSIC_FORMATS:
        raise HTTPException(status_code=400, detail="Not a music track.")
    path.unlink()
    return Response(status_code=204)
