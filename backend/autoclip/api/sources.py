"""Source ingestion endpoints."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..config import load as load_settings
from ..db import store
from ..db.models import Source, new_id, utcnow
from ..pipeline import ingest
from .schemas import SourceOut, YouTubeIngestIn

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sources", tags=["sources"])

#: Uploads stream to disk in chunks rather than being buffered whole — source
#: videos routinely exceed available RAM.
UPLOAD_CHUNK = 4 * 1024 * 1024


@router.get("", response_model=list[SourceOut])
async def list_sources(limit: int = 50) -> list[SourceOut]:
    sources = await asyncio.to_thread(store.list_sources, limit)
    return [SourceOut.of(s) for s in sources]


@router.get("/{source_id}", response_model=SourceOut)
async def get_source(source_id: str) -> SourceOut:
    source = await asyncio.to_thread(store.get_source, source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found.")
    return SourceOut.of(source)


@router.post("/youtube", status_code=202)
async def ingest_youtube(
    payload: YouTubeIngestIn,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """Start a YouTube download in the background and return immediately.

    Returns 202 with ``{"id": source_id, "ingest_status": "pending"}``.
    Poll ``GET /api/sources/{id}`` until ``ingest_status`` is ``"ready"``
    (or ``"failed"``).
    """
    if not ingest.is_youtube_url(payload.url):
        raise HTTPException(status_code=400, detail="That is not a YouTube URL.")

    settings = load_settings().ingest
    if payload.cookies_from_browser is not None:
        settings = settings.model_copy(
            update={"cookies_from_browser": payload.cookies_from_browser}
        )

    # Re-scanning a video is the common case: reuse the download (and with it the
    # source-keyed transcript cache) instead of fetching gigabytes again.
    existing = await asyncio.to_thread(store.find_ready_source_by_url, payload.url)
    if existing is not None:
        if settings.prefer_youtube_captions:
            await asyncio.to_thread(
                ingest.fetch_youtube_captions, payload.url, Path(existing.path).parent, settings
            )
        log.info("Reusing source %s for %s.", existing.id, payload.url)
        return JSONResponse(status_code=201, content=SourceOut.of(existing).model_dump(mode="json"))

    source_id = new_id()
    pending = Source(
        id=source_id,
        type="youtube",
        url=payload.url,
        path="",
        title="",
        created_at=utcnow(),
        ingest_status="pending",
    )
    await asyncio.to_thread(store.create_source, pending)

    background_tasks.add_task(_do_ingest_youtube, source_id, payload.url, settings)
    return JSONResponse(
        status_code=202,
        content={"id": source_id, "ingest_status": "pending"},
    )


async def _do_ingest_youtube(source_id: str, url: str, settings) -> None:
    """Background task: download the video and update the source record."""
    try:
        source = await asyncio.to_thread(ingest.ingest_youtube, url, settings)
        # Carry over the pre-assigned id so the DB record stays consistent.
        source = Source(
            id=source_id,
            type=source.type,
            url=source.url,
            path=source.path,
            filename=source.filename,
            title=source.title,
            channel=source.channel,
            duration_s=source.duration_s,
            width=source.width,
            height=source.height,
            fps=source.fps,
            has_audio=source.has_audio,
            has_video=source.has_video,
            created_at=source.created_at,
            ingest_status="ready",
        )
        await asyncio.to_thread(
            store.update_source_after_ingest,
            source_id,
            ingest_status="ready",
            source=source,
        )
        log.info("Async ingest complete for source %s (%s).", source_id, source.title)
    except Exception as exc:
        log.error("Async ingest failed for source %s: %s", source_id, exc)
        await asyncio.to_thread(
            store.update_source_after_ingest,
            source_id,
            ingest_status="failed",
            ingest_error=str(exc),
        )


@router.post("/upload", response_model=SourceOut, status_code=201)
async def upload_source(file: UploadFile = File(...)) -> SourceOut:
    """Accept a media upload and register it as a source."""
    filename = Path(file.filename or "upload")
    suffix = filename.suffix.lower()
    if suffix not in ingest.ACCEPTED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail={
                "message": f"{suffix or 'This file'} is not a supported format.",
                "hint": f"Accepted: {', '.join(sorted(ingest.ACCEPTED_SUFFIXES))}",
            },
        )

    # Stage to a temp file first so a failed or abandoned upload never leaves a
    # half-written file in the media store.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as staging:
        staging_path = Path(staging.name)
        try:
            while chunk := await file.read(UPLOAD_CHUNK):
                staging.write(chunk)
        except Exception as exc:
            staging_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="Upload failed.") from exc

    try:
        source = await asyncio.to_thread(
            ingest.ingest_file, staging_path, move=True, title=filename.stem
        )
    except ingest.IngestError as exc:
        staging_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422, detail={"message": str(exc), "hint": exc.hint}
        ) from exc
    finally:
        shutil.rmtree(staging_path.parent / staging_path.name, ignore_errors=True)

    await asyncio.to_thread(store.create_source, source)
    return SourceOut.of(source)
