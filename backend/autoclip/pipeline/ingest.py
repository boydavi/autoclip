"""Ingestion — YouTube URLs and local file uploads become source records.

Both paths converge on a validated :class:`~autoclip.db.models.Source` with a
file on disk and probed metadata.

A note on YouTube: as of 2026 most anonymous downloads hit a bot check, and
proof-of-origin tokens no longer clear it reliably — browser cookies do. So
:class:`IngestError` distinguishes that specific failure and tells the user how
to fix it, rather than surfacing a raw yt-dlp traceback.
"""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Callable
from pathlib import Path

from .. import paths
from ..config import IngestSettings
from ..db.models import Source, new_id
from . import ffmpeg

log = logging.getLogger(__name__)

#: Container and audio formats we accept for upload.
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
ACCEPTED_SUFFIXES = VIDEO_SUFFIXES | AUDIO_SUFFIXES

_YOUTUBE_HOSTS = ("youtube.com", "youtu.be", "www.youtube.com", "m.youtube.com")

#: yt-dlp error fragments that mean "I have no extractor for this URL".
_UNSUPPORTED_MARKERS = (
    "unsupported url",
    "no suitable extractor",
    "this url is not supported",
)

#: yt-dlp error fragments that mean "YouTube wants proof you're a human".
_BOT_CHECK_MARKERS = (
    "sign in to confirm",
    "confirm you're not a bot",
    "confirm you are not a bot",
    "this content isn't available",
    "player response",
)


class IngestError(RuntimeError):
    """Ingestion failed in a way worth explaining to the user."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base}\n\n{self.hint}" if self.hint else base


def is_youtube_url(url: str) -> bool:
    from urllib.parse import urlparse

    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host in _YOUTUBE_HOSTS


def is_http_url(url: str) -> bool:
    """True for any http:// or https:// URL."""
    from urllib.parse import urlparse

    try:
        return urlparse(url).scheme.lower() in ("http", "https")
    except ValueError:
        return False


#: Headers a caller must never be able to inject — they can be used to reach
#: cloud-provider metadata services or to forge forwarding context.
_BLOCKED_REQUEST_HEADERS: frozenset[str] = frozenset({
    "host",
    "metadata-flavor",                   # GCP metadata service
    "x-aws-ec2-metadata-token",          # AWS IMDSv2
    "x-aws-ec2-metadata-token-ttl-seconds",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-real-ip",
})

#: Max redirects to follow in the httpx fallback path.
_MAX_REDIRECTS = 5


def _validate_ssrf_safe(url: str) -> None:
    """Resolve the URL's hostname and raise IngestError if any address is internal.

    Resolving and blocking at the Python level (rather than relying on the
    network boundary alone) prevents SSRF against metadata services, internal
    APIs, and link-local addresses reachable from the host running AutoClip.
    """
    import ipaddress
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise IngestError("Only http:// and https:// URLs are supported.")

    hostname = (parsed.hostname or "").rstrip(".").lower()
    if not hostname:
        raise IngestError("URL has no hostname.")

    try:
        results = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise IngestError(
            f"Could not resolve hostname '{hostname}'.",
            hint=str(exc),
        ) from exc

    for _family, _type, _proto, _canonname, sockaddr in results:
        ip_str = sockaddr[0]
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_multicast
            or addr.is_reserved
            or addr.is_unspecified
        ):
            raise IngestError(
                "URL resolves to a private or reserved network address.",
                hint=(
                    "AutoClip cannot fetch from internal addresses to prevent "
                    "unintended access to services on your network."
                ),
            )


def _sanitize_headers(headers: dict[str, str] | None) -> dict[str, str]:
    """Return a copy of headers with blocked keys removed."""
    if not headers:
        return {}
    return {
        k: v for k, v in headers.items()
        if k.lower() not in _BLOCKED_REQUEST_HEADERS
    }


def slugify(text: str, *, max_length: int = 60) -> str:
    """Turn a title into a filesystem- and URL-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug[:max_length] or "clip"


# --------------------------------------------------------------------------
# YouTube
# --------------------------------------------------------------------------


def fetch_youtube_captions(url: str, target_dir: Path, settings: IngestSettings | None = None) -> bool:
    """Download only YouTube's auto-captions (json3) beside an existing source.

    Never raises: captions are an optimisation and Whisper remains the fallback.
    """
    if any(target_dir.glob("*.json3")):
        return True
    import yt_dlp

    settings = settings or IngestSettings()
    options: dict = {
        "skip_download": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en.*"],
        "subtitlesformat": "json3",
        "outtmpl": str(target_dir / "source.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    if settings.cookies_from_browser:
        options["cookiesfrombrowser"] = (settings.cookies_from_browser,)
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.extract_info(url, download=True)
    except Exception as exc:
        log.warning("Caption-only fetch failed for %s: %s", url, exc)
        return False
    return any(target_dir.glob("*.json3"))


def ingest_youtube(
    url: str,
    settings: IngestSettings | None = None,
    *,
    on_progress: Callable[[float], None] | None = None,
) -> Source:
    """Download a YouTube video and return a validated source record.

    Preconditions:
        url points at content the user owns or has the rights to process.
    """
    import yt_dlp

    settings = settings or IngestSettings()
    source_id = new_id()
    target_dir = paths.source_media_dir(source_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    free_bytes = shutil.disk_usage(target_dir).free
    if free_bytes < 1 * 1024 ** 3:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise IngestError(
            "Not enough disk space to download this video.",
            hint=f"Only {free_bytes // (1024 ** 2)} MB free. Free up at least 1 GB and retry.",
        )

    def hook(status: dict) -> None:
        if not on_progress or status.get("status") != "downloading":
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate")
        done = status.get("downloaded_bytes")
        if total and done:
            on_progress(min(1.0, done / total))

    options: dict = {
        "format": settings.ytdlp_format,
        "outtmpl": str(target_dir / "source.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
        "retries": 3,
        "fragment_retries": 3,
    }
    if settings.cookies_from_browser:
        # yt-dlp expects a tuple; only the browser name is required.
        options["cookiesfrombrowser"] = (settings.cookies_from_browser,)
    if settings.prefer_youtube_captions:
        options["writeautomaticsub"] = True
        options["subtitleslangs"] = ["en.*"]
        options["subtitlesformat"] = "json3"

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            metadata = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise _translate_ytdlp_error(exc, settings) from exc
    except Exception as exc:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise IngestError(f"Could not download {url}: {exc}") from exc

    downloaded = _find_downloaded_file(target_dir)
    if downloaded is None:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise IngestError("yt-dlp reported success but produced no media file.")

    info = _probe_and_validate(downloaded)

    return Source(
        id=source_id,
        type="youtube",
        url=url,
        path=str(downloaded),
        filename=downloaded.name,
        title=(metadata or {}).get("title") or downloaded.stem,
        channel=(metadata or {}).get("uploader") or (metadata or {}).get("channel"),
        duration_s=info.duration_s or float((metadata or {}).get("duration") or 0.0),
        width=info.width,
        height=info.height,
        fps=info.fps,
        has_audio=info.has_audio,
        has_video=info.has_video,
    )


def _translate_ytdlp_error(exc: Exception, settings: IngestSettings) -> IngestError:
    """Turn a yt-dlp failure into something the user can act on."""
    message = str(exc).lower()

    if any(marker in message for marker in _BOT_CHECK_MARKERS):
        if settings.cookies_from_browser:
            hint = (
                f"Cookies are already being read from {settings.cookies_from_browser}, but "
                "YouTube still refused. Make sure you are signed in to YouTube in that "
                "browser and that the browser is fully closed — it locks its cookie "
                "database while running."
            )
        else:
            hint = (
                "YouTube is asking for proof you're not a bot. Set a browser to pull "
                "cookies from — in Settings, or via config.json's "
                '`ingest.cookies_from_browser` (e.g. "chrome", "firefox", "edge"). '
                "You must be signed in to YouTube in that browser, and it must be closed "
                "while AutoClip downloads."
            )
        return IngestError("YouTube blocked this download with a bot check.", hint=hint)

    if "private video" in message or "members-only" in message:
        return IngestError(
            "This video is private or members-only.",
            hint="AutoClip only downloads content you can access and have the rights to use.",
        )
    if "unavailable" in message or "removed" in message:
        return IngestError("This video is unavailable or has been removed.")
    if "drm" in message:
        return IngestError(
            "This content is DRM-protected.",
            hint="AutoClip does not and will not circumvent DRM.",
        )

    return IngestError(
        "yt-dlp could not download this video.",
        hint=(
            "YouTube changes frequently and yt-dlp is updated often. Try "
            "`autoclip update-ytdlp` to pull the latest version.\n\n"
            f"Original error: {exc}"
        ),
    )


def _find_downloaded_file(directory: Path) -> Path | None:
    """Return the largest media file in a directory, ignoring sidecars."""
    candidates = [
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in ACCEPTED_SUFFIXES
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


# --------------------------------------------------------------------------
# Generic URL (yt-dlp → httpx fallback)
# --------------------------------------------------------------------------


def ingest_url(
    url: str,
    settings: IngestSettings | None = None,
    *,
    headers: dict[str, str] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> Source:
    """Download any HTTP(S) URL and return a validated source record.

    Strategy:
      1. yt-dlp handles 1000+ platforms (Vimeo, Twitch, TikTok, Twitter/X,
         Loom, etc.) and its generic extractor covers plain media links.
      2. When auth headers are present, or when yt-dlp raises UnsupportedError,
         fall back to a direct httpx download. Auth headers are never forwarded
         to yt-dlp to prevent credential leakage to third-party CDNs.
      3. Every URL (including redirect hops) is validated against SSRF before
         any network connection is made.

    ``headers`` example for a Twilio recording::

        {"Authorization": "Basic <base64(AccountSid:AuthToken)>"}
    """
    import yt_dlp

    # Reject internal/private addresses before touching the network.
    _validate_ssrf_safe(url)
    safe_headers = _sanitize_headers(headers)

    # When auth headers are present, bypass yt-dlp entirely. yt-dlp passes
    # http_headers to every host it contacts (CDN, manifest, segment URLs),
    # which would leak the caller's credentials to third-party servers.
    has_auth = any(k.lower() == "authorization" for k in safe_headers)
    use_http_fallback = has_auth

    settings = settings or IngestSettings()
    source_id = new_id()
    target_dir = paths.source_media_dir(source_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    free_bytes = shutil.disk_usage(target_dir).free
    if free_bytes < 1 * 1024 ** 3:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise IngestError(
            "Not enough disk space to download this URL.",
            hint=f"Only {free_bytes // (1024 ** 2)} MB free. Free up at least 1 GB and retry.",
        )

    # ------------------------------------------------------------------
    # Path 1: yt-dlp (1000+ platforms, no auth headers)
    # ------------------------------------------------------------------
    if not use_http_fallback:
        def _progress_hook(status: dict) -> None:
            if not on_progress or status.get("status") != "downloading":
                return
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            done = status.get("downloaded_bytes")
            if total and done:
                on_progress(min(1.0, done / total))

        options: dict = {
            "format": settings.ytdlp_format,
            "outtmpl": str(target_dir / "source.%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "progress_hooks": [_progress_hook],
            "retries": 3,
            "fragment_retries": 3,
        }
        if settings.cookies_from_browser:
            options["cookiesfrombrowser"] = (settings.cookies_from_browser,)

        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                metadata = ydl.extract_info(url, download=True)

            downloaded = _find_downloaded_file(target_dir)
            if downloaded is None:
                shutil.rmtree(target_dir, ignore_errors=True)
                raise IngestError("yt-dlp reported success but produced no media file.")

            info = _probe_and_validate(downloaded)
            return Source(
                id=source_id,
                type="url",
                url=url,
                path=str(downloaded),
                filename=downloaded.name,
                title=(metadata or {}).get("title") or downloaded.stem,
                channel=(metadata or {}).get("uploader") or (metadata or {}).get("channel"),
                duration_s=info.duration_s or float((metadata or {}).get("duration") or 0.0),
                width=info.width,
                height=info.height,
                fps=info.fps,
                has_audio=info.has_audio,
                has_video=info.has_video,
            )

        except yt_dlp.utils.UnsupportedError:
            use_http_fallback = True
        except yt_dlp.utils.DownloadError as exc:
            msg = str(exc).lower()
            if any(m in msg for m in _UNSUPPORTED_MARKERS):
                use_http_fallback = True
            else:
                shutil.rmtree(target_dir, ignore_errors=True)
                raise _translate_ytdlp_error(exc, settings) from exc
        except IngestError:
            raise
        except Exception as exc:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise IngestError(f"Could not download the URL.") from exc

    if not use_http_fallback:
        # Both yt-dlp paths returned without raising — should not happen.
        shutil.rmtree(target_dir, ignore_errors=True)
        raise IngestError("yt-dlp reported success but produced no usable output.")

    # ------------------------------------------------------------------
    # Path 2: direct httpx download — Twilio, S3, CDN, plain media URLs.
    # Auth headers stay on the origin; they are dropped on cross-origin
    # redirects by stripping the header after the first hop.
    # ------------------------------------------------------------------
    import httpx
    from urllib.parse import unquote, urljoin, urlparse

    UPLOAD_MAX_BYTES = 20 * 1024 * 1024 * 1024  # 20 GB — same ceiling as the upload endpoint

    req_headers = {"User-Agent": "AutoClip/1.0", **safe_headers}
    fname = "source"
    suffix = ".mp4"
    current_url = url
    origin_host = (urlparse(url).hostname or "").lower()

    try:
        for _hop in range(_MAX_REDIRECTS + 1):
            if _hop == _MAX_REDIRECTS:
                shutil.rmtree(target_dir, ignore_errors=True)
                raise IngestError(
                    "Too many redirects.",
                    hint=f"Followed {_MAX_REDIRECTS} redirects without reaching the media file.",
                )

            # Re-validate every redirect target against SSRF.
            _validate_ssrf_safe(current_url)

            # Drop auth headers when the redirect leaves the original host.
            hop_host = (urlparse(current_url).hostname or "").lower()
            hop_headers = req_headers if hop_host == origin_host else {
                k: v for k, v in req_headers.items() if k.lower() != "authorization"
            }

            with httpx.stream(
                "GET", current_url, headers=hop_headers,
                follow_redirects=False, timeout=30,
            ) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("location", "")
                    if not location:
                        shutil.rmtree(target_dir, ignore_errors=True)
                        raise IngestError("Server sent a redirect with no Location header.")
                    current_url = urljoin(current_url, location)
                    continue

                resp.raise_for_status()

                # Reject oversized responses early from Content-Length.
                cl = resp.headers.get("content-length")
                if cl and int(cl) > UPLOAD_MAX_BYTES:
                    shutil.rmtree(target_dir, ignore_errors=True)
                    raise IngestError(
                        "Remote file is too large.",
                        hint=f"Maximum allowed size is {UPLOAD_MAX_BYTES // (1024**3)} GB.",
                    )

                # Infer filename from Content-Disposition, then URL path.
                cd = resp.headers.get("Content-Disposition", "")
                if "filename=" in cd:
                    fname = cd.split("filename=")[-1].strip().strip('"\'')
                else:
                    url_path = unquote(urlparse(current_url).path)
                    fname = url_path.rsplit("/", 1)[-1] or "source"

                detected_suffix = Path(fname).suffix.lower()
                if detected_suffix in ACCEPTED_SUFFIXES:
                    suffix = detected_suffix

                staging = target_dir / f"source{suffix}"
                received = 0
                with staging.open("wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=4 * 1024 * 1024):
                        received += len(chunk)
                        if received > UPLOAD_MAX_BYTES:
                            shutil.rmtree(target_dir, ignore_errors=True)
                            raise IngestError(
                                "Remote file exceeded the size limit during download.",
                                hint=f"Maximum allowed size is {UPLOAD_MAX_BYTES // (1024**3)} GB.",
                            )
                        fh.write(chunk)
                        if on_progress and cl:
                            on_progress(min(1.0, received / int(cl)))
                break  # success

    except IngestError:
        raise
    except httpx.HTTPStatusError as exc:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise IngestError(
            f"URL returned HTTP {exc.response.status_code}.",
            hint=(
                "Check the URL is correct and that any required authentication "
                "headers are included (e.g. Authorization: Bearer <token>)."
            ),
        ) from exc
    except Exception as exc:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise IngestError("Could not download the URL.") from exc

    try:
        info = _probe_and_validate(staging)
    except Exception:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise

    return Source(
        id=source_id,
        type="url",
        url=url,
        path=str(staging),
        filename=staging.name,
        title=Path(fname).stem,
        duration_s=info.duration_s,
        width=info.width,
        height=info.height,
        fps=info.fps,
        has_audio=info.has_audio,
        has_video=info.has_video,
    )


# --------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------


def ingest_file(path: Path, *, move: bool = False, title: str | None = None) -> Source:
    """Register a local file as a source, copying it into AutoClip's media store.

    Copying rather than referencing in place means a job stays reproducible even
    if the user moves or deletes the original.

    Preconditions:
        path exists and is a readable media file.
    """
    path = Path(path)
    if not path.exists():
        raise IngestError(f"{path} does not exist.")
    if not path.is_file():
        raise IngestError(f"{path} is not a file.")

    suffix = path.suffix.lower()
    if suffix not in ACCEPTED_SUFFIXES:
        accepted = ", ".join(sorted(ACCEPTED_SUFFIXES))
        raise IngestError(
            f"{suffix or 'This file'} is not a supported format.",
            hint=f"Accepted formats: {accepted}",
        )

    source_id = new_id()
    target_dir = paths.source_media_dir(source_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"source{suffix}"

    file_size = path.stat().st_size
    free_bytes = shutil.disk_usage(target_dir).free
    if not move and free_bytes < file_size + 256 * 1024 * 1024:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise IngestError(
            f"Not enough disk space to store {path.name}.",
            hint=f"Need {(file_size + 256 * 1024 * 1024) // (1024 ** 2)} MB free, "
                 f"only {free_bytes // (1024 ** 2)} MB available.",
        )

    try:
        if move:
            shutil.move(str(path), target)
        else:
            shutil.copy2(path, target)
    except OSError as exc:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise IngestError(f"Could not store {path.name}: {exc}") from exc

    try:
        info = _probe_and_validate(target)
    except Exception:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise

    return Source(
        id=source_id,
        type="upload",
        path=str(target),
        filename=path.name,
        title=title or info.title or path.stem,
        duration_s=info.duration_s,
        width=info.width,
        height=info.height,
        fps=info.fps,
        has_audio=info.has_audio,
        has_video=info.has_video,
    )


def _probe_and_validate(path: Path) -> ffmpeg.MediaInfo:
    """Probe a media file and reject anything the pipeline can't process."""
    try:
        info = ffmpeg.probe(path)
    except ffmpeg.FFmpegError as exc:
        raise IngestError(f"{path.name} could not be read as media.", hint=str(exc)) from exc

    if not info.has_audio:
        raise IngestError(
            f"{path.name} has no audio track.",
            hint=(
                "AutoClip finds clips by transcribing speech, so a file with no audio "
                "has nothing to work from."
            ),
        )
    if info.duration_s <= 0:
        raise IngestError(
            f"{path.name} reports zero duration.",
            hint="The file may be corrupt or still being written.",
        )

    return info
