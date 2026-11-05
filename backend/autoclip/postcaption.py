"""The post text that goes beside a clip: title options, description, call to action, link, hashtags.

One layout, used for the ``.txt`` file next to every exported clip. Empty parts are left out, so a
clip with no CTA or link just has no such lines. (ATALLA's ``core/ops/post_text.py`` writes the
same layout for clips it renders itself; its tests pin the two to the same output.)
"""

from __future__ import annotations

import re


def compose_post_text(
    *,
    titles: list[str] | tuple[str, ...] = (),
    description: str = "",
    cta: str = "",
    link: str = "",
    episode: str = "",
    hashtags: str = "",
) -> str:
    """Title options, then description, then CTA, link and full-episode line together, then hashtags.

    ::

        TITLE OPTIONS
        1. ...
        2. ...

        <description>

        <cta>
        <link>
        <episode line>

        <hashtags>
    """
    blocks: list[str] = []
    shown = [t.strip() for t in titles if t and t.strip()]
    if shown:
        blocks.append("TITLE OPTIONS\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(shown, 1)))
    if description.strip():
        blocks.append(description.strip())
    call = "\n".join(part.strip() for part in (cta, link, episode) if part and part.strip())
    if call:
        blocks.append(call)
    if hashtags.strip():
        blocks.append(hashtags.strip())
    return "\n\n".join(blocks)


_YOUTUBE_ID = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|embed/|live/))([\w-]{11})", re.IGNORECASE
)


def episode_line(url: str | None, start_s: float = 0.0) -> str:
    """"Full episode: <link>" pointing at the moment the clip starts; "" when there is no source URL.

    A YouTube URL becomes a short link with the start time (``https://youtu.be/ID?t=217``); any other
    URL is used as it is, since it may not take a time.
    """
    url = (url or "").strip()
    if not url:
        return ""
    found = _YOUTUBE_ID.search(url)
    if found:
        return f"Full episode: https://youtu.be/{found.group(1)}?t={max(0, int(start_s))}"
    return f"Full episode: {url}"
