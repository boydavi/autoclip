"""The post text that goes beside a clip: title options, description, call to action, link, hashtags.

One layout, used for the ``.txt`` file next to every exported clip. Empty parts are left out, so a
clip with no CTA or link just has no such lines. (ATALLA's ``core/ops/post_text.py`` writes the
same layout for clips it renders itself; its tests pin the two to the same output.)
"""

from __future__ import annotations


def compose_post_text(
    *,
    titles: list[str] | tuple[str, ...] = (),
    description: str = "",
    cta: str = "",
    link: str = "",
    hashtags: str = "",
) -> str:
    """Title options, then description, then CTA and link together, then hashtags.

    ::

        TITLE OPTIONS
        1. ...
        2. ...

        <description>

        <cta>
        <link>

        <hashtags>
    """
    blocks: list[str] = []
    shown = [t.strip() for t in titles if t and t.strip()]
    if shown:
        blocks.append("TITLE OPTIONS\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(shown, 1)))
    if description.strip():
        blocks.append(description.strip())
    call = "\n".join(part.strip() for part in (cta, link) if part and part.strip())
    if call:
        blocks.append(call)
    if hashtags.strip():
        blocks.append(hashtags.strip())
    return "\n\n".join(blocks)
