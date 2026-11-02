"""Smoke tests for the prompt_style → prompt_version wiring.

No API keys or running services required — all assertions are local file-system
and in-process checks.
"""

from __future__ import annotations

import pytest
from autoclip import config
from autoclip.providers import STYLE_TO_PROMPT, detection_config
from autoclip.providers.base import PROMPT_DIR, load_prompt


# --------------------------------------------------------------------------
# 1. Every declared style maps to a file that actually exists on disk
# --------------------------------------------------------------------------

@pytest.mark.parametrize("style,version", STYLE_TO_PROMPT.items())
def test_prompt_file_exists_for_every_style(style: str, version: str) -> None:
    path = PROMPT_DIR / f"{version}.txt"
    assert path.exists(), (
        f"prompt_style={style!r} maps to {version!r} but "
        f"{path} does not exist on disk"
    )


# --------------------------------------------------------------------------
# 2. load_prompt() returns non-empty content for every style
# --------------------------------------------------------------------------

@pytest.mark.parametrize("style,version", STYLE_TO_PROMPT.items())
def test_load_prompt_returns_content_for_every_style(style: str, version: str) -> None:
    text = load_prompt(version)
    assert len(text) > 100, (
        f"prompt_style={style!r} → {version!r}: file loaded but looks empty"
    )


# --------------------------------------------------------------------------
# 3. detection_config() passes the right prompt_version for each style
# --------------------------------------------------------------------------

@pytest.mark.parametrize("style,expected_version", STYLE_TO_PROMPT.items())
def test_detection_config_maps_style_to_correct_prompt_version(
    style: str, expected_version: str
) -> None:
    settings = config.Settings()
    settings.clips.prompt_style = style

    dc = detection_config(settings)

    assert dc.prompt_version == expected_version, (
        f"prompt_style={style!r} should produce prompt_version={expected_version!r}, "
        f"got {dc.prompt_version!r}"
    )


# --------------------------------------------------------------------------
# 4. Unknown style (not in the map) falls back to highlight_v1, not a crash
# --------------------------------------------------------------------------

def test_unknown_style_falls_back_to_highlight_v1() -> None:
    settings = config.Settings()
    settings.clips.prompt_style = "nonexistent_style"

    dc = detection_config(settings)

    assert dc.prompt_version == "highlight_v1"


# --------------------------------------------------------------------------
# 5. Default settings use "general" → "highlight_v1"
# --------------------------------------------------------------------------

def test_default_settings_use_general_prompt() -> None:
    settings = config.Settings()

    assert settings.clips.prompt_style == "general"

    dc = detection_config(settings)
    assert dc.prompt_version == "highlight_v1"
