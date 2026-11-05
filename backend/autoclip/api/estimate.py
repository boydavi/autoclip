"""Job cost estimation endpoint.

Returns a rough USD breakdown before the user commits a job.  Numbers are
estimates derived from average speech rates and public provider pricing —
not invoices.  Actual costs differ based on transcript density, model output
length, and provider rate changes.
"""

from __future__ import annotations

import asyncio
import math

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import load as load_settings
from ..db import store
from ..pipeline.highlights import OVERLAP_S, WINDOW_S

router = APIRouter(prefix="/api/estimate", tags=["estimate"])

ADVANCE_S = WINDOW_S - OVERLAP_S  # 420 s per step

# Average speech rate (words / second): ~150 wpm, between a lecture and a fast conversation.
_WORDS_PER_S = 150 / 60

# Tokens per rendered word. Each word goes out as "[index]word", and the bracketed number costs
# about three tokens on its own. Measured on a real Groq run (gpt-oss-120b): ~4.8 per word.
_TOKENS_PER_WORD = 4.8

# System prompt + the per-window wrapper (tokens): the prompt text is ~5,000 characters.
_SYSTEM_TOKENS = 1400

# Output tokens per window, reasoning included (Groq counts it): ~730 measured.
_OUTPUT_TOKENS = 750

# Groq Whisper pricing per audio second.
_GROQ_WHISPER_PER_S: dict[str, float] = {
    "whisper-large-v3-turbo": 0.04 / 3600,   # $0.04 / hr
    "whisper-large-v3":       0.111 / 3600,  # $0.111 / hr
}
_GROQ_WHISPER_DEFAULT = 0.04 / 3600

# LLM pricing: provider → [(model_substring, input_$/MTok, output_$/MTok)].
# First match (case-insensitive substring) wins; empty string = fallback.
# Anthropic 5.x rows are the standard-tier prices from
# platform.claude.com/docs/en/about-claude/pricing (prompts ≤100K tokens, which
# every AutoClip window is; Haiku 5.5 bills $0.50/$2.50 above that). The older
# rows are legacy generations kept for users who still pin them.
_LLM_PRICING: dict[str, list[tuple[str, float, float]]] = {
    "anthropic": [
        ("haiku-5",  0.10,   0.50),
        ("sonnet-5", 2.00,  10.00),
        ("opus-5",   4.00,  20.00),
        ("haiku",    1.00,   5.00),   # legacy claude-haiku-4-5
        ("opus",    15.00,  75.00),   # legacy claude-opus-4 / 4.1 / 3
        ("sonnet",   3.00,  15.00),   # legacy claude-sonnet-4 / 3.7
        ("",         3.00,  15.00),
    ],
    "openai": [
        ("gpt-4o-mini",  0.15,  0.60),
        ("o1-mini",      3.00, 12.00),
        ("o1",          15.00, 60.00),
        ("gpt-4o",       2.50, 10.00),
        ("",             2.50, 10.00),
    ],
    "gemini": [
        ("2.0-flash",  0.075,  0.30),
        ("1.5-flash",  0.075,  0.30),
        ("1.5-pro",    1.25,   5.00),
        ("",           0.075,  0.30),
    ],
    "groq": [
        ("gpt-oss-20b",  0.075, 0.30),
        ("gpt-oss-120b", 0.15,  0.60),
        ("",             0.15,  0.60),  # unknown Groq model: assume the pricier gpt-oss
    ],
    "ollama": [
        ("", 0.0, 0.0),
    ],
}


def _window_count(duration_s: float) -> int:
    """Number of sliding windows for a transcript of *duration_s* seconds."""
    return max(1, math.ceil((duration_s - OVERLAP_S) / ADVANCE_S))


def _llm_per_window(provider: str, model: str) -> float:
    rows = _LLM_PRICING.get(provider.lower(), [("", 3.00, 15.00)])
    inp_mtok, out_mtok = rows[-1][1], rows[-1][2]  # fallback
    m = model.lower()
    for substr, i, o in rows:
        if substr and substr in m:
            inp_mtok, out_mtok = i, o
            break

    # Input tokens = system prompt + indexed transcript words in the window.
    input_tokens = _SYSTEM_TOKENS + _WORDS_PER_S * WINDOW_S * _TOKENS_PER_WORD
    return (input_tokens * inp_mtok + _OUTPUT_TOKENS * out_mtok) / 1_000_000


def _whisper_cost(provider: str, model: str, duration_s: float) -> float:
    if provider != "groq":
        return 0.0
    rate = _GROQ_WHISPER_PER_S.get(model, _GROQ_WHISPER_DEFAULT)
    return duration_s * rate


class CostEstimateOut(BaseModel):
    duration_s: float
    windows: int
    whisper_provider: str
    whisper_cost_usd: float
    llm_provider: str
    llm_model: str
    llm_cost_per_window_usd: float
    llm_cost_usd: float
    total_usd: float


@router.get("/{source_id}", response_model=CostEstimateOut)
async def estimate_job_cost(source_id: str) -> CostEstimateOut:
    source = await asyncio.to_thread(store.get_source, source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found.")
    if source.ingest_status != "ready":
        raise HTTPException(status_code=409, detail="Source is not ready yet.")

    settings = await asyncio.to_thread(load_settings)

    llm_provider = settings.active_provider
    provider_cfg = settings.providers.get(llm_provider)
    llm_model = provider_cfg.model if provider_cfg else ""

    whisper_provider = settings.whisper.provider
    groq_model = settings.whisper.groq_model

    duration_s = source.duration_s
    windows = _window_count(duration_s)

    whisper_cost = _whisper_cost(whisper_provider, groq_model, duration_s)
    llm_per_window = _llm_per_window(llm_provider, llm_model)
    llm_total = llm_per_window * windows

    return CostEstimateOut(
        duration_s=round(duration_s, 1),
        windows=windows,
        whisper_provider=whisper_provider,
        whisper_cost_usd=round(whisper_cost, 4),
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_cost_per_window_usd=round(llm_per_window, 5),
        llm_cost_usd=round(llm_total, 4),
        total_usd=round(whisper_cost + llm_total, 4),
    )
