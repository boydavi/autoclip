"""What a job costs on Groq: tokens, audio minutes, and an estimated price.

Counts are exact where the API reports them (chat usage); the price is an ESTIMATE from the
list prices below and can drift - the Groq console's usage page is the source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

#: USD per 1M tokens (input, output) for chat models served by Groq. Update when Groq changes them.
LLM_PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
}
#: USD per hour of audio transcribed by Groq Whisper.
WHISPER_PRICES: dict[str, float] = {
    "whisper-large-v3-turbo": 0.04,
    "whisper-large-v3": 0.111,
}


@dataclass
class Usage:
    """Token counts for one provider instance (one job's clip-selection stage)."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, prompt_tokens: int | None, completion_tokens: int | None) -> None:
        self.calls += 1
        self.input_tokens += int(prompt_tokens or 0)
        self.output_tokens += int(completion_tokens or 0)


def llm_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimated cost of chat tokens; 0.0 for a model with no known price (never guessed)."""
    price = LLM_PRICES.get(model)
    if price is None:
        return 0.0
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


def whisper_cost_usd(model: str, seconds: float) -> float:
    rate = WHISPER_PRICES.get(model)
    return 0.0 if rate is None else rate * seconds / 3600.0
