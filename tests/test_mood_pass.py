"""The focused mood pass: each chosen clip's mood is re-judged from its own text at temperature 0."""

from __future__ import annotations

import asyncio

from autoclip.db.models import Clip
from autoclip.pipeline import highlights
from autoclip.providers.base import MOOD_RULES, DetectionConfig
from autoclip.pipeline.transcript import Transcript, Word


def transcript():
    words = [Word(f"w{i}", float(i), i + 0.9) for i in range(60)]
    return Transcript(language="en", model="m", words=words)


def clip(start, end, mood="dramatic"):
    return Clip(id=f"c{start}", job_id="j", start_s=start, end_s=end, start_word=start, end_word=end, mood=mood)


class FakeProvider:
    def __init__(self, replies):
        self.replies, self.calls = replies, []

    async def complete_json(self, system, user, config):
        self.calls.append((system, user, config))
        reply = self.replies[len(self.calls) - 1] if isinstance(self.replies, list) else self.replies(user)
        if isinstance(reply, Exception):
            raise reply
        return reply


def run(clips, provider, **kw):
    return asyncio.run(highlights.refine_moods(clips, transcript(), provider, DetectionConfig(temperature=0.3), **kw))


def test_each_clip_is_judged_from_its_own_text_at_temperature_zero():
    clips = [clip(0, 9), clip(20, 29)]
    provider = FakeProvider(lambda user: {"mood": "calm"})
    assert run(clips, provider) == 2
    assert [c.mood for c in clips] == ["calm", "calm"]
    system, user, config = provider.calls[0]
    assert config.temperature == 0.0 and "w0 w1" in user and "w20" not in user
    assert MOOD_RULES in system and "Decide in this order" in system          # the same rubric as the window prompt


def test_a_failed_or_unusable_reply_keeps_the_window_prompts_mood():
    clips = [clip(0, 9, "dramatic"), clip(10, 19, "energetic"), clip(20, 29, "calm"), clip(30, 39, "dramatic")]
    provider = FakeProvider([RuntimeError("429"), {"mood": "sad"}, {}, {"mood": " ENERGETIC "}])
    assert run(clips, provider) == 1
    assert [c.mood for c in clips] == ["dramatic", "energetic", "calm", "energetic"]


def test_an_unchanged_mood_is_not_counted_and_no_clips_means_no_calls():
    clips = [clip(0, 9, "calm")]
    assert run(clips, FakeProvider([{"mood": "calm"}])) == 0
    provider = FakeProvider([])
    assert run([], provider) == 0 and provider.calls == []


def test_a_cancelled_job_makes_no_calls():
    provider = FakeProvider(lambda user: {"mood": "calm"})
    assert run([clip(0, 9)], provider, is_cancelled=lambda: True) == 0
    assert provider.calls == []


def test_detect_runs_the_mood_pass_after_the_clips_are_built():
    import inspect

    source = inspect.getsource(highlights.detect)
    assert source.index("build_clips(") < source.index("await refine_moods(")
    assert "concurrency=concurrency or 4" in source
