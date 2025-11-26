from __future__ import annotations

import time
from typing import Iterable, Optional

from src.v2.interfaces import (
    AudioChunk,
    AudioSource,
    SilencePolicy,
    SilenceEvent,
    STTEngine,
    IntentClassifier,
    InferenceResult,
    IntentResult,
    RepeatFilter,
)


class MockAudioSource(AudioSource):
    def __init__(self, chunks: Optional[Iterable[AudioChunk]] = None) -> None:
        if chunks is None:
            now = time.time()
            chunks = [AudioChunk(data=[0.0], timestamp=now + i * 0.1) for i in range(3)]
        self._chunks = list(chunks)

    def stream(self) -> Iterable[AudioChunk]:
        for chunk in self._chunks:
            yield chunk


class ImmediateSilencePolicy(SilencePolicy):
    def observe(self, chunk: AudioChunk) -> Optional[SilenceEvent]:
        return SilenceEvent(event_id="mock-event", timestamp=chunk.timestamp, audio=chunk.data)


class MockSTTEngine(STTEngine):
    def transcribe(self, audio):
        return InferenceResult(text="hello world", latency_ms=10.0)


class MockIntentClassifier(IntentClassifier):
    def classify(self, text: str) -> IntentResult:
        return IntentResult(prompt_id="prompt-1", score=0.9, latency_ms=5.0)


class MockRepeatFilter(RepeatFilter):
    def __init__(self) -> None:
        self.calls = 0

    def should_filter(self, prompt_id: str, score: float) -> bool:
        self.calls += 1
        return False
