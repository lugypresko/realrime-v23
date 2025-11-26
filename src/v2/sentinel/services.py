from __future__ import annotations

import time
from typing import Iterable, Optional

from src.v2.interfaces import AudioChunk, AudioSource, SilenceEvent, SilencePolicy, TelemetryClient, ReplayStore


class StaticAudioSource(AudioSource):
    """Produces a finite sequence of audio chunks for deterministic tests."""

    def __init__(self, chunks: Iterable[AudioChunk]):
        self._chunks = list(chunks)

    def stream(self) -> Iterable[AudioChunk]:
        for chunk in self._chunks:
            yield chunk


class ThresholdSilencePolicy(SilencePolicy):
    """Emits a silence event once a configured count of chunks is seen."""

    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self._count = 0

    def observe(self, chunk: AudioChunk) -> Optional[SilenceEvent]:
        self._count += 1
        if self._count >= self.threshold:
            return SilenceEvent(event_id=f"event-{int(time.time()*1000)}", timestamp=chunk.timestamp, audio=chunk.data)
        return None


class InMemoryReplayStore(ReplayStore):
    def __init__(self) -> None:
        self._buffer: list[AudioChunk] = []

    def add(self, chunk: AudioChunk) -> None:
        self._buffer.append(chunk)

    def snapshot(self):
        return list(self._buffer)


class SentinelTelemetry:
    def __init__(self, client: TelemetryClient):
        self._client = client

    def log_trigger(self, event: SilenceEvent) -> None:
        self._client.record({"type": "silence_trigger", "event_id": event.event_id, "timestamp": event.timestamp})

    def log_chunk(self, chunk: AudioChunk) -> None:
        self._client.record({"type": "audio_chunk", "timestamp": chunk.timestamp})
