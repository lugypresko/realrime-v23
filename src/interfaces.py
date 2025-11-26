from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable, Any, Dict, Iterable, Optional


@runtime_checkable
class AudioSource(Protocol):
    def start(self, callback) -> None:
        ...

    def stop(self) -> None:
        ...


@runtime_checkable
class SilencePolicy(Protocol):
    def handle_prob(self, prob: float, timestamp: float, frames: int, sample_rate: int) -> Optional[Dict[str, Any]]:
        """Return SILENCE_TRIGGER event dict when ready, otherwise None."""
        ...


@runtime_checkable
class SilenceDetector(Protocol):
    def update(self, prob: float, timestamp: float, frames: int) -> Optional[Dict[str, Any]]:
        ...


@runtime_checkable
class STTEngine(Protocol):
    def transcribe(self, audio) -> Dict[str, Any]:
        ...


@runtime_checkable
class IntentClassifier(Protocol):
    def classify(self, text: str) -> Dict[str, Any]:
        ...


@runtime_checkable
class Governor(Protocol):
    def decide(self, event: Dict[str, Any], transport_delay: float, whisper_latency: float, intent_latency: float) -> Dict[str, Any]:
        ...

    def record_backpressure(self) -> None:
        ...


@runtime_checkable
class TelemetryClient(Protocol):
    def emit(self, event: Dict[str, Any]) -> None:
        ...


@runtime_checkable
class ResultFormatter(Protocol):
    def format(self, event: Dict[str, Any]) -> str:
        ...


@runtime_checkable
class WarmStarter(Protocol):
    def warm_start(self) -> None:
        ...


@runtime_checkable
class ReplayStore(Protocol):
    def add(self, audio_chunk, vad_prob: float) -> None:
        ...

    def dump_to_disk(self, path: str, event_id: str) -> None:
        ...


@runtime_checkable
class LatencyTracker(Protocol):
    def record(self, whisper_latency: float, intent_latency: float, total_age: float, decision: str) -> None:
        ...


@runtime_checkable
class RepeatFilter(Protocol):
    def should_suppress(self, prompt_id: str, score_diff: float) -> bool:
        ...
