from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Iterable, Any, Dict, Optional


@dataclass
class AudioChunk:
    data: Any
    timestamp: float


@dataclass
class SilenceEvent:
    event_id: str
    timestamp: float
    audio: Any


@dataclass
class InferenceResult:
    text: str
    latency_ms: float
    success: bool = True
    error: Optional[str] = None


@dataclass
class IntentResult:
    prompt_id: str
    score: float
    latency_ms: float


@dataclass
class WorkerDecision:
    decision: str
    reason: Optional[str] = None


@dataclass
class WorkerOutput:
    event_id: str
    text: str
    prompt_id: str
    score: float
    whisper_latency_ms: float
    intent_latency_ms: float
    total_latency_ms: float
    decision: str
    metadata: Dict[str, Any]


@runtime_checkable
class AudioSource(Protocol):
    def stream(self) -> Iterable[AudioChunk]:
        ...


@runtime_checkable
class SilencePolicy(Protocol):
    def observe(self, chunk: AudioChunk) -> Optional[SilenceEvent]:
        ...


@runtime_checkable
class STTEngine(Protocol):
    def transcribe(self, audio: Any) -> InferenceResult:
        ...


@runtime_checkable
class IntentClassifier(Protocol):
    def classify(self, text: str) -> IntentResult:
        ...


@runtime_checkable
class GovernorService(Protocol):
    def decide(self, event: SilenceEvent, inference: InferenceResult, intent: IntentResult) -> WorkerDecision:
        ...


@runtime_checkable
class TelemetryClient(Protocol):
    def record(self, event: Dict[str, Any]) -> None:
        ...


@runtime_checkable
class ResultFormatter(Protocol):
    def format(self, output: WorkerOutput) -> str:
        ...


@runtime_checkable
class WarmStarter(Protocol):
    def warm_start(self) -> None:
        ...


@runtime_checkable
class ReplayStore(Protocol):
    def add(self, chunk: AudioChunk) -> None:
        ...

    def snapshot(self) -> Iterable[AudioChunk]:
        ...


@runtime_checkable
class LatencyTracker(Protocol):
    def record(self, metrics: Dict[str, float]) -> None:
        ...


@runtime_checkable
class RepeatFilter(Protocol):
    def should_filter(self, prompt_id: str, score: float) -> bool:
        ...


def validate_silence_event(event: SilenceEvent) -> bool:
    return bool(event.event_id) and event.timestamp is not None and event.audio is not None


def validate_worker_output(output: WorkerOutput) -> bool:
    has_core_fields = all([
        output.event_id,
        output.text is not None,
        output.prompt_id,
        output.decision,
    ])
    latencies_non_negative = (
        output.whisper_latency_ms >= 0
        and output.intent_latency_ms >= 0
        and output.total_latency_ms >= 0
    )
    return bool(has_core_fields and latencies_non_negative)
