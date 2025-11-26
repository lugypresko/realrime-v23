from __future__ import annotations

import time
from typing import Callable, Iterable

from src.v2.interfaces import (
    SilenceEvent,
    WorkerOutput,
    TelemetryClient,
    STTEngine,
    IntentClassifier,
    GovernorService,
    WarmStarter,
    LatencyTracker,
)
from src.v2.interfaces import validate_silence_event, validate_worker_output


class WorkerProcess:
    """Consumes silence events and emits worker outputs using injected interfaces only."""

    def __init__(
        self,
        stt: STTEngine,
        intent_classifier: IntentClassifier,
        governor: GovernorService,
        telemetry_client: TelemetryClient,
        warmstarter: WarmStarter | None = None,
        latency_tracker: LatencyTracker | None = None,
        max_pending_events: int | None = None,
    ) -> None:
        self.stt = stt
        self.intent_classifier = intent_classifier
        self.governor = governor
        self.telemetry = telemetry_client
        self.warmstarter = warmstarter
        self.latency_tracker = latency_tracker
        self.max_pending_events = max_pending_events

    def run(
        self,
        events: Iterable[SilenceEvent],
        emit: Callable[[WorkerOutput], None],
        pending_events: Callable[[], int] | None = None,
    ) -> None:
        if self.warmstarter:
            self.warmstarter.warm_start()

        for event in events:
            if not validate_silence_event(event):
                continue
            if self.max_pending_events is not None and pending_events and pending_events() > self.max_pending_events:
                self.telemetry.record({"type": "worker_drop", "event_id": event.event_id, "reason": "backpressure"})
                continue

            start = time.monotonic()
            inference_result = self.stt.transcribe(event.audio)
            whisper_latency = (time.monotonic() - start) * 1000
            intent_result = self.intent_classifier.classify(inference_result.text)
            intent_latency = (time.monotonic() - start) * 1000 - whisper_latency
            decision = self.governor.decide(event, inference_result, intent_result)
            total_latency = (time.monotonic() - start) * 1000

            output = WorkerOutput(
                event_id=event.event_id,
                text=inference_result.text,
                prompt_id=intent_result.prompt_id,
                score=intent_result.score,
                whisper_latency_ms=whisper_latency,
                intent_latency_ms=intent_latency,
                total_latency_ms=total_latency,
                decision=decision.decision,
                metadata={"reason": decision.reason},
            )

            if self.latency_tracker:
                self.latency_tracker.record({
                    "whisper": whisper_latency,
                    "intent": intent_latency,
                    "total": total_latency,
                })

            if validate_worker_output(output):
                self.telemetry.record({
                    "type": "worker_output",
                    "event_id": output.event_id,
                    "decision": output.decision,
                    "latency_ms": output.total_latency_ms,
                })
                emit(output)
