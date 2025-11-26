from __future__ import annotations

import time
from typing import Optional

from src.v2.interfaces import (
    SilenceEvent,
    InferenceResult,
    IntentResult,
    GovernorService,
    RepeatFilter,
    WorkerDecision,
)


class SimpleGovernor(GovernorService):
    """Applies latency cutoff and optional repeat suppression."""

    def __init__(self, repeat_filter: Optional[RepeatFilter] = None, max_latency_ms: float = 1500.0) -> None:
        self.repeat_filter = repeat_filter
        self.max_latency_ms = max_latency_ms

    def decide(self, event: SilenceEvent, inference: InferenceResult, intent: IntentResult) -> WorkerDecision:
        now_ms = time.monotonic() * 1000
        age = now_ms - event.timestamp * 1000
        if age > self.max_latency_ms:
            return WorkerDecision(decision="SUPPRESSED_LATE", reason="age_exceeded")
        if self.repeat_filter and self.repeat_filter.should_filter(intent.prompt_id, intent.score):
            return WorkerDecision(decision="SUPPRESSED_REPEAT", reason="repeat_filter")
        return WorkerDecision(decision="SUCCESS", reason=None)
