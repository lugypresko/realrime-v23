from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class HealthStats:
    total_triggers: int
    avg_latency_ms: Optional[float]
    p95_latency_ms: Optional[float]
    duplicate_triggers: int
    missed_triggers: int
    audio_overruns: int
    stt_failures: int
    intent_failures: int
    memory_drift: Optional[float]
    cpu_spikes: int
    stt_error_rate: float
    intent_quality_flags: int
    stall_flags: dict
    event_id_order_ok: bool
    timestamp_monotonic: bool
    missed_trigger_intervals: int
    duplicate_trigger_intervals: int
    failures: list


class HealthReport:
    def __init__(self, stats: HealthStats, ready: bool) -> None:
        self.stats = stats
        self.ready = ready

    @classmethod
    def from_stats(cls, stats: HealthStats) -> "HealthReport":
        ready = (
            (stats.p95_latency_ms is not None and stats.p95_latency_ms < 1500)
            and stats.stt_error_rate < 5
            and stats.missed_triggers <= 3
            and stats.audio_overruns == 0
            and (stats.memory_drift is None or stats.memory_drift < 20)
            and stats.event_id_order_ok
            and stats.timestamp_monotonic
            and not any(stats.stall_flags.values())
        )
        return cls(stats, ready)

    def render(self) -> str:
        lines = ["--- DRY LIVE PIPELINE REPORT ---"]
        lines.append(f"Total triggers: {self.stats.total_triggers}")
        lines.append(f"Average end-to-end latency: {self._fmt(self.stats.avg_latency_ms)} ms")
        lines.append(f"p95 latency: {self._fmt(self.stats.p95_latency_ms)} ms")
        lines.append(f"Duplicate triggers: {self.stats.duplicate_triggers}")
        lines.append(f"Missed triggers: {self.stats.missed_triggers}")
        lines.append(f"Audio overruns: {self.stats.audio_overruns}")
        lines.append(f"STT failures: {self.stats.stt_failures}")
        lines.append(f"Intent failures: {self.stats.intent_failures}")
        lines.append(f"Memory drift: {self._fmt(self.stats.memory_drift)} MB")
        lines.append(f"CPU spikes: {self.stats.cpu_spikes}")
        lines.append(f"STT error rate: {self.stats.stt_error_rate:.2f}%")
        lines.append(f"Intent quality flags: {self.stats.intent_quality_flags}")
        lines.append(f"Stall flags: {self.stats.stall_flags}")
        lines.append(f"Event ID order OK: {self.stats.event_id_order_ok}")
        lines.append(f"Timestamp monotonic: {self.stats.timestamp_monotonic}")
        lines.append(f"Duplicate intervals (<300ms): {self.stats.duplicate_trigger_intervals}")
        lines.append(f"Missed trigger intervals (>2s): {self.stats.missed_trigger_intervals}")
        lines.append(f"Failures: {self.stats.failures}")
        lines.append(f"READY_FOR_LIVE: {'YES' if self.ready else 'NO'}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "stats": self.stats.__dict__,
            "ready": self.ready,
        }

    def _fmt(self, value: Optional[float]) -> str:
        return "n/a" if value is None else f"{value:.2f}"
