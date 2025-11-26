import statistics
from collections import deque
from typing import Deque, Dict, List, Optional

from src.telemetry.telemetry_writer import write_event


class TelemetryAggregator:
    def __init__(self, max_events: int = 50) -> None:
        self.max_events = max_events
        self.whisper_latencies: Deque[float] = deque(maxlen=max_events)
        self.intent_latencies: Deque[float] = deque(maxlen=max_events)
        self.total_latencies: Deque[float] = deque(maxlen=max_events)
        self.decisions: Deque[str] = deque(maxlen=max_events)

    def add_measurement(
        self,
        whisper_latency: float,
        intent_latency: float,
        total_latency: float,
        decision: str,
    ) -> Dict[str, float]:
        self.whisper_latencies.append(whisper_latency)
        self.intent_latencies.append(intent_latency)
        self.total_latencies.append(total_latency)
        self.decisions.append(decision)
        metrics = self.summary()
        write_event({"type": "TELEMETRY_METRIC", **metrics, "decision": decision})
        return metrics

    def _percentile(self, values: List[float], percentile: float) -> Optional[float]:
        if not values:
            return None
        values_sorted = sorted(values)
        k = (len(values_sorted) - 1) * percentile
        f = int(k)
        c = min(f + 1, len(values_sorted) - 1)
        if f == c:
            return values_sorted[int(k)]
        d0 = values_sorted[f] * (c - k)
        d1 = values_sorted[c] * (k - f)
        return d0 + d1

    def summary(self) -> Dict[str, float]:
        whisper_list = list(self.whisper_latencies)
        intent_list = list(self.intent_latencies)
        total_list = list(self.total_latencies)
        return {
            "whisper_p50": self._percentile(whisper_list, 0.5) or 0.0,
            "whisper_p95": self._percentile(whisper_list, 0.95) or 0.0,
            "intent_p95": self._percentile(intent_list, 0.95) or 0.0,
            "total_p95": self._percentile(total_list, 0.95) or 0.0,
            "suppression_rate": (self.decisions.count("SUPPRESSED_LATE") / len(self.decisions)) if self.decisions else 0.0,
        }
