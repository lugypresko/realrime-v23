from collections import deque
from typing import Deque, Dict, List

try:
    import numpy as np
except Exception:  # pragma: no cover
    from src.mocks import mock_numpy as np

from src.logging.structured_logger import log_event


class LatencyHistory:
    def __init__(self, max_events: int = 50):
        self.max_events = max_events
        self.records: Deque[Dict] = deque(maxlen=max_events)

    def add(self, record: Dict) -> None:
        self.records.append(record)

    def _percentile(self, values: List[float], percentile: float) -> float:
        if not values:
            return 0.0
        try:
            return float(np.percentile(values, percentile))
        except Exception:
            sorted_vals = sorted(values)
            k = int(len(sorted_vals) * percentile / 100)
            k = min(k, len(sorted_vals) - 1)
            return float(sorted_vals[k])

    def metrics(self) -> Dict:
        whisper = [rec.get("whisper_latency", 0) * 1000 for rec in self.records]
        intent = [rec.get("intent_latency", 0) * 1000 for rec in self.records]
        total = [rec.get("event_age", 0) * 1000 for rec in self.records]
        decisions = [rec.get("decision") for rec in self.records]
        suppression_rate = 0.0
        if decisions:
            suppressed = sum(1 for d in decisions if d != "SUCCESS")
            suppression_rate = suppressed / len(decisions)

        return {
            "p50": {
                "whisper_ms": self._percentile(whisper, 50),
                "intent_ms": self._percentile(intent, 50),
                "total_ms": self._percentile(total, 50),
            },
            "p95": {
                "whisper_ms": self._percentile(whisper, 95),
                "intent_ms": self._percentile(intent, 95),
                "total_ms": self._percentile(total, 95),
            },
            "p99": {
                "whisper_ms": self._percentile(whisper, 99),
                "intent_ms": self._percentile(intent, 99),
                "total_ms": self._percentile(total, 99),
            },
            "suppression_rate": suppression_rate,
        }

    def log_warnings(self):
        metrics = self.metrics()
        if metrics["p95"]["whisper_ms"] > 1300:
            log_event({"type": "LATENCY_WARNING", "message": "Whisper slowing down", "p95_ms": metrics["p95"]["whisper_ms"]})
        return metrics
