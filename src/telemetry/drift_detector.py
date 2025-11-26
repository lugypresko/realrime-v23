from collections import deque
from typing import Deque

from src.telemetry.telemetry_writer import write_event


class DriftDetector:
    def __init__(self, window: int = 20, threshold_ms: float = 1300.0) -> None:
        self.window = window
        self.threshold_ms = threshold_ms
        self.latencies: Deque[float] = deque(maxlen=window)

    def add(self, latency_ms: float) -> bool:
        self.latencies.append(latency_ms)
        if len(self.latencies) < self.window:
            return False
        p95 = sorted(self.latencies)[int(0.95 * (len(self.latencies) - 1))]
        if p95 > self.threshold_ms:
            write_event({"type": "DRIFT_WARNING", "p95": p95, "threshold_ms": self.threshold_ms})
            return True
        return False
