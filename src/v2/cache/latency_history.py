from __future__ import annotations

from collections import deque
from typing import Deque, Dict

from src.v2.interfaces import LatencyTracker


class RollingLatencyTracker(LatencyTracker):
    def __init__(self, max_points: int = 50) -> None:
        self.max_points = max_points
        self._values: Deque[Dict[str, float]] = deque(maxlen=max_points)

    def record(self, metrics: Dict[str, float]) -> None:
        self._values.append(metrics)

    @property
    def values(self):
        return list(self._values)
