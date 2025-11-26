from __future__ import annotations

from typing import Dict, Any, List

from src.v2.interfaces import TelemetryClient


class InMemoryTelemetry(TelemetryClient):
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def record(self, event: Dict[str, Any]) -> None:
        self.events.append(event)
