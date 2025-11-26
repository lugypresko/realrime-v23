import time
from typing import Dict

from src.telemetry.telemetry_writer import write_event


class PipelineWatchdog:
    def __init__(self, timeout_sec: float = 2.0) -> None:
        self.timeout_sec = timeout_sec
        self.inflight: Dict[str, float] = {}

    def start(self, event_id: str) -> None:
        self.inflight[event_id] = time.monotonic()

    def clear(self, event_id: str) -> None:
        self.inflight.pop(event_id, None)

    def check(self) -> None:
        now = time.monotonic()
        expired = [eid for eid, ts in self.inflight.items() if now - ts > self.timeout_sec]
        for eid in expired:
            write_event({"type": "WATCHDOG_TIMEOUT", "event_id": eid, "timeout_sec": self.timeout_sec})
            self.clear(eid)
