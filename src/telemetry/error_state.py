import time
from collections import deque
from typing import Deque

from src.telemetry.telemetry_writer import write_event


class ErrorStateManager:
    STATES = {"SILENT_FAIL", "RETRY", "RECOVERED", "SUPPRESSED", "FATAL"}

    def __init__(self, window_sec: float = 10.0) -> None:
        self.window_sec = window_sec
        self.whisper_failures: Deque[float] = deque()
        self.vad_inactive_since: float | None = None
        self.safe_mode = False

    def record_whisper_failure(self) -> None:
        now = time.monotonic()
        self.whisper_failures.append(now)
        self._trim(now)
        if len(self.whisper_failures) >= 2:
            self.safe_mode = True
            write_event({"type": "ERROR", "state": "RETRY", "message": "Whisper failures exceeded threshold"})

    def record_whisper_success(self) -> None:
        if self.safe_mode and not self.whisper_failures:
            self.safe_mode = False
            write_event({"type": "ERROR", "state": "RECOVERED"})

    def record_vad_inactivity(self) -> None:
        now = time.monotonic()
        if self.vad_inactive_since is None:
            self.vad_inactive_since = now
        elif now - self.vad_inactive_since > self.window_sec:
            self.safe_mode = True
            write_event({"type": "ERROR", "state": "SILENT_FAIL", "message": "VAD inactive"})

    def clear_vad_inactivity(self) -> None:
        self.vad_inactive_since = None

    def _trim(self, now: float) -> None:
        while self.whisper_failures and now - self.whisper_failures[0] > self.window_sec:
            self.whisper_failures.popleft()

    def should_use_safe_mode(self) -> bool:
        return self.safe_mode
