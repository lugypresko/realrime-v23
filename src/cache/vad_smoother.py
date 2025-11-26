import time
from collections import deque
from typing import Deque, Tuple


class VADSmoother:
    def __init__(self, window_ms: int = 400, enter_threshold: float = 0.6, exit_threshold: float = 0.3):
        self.window_ms = window_ms
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.samples: Deque[Tuple[float, float]] = deque()
        self.state = False  # False = silent, True = speaking

    def _prune(self, now: float):
        cutoff = now - (self.window_ms / 1000.0)
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def update(self, vad_score: float, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        self.samples.append((now, vad_score))
        self._prune(now)
        if not self.samples:
            return self.state
        avg = sum(score for _, score in self.samples) / len(self.samples)
        if self.state:
            if avg < self.exit_threshold:
                self.state = False
        else:
            if avg > self.enter_threshold:
                self.state = True
        return self.state

    def is_speaking(self) -> bool:
        return self.state
