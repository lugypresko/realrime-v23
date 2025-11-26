from __future__ import annotations

import time
from src.v2.interfaces import WarmStarter


class NoOpWarmStarter(WarmStarter):
    def warm_start(self) -> None:
        return None


class TimedWarmStarter(WarmStarter):
    def __init__(self, warm_func) -> None:
        self._warm_func = warm_func
        self.duration_ms: float | None = None

    def warm_start(self) -> None:
        start = time.time()
        self._warm_func()
        self.duration_ms = (time.time() - start) * 1000
