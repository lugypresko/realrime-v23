from __future__ import annotations

import time
from collections import deque
from typing import Deque, List, Optional


MEMORY_RESET_SECONDS = 90
MEMORY_RESET_SECONDS_DEBUG = 10


class RollingMemory:
    """Track recent intents and suggestions with an inactivity reset."""

    def __init__(self, debug: bool = False, max_items: int = 5):
        self.max_items = max_items
        self.reset_seconds = MEMORY_RESET_SECONDS_DEBUG if debug else MEMORY_RESET_SECONDS
        self._intent_history: Deque[str] = deque(maxlen=max_items)
        self._suggestion_history: Deque[str] = deque(maxlen=max_items)
        self._last_access = time.monotonic()

    def _maybe_reset(self) -> None:
        if time.monotonic() - self._last_access > self.reset_seconds:
            self._intent_history.clear()
            self._suggestion_history.clear()
        self._last_access = time.monotonic()

    def record_intent(self, intent: str) -> None:
        self._maybe_reset()
        self._intent_history.append(intent.lower())

    def record_suggestion(self, suggestion: str) -> None:
        self._maybe_reset()
        self._suggestion_history.append(suggestion)

    def recent_intents(self) -> List[str]:
        self._maybe_reset()
        return list(self._intent_history)

    def recent_suggestions(self) -> List[str]:
        self._maybe_reset()
        return list(self._suggestion_history)

    def contains_suggestion(self, suggestion: str) -> bool:
        self._maybe_reset()
        return suggestion in self._suggestion_history

    def reset(self) -> None:
        self._intent_history.clear()
        self._suggestion_history.clear()
        self._last_access = time.monotonic()

    def last_intent(self) -> Optional[str]:
        self._maybe_reset()
        return self._intent_history[-1] if self._intent_history else None

    def last_suggestion(self) -> Optional[str]:
        self._maybe_reset()
        return self._suggestion_history[-1] if self._suggestion_history else None

