from __future__ import annotations

from collections import deque
from typing import Deque

from src.v2.interfaces import RepeatFilter


class LastNRepeatFilter(RepeatFilter):
    def __init__(self, capacity: int = 5, score_delta: float = 0.1) -> None:
        self.capacity = capacity
        self.score_delta = score_delta
        self.history: Deque[tuple[str, float]] = deque(maxlen=capacity)

    def should_filter(self, prompt_id: str, score: float) -> bool:
        for prev_id, prev_score in self.history:
            if prev_id == prompt_id and abs(prev_score - score) < self.score_delta:
                return True
        self.history.append((prompt_id, score))
        return False
