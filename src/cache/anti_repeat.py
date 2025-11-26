from collections import deque
from typing import Deque

from src.logging.structured_logger import log_event


class AntiRepeatCache:
    def __init__(self, max_len: int = 5):
        self.history: Deque[str] = deque(maxlen=max_len)

    def should_suppress(self, prompt_id: str, score_diff: float) -> bool:
        if prompt_id in self.history and score_diff < 0.1:
            log_event(
                {
                    "type": "SUPPRESS_REPEAT",
                    "prompt_id": prompt_id,
                    "score_diff": score_diff,
                }
            )
            return True
        return False

    def record(self, prompt_id: str) -> None:
        self.history.append(prompt_id)
