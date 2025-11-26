import time
from typing import Tuple


def classify_mock_intent(text: str) -> Tuple[int, float, float]:
    start = time.monotonic()
    prompt_id = 0
    score = 0.99
    latency = time.monotonic() - start
    return prompt_id, score, latency
