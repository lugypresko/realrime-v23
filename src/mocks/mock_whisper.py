import time
from typing import Tuple


def transcribe_mock(audio) -> Tuple[str, float]:
    start = time.monotonic()
    text = "mock transcript"
    latency = time.monotonic() - start
    return text, latency
