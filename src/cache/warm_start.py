import time
from typing import Any

from src.logging.structured_logger import log_event


SILENCE_DURATION_SEC = 0.4
SAMPLE_RATE = 16000


class _DummyAudio:
    def __init__(self, length: int):
        self.length = length

    def astype(self, *_args, **_kwargs):
        return self


def warm_start_whisper(model: Any, sample_rate: int = SAMPLE_RATE) -> float:
    """Run a dummy whisper inference to remove cold-start latency."""
    silence_frames = int(sample_rate * SILENCE_DURATION_SEC)
    try:
        import numpy as np
    except Exception:  # pragma: no cover - fallback when numpy unavailable
        from src.mocks import mock_numpy as np

    silent_audio = np.zeros(silence_frames, dtype="float32")

    start = time.monotonic()
    try:
        _segments, _info = model.transcribe(
            silent_audio,
            beam_size=1,
            language="en",
            temperature=0.0,
        )
    except Exception:
        # Ignore inference failures in warm start; treat as best-effort.
        pass
    duration_ms = (time.monotonic() - start) * 1000
    log_event({"type": "CACHE_WARM_START", "message": "Whisper warm-start complete", "duration_ms": duration_ms})
    return duration_ms
