try:
    import numpy as np
except Exception:  # pragma: no cover - fallback for environments without numpy
    from src.mocks import mock_numpy as np

SAMPLE_RATE = 16000
BUFFER_DURATION = 1.2


def generate_mock_buffer() -> np.ndarray:
    length = int(SAMPLE_RATE * BUFFER_DURATION)
    return np.zeros(length, dtype=np.float32)
