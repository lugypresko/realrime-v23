from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    from src.mocks import mock_numpy as np

from src.logging.structured_logger import log_event


class ReplayBuffer:
    def __init__(self, sample_rate: int = 16000, duration_sec: int = 20):
        self.sample_rate = sample_rate
        self.max_len = sample_rate * duration_sec
        self.audio: Deque[float] = deque(maxlen=self.max_len)
        self.vad: Deque[float] = deque(maxlen=self.max_len)

    def add(self, audio_chunk, vad_prob: float) -> None:
        if audio_chunk is None:
            return
        for sample in audio_chunk:
            self.audio.append(float(sample))
            self.vad.append(float(vad_prob))

    def dump_to_disk(self, path: str | Path, event_id: str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        audio_array = np.array(list(self.audio))
        vad_array = np.array(list(self.vad))
        np.savez(target, audio=audio_array, vad=vad_array, event_id=event_id)
        log_event({"type": "REPLAY_DUMP", "path": str(target), "event_id": event_id})
        return target
