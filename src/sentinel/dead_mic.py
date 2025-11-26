import time
from typing import Optional

from src.logging.structured_logger import log_event


class DeadMicDetector:
    def __init__(self, max_silence_sec: float = 4.0, rms_floor: float = 1e-4):
        self.max_silence_sec = max_silence_sec
        self.rms_floor = rms_floor
        self.last_signal_ts: Optional[float] = None

    def update(self, audio_chunk, vad_prob: float, now: float) -> bool:
        amplitude = 0.0
        if audio_chunk is not None:
            amplitude = sum(abs(float(x)) for x in audio_chunk) / max(len(audio_chunk), 1)
        if amplitude > self.rms_floor or vad_prob > 0.05:
            self.last_signal_ts = now
            return False

        if self.last_signal_ts is None:
            self.last_signal_ts = now
            return False

        if now - self.last_signal_ts > self.max_silence_sec:
            log_event({"type": "MIC_DEAD", "timestamp": now})
            return True
        return False
