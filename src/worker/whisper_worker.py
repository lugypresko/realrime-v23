from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from src.audio_ring_buffer import AudioRingBuffer
from src.event_bus import EventBus
from src.sentinel.services import SilencePolicy
from src.worker.services import InferenceService

try:
    import numpy as np
except Exception:  # pragma: no cover
    from src.mocks import mock_numpy as np


class WhisperWorker(threading.Thread):
    """Dedicated worker that pulls audio frames and runs Whisper off the audio callback thread."""

    def __init__(
        self,
        ring_buffer: AudioRingBuffer,
        inference_service: InferenceService,
        event_bus: EventBus,
        sample_rate: int,
        frames_per_window: int,
        vad_model: Optional[Callable[[np.ndarray, int], float]] = None,
        silence_policy: Optional[SilencePolicy] = None,
    ):
        super().__init__(daemon=True, name="whisper-worker")
        self.ring_buffer = ring_buffer
        self.inference_service = inference_service
        self.event_bus = event_bus
        self.sample_rate = sample_rate
        self.frames_per_window = frames_per_window
        self.vad_model = vad_model
        self.silence_policy = silence_policy
        self._stop_event = threading.Event()
        self._last_sequence = ring_buffer.sequence

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:  # pragma: no cover - threading
        while not self._stop_event.is_set():
            try:
                self._last_sequence = self.ring_buffer.wait_for_new_data(
                    last_sequence=self._last_sequence, timeout=0.5
                )
            except Exception:
                continue

            if self._stop_event.is_set():
                break

            audio = self.ring_buffer.read_latest(max_frames=self.frames_per_window)
            if audio is None or len(audio) == 0:
                continue

            speech_prob = None
            if self.vad_model:
                try:
                    speech_prob = float(self.vad_model(audio, self.sample_rate))
                except Exception:
                    speech_prob = 0.0

            now = time.monotonic()
            if self.silence_policy and speech_prob is not None:
                silence_event = self.silence_policy.handle_prob(
                    prob=speech_prob,
                    timestamp=now,
                    frames=len(audio),
                    sample_rate=self.sample_rate,
                )
                if silence_event:
                    self.event_bus.publish("silence_gap_event", silence_event)

            inference_start = time.perf_counter()
            result = self.inference_service.transcribe(audio)
            latency_ms = (time.perf_counter() - inference_start) * 1000

            event = {
                "timestamp": now,
                "text": result.get("text", ""),
                "tokens": result.get("tokens"),
                "latency_ms": latency_ms,
                "speech_prob": speech_prob,
            }
            self.event_bus.publish("transcription_event", event)
