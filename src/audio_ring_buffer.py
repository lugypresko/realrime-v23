from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - fallback when numpy missing
    from src.mocks import mock_numpy as np


class AudioRingBuffer:
    """Bounded, thread-safe ring buffer for audio frames with drop-oldest semantics."""

    def __init__(self, max_frames: int):
        self._buffer: Deque[np.ndarray] = deque(maxlen=max_frames)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._sequence = 0

    def push(self, frame) -> None:
        try:
            frame_array = np.asarray(frame, dtype=np.float32).copy()
        except Exception:
            frame_array = np.array(frame).copy()
        with self._not_empty:
            if len(self._buffer) == self._buffer.maxlen:
                self._buffer.popleft()
            self._buffer.append(frame_array)
            self._sequence += 1
            self._not_empty.notify_all()

    def read_latest(self, max_frames: Optional[int] = None) -> Optional[np.ndarray]:
        with self._lock:
            if not self._buffer:
                return None
            frames = list(self._buffer)
        if max_frames is not None:
            frames = frames[-max_frames:]
        if not frames:
            return None
        try:
            return np.concatenate(frames)
        except Exception:
            combined = []
            for frame in frames:
                combined.extend(list(frame))
            return combined

    def wait_for_data(self, timeout: Optional[float] = None) -> bool:
        with self._not_empty:
            return self._not_empty.wait_for(lambda: len(self._buffer) > 0, timeout=timeout)

    def wait_for_new_data(self, last_sequence: int, timeout: Optional[float] = None) -> int:
        with self._not_empty:
            self._not_empty.wait_for(lambda: self._sequence != last_sequence, timeout=timeout)
            return self._sequence

    @property
    def sequence(self) -> int:
        with self._lock:
            return self._sequence

    def __len__(self) -> int:  # pragma: no cover - trivial
        with self._lock:
            return len(self._buffer)
