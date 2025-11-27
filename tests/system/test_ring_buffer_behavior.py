from __future__ import annotations

import random
import time

try:
    import numpy as np
except Exception:  # pragma: no cover
    from src.mocks import mock_numpy as np

from src.audio_ring_buffer import AudioRingBuffer


def test_ring_buffer_bounded_and_contiguous():
    max_frames = 32
    buffer = AudioRingBuffer(max_frames=max_frames)
    max_observed = 0
    history = []

    for idx in range(100_000):
        frame = [float(idx)] * 10
        buffer.push(frame)
        history.append(frame)
        max_observed = max(max_observed, len(buffer))
    assert max_observed <= max_frames, "Ring buffer exceeded max capacity"

    expected = []
    for item in history[-max_frames:]:
        expected.extend(item)
    combined = buffer.read_latest()
    assert list(combined) == expected, "Contiguous chunk mismatch"

    buffer = AudioRingBuffer(max_frames=20)
    slices = []
    for _ in range(30):
        size = random.randint(50, 300)
        chunk = list(range(size))
        slices.append(chunk)
        buffer.push(chunk)
    expected_concat = []
    for chunk in slices[-20:]:
        expected_concat.extend(chunk)
    assert list(buffer.read_latest()) == expected_concat


def test_ring_buffer_read_performance():
    buffer = AudioRingBuffer(max_frames=20)
    for i in range(20):
        buffer.push([float(x + i) for x in range(200)])

    durations = []
    for _ in range(10_000):
        start = time.perf_counter()
        _ = buffer.read_latest()
        durations.append((time.perf_counter() - start) * 1000)

    durations.sort()
    p99 = durations[int(0.99 * len(durations))]
    assert p99 < 3.0, f"Ring buffer read too slow: p99={p99}ms"
