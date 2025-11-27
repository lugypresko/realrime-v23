from __future__ import annotations

import threading
import time

try:
    import numpy as np
except Exception:  # pragma: no cover
    from src.mocks import mock_numpy as np

from src.audio_ring_buffer import AudioRingBuffer
from src.event_bus import EventBus
from src.worker.whisper_worker import WhisperWorker


class MockInferenceService:
    def __init__(self, delays=None):
        self.calls = []
        self.delays = delays or [0.05, 0.1, 0.15]

    def transcribe(self, audio):
        delay = self.delays[min(len(self.calls), len(self.delays) - 1)]
        time.sleep(delay)
        self.calls.append(delay)
        return {"text": f"mock-{len(self.calls)}", "tokens": [1, 2, 3]}


def _start_worker_with_frames(frames, delays):
    ring = AudioRingBuffer(max_frames=1)
    bus = EventBus(max_queue_size=32, max_workers=2)
    inference = MockInferenceService(delays=delays)
    results = []
    done = threading.Event()

    def handler(event):
        results.append(event)
        done.set()

    bus.subscribe("transcription_event", handler)

    for frame in frames:
        ring.push(frame)

    worker = WhisperWorker(
        ring_buffer=ring,
        inference_service=inference,
        event_bus=bus,
        sample_rate=16000,
        frames_per_window=1600,
    )
    worker.start()
    return worker, bus, results, done, inference


def test_whisper_worker_cancels_stale_jobs():
    frames = [[0.0] * 1600, [1.0] * 1600, [2.0] * 1600]
    worker, bus, results, done, inference = _start_worker_with_frames(frames, delays=[0.1])

    assert done.wait(timeout=1.0), "Whisper worker did not emit transcription"
    worker.stop()
    worker.join(timeout=1.0)
    bus.shutdown()

    assert len(results) == 1, "Only the latest job should be processed"
    assert inference.calls == [0.1]


def test_whisper_worker_burst_processes_latest_only():
    frames = [[float(i)] * 1600 for i in range(50)]
    worker, bus, results, done, inference = _start_worker_with_frames(frames, delays=[0.05])

    assert done.wait(timeout=1.0), "Transcription not produced from burst"
    worker.stop()
    worker.join(timeout=1.0)
    bus.shutdown()

    assert len(results) == 1, "Burst should coalesce to latest frame"
    assert results[0]["text"].startswith("mock-1")


def test_whisper_worker_latency_under_budget():
    ring = AudioRingBuffer(max_frames=2)
    bus = EventBus(max_queue_size=8, max_workers=2)
    inference = MockInferenceService(delays=[0.12])
    durations = []
    done = threading.Event()

    def handler(event):
        durations.append(event["latency_ms"])
        done.set()

    bus.subscribe("transcription_event", handler)

    worker = WhisperWorker(
        ring_buffer=ring,
        inference_service=inference,
        event_bus=bus,
        sample_rate=16000,
        frames_per_window=1600,
    )
    worker.start()
    ring.push([1.0] * 1600)

    assert done.wait(timeout=1.0), "Latency event not produced"
    worker.stop()
    worker.join(timeout=1.0)
    bus.shutdown()

    assert durations and durations[0] < 200, "Whisper worker latency exceeded budget"
