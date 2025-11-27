import threading

import pytest

from src.audio_ring_buffer import AudioRingBuffer
from src.event_bus import EventBus


def test_audio_ring_buffer_drops_oldest_and_tracks_sequence():
    buffer = AudioRingBuffer(max_frames=2)
    buffer.push([1])
    first_seq = buffer.sequence
    buffer.push([2])
    buffer.push([3])
    assert buffer.sequence > first_seq
    latest = buffer.read_latest()
    assert list(latest) == [2.0, 3.0]


def test_event_bus_bounded_queue_drops_oldest():
    bus = EventBus(max_queue_size=2, max_workers=2)
    seen = []
    done = threading.Event()

    def handler(event):
        seen.append(event["id"])
        if event["id"] == 3:
            done.set()

    bus.subscribe("test_event", handler)
    bus.publish("test_event", {"id": 1})
    bus.publish("test_event", {"id": 2})
    bus.publish("test_event", {"id": 3})

    done.wait(timeout=1)
    snapshot = bus.get_queue_snapshot("test_event")
    assert [item["id"] for item in snapshot] == [2, 3]
    assert 3 in seen
