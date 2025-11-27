from __future__ import annotations

import statistics
import threading
import time
from typing import List

from src.event_bus import EventBus


def test_event_bus_ordering_and_latency():
    events = []
    latencies_ms: List[float] = []
    done = threading.Event()
    total = 10_000

    bus = EventBus(max_queue_size=total + 10, max_workers=1)
    original_submit = bus._executor.submit
    bus._executor.submit = lambda fn, payload: fn(payload)  # type: ignore

    def handler(payload):
        latencies_ms.append((time.perf_counter() - payload["sent"]) * 1000)
        events.append(payload["id"])
        if len(events) == total:
            done.set()

    bus.subscribe("seq", handler)

    for idx in range(total):
        bus.publish("seq", {"id": idx, "sent": time.perf_counter()})

    assert done.wait(timeout=2), "Event bus did not deliver all events in time"
    assert events == list(range(total)), "Events delivered out of order"

    avg_latency = statistics.mean(latencies_ms)
    p99_latency = sorted(latencies_ms)[int(0.99 * len(latencies_ms))]
    assert avg_latency < 2.0, f"Average latency too high: {avg_latency}ms"
    assert p99_latency < 5.0, f"p99 latency too high: {p99_latency}ms"

    bus._executor.submit = original_submit  # type: ignore
    bus.shutdown()


def test_event_bus_bounded_queue_drops_oldest():
    bus = EventBus(max_queue_size=200, max_workers=2)
    total = 1_000

    for idx in range(total):
        bus.publish("bounded", {"id": idx})

    snapshot = bus.get_queue_snapshot("bounded")
    assert len(snapshot) == 200, "Queue exceeded bound or not filled as expected"
    assert [item["id"] for item in snapshot] == list(range(total - 200, total)), "Oldest events were not dropped"
    bus.shutdown()
