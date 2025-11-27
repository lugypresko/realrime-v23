from __future__ import annotations

import threading
import time
from collections import deque

from src.dialogue_brain.brain import DialogueBrain
from src.event_bus import EventBus

MOCK_TEXT = "we are evaluating options"
MOCK_LATENCY_MS = 120.0
INTENT_LABEL = "Explore Options (probe)"


def main() -> None:
    event_bus = EventBus(max_queue_size=32, max_workers=4)
    brain = DialogueBrain(debug=True, event_bus=event_bus)
    frame_queue: deque = deque(maxlen=10)
    completion = threading.Event()

    def on_silence(_payload):
        frame_queue.append(_payload)
        event_bus.publish(
            "stt_result",
            {"timestamp": time.time(), "text": MOCK_TEXT, "latency_ms": MOCK_LATENCY_MS},
        )

    def on_stt(event):
        print(f"[MOCK STT]   \"{event.get('text', '')}\"")
        intent_payload = {
            "timestamp": time.time(),
            "text": event.get("text", ""),
            "intent": INTENT_LABEL,
            "intent_ms": 10.0,
            "latency_ms": event.get("latency_ms", 0.0),
        }
        event_bus.publish("intent_result", intent_payload)

    def on_intent(event):
        result = brain.process(text=event.get("text", ""), intent=event.get("intent", ""))
        total_latency = (
            event.get("latency_ms", 0.0)
            + event.get("intent_ms", 0.0)
            + result.get("latency_ms", 0.0)
        )
        print(f"[INTENT]     {event.get('intent', '')} (0.41)")
        print(f"[STATE]      {result.get('state')}")
        print(f"[SUGGEST]    \"{result.get('suggestion')}\"")
        print(f"[LATENCY]    {total_latency:.0f}ms")
        completion.set()

    event_bus.subscribe("silence_gap", on_silence)
    event_bus.subscribe("stt_result", on_stt)
    event_bus.subscribe("intent_result", on_intent)

    for idx in range(10):
        completion.clear()
        event_bus.publish("silence_gap", {"buffer": b"FAKE_AUDIO", "idx": idx})
        completion.wait(timeout=1)

    event_bus.shutdown()


if __name__ == "__main__":
    main()
