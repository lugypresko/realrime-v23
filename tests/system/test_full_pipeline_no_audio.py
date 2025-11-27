from __future__ import annotations

import threading
import time
from collections import deque
from typing import Dict, List

from src.event_bus import EventBus
from src.dialogue_brain.brain import DialogueBrain


MOCK_TEXT = "mock transcription: we are evaluating options"
MOCK_LATENCY_MS = 120.0


class MockWhisperWorker:
    """Deterministic whisper stand-in that processes silence events immediately."""

    def __init__(self, event_bus: EventBus, events_log: List[str]):
        self.event_bus = event_bus
        self.queue: deque = deque(maxlen=10)
        self.max_queue_depth = 0
        self.events_log = events_log
        self.event_bus.subscribe("silence_gap", self._on_silence_gap)

    def _on_silence_gap(self, payload: Dict):
        self.queue.append(payload)
        self.max_queue_depth = max(self.max_queue_depth, len(self.queue))
        newest = self.queue[-1]
        self.queue.clear()
        event = {
            "timestamp": time.time(),
            "text": MOCK_TEXT,
            "latency_ms": MOCK_LATENCY_MS,
            "buffer": newest.get("buffer"),
        }
        self.events_log.append("stt_result")
        self.event_bus.publish("stt_result", event)


class IntentClassifier:
    def __init__(self, event_bus: EventBus, events_log: List[str], intent_target: Dict):
        self.event_bus = event_bus
        self.events_log = events_log
        self.intent_target = intent_target
        self.event_bus.subscribe("stt_result", self._on_stt)

    def _on_stt(self, event: Dict):
        start = time.perf_counter()
        intent_label = "Explore Options (probe)"
        latency_ms = (time.perf_counter() - start) * 1000
        payload = {
            "timestamp": time.time(),
            "text": event.get("text", ""),
            "intent": intent_label,
            "intent_ms": latency_ms,
            "latency_ms": event.get("latency_ms", 0.0),
        }
        self.intent_target.update(payload)
        self.events_log.append("intent_result")
        self.event_bus.publish("intent_result", payload)


class BrainConsumer:
    def __init__(self, event_bus: EventBus, brain: DialogueBrain, events_log: List[str], brain_target: Dict):
        self.event_bus = event_bus
        self.brain = brain
        self.events_log = events_log
        self.brain_target = brain_target
        self.event_bus.subscribe("intent_result", self._on_intent)

    def _on_intent(self, event: Dict):
        start = time.perf_counter()
        result = self.brain.process(text=event.get("text", ""), intent=event.get("intent", ""))
        brain_ms = (time.perf_counter() - start) * 1000
        payload = {
            "timestamp": time.time(),
            "state": result.get("state"),
            "suggestion": result.get("suggestion"),
            "cache_hit": result.get("cache_hit", False),
            "latency_ms": result.get("latency_ms", brain_ms),
            "brain_ms": result.get("latency_ms", brain_ms),
            "intent_ms": event.get("intent_ms", 0.0),
            "whisper_ms": event.get("latency_ms", 0.0),
            "intent": event.get("intent", ""),
        }
        self.brain_target.update(payload)
        self.events_log.append("brain_result")
        self.event_bus.publish("brain_result", payload)


class HUDCapture:
    def __init__(self, event_bus: EventBus, events_log: List[str]):
        self.rendered = []
        self.events_log = events_log
        self.event_bus = event_bus
        self.event_bus.subscribe("brain_result", self._on_brain)
        self.lock = threading.Lock()

    def _on_brain(self, event: Dict):
        with self.lock:
            line = (
                f"INTENT: {event.get('intent', '')} | "
                f"STATE: {event.get('state', '')} | "
                f"SUGGEST: {event.get('suggestion', '')} | "
                f"LATENCY: {event.get('whisper_ms', 0)+event.get('intent_ms', 0)+event.get('brain_ms', 0):.0f}ms"
            )
            self.rendered.append(line)
            self.events_log.append("hud_render")
            self.event_bus.publish("hud_render", event)


def _assert_subsequence(subsequence: List[str], sequence: List[str]):
    it = iter(sequence)
    assert all(item in it for item in subsequence), f"Missing subsequence {subsequence} in {sequence}"


def test_full_pipeline_no_audio():
    event_bus = EventBus(max_queue_size=50, max_workers=4)
    events_log: List[str] = []
    intent_capture: Dict = {}
    brain_capture: Dict = {}

    whisper = MockWhisperWorker(event_bus, events_log)
    brain = DialogueBrain(debug=True, event_bus=event_bus)
    IntentClassifier(event_bus, events_log, intent_capture)
    BrainConsumer(event_bus, brain, events_log, brain_capture)
    hud = HUDCapture(event_bus, events_log)
    _ = hud  # silence unused

    hud_event = threading.Event()

    def _hud_done(_payload: Dict):
        hud_event.set()

    event_bus.subscribe("hud_render", _hud_done)

    for idx in range(5):
        events_log.append("silence_gap")
        event_bus.publish("silence_gap", {"buffer": b"FAKE_AUDIO_FRAME", "idx": idx})

    assert hud_event.wait(timeout=2), "Pipeline did not complete in time"

    _assert_subsequence([
        "silence_gap",
        "stt_result",
        "intent_result",
        "brain_result",
        "hud_render",
    ], events_log)

    assert any(evt == "stt_result" for evt in events_log), "No stt_result emitted"
    assert intent_capture.get("intent") is not None, "Intent classifier did not run"

    assert isinstance(brain_capture, dict)
    assert "state" in brain_capture
    assert "suggestion" in brain_capture
    assert "latency_ms" in brain_capture

    total_latency = (
        brain_capture.get("whisper_ms", 0)
        + brain_capture.get("intent_ms", 0)
        + brain_capture.get("brain_ms", 0)
    )
    assert total_latency < 200, f"Pipeline latency too high: {total_latency}ms"

    hud_payload = "\n".join(hud.rendered)
    assert "INTENT" in hud_payload
    assert "SUGGEST" in hud_payload
    assert "STATE" in hud_payload

    assert len(event_bus.get_queue_snapshot("silence_gap")) <= 50
    assert len(event_bus.get_queue_snapshot("brain_result")) <= 50
    assert whisper.max_queue_depth <= 10

    event_bus.shutdown()
