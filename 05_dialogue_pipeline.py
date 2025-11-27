"""
Event-driven Dialogue Pipeline (audio -> whisper -> intent -> brain -> HUD)
This runner avoids polling/sleeps by wiring each stage to the EventBus.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Iterable, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    from src.mocks import mock_numpy as np

from rich.live import Live

from src.audio_ring_buffer import AudioRingBuffer
from src.dialogue_brain.brain import DialogueBrain
from src.event_bus import EventBus
from src.worker.whisper_worker import WhisperWorker
from ui.minimal_hud import HUDState, MinimalHUD

SAMPLE_RATE = 16000
BLOCK_SIZE = 512
BUFFER_DURATION = 1.2
BUFFER_FRAMES = int((SAMPLE_RATE * BUFFER_DURATION) / BLOCK_SIZE) + 1

SAMPLE_TEXT = [
    "We're ready to move once the decision maker signs.",
    "That timeline is tight for us.",
    "Price might be an issue with finance.",
    "I just want to understand who will own rollout.",
    "What happens if the plan slips again?",
]

INTENT_MAP = {
    "ready": "Decision Ready",
    "timeline": "Objection - Timeline",
    "price": "Objection - Price",
    "cost": "Objection - Price",
    "decision": "Decision Inquiry",
    "owner": "Discovery - Stakeholders",
}


def classify_intent(text: str) -> Tuple[str, float]:
    start = time.perf_counter()
    lowered = text.lower()
    for key, intent in INTENT_MAP.items():
        if key in lowered:
            latency_ms = (time.perf_counter() - start) * 1000
            return intent, latency_ms
    latency_ms = (time.perf_counter() - start) * 1000
    return "General Inquiry", latency_ms


@dataclass
class ScriptedInferenceService:
    texts: Iterable[str]

    def __post_init__(self):
        self._iterator = iter(self.texts)

    def transcribe(self, audio):
        start = time.perf_counter()
        try:
            text = next(self._iterator)
        except StopIteration:
            text = ""
        latency_ms = (time.perf_counter() - start) * 1000
        return {"text": text, "latency": latency_ms}


class IntentDetector:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.event_bus.subscribe("transcription_event", self._on_transcription)

    def _on_transcription(self, event):
        intent, latency_ms = classify_intent(event.get("text", ""))
        payload = {
            "timestamp": time.time(),
            "intent": intent,
            "text": event.get("text", ""),
            "whisper_ms": event.get("latency_ms", 0.0),
            "intent_ms": latency_ms,
        }
        self.event_bus.publish("intent_event", payload)


class BrainConsumer:
    def __init__(self, event_bus: EventBus):
        self.brain = DialogueBrain(debug=True, event_bus=event_bus)
        self.event_bus = event_bus
        self.event_bus.subscribe("intent_event", self._on_intent)

    def _on_intent(self, event):
        start = time.perf_counter()
        result = self.brain.process(text=event.get("text", ""), intent=event.get("intent", ""))
        brain_ms = (time.perf_counter() - start) * 1000
        payload = {
            "timestamp": time.time(),
            "state": result.get("state", "opening"),
            "suggestion": result.get("suggestion", ""),
            "cache_hit": result.get("cache_hit", False),
            "brain_ms": result.get("latency_ms", brain_ms),
            "intent_ms": event.get("intent_ms", 0.0),
            "whisper_ms": event.get("whisper_ms", 0.0),
            "text": event.get("text", ""),
            "intent": event.get("intent", ""),
        }
        self.event_bus.publish("brain_event", payload)


class HUDSubscriber:
    def __init__(self, hud: MinimalHUD, event_bus: EventBus):
        self.hud = hud
        self._state = HUDState()
        self._lock = threading.Lock()
        self._live: Live | None = None
        event_bus.subscribe("transcription_event", self._on_transcription)
        event_bus.subscribe("intent_event", self._on_intent)
        event_bus.subscribe("brain_event", self._on_brain)
        event_bus.subscribe("suggestion_event", self._on_suggestion)
        event_bus.subscribe("state_transition_event", self._on_state_transition)

    def set_live(self, live: Live) -> None:
        self._live = live

    def _push(self):
        if self._live:
            self._live.update(self.hud.render(self._state))
        else:
            self.hud.refresh(self._state)

    def _on_transcription(self, event):
        with self._lock:
            self._state.text = event.get("text", "")
            self._state.whisper_ms = event.get("latency_ms", 0.0)
            self._push()

    def _on_intent(self, event):
        with self._lock:
            self._state.intent = event.get("intent", "")
            self._state.intent_ms = event.get("intent_ms", 0.0)
            self._push()

    def _on_state_transition(self, event):
        with self._lock:
            self._state.state = event.get("new_state", self._state.state)
            self._push()

    def _on_suggestion(self, event):
        with self._lock:
            self._state.state = event.get("state", self._state.state)
            self._state.suggestion = event.get("suggestion", "")
            self._state.brain_ms = event.get("brain_ms", self._state.brain_ms)
            self._push()

    def _on_brain(self, event):
        with self._lock:
            self._state.state = event.get("state", self._state.state)
            self._state.suggestion = event.get("suggestion", self._state.suggestion)
            self._state.brain_ms = event.get("brain_ms", self._state.brain_ms)
            self._state.whisper_ms = event.get("whisper_ms", self._state.whisper_ms)
            self._state.intent_ms = event.get("intent_ms", self._state.intent_ms)
            self._state.total_ms = self._state.whisper_ms + self._state.intent_ms + self._state.brain_ms
            self._push()


def run_pipeline():
    event_bus = EventBus(max_queue_size=64)
    ring_buffer = AudioRingBuffer(max_frames=BUFFER_FRAMES)
    inference_service = ScriptedInferenceService(SAMPLE_TEXT)
    hud = MinimalHUD()
    hud_subscriber = HUDSubscriber(hud, event_bus)
    intent_detector = IntentDetector(event_bus)
    _ = intent_detector  # appease linters for unused variable
    BrainConsumer(event_bus)

    whisper_worker = WhisperWorker(
        ring_buffer=ring_buffer,
        inference_service=inference_service,  # type: ignore[arg-type]
        event_bus=event_bus,
        sample_rate=SAMPLE_RATE,
        frames_per_window=BUFFER_FRAMES,
    )
    whisper_worker.start()

    # Simulate audio producer writing frames into the bounded buffer without sleeps
    for _ in SAMPLE_TEXT:
        ring_buffer.push(np.zeros(BLOCK_SIZE, dtype=np.float32))

    stop_event = threading.Event()

    def on_completion(_event):
        if len(event_bus.get_queue_snapshot("brain_event")) >= len(SAMPLE_TEXT):
            stop_event.set()

    event_bus.subscribe("brain_event", on_completion)

    with Live(hud.render(HUDState()), console=hud.console, refresh_per_second=12) as live:
        hud_subscriber.set_live(live)
        stop_event.wait(timeout=2)

    whisper_worker.stop()
    whisper_worker.join(timeout=1)


if __name__ == "__main__":
    run_pipeline()
