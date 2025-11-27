from __future__ import annotations

import random
import threading
import time
from typing import Dict, List

from src.dialogue_brain.brain import DialogueBrain
from src.event_bus import EventBus


class MockPipeline:
    def __init__(self, event_count: int = 200):
        self.event_count = event_count
        self.bus = EventBus(max_queue_size=256, max_workers=8)
        self._original_submit = self.bus._executor.submit
        self.bus._executor.submit = lambda fn, payload: fn(payload)  # type: ignore
        self.brain = DialogueBrain(debug=True)
        self.outputs: List[Dict] = []
        self._hud_render_count = 0
        self._setup_pipeline()

    def _setup_pipeline(self):
        self.stt_seq = 0
        self.intent_seq = 0
        self.brain_seq = 0
        self.start_times: Dict[int, float] = {}

        self.bus.subscribe("silence_gap_event", self._on_silence)
        self.bus.subscribe("transcription_event", self._on_transcription)
        self.bus.subscribe("intent_event", self._on_intent)
        self.bus.subscribe("brain_event", self._on_brain)

    def _on_silence(self, payload: Dict):
        self.stt_seq += 1
        idx = payload.get("idx", self.stt_seq)
        self.start_times[idx] = payload.get("start", time.perf_counter())
        self.bus.publish(
            "transcription_event",
            {
                "text": "we are evaluating options",
                "latency_ms": 90.0,
                "stt_seq": self.stt_seq,
                "idx": idx,
            },
        )

    def _on_transcription(self, payload: Dict):
        self.intent_seq += 2
        self.bus.publish(
            "intent_event",
            {
                "intent": "probe",
                "intent_ms": 12.0,
                "stt_seq": payload["stt_seq"],
                "intent_seq": self.intent_seq,
                "idx": payload.get("idx"),
                "text": payload.get("text"),
            },
        )

    def _on_intent(self, payload: Dict):
        self.brain_seq += 3
        result = self.brain.process(payload.get("text", ""), payload.get("intent", ""))
        self.bus.publish(
            "brain_event",
            {
                "brain_seq": self.brain_seq,
                "intent_seq": payload["intent_seq"],
                "stt_seq": payload["stt_seq"],
                "idx": payload.get("idx"),
                "state": result["state"],
                "suggestion": result["suggestion"],
                "brain_ms": result["latency_ms"],
                "intent_ms": payload.get("intent_ms", 0.0),
                "whisper_ms": payload.get("latency_ms", 0.0),
            },
        )

    def _on_brain(self, payload: Dict):
        idx = payload.get("idx", len(self.outputs))
        total_ms = (
            payload.get("brain_ms", 0.0)
            + payload.get("intent_ms", 0.0)
            + payload.get("whisper_ms", 0.0)
        )
        pipeline_latency = (time.perf_counter() - self.start_times.get(idx, time.perf_counter())) * 1000
        record = {
            "idx": idx,
            "stt_seq": payload["stt_seq"],
            "intent_seq": payload["intent_seq"],
            "brain_seq": payload["brain_seq"],
            "state": payload["state"],
            "suggestion": payload["suggestion"],
            "latency": total_ms,
            "pipeline_latency": pipeline_latency,
        }
        self.outputs.append(record)
        self._hud_render_count += 1

    def run(self) -> List[Dict]:
        done = threading.Event()

        def maybe_done(_):
            if len(self.outputs) >= self.event_count:
                done.set()

        self.bus.subscribe("brain_event", maybe_done)

        for idx in range(self.event_count):
            self.bus.publish("silence_gap_event", {"buffer": b"frame", "idx": idx, "start": time.perf_counter()})

        assert done.wait(timeout=3), "Pipeline did not finish in time"
        return self.outputs

    def shutdown(self) -> None:
        self.bus._executor.submit = self._original_submit  # type: ignore



def _normalize_outputs(outputs: List[Dict]) -> List[Dict]:
    return [{"idx": o["idx"], "state": o["state"], "suggestion": o["suggestion"]} for o in outputs]


def test_pipeline_ordering_latency_and_determinism():
    random.seed(42)
    runner = MockPipeline(event_count=200)
    outputs_first = runner.run()
    runner.shutdown()

    assert all(o["stt_seq"] < o["intent_seq"] < o["brain_seq"] for o in outputs_first)

    latencies = [o["pipeline_latency"] for o in outputs_first]
    avg_latency = sum(latencies) / len(latencies)
    p99_latency = sorted(latencies)[int(0.99 * len(latencies))]
    assert avg_latency < 200, f"Average latency too high: {avg_latency}ms"
    assert p99_latency < 300, f"p99 latency too high: {p99_latency}ms"

    assert len(outputs_first) == 200
    assert runner._hud_render_count == 200

    assert len(runner.bus.get_queue_snapshot("silence_gap_event")) <= runner.bus.max_queue_size
    assert len(runner.bus.get_queue_snapshot("brain_event")) <= runner.bus.max_queue_size

    random.seed(42)
    runner2 = MockPipeline(event_count=200)
    outputs_second = runner2.run()
    runner2.shutdown()
    assert _normalize_outputs(outputs_first) == _normalize_outputs(outputs_second)
