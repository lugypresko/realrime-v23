import time
from typing import Dict, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    from src.mocks import mock_numpy as np

from src.cache.anti_repeat import AntiRepeatCache
from src.cache.latency_history import LatencyHistory
from src.cache.warm_start import warm_start_whisper
from src.interfaces import Governor, IntentClassifier, LatencyTracker, RepeatFilter, STTEngine
from src.telemetry.drift_detector import DriftDetector
from src.telemetry.error_state import ErrorStateManager
from src.telemetry.prompt_quality import PromptQualityMonitor
from src.telemetry.telemetry_aggregator import TelemetryAggregator


class InferenceService(STTEngine):
    def __init__(self, use_mock: bool, error_state: ErrorStateManager):
        self.use_mock = use_mock
        self.error_state = error_state
        if use_mock:
            from src.mocks.mock_whisper import transcribe_mock

            self._mock = transcribe_mock
            self._stt = None
        else:
            from faster_whisper import WhisperModel

            self._stt = WhisperModel("tiny.en", device="cpu", compute_type="int8")
            warm_start_whisper(self._stt)
            self._mock = None

    def transcribe(self, audio) -> Dict[str, float | str]:
        if self.use_mock:
            text, latency = self._mock(audio)
            return {"text": text, "latency": latency}

        text = ""
        latency = 0.0
        try:
            if self.error_state.should_use_safe_mode():
                text = "[safe-mode-transcript]"
                latency = 0.0
            else:
                start = time.monotonic()
                segments, info = self._stt.transcribe(audio, beam_size=1, language="en", temperature=0.0)
                text = " ".join([segment.text for segment in segments]).strip() or "[no-transcript]"
                latency = time.monotonic() - start
                self.error_state.clear_vad_inactivity()
        except Exception:
            self.error_state.record_whisper_failure()
            time.sleep(0.32)
            try:
                segments, info = self._stt.transcribe(audio, beam_size=1, language="en", temperature=0.0)
                text = " ".join([segment.text for segment in segments]).strip() or "[retry-no-transcript]"
                latency = time.monotonic() - start  # type: ignore[name-defined]
            except Exception:
                self.error_state.record_whisper_failure()
                text = "[whisper-failed]"
                latency = 0.0
        return {"text": text, "latency": latency}


class IntentService(IntentClassifier):
    def __init__(self, use_mock: bool):
        self.use_mock = use_mock
        if use_mock:
            from src.mocks.mock_intent import classify_mock_intent

            self._mock = classify_mock_intent
            self._encoder = None
            self._embeddings = None
        else:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
            self._embeddings = np.load("embeddings.npy")
            self._mock = None

    def classify(self, text: str) -> Dict[str, float | int]:
        if self.use_mock:
            best_idx, best_score, latency = self._mock(text)
            return {"prompt_id": str(best_idx), "score": float(best_score), "latency": latency}
        intent_start = time.monotonic()
        try:
            utterance_embedding = self._encoder.encode(text, convert_to_numpy=True, show_progress_bar=False)
            scores = self._embeddings @ utterance_embedding
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
        except Exception:
            best_idx = 0
            best_score = 0.0
        intent_latency = time.monotonic() - intent_start
        return {"prompt_id": str(best_idx), "score": best_score, "latency": intent_latency}


class GovernorService(Governor):
    def __init__(self, repeat_filter: RepeatFilter, error_state: ErrorStateManager, prompt_quality: PromptQualityMonitor):
        self.repeat_filter = repeat_filter
        self.error_state = error_state
        self.prompt_quality = prompt_quality

    def decide(self, event, transport_delay: float, whisper_latency: float, intent_latency: float) -> Dict[str, str]:
        event_age = time.monotonic() - event["timestamp"]
        decision = "SUCCESS" if event_age <= 1.5 else "SUPPRESSED_LATE"
        score_diff = abs(event.get("score", 0.0) - event.get("previous_score", 0.0)) if "score" in event else 0.0
        prompt_id = str(event.get("prompt_id", ""))
        if decision == "SUCCESS" and self.repeat_filter.should_suppress(prompt_id, score_diff):
            decision = "SUPPRESSED_REPEAT"
        if self.error_state.should_use_safe_mode():
            decision = "SUPPRESSED_SAFE_MODE"
        if decision == "SUCCESS":
            self.repeat_filter.record(prompt_id) if hasattr(self.repeat_filter, "record") else None
            self.prompt_quality.evaluate(prompt_id, event.get("score", 0.0))
        return {"decision": decision, "event_age": event_age}

    def record_backpressure(self) -> None:
        # handled externally for now
        return None


class LatencyMonitor(LatencyTracker):
    def __init__(self):
        self.history = LatencyHistory()
        self.telemetry = TelemetryAggregator()
        self.drift_detector = DriftDetector()

    def record(self, whisper_latency: float, intent_latency: float, total_age: float, decision: str):
        record = {
            "whisper_latency": whisper_latency,
            "intent_latency": intent_latency,
            "event_age": total_age,
            "decision": decision,
        }
        self.history.add(record)
        metrics = self.history.log_warnings()
        metrics.update(self.telemetry.add_measurement(whisper_latency, intent_latency, total_age, decision))
        self.drift_detector.add(whisper_latency * 1000)
        return metrics


class RepeatFilterAdapter(RepeatFilter):
    def __init__(self):
        self.cache = AntiRepeatCache()

    def should_suppress(self, prompt_id: str, score_diff: float) -> bool:
        return self.cache.should_suppress(prompt_id, score_diff)

    def record(self, prompt_id: str) -> None:
        self.cache.record(prompt_id)


class BackpressureController:
    def __init__(self, threshold: int = 3):
        self.threshold = threshold

    def drop_oldest(self, queue_sw):
        dropped_ids = []
        try:
            while queue_sw.qsize() > self.threshold:
                dropped = queue_sw.get_nowait()
                if dropped:
                    dropped_ids.append(dropped.get("id"))
        except NotImplementedError:
            pass
        return dropped_ids
