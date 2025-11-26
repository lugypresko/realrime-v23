from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Tuple
from uuid import uuid4

from src.v2.interfaces import (
    AudioChunk,
    AudioSource,
    GovernorService,
    IntentClassifier,
    STTEngine,
    SilenceEvent,
    SilencePolicy,
    WorkerOutput,
    validate_worker_output,
)
from src.v2.worker.services import SimpleGovernor
from src.v2.interfaces import IntentResult, InferenceResult  # late import for Protocol compat


@dataclass
class ValidationOutcome:
    success: bool
    message: str


class LiveValidationComposition:
    """Build real implementations for live validation without multiprocessing or telemetry."""

    def __init__(self, sample_rate: int = 16000, frame_size: int = 1024) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size

    def build_audio_source(self, duration_sec: float) -> "LiveAudioSource":
        return LiveAudioSource(duration_sec=duration_sec, sample_rate=self.sample_rate, frame_size=self.frame_size)

    def build_silence_policy(self) -> "RmsSilencePolicy":
        return RmsSilencePolicy(sample_rate=self.sample_rate, frame_size=self.frame_size)

    def build_stt(self) -> "WhisperSTTEngine":
        return WhisperSTTEngine()

    def build_intent(self) -> "EmbeddingIntentClassifier":
        return EmbeddingIntentClassifier()

    def build_governor(self) -> GovernorService:
        return SimpleGovernor()


class LiveAudioSource(AudioSource):
    def __init__(self, duration_sec: float, sample_rate: int, frame_size: int) -> None:
        self.duration_sec = duration_sec
        self.sample_rate = sample_rate
        self.frame_size = frame_size

    def _record(self):
        try:
            import numpy as np
            import sounddevice as sd
        except Exception as exc:  # pragma: no cover - runtime dependency missing in CI
            raise RuntimeError(f"Audio dependencies unavailable: {exc}") from exc

        frames = int(self.duration_sec * self.sample_rate)
        try:
            recording = sd.rec(frames, samplerate=self.sample_rate, channels=1, dtype="float32")
            sd.wait(timeout=self.duration_sec + 2)
        except Exception as exc:  # pragma: no cover - hardware/environment specific
            raise RuntimeError(f"Audio capture failed: {exc}") from exc

        if recording is None:
            raise RuntimeError("No audio captured")

        flat = np.array(recording).flatten()
        return flat

    def stream(self) -> Iterable[AudioChunk]:
        audio = self._record()
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover - runtime dependency missing in CI
            raise RuntimeError(f"NumPy unavailable: {exc}") from exc

        start_ts = time.monotonic()
        frame_samples = self.frame_size
        for idx in range(0, len(audio), frame_samples):
            frame = audio[idx : idx + frame_samples]
            ts = start_ts + (idx / self.sample_rate)
            if frame.size == 0:
                continue
            yield AudioChunk(data=np.copy(frame), timestamp=ts)


class RmsSilencePolicy(SilencePolicy):
    def __init__(self, sample_rate: int, frame_size: int, speech_threshold: float = 0.01, silence_frames: int = 6) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.speech_threshold = speech_threshold
        self.silence_frames = silence_frames
        self._silence_count = 0
        self._speech_seen = False

    def observe(self, chunk: AudioChunk) -> SilenceEvent | None:
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover - runtime dependency missing in CI
            raise RuntimeError(f"NumPy unavailable for VAD: {exc}") from exc

        rms = float(np.sqrt(np.mean(np.square(chunk.data))))
        if rms > self.speech_threshold:
            self._speech_seen = True
            self._silence_count = 0
            return None
        if self._speech_seen:
            self._silence_count += 1
            if self._silence_count >= self.silence_frames:
                return SilenceEvent(event_id=str(uuid4()), timestamp=chunk.timestamp, audio=chunk.data)
        return None


class WhisperSTTEngine(STTEngine):
    def __init__(self, model_size: str = "tiny.en") -> None:
        self.model_size = model_size
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # pragma: no cover - runtime dependency missing in CI
            raise RuntimeError(f"Whisper not available: {exc}") from exc
        self._model = WhisperModel(self.model_size, device="cpu")
        return self._model

    def transcribe(self, audio: object):
        import numpy as np  # type: ignore

        model = self._load()
        start = time.monotonic()
        segments, _ = model.transcribe(np.array(audio), language="en")
        text_parts = [segment.text for segment in segments]
        latency_ms = (time.monotonic() - start) * 1000
        text = " ".join(text_parts).strip()
        return InferenceResult(text=text, latency_ms=latency_ms, success=bool(text), error=None if text else "empty")


class EmbeddingIntentClassifier(IntentClassifier):
    def __init__(self) -> None:
        self._embeddings = None
        self._encoder = None
        self._load_assets()

    def _load_assets(self):
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - runtime dependency missing in CI
            raise RuntimeError(f"Intent dependencies unavailable: {exc}") from exc

        root = Path(__file__).resolve().parents[3]
        prompts_path = root / "prompts.json"
        embeddings_path = root / "embeddings.npy"
        if not prompts_path.exists() or not embeddings_path.exists():
            raise RuntimeError("Prompt assets missing")

        with prompts_path.open("r", encoding="utf-8") as f:
            json.load(f)  # prompts content not required for id mapping here
        self._embeddings = np.load(embeddings_path)
        self._encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def classify(self, text: str):
        import numpy as np  # type: ignore

        if not text:
            return IntentResult(prompt_id="", score=0.0, latency_ms=0.0)
        start = time.monotonic()
        query_vec = self._encoder.encode(text)
        scores = np.dot(self._embeddings, query_vec)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        latency_ms = (time.monotonic() - start) * 1000
        return IntentResult(prompt_id=str(best_idx), score=best_score, latency_ms=latency_ms)


# Validation helpers

def _safe_status(fn: Callable[[], ValidationOutcome]) -> ValidationOutcome:
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - defensive guard for live runs
        return ValidationOutcome(False, str(exc))


def validate_audio_device(duration_sec: float = 3.0) -> ValidationOutcome:
    def _run():
        comp = LiveValidationComposition()
        source = comp.build_audio_source(duration_sec)
        try:
            import sounddevice as sd
        except Exception as exc:  # pragma: no cover - runtime dependency missing in CI
            return ValidationOutcome(False, f"sounddevice unavailable: {exc}")

        devices = sd.query_devices()
        if not devices:
            return ValidationOutcome(False, "No audio devices detected")
        default_input = None
        if isinstance(sd.default.device, (list, tuple)) and sd.default.device:
            default_input = sd.default.device[0]
        if default_input is None:
            return ValidationOutcome(False, "No default input device configured")

        audio = source._record()
        peak = float(abs(audio).max())
        if peak == 0.0:
            return ValidationOutcome(False, "Captured audio is silent")
        return ValidationOutcome(True, f"Device #{default_input} active; captured {len(audio)} samples; peak {peak:.4f}")

    return _safe_status(_run)


def validate_vad_live(duration_sec: float = 3.0) -> ValidationOutcome:
    def _run():
        comp = LiveValidationComposition()
        source = comp.build_audio_source(duration_sec)
        policy = comp.build_silence_policy()
        speech_frames = 0
        silence_events = 0
        for chunk in source.stream():
            event = policy.observe(chunk)
            rms = float((chunk.data ** 2).mean() ** 0.5)
            if rms > policy.speech_threshold:
                speech_frames += 1
            if event:
                silence_events += 1
                break
        if speech_frames == 0:
            return ValidationOutcome(False, "No speech detected; speak during capture")
        if silence_events == 0:
            return ValidationOutcome(False, "Silence not detected after speech")
        return ValidationOutcome(True, f"Speech frames: {speech_frames}; silence events: {silence_events}")

    return _safe_status(_run)


def validate_stt_live(duration_sec: float = 3.0) -> ValidationOutcome:
    def _run():
        comp = LiveValidationComposition()
        source = comp.build_audio_source(duration_sec)
        stt = comp.build_stt()
        audio = source._record()
        start = time.monotonic()
        result = stt.transcribe(audio)
        latency_ms = (time.monotonic() - start) * 1000
        if not result.success or not result.text:
            return ValidationOutcome(False, "STT returned empty result")
        return ValidationOutcome(True, f"Text: '{result.text[:50]}', latency={latency_ms:.1f}ms")

    return _safe_status(_run)


def validate_intent_live(duration_sec: float = 3.0) -> ValidationOutcome:
    def _run():
        comp = LiveValidationComposition()
        source = comp.build_audio_source(duration_sec)
        stt = comp.build_stt()
        intent = comp.build_intent()
        audio = source._record()
        stt_result = stt.transcribe(audio)
        if not stt_result.text:
            return ValidationOutcome(False, "No text from STT for intent classification")
        intent_result = intent.classify(stt_result.text)
        if not intent_result.prompt_id:
            return ValidationOutcome(False, "Intent classification returned no prompt")
        return ValidationOutcome(True, f"Prompt={intent_result.prompt_id}, score={intent_result.score:.4f}, intent_latency={intent_result.latency_ms:.1f}ms")

    return _safe_status(_run)


def validate_end_to_end_live(duration_sec: float = 3.0) -> ValidationOutcome:
    def _run():
        comp = LiveValidationComposition()
        source = comp.build_audio_source(duration_sec)
        policy = comp.build_silence_policy()
        stt = comp.build_stt()
        intent = comp.build_intent()
        governor = comp.build_governor()
        start = time.monotonic()
        event: SilenceEvent | None = None
        for chunk in source.stream():
            maybe_event = policy.observe(chunk)
            if maybe_event:
                event = maybe_event
                break
        if event is None:
            return ValidationOutcome(False, "No silence trigger produced; ensure speech then pause")
        stt_result = stt.transcribe(event.audio)
        intent_result = intent.classify(stt_result.text)
        decision = governor.decide(event, stt_result, intent_result)
        total_latency = (time.monotonic() - start) * 1000
        output = WorkerOutput(
            event_id=event.event_id,
            text=stt_result.text,
            prompt_id=intent_result.prompt_id,
            score=intent_result.score,
            whisper_latency_ms=stt_result.latency_ms,
            intent_latency_ms=intent_result.latency_ms,
            total_latency_ms=total_latency,
            decision=decision.decision,
            metadata={"reason": decision.reason},
        )
        if not validate_worker_output(output):
            return ValidationOutcome(False, "Worker output schema invalid")
        if total_latency > 1500:
            return ValidationOutcome(False, f"Latency budget exceeded: {total_latency:.1f}ms")
        return ValidationOutcome(True, f"decision={output.decision}, total_latency={total_latency:.1f}ms")

    return _safe_status(_run)


def run_all_validations() -> Tuple[bool, list[str]]:
    validations: list[Tuple[str, Callable[[], ValidationOutcome]]] = [
        ("Audio device", validate_audio_device),
        ("VAD", validate_vad_live),
        ("STT", validate_stt_live),
        ("Intent", validate_intent_live),
        ("End-to-end", validate_end_to_end_live),
    ]
    messages: list[str] = []
    all_ok = True
    for name, fn in validations:
        outcome = fn()
        status = "OK" if outcome.success else "FAIL"
        messages.append(f"[{status}] {name}: {outcome.message}")
        if not outcome.success:
            all_ok = False
    return all_ok, messages


if __name__ == "__main__":
    ok, lines = run_all_validations()
    for line in lines:
        print(line)
    sys.exit(0 if ok else 1)
