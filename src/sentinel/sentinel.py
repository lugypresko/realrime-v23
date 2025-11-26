import sys
import time
from typing import Deque, Optional

from src.cache.silence_jitter import SilenceJitter
from src.cache.vad_smoother import VADSmoother

try:
    import numpy as np
except Exception:  # pragma: no cover - fallback for environments without numpy
    from src.mocks import mock_numpy as np

from src.contracts import SILENCE_TRIGGER_FIELDS, create_silence_trigger, ensure_schema_keys
from src.logging.structured_logger import log_event
from src.sentinel.services import (
    AudioInputService,
    build_ring_buffer,
    SentinelTelemetry,
    SilencePolicy,
    SoundDeviceAudioSource,
)
from src.telemetry.device_monitor import detect_sampling_drift, enumerate_microphones
from src.telemetry.error_state import ErrorStateManager


SAMPLE_RATE = 16000
BLOCK_SIZE = 512
BUFFER_DURATION = 1.2
BUFFER_FRAMES = int((SAMPLE_RATE * BUFFER_DURATION) / BLOCK_SIZE) + 1


def _load_vad_model():
    import torch

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    return model


def sentinel_process(queue_sw, use_mock: bool = False, mock_event_count: int = 3, services: Optional[dict] = None):
    sys.stdout.reconfigure(encoding="utf-8")

    if use_mock:
        from src.mocks.mock_vad import emit_mock_triggers

        emit_mock_triggers(queue_sw, mock_event_count)
        return

    svc = services or {}
    ring_buffer: Deque[np.ndarray] = svc.get("ring_buffer") or build_ring_buffer()
    silence_policy: SilencePolicy = svc.get("silence_policy") or SilencePolicy(
        ring_buffer, VADSmoother(), SilenceJitter()
    )
    telemetry: SentinelTelemetry = svc.get("telemetry") or SentinelTelemetry()
    audio_service: AudioInputService = svc.get("audio_service") or AudioInputService(
        SoundDeviceAudioSource(), svc.get("replay_recorder"), svc.get("dead_mic"), svc.get("error_state") or ErrorStateManager()
    )
    error_state: ErrorStateManager = svc.get("error_state") or ErrorStateManager()

    import sounddevice as sd

    model = _load_vad_model()
    enumerate_microphones()

    def audio_callback(indata, frames, time_info, status):
        if status:
            log_event({"type": "SENTINEL_STATUS", "status": str(status)})

        audio_chunk = indata[:, 0]
        ring_buffer.append(audio_chunk.copy())
        now = time.monotonic()

        try:
            import torch

            audio_tensor = torch.from_numpy(audio_chunk)
            speech_prob = model(audio_tensor, SAMPLE_RATE).item()
        except Exception:
            speech_prob = 0.0

        if hasattr(audio_service, "replay_recorder") and audio_service.replay_recorder:
            audio_service.replay_recorder.add(audio_chunk, speech_prob)
        if hasattr(audio_service, "dead_mic") and audio_service.dead_mic and audio_service.dead_mic.update(
            audio_chunk, speech_prob, now
        ):
            print("⚠ Microphone disconnected")
            dead_event = {"type": "MIC_DEAD", "timestamp": now}
            queue_sw.put(dead_event)
            log_event(dead_event)

        trigger = silence_policy.handle_prob(speech_prob, now, frames, SAMPLE_RATE)
        if trigger:
            event = trigger["event"]
            event_id = trigger["event_id"]
            ensure_schema_keys(event, SILENCE_TRIGGER_FIELDS, "SILENCE_TRIGGER")
            queue_sw.put(event)
            telemetry.emit_trigger(event_id, now, trigger["silence_ms"])

        error_state.record_vad_inactivity()

    telemetry.emit_start(SAMPLE_RATE)

    sd.rec(int(0.1 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        telemetry.emit_stop()
