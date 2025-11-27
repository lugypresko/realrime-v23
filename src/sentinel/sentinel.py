from __future__ import annotations

import sys
import threading
import time
from typing import Optional

from src.audio_ring_buffer import AudioRingBuffer
from src.cache.silence_jitter import SilenceJitter
from src.cache.vad_smoother import VADSmoother

try:
    import numpy as np
except Exception:  # pragma: no cover - fallback for environments without numpy
    from src.mocks import mock_numpy as np

from src.contracts import SILENCE_TRIGGER_FIELDS, create_silence_trigger, ensure_schema_keys
from src.logging.structured_logger import log_event
from src.sentinel.services import (
    BLOCK_SIZE,
    BUFFER_FRAMES,
    SAMPLE_RATE,
    AudioInputService,
    SentinelTelemetry,
    SilencePolicy,
    SoundDeviceAudioSource,
    build_ring_buffer,
)
from src.telemetry.device_monitor import enumerate_microphones
from src.telemetry.error_state import ErrorStateManager


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
    ring_buffer: AudioRingBuffer = svc.get("ring_buffer") or build_ring_buffer()
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
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        if status:
            log_event({"type": "SENTINEL_STATUS", "status": str(status)})
        audio_chunk = indata[:, 0]
        ring_buffer.push(audio_chunk)

    def silence_worker():
        last_seq = ring_buffer.sequence
        while not stop_event.is_set():
            last_seq = ring_buffer.wait_for_new_data(last_sequence=last_seq, timeout=0.5)
            audio = ring_buffer.read_latest(max_frames=BUFFER_FRAMES)
            if audio is None:
                continue

            try:
                import torch

                chunk = audio[-BLOCK_SIZE:]
                speech_prob = float(model(torch.from_numpy(chunk), SAMPLE_RATE).item())
            except Exception:
                chunk = audio[-BLOCK_SIZE:]
                speech_prob = 0.0

            now = time.monotonic()
            if getattr(audio_service, "replay_recorder", None):
                audio_service.replay_recorder.add(chunk, speech_prob)
            if getattr(audio_service, "dead_mic", None) and audio_service.dead_mic.update(chunk, speech_prob, now):
                dead_event = {"type": "MIC_DEAD", "timestamp": now}
                queue_sw.put(dead_event)
                log_event(dead_event)

            trigger = silence_policy.handle_prob(speech_prob, now, len(chunk), SAMPLE_RATE)
            if trigger:
                event = trigger["event"]
                event_id = trigger["event_id"]
                ensure_schema_keys(event, SILENCE_TRIGGER_FIELDS, "SILENCE_TRIGGER")
                queue_sw.put(event)
                telemetry.emit_trigger(event_id, now, trigger["silence_ms"])

            error_state.record_vad_inactivity()

    telemetry.emit_start(SAMPLE_RATE)
    silence_thread = threading.Thread(target=silence_worker, daemon=True, name="silence-worker")
    silence_thread.start()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        ):
            stop_event.wait()
    except KeyboardInterrupt:
        telemetry.emit_stop()
    finally:
        stop_event.set()
        silence_thread.join(timeout=1)
