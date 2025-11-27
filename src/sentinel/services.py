import threading
import time
import uuid

from src.audio_ring_buffer import AudioRingBuffer
from src.cache.replay_buffer import ReplayBuffer
from src.cache.silence_jitter import SilenceJitter
from src.cache.vad_smoother import VADSmoother
from src.contracts import SILENCE_TRIGGER_FIELDS, create_silence_trigger, ensure_schema_keys
from src.interfaces import AudioSource, ReplayStore, SilencePolicy as SilencePolicyInterface
from src.logging.structured_logger import log_event
from src.sentinel.dead_mic import DeadMicDetector
from src.telemetry.device_monitor import enumerate_microphones
from src.telemetry.error_state import ErrorStateManager
from src.telemetry.telemetry_writer import write_event

try:
    import numpy as np
except Exception:  # pragma: no cover
    from src.mocks import mock_numpy as np

SAMPLE_RATE = 16000
BLOCK_SIZE = 512
BUFFER_DURATION = 1.2
BUFFER_FRAMES = int((SAMPLE_RATE * BUFFER_DURATION) / BLOCK_SIZE) + 1


def build_ring_buffer() -> AudioRingBuffer:
    return AudioRingBuffer(max_frames=BUFFER_FRAMES)


class SoundDeviceAudioSource(AudioSource):
    def __init__(self, sample_rate: int = SAMPLE_RATE, block_size: int = BLOCK_SIZE):
        import sounddevice as sd

        self._sd = sd
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._stream = None

    def start(self, callback):
        self._stream = self._sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            callback=callback,
        )
        self._stream.__enter__()
        return self

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.__exit__(None, None, None)
            self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()


class SilencePolicy(SilencePolicyInterface):
    def __init__(self, ring_buffer: AudioRingBuffer, smoother: VADSmoother, jitter: SilenceJitter):
        self.ring_buffer = ring_buffer
        self.smoother = smoother
        self.jitter = jitter

    def handle_prob(self, prob: float, timestamp: float, frames: int, sample_rate: int):
        speaking = self.smoother.update(prob, timestamp)
        if speaking:
            self.jitter.reset_on_speech()
            return None

        delta_ms = (frames / sample_rate) * 1000
        self.jitter.update_silence(delta_ms)
        if self.jitter.is_trigger_ready():
            full_audio = self.ring_buffer.read_latest()
            if full_audio is None:
                return None
            event_id = str(uuid.uuid4())
            event = create_silence_trigger(
                event_id=event_id,
                audio=full_audio,
                timestamp=timestamp,
            )
            ensure_schema_keys(event, SILENCE_TRIGGER_FIELDS, "SILENCE_TRIGGER")
            self.jitter.reset_on_speech()
            return {"event": event, "event_id": event_id, "silence_ms": self.jitter.silence_ms}
        return None


class ReplayRecorder(ReplayStore):
    def __init__(self, sample_rate: int = SAMPLE_RATE, duration_sec: float = 20):
        self.buffer = ReplayBuffer(sample_rate=sample_rate, duration_sec=duration_sec)

    def add(self, audio_chunk, vad_prob: float) -> None:
        self.buffer.add(audio_chunk, vad_prob)

    def dump_to_disk(self, path: str, event_id: str) -> None:
        self.buffer.dump_to_disk(path, event_id)


class SentinelTelemetry:
    def emit_trigger(self, event_id: str, timestamp: float, silence_ms: float) -> None:
        log_event(
            {
                "type": "SILENCE_TRIGGER",
                "event_id": event_id,
                "sentinel_timestamp": timestamp,
                "sentinel_detect_latency": silence_ms,
            }
        )
        write_event({"type": "SENTINEL_DETECT", "event_id": event_id, "latency_ms": silence_ms})

    def emit_start(self, sample_rate: int) -> None:
        write_event({"type": "SENTINEL_START", "sample_rate": sample_rate})

    def emit_stop(self) -> None:
        log_event({"type": "SENTINEL_STOP"})

    def emit_status(self, status: str) -> None:
        log_event({"type": "SENTINEL_STATUS", "status": status})


class AudioInputService:
    def __init__(
        self,
        audio_source: AudioSource,
        replay_recorder: ReplayStore,
        dead_mic: DeadMicDetector,
        error_state: ErrorStateManager,
    ):
        self.audio_source = audio_source
        self.replay_recorder = replay_recorder
        self.dead_mic = dead_mic
        self.error_state = error_state
        self._stop_event = threading.Event()

    def enumerate_devices(self):
        enumerate_microphones()

    def run(self, callback):
        with self.audio_source.start(callback):
            self._stop_event.wait()

    def stop(self) -> None:
        self._stop_event.set()
