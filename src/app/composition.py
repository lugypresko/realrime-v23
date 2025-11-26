from src.interfaces import TelemetryClient
from src.logging.structured_logger import log_event
from src.sentinel.services import (
    AudioInputService,
    build_ring_buffer,
    ReplayRecorder,
    SilencePolicy,
    SoundDeviceAudioSource,
    SentinelTelemetry,
)
from src.sentinel.dead_mic import DeadMicDetector
from src.telemetry.error_state import ErrorStateManager
from src.cache.silence_jitter import SilenceJitter
from src.cache.vad_smoother import VADSmoother
from src.worker.services import (
    BackpressureController,
    GovernorService,
    InferenceService,
    IntentService,
    LatencyMonitor,
    RepeatFilterAdapter,
)
from src.telemetry.prompt_quality import PromptQualityMonitor
from src.presenter.services import PresenterTelemetry, ResultValidator, SimpleResultFormatter


class StructuredLoggerClient(TelemetryClient):
    def emit(self, event):
        log_event(event)


def build_sentinel_dependencies(use_mock: bool = False):
    ring_buffer = build_ring_buffer()
    smoother = VADSmoother()
    jitter = SilenceJitter()
    silence_policy = SilencePolicy(ring_buffer, smoother, jitter)
    telemetry = SentinelTelemetry()
    if use_mock:
        audio_source = None
    else:
        audio_source = SoundDeviceAudioSource()
    replay_recorder = ReplayRecorder()
    error_state = ErrorStateManager()
    dead_mic = DeadMicDetector()
    audio_service = AudioInputService(audio_source, replay_recorder, dead_mic, error_state) if audio_source else None
    return {
        "ring_buffer": ring_buffer,
        "silence_policy": silence_policy,
        "telemetry": telemetry,
        "audio_service": audio_service,
        "replay_recorder": replay_recorder,
        "error_state": error_state,
        "dead_mic": dead_mic,
    }


def build_worker_dependencies(use_mock: bool = False):
    error_state = ErrorStateManager()
    inference_service = InferenceService(use_mock=use_mock, error_state=error_state)
    intent_service = IntentService(use_mock=use_mock)
    prompt_quality = PromptQualityMonitor()
    governor = GovernorService(repeat_filter=RepeatFilterAdapter(), error_state=error_state, prompt_quality=prompt_quality)
    latency_monitor = LatencyMonitor()
    backpressure = BackpressureController()
    return {
        "error_state": error_state,
        "inference_service": inference_service,
        "intent_service": intent_service,
        "governor": governor,
        "latency_monitor": latency_monitor,
        "backpressure": backpressure,
        "prompt_quality": prompt_quality,
    }


def build_presenter_dependencies():
    return {
        "validator": ResultValidator(),
        "formatter": SimpleResultFormatter(),
        "telemetry": PresenterTelemetry(),
    }
