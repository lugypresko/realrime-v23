from __future__ import annotations

from typing import Callable

from src.v2.interfaces import AudioSource, SilencePolicy, TelemetryClient, ReplayStore, validate_silence_event
from src.v2.sentinel.services import SentinelTelemetry


class SentinelProcess:
    def __init__(
        self,
        audio_source: AudioSource,
        silence_policy: SilencePolicy,
        telemetry_client: TelemetryClient,
        replay_store: ReplayStore,
    ) -> None:
        self.audio_source = audio_source
        self.silence_policy = silence_policy
        self.telemetry = SentinelTelemetry(telemetry_client)
        self.replay_store = replay_store

    def run(self, emit: Callable[[object], None]) -> None:
        for chunk in self.audio_source.stream():
            self.telemetry.log_chunk(chunk)
            self.replay_store.add(chunk)
            event = self.silence_policy.observe(chunk)
            if event and validate_silence_event(event):
                self.telemetry.log_trigger(event)
                emit(event)
