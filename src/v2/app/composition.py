from __future__ import annotations

import queue
from typing import Iterable, Tuple

from src.v2.interfaces import AudioChunk
from src.v2.cache.anti_repeat import LastNRepeatFilter
from src.v2.cache.latency_history import RollingLatencyTracker
from src.v2.cache.warm_start import NoOpWarmStarter
from src.v2.mocks.mocks import (
    MockAudioSource,
    ImmediateSilencePolicy,
    MockSTTEngine,
    MockIntentClassifier,
    MockRepeatFilter,
)
from src.v2.presenter.presenter import PresenterProcess
from src.v2.presenter.services import SimpleResultFormatter
from src.v2.sentinel.sentinel import SentinelProcess
from src.v2.sentinel.services import StaticAudioSource, ThresholdSilencePolicy, InMemoryReplayStore
from src.v2.telemetry.client import InMemoryTelemetry
from src.v2.worker.services import SimpleGovernor
from src.v2.worker.worker import WorkerProcess


class CompositionRoot:
    """Builds pipelines with real or mock bindings without mode branching inside processes."""

    def __init__(self, mode: str = "mock") -> None:
        self.mode = mode
        self.telemetry = InMemoryTelemetry()

    def _build_sentinel(self) -> SentinelProcess:
        if self.mode == "mock":
            audio_source = MockAudioSource()
            silence_policy = ImmediateSilencePolicy()
            replay = InMemoryReplayStore()
        else:
            now = 0.0
            chunks = [AudioChunk(data=[0.0], timestamp=now + i * 0.1) for i in range(5)]
            audio_source = StaticAudioSource(chunks)
            silence_policy = ThresholdSilencePolicy()
            replay = InMemoryReplayStore()
        return SentinelProcess(audio_source, silence_policy, self.telemetry, replay)

    def _build_worker(self) -> WorkerProcess:
        if self.mode == "mock":
            stt = MockSTTEngine()
            classifier = MockIntentClassifier()
            repeat_filter = MockRepeatFilter()
        else:
            stt = MockSTTEngine()
            classifier = MockIntentClassifier()
            repeat_filter = LastNRepeatFilter()
        governor = SimpleGovernor(repeat_filter=repeat_filter)
        return WorkerProcess(
            stt=stt,
            intent_classifier=classifier,
            governor=governor,
            telemetry_client=self.telemetry,
            warmstarter=NoOpWarmStarter(),
            latency_tracker=RollingLatencyTracker(),
            max_pending_events=3,
        )

    def _build_presenter(self) -> PresenterProcess:
        formatter = SimpleResultFormatter()
        return PresenterProcess(formatter, self.telemetry)

    def build_pipeline(self) -> Tuple[SentinelProcess, WorkerProcess, PresenterProcess]:
        return self._build_sentinel(), self._build_worker(), self._build_presenter()

    def run_pipeline(self) -> Iterable[str]:
        sentinel, worker, presenter = self.build_pipeline()
        event_queue: queue.Queue = queue.Queue()
        output_queue: queue.Queue = queue.Queue()

        sentinel.run(event_queue.put)

        def pending_events():
            return event_queue.qsize()

        def event_iter():
            while not event_queue.empty():
                yield event_queue.get()

        worker.run(event_iter(), output_queue.put, pending_events=pending_events)

        def output_iter():
            while not output_queue.empty():
                yield output_queue.get()

        return presenter.run(output_iter())
