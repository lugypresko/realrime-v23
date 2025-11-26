import time
from queue import Queue

from src.v2.interfaces import SilenceEvent
from src.v2.mocks.mocks import MockSTTEngine, MockIntentClassifier, MockRepeatFilter
from src.v2.cache.anti_repeat import LastNRepeatFilter
from src.v2.cache.latency_history import RollingLatencyTracker
from src.v2.cache.warm_start import NoOpWarmStarter
from src.v2.worker.services import SimpleGovernor
from src.v2.worker.worker import WorkerProcess
from src.v2.telemetry.client import InMemoryTelemetry


def test_worker_process_produces_output():
    telemetry = InMemoryTelemetry()
    governor = SimpleGovernor(repeat_filter=MockRepeatFilter())
    worker = WorkerProcess(
        stt=MockSTTEngine(),
        intent_classifier=MockIntentClassifier(),
        governor=governor,
        telemetry_client=telemetry,
        warmstarter=NoOpWarmStarter(),
        latency_tracker=RollingLatencyTracker(),
        max_pending_events=10,
    )

    event = SilenceEvent(event_id="evt-1", timestamp=time.time(), audio=[0.0])
    out_queue: Queue = Queue()
    worker.run([event], out_queue.put, pending_events=lambda: 0)
    assert not out_queue.empty()
    output = out_queue.get()
    assert output.event_id == "evt-1"
    assert output.text == "hello world"
