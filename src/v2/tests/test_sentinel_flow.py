import time
from queue import Queue

from src.v2.interfaces import AudioChunk
from src.v2.sentinel.services import StaticAudioSource, ThresholdSilencePolicy, InMemoryReplayStore
from src.v2.sentinel.sentinel import SentinelProcess
from src.v2.telemetry.client import InMemoryTelemetry


def test_sentinel_emits_event():
    now = time.time()
    chunks = [AudioChunk(data=[0.0], timestamp=now + i * 0.1) for i in range(3)]
    source = StaticAudioSource(chunks)
    policy = ThresholdSilencePolicy(threshold=2)
    telemetry = InMemoryTelemetry()
    replay = InMemoryReplayStore()
    sentinel = SentinelProcess(source, policy, telemetry, replay)

    q = Queue()
    sentinel.run(q.put)
    assert not q.empty()
    event = q.get()
    assert event.audio == [0.0]
    assert replay.snapshot()
