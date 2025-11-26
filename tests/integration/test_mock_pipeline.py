import json
import multiprocessing as mp
import queue
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.composition import build_sentinel_dependencies, build_worker_dependencies
from src.logging.structured_logger import LOG_PATH
from src.sentinel.sentinel import sentinel_process
from src.worker.worker import worker_process


def _drain_queue(q: mp.Queue):
    items = []
    try:
        while True:
            items.append(q.get_nowait())
    except queue.Empty:
        pass
    return items


def test_mock_pipeline_integration(tmp_path):
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    queue_sw = mp.Queue()
    queue_wp = mp.Queue()
    event_count = 2

    sentinel = mp.Process(
        target=sentinel_process,
        args=(queue_sw,),
        kwargs={"use_mock": True, "mock_event_count": event_count, "services": build_sentinel_dependencies(use_mock=True)},
    )
    worker = mp.Process(
        target=worker_process,
        args=(queue_sw, queue_wp),
        kwargs={"use_mock": True, "mock_event_limit": event_count, "services": build_worker_dependencies(use_mock=True)},
    )

    sentinel.start()
    worker.start()

    sentinel.join(timeout=3)
    worker.join(timeout=3)

    if sentinel.is_alive():
        sentinel.terminate()
        sentinel.join()
    if worker.is_alive():
        worker.terminate()
        worker.join()

    results = _drain_queue(queue_wp)
    assert len(results) == event_count

    required_fields = set(
        [
            "id",
            "event_id",
            "event_timestamp",
            "sentinel_timestamp",
            "worker_start_ts",
            "whisper_latency",
            "intent_latency",
            "event_age",
            "decision",
            "text",
            "prompt_id",
            "score",
            "transport_latency_ms",
            "total_latency_ms",
        ]
    )

    for result in results:
        assert required_fields.issubset(result.keys())
        assert result["decision"] in {"SUCCESS", "SUPPRESSED_LATE", "SUPPRESSED_REPEAT"}
        assert result["id"] == result["event_id"]
        assert isinstance(result["text"], str)

    assert LOG_PATH.exists()
    with LOG_PATH.open() as f:
        lines = [json.loads(line) for line in f]
    assert any(entry.get("type") == "SILENCE_TRIGGER" for entry in lines)
    assert any(entry.get("type") == "WORKER_RESULT" for entry in lines)
