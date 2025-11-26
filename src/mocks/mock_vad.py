import time
import uuid
from typing import Any

from src.contracts import SILENCE_TRIGGER_FIELDS, create_silence_trigger, ensure_schema_keys
from src.logging.structured_logger import log_event
from src.mocks.mock_audio import generate_mock_buffer


def emit_mock_triggers(queue_sw: Any, count: int = 3, interval: float = 0.1) -> None:
    for idx in range(count):
        event_id = f"mock-event-{idx+1}"
        event = create_silence_trigger(
            event_id=event_id,
            audio=generate_mock_buffer(),
            timestamp=time.monotonic(),
        )
        ensure_schema_keys(event, SILENCE_TRIGGER_FIELDS, "SILENCE_TRIGGER")
        queue_sw.put(event)
        log_event({"type": "SILENCE_TRIGGER", "event_id": event_id, "sentinel_timestamp": event["timestamp"]})
        time.sleep(interval)
