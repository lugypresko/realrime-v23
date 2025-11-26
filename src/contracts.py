from typing import Dict, Iterable

try:
    import numpy as np
except Exception:  # pragma: no cover - fallback for environments without numpy
    from src.mocks import mock_numpy as np


SILENCE_TRIGGER_FIELDS = (
    "type",
    "id",
    "event_id",
    "audio",
    "timestamp",
    "sentinel_timestamp",
)

WORKER_RESULT_FIELDS = (
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
)


def ensure_schema_keys(event: Dict, required_keys: Iterable[str], name: str) -> None:
    missing = set(required_keys) - set(event.keys())
    if missing:
        raise ValueError(f"{name} schema mismatch. Missing: {sorted(missing)}")


def create_silence_trigger(event_id: str, audio: np.ndarray, timestamp: float) -> Dict:
    event = {
        "type": "SILENCE_TRIGGER",
        "id": event_id,
        "event_id": event_id,
        "audio": audio.astype(np.float32),
        "timestamp": float(timestamp),
        "sentinel_timestamp": float(timestamp),
    }
    ensure_schema_keys(event, SILENCE_TRIGGER_FIELDS, "SILENCE_TRIGGER")
    return event


def create_worker_result(
    *,
    event_id: str,
    event_timestamp: float,
    sentinel_timestamp: float,
    worker_start_ts: float,
    whisper_latency: float,
    intent_latency: float,
    event_age: float,
    decision: str,
    text: str,
    prompt_id: str,
    score: float,
    transport_latency_ms: float,
    total_latency_ms: float,
) -> Dict:
    result = {
        "id": event_id,
        "event_id": event_id,
        "event_timestamp": float(event_timestamp),
        "sentinel_timestamp": float(sentinel_timestamp),
        "worker_start_ts": float(worker_start_ts),
        "whisper_latency": float(whisper_latency),
        "intent_latency": float(intent_latency),
        "event_age": float(event_age),
        "decision": decision,
        "text": text,
        "prompt_id": prompt_id,
        "score": float(score),
        "transport_latency_ms": float(transport_latency_ms),
        "total_latency_ms": float(total_latency_ms),
    }
    ensure_schema_keys(result, WORKER_RESULT_FIELDS, "WORKER_RESULT")
    return result
