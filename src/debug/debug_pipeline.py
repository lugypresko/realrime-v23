from src.logging.structured_logger import log_event


def log_latency(event_id: str, transport_ms: float, whisper_ms: float, intent_ms: float, total_ms: float) -> None:
    log_event(
        {
            "type": "DEBUG_PIPELINE",
            "event_id": event_id,
            "transport_ms": transport_ms,
            "whisper_ms": whisper_ms,
            "intent_ms": intent_ms,
            "total_ms": total_ms,
        }
    )


if __name__ == "__main__":
    log_event({"type": "DEBUG_PIPELINE", "message": "Use log_latency to trace per-event timings."})
