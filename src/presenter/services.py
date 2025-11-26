from src.contracts import ensure_schema_keys, WORKER_RESULT_FIELDS
from src.interfaces import ResultFormatter
from src.logging.structured_logger import log_event


class ResultValidator:
    def validate(self, result):
        ensure_schema_keys(result, WORKER_RESULT_FIELDS, "WORKER_RESULT")
        return True


class SimpleResultFormatter(ResultFormatter):
    def format(self, event: dict) -> str:
        return (
            f"[P] id={event['id']} decision={event['decision']} text=\"{event['text']}\" "
            f"prompt={event['prompt_id']} score={event['score']} transport={event['transport_latency_ms']:.1f}ms "
            f"whisper={event['whisper_latency']*1000:.1f}ms intent={event['intent_latency']*1000:.1f}ms "
            f"total_age={event['event_age']*1000:.1f}ms"
        )


class PresenterTelemetry:
    def emit(self, result):
        log_event({"type": "PRESENTER_RESULT", **result})
