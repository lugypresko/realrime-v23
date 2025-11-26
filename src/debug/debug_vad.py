from src.logging.structured_logger import log_event


def report_vad_trigger(event_id: str, speaking: bool, prob: float) -> None:
    log_event({"type": "DEBUG_VAD", "event_id": event_id, "speaking": speaking, "prob": prob})


if __name__ == "__main__":
    log_event({"type": "DEBUG_VAD", "message": "Hook this into VAD callback for detailed traces."})
