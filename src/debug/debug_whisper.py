import time

from src.logging.structured_logger import log_event


def run_debug_whisper(transcribe_fn, audio):
    start = time.monotonic()
    text = transcribe_fn(audio)
    duration_ms = (time.monotonic() - start) * 1000
    log_event({"type": "DEBUG_WHISPER", "text": text, "duration_ms": duration_ms})
    return text, duration_ms


if __name__ == "__main__":
    log_event({"type": "DEBUG_WHISPER", "message": "Provide a transcribe_fn and audio buffer in code."})
