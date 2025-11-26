from typing import Sequence

from src.logging.structured_logger import log_event


def report_top_intents(text: str, prompt_ids: Sequence[str], scores: Sequence[float]) -> None:
    top = list(zip(prompt_ids, scores))[:3]
    log_event({"type": "DEBUG_INTENT", "text": text, "top": top})


if __name__ == "__main__":
    log_event({"type": "DEBUG_INTENT", "message": "Call report_top_intents from intent engine for debugging."})
