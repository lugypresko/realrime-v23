from src.telemetry.telemetry_writer import write_event


class PromptQualityMonitor:
    def __init__(self, threshold: float = 0.2) -> None:
        self.threshold = threshold

    def evaluate(self, prompt_id: str, score: float) -> bool:
        if score < self.threshold:
            write_event({"type": "PROMPT_QUALITY_LOW", "prompt_id": prompt_id, "score": score})
            return False
        return True
