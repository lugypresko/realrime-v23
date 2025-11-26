from __future__ import annotations

from typing import Dict

from src.v2.interfaces import WorkerOutput, ResultFormatter, TelemetryClient, validate_worker_output


class SimpleResultFormatter(ResultFormatter):
    def format(self, output: WorkerOutput) -> str:
        return (
            f"[P] id={output.event_id} decision={output.decision} text=\"{output.text}\" "
            f"prompt={output.prompt_id} score={output.score:.3f} total={output.total_latency_ms:.1f}ms"
        )


class ResultValidator:
    def validate(self, output: WorkerOutput) -> bool:
        return validate_worker_output(output)


class PresenterTelemetry:
    def __init__(self, client: TelemetryClient):
        self.client = client

    def log_output(self, output: WorkerOutput) -> None:
        self.client.record({"type": "presenter_output", "event_id": output.event_id, "decision": output.decision})
