from __future__ import annotations

from typing import Iterable

from src.v2.interfaces import ResultFormatter, TelemetryClient, WorkerOutput
from src.v2.presenter.services import PresenterTelemetry, ResultValidator


class PresenterProcess:
    def __init__(self, formatter: ResultFormatter, telemetry_client: TelemetryClient) -> None:
        self.formatter = formatter
        self.validator = ResultValidator()
        self.telemetry = PresenterTelemetry(telemetry_client)

    def run(self, outputs: Iterable[WorkerOutput]) -> list[str]:
        rendered: list[str] = []
        for output in outputs:
            if not self.validator.validate(output):
                continue
            self.telemetry.log_output(output)
            rendered.append(self.formatter.format(output))
        return rendered
