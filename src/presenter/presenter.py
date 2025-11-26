import sys
from typing import Optional

from src.contracts import WORKER_RESULT_FIELDS
from src.logging.structured_logger import log_event
from src.presenter.services import PresenterTelemetry, ResultValidator, SimpleResultFormatter
from src.telemetry.event_inspector import inspect_event
from src.telemetry.telemetry_writer import write_event


def presenter_process(queue_wp, use_mock: bool = False, mock_event_limit: int | None = None, services: Optional[dict] = None):
    sys.stdout.reconfigure(encoding="utf-8")
    processed = 0
    svc = services or {}
    validator: ResultValidator = svc.get("validator") or ResultValidator()
    formatter: SimpleResultFormatter = svc.get("formatter") or SimpleResultFormatter()
    telemetry: PresenterTelemetry = svc.get("telemetry") or PresenterTelemetry()

    while True:
        result = queue_wp.get()
        if not result:
            continue
        validator.validate(result)
        if not inspect_event(result, WORKER_RESULT_FIELDS, "WORKER_RESULT"):
            continue

        line = formatter.format(result)
        print(line)
        telemetry.emit(result)
        write_event({"type": "PRESENTER", "event_id": result["id"], "decision": result["decision"]})
        log_event({"type": "PRESENTER", "event_id": result["id"], "decision": result["decision"], "line": line})

        processed += 1
        if use_mock and mock_event_limit is not None and processed >= mock_event_limit:
            break
