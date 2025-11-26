import sys
import time
from typing import Tuple, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - fallback for environments without numpy
    from src.mocks import mock_numpy as np

from src.contracts import (
    SILENCE_TRIGGER_FIELDS,
    WORKER_RESULT_FIELDS,
    create_worker_result,
    ensure_schema_keys,
)
from src.debug.debug_pipeline import log_latency
from src.logging.structured_logger import log_event
from src.telemetry.error_state import ErrorStateManager
from src.telemetry.event_inspector import inspect_event
from src.telemetry.prompt_quality import PromptQualityMonitor
from src.telemetry.telemetry_writer import write_event
from src.telemetry.watchdog import PipelineWatchdog
from src.worker.services import (
    BackpressureController,
    GovernorService,
    InferenceService,
    IntentService,
    LatencyMonitor,
    RepeatFilterAdapter,
)


BACKPRESSURE_THRESHOLD = 3


def worker_process(queue_sw, queue_wp, use_mock: bool = False, mock_event_limit: int | None = None, services: Optional[dict] = None):
    sys.stdout.reconfigure(encoding="utf-8")

    svc = services or {}
    error_state = svc.get("error_state") or ErrorStateManager()
    inference_service: InferenceService = svc.get("inference_service") or InferenceService(
        use_mock=use_mock, error_state=error_state
    )
    intent_service: IntentService = svc.get("intent_service") or IntentService(use_mock=use_mock)
    governor: GovernorService = svc.get("governor") or GovernorService(
        RepeatFilterAdapter(), inference_service.error_state, svc.get("prompt_quality") if svc else PromptQualityMonitor()
    )
    latency_monitor: LatencyMonitor = svc.get("latency_monitor") or LatencyMonitor()
    backpressure: BackpressureController = svc.get("backpressure") or BackpressureController(BACKPRESSURE_THRESHOLD)
    error_state = getattr(inference_service, "error_state", None)
    watchdog = PipelineWatchdog()

    processed = 0

    while True:
        dropped_ids = backpressure.drop_oldest(queue_sw)
        for dropped_id in dropped_ids:
            log_event({"type": "SUPPRESSED_BACKPRESSURE", "event_id": dropped_id})
            write_event({"type": "SUPPRESSED_BACKPRESSURE", "event_id": dropped_id})

        event = queue_sw.get()
        if not event or event.get("type") == "MIC_DEAD":
            continue
        ensure_schema_keys(event, SILENCE_TRIGGER_FIELDS, "SILENCE_TRIGGER")
        if not inspect_event(event, SILENCE_TRIGGER_FIELDS, "SILENCE_TRIGGER"):
            continue

        worker_start_ts = time.monotonic()
        watchdog.start(event["id"])
        audio = event["audio"].astype(np.float32)

        inference_result = inference_service.transcribe(audio)
        text = inference_result["text"]
        whisper_latency = float(inference_result["latency"])

        intent_result = intent_service.classify(text)
        best_idx = intent_result["prompt_id"]
        best_score = intent_result["score"]
        intent_latency = float(intent_result["latency"])

        event_age = time.monotonic() - event["timestamp"]
        transport_latency = worker_start_ts - event["timestamp"]
        transport_ms = transport_latency * 1000
        whisper_ms = whisper_latency * 1000
        intent_ms = intent_latency * 1000
        total_ms = event_age * 1000

        decision_info = governor.decide({"timestamp": event["timestamp"], "prompt_id": best_idx, "score": best_score}, transport_latency, whisper_latency, intent_latency)
        decision = decision_info["decision"]
        metrics = latency_monitor.record(whisper_latency, intent_latency, event_age, decision)

        result = create_worker_result(
            event_id=event["id"],
            event_timestamp=event["timestamp"],
            sentinel_timestamp=event.get("sentinel_timestamp", event["timestamp"]),
            worker_start_ts=worker_start_ts,
            whisper_latency=whisper_latency,
            intent_latency=intent_latency,
            event_age=event_age,
            decision=decision,
            text=text,
            prompt_id=str(best_idx),
            score=best_score,
            transport_latency_ms=transport_ms,
            total_latency_ms=total_ms,
        )
        ensure_schema_keys(result, WORKER_RESULT_FIELDS, "WORKER_RESULT")

        queue_wp.put(result)
        log_event(
            {
                "type": "WORKER_RESULT",
                "decision": decision,
                "event_id": event["id"],
                "transport_ms": transport_ms,
                "whisper_ms": whisper_ms,
                "intent_ms": intent_ms,
                "total_ms": total_ms,
                "latency_metrics": metrics,
            }
        )
        write_event(
            {
                "type": "WORKER_RESULT",
                "decision": decision,
                "event_id": event["id"],
                "transport_ms": transport_ms,
                "whisper_ms": whisper_ms,
                "intent_ms": intent_ms,
                "total_ms": total_ms,
            }
        )
        log_latency(event["id"], transport_ms, whisper_ms, intent_ms, total_ms)
        watchdog.clear(event["id"])
        watchdog.check()

        processed += 1
        if use_mock and mock_event_limit is not None and processed >= mock_event_limit:
            break
