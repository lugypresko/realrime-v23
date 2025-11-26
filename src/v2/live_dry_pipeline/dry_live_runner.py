from __future__ import annotations

import json
import queue
import statistics
import sys
import time
from multiprocessing import Event, Process, Queue
from threading import Thread
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

from src.v2.app.composition import CompositionRoot
from src.v2.interfaces import (
    AudioSource,
    GovernorService,
    IntentClassifier,
    ResultFormatter,
    SilenceEvent,
    SilencePolicy,
    STTEngine,
    TelemetryClient,
    WorkerOutput,
    validate_silence_event,
    validate_worker_output,
)
from src.v2.live_dry_pipeline.health_report import HealthReport, HealthStats


class MetricsCollector:
    def __init__(self) -> None:
        self.triggers: int = 0
        self.outputs: List[WorkerOutput] = []
        self.trigger_ids: List[str] = []
        self.output_ids: List[str] = []
        self.stt_failures: int = 0
        self.intent_failures: int = 0
        self.cpu_samples: List[float] = []
        self.mem_samples: List[float] = []
        self.audio_overruns: int = 0
        self.event_id_order_ok: bool = True
        self.timestamp_monotonic: bool = True
        self.stalls: Dict[str, bool] = {"sentinel": False, "worker": False, "presenter": False}
        self.last_seen: Dict[str, float] = {"sentinel": time.monotonic(), "worker": time.monotonic(), "presenter": time.monotonic()}
        self.events_seen: Dict[str, bool] = {"sentinel": False, "worker": False, "presenter": False}
        self.failures: List[str] = []
        self.stt_checks: int = 0
        self.stt_quality_errors: int = 0
        self.intent_checks: int = 0
        self.intent_quality_flags: int = 0
        self.duplicate_trigger_intervals: int = 0
        self.missed_trigger_intervals: int = 0
        self.fail_fast_error: Optional[str] = None
        self.last_trigger_timestamp: Optional[float] = None
        self.last_output_timestamp: Optional[float] = None

    def record_trigger(self, event: SilenceEvent) -> None:
        self.triggers += 1
        self.trigger_ids.append(event.event_id)
        self._check_event_order(event.event_id, "sentinel")
        self._update_last("sentinel")
        self.events_seen["sentinel"] = True
        previous_ts = self.last_trigger_timestamp
        if previous_ts is not None and event.timestamp < previous_ts:
            self.timestamp_monotonic = False
            self.failures.append("non-monotonic trigger timestamp")
        self.last_trigger_timestamp = event.timestamp
        self._check_duplicate_interval(event, previous_ts)

    def record_output(self, output: WorkerOutput) -> None:
        self.outputs.append(output)
        self.output_ids.append(output.event_id)
        self._check_event_order(output.event_id, "worker")
        self._update_last("worker")
        self.events_seen["worker"] = True
        if self.last_output_timestamp is not None and output.metadata.get("timestamp") is not None:
            try:
                ts = float(output.metadata.get("timestamp"))
                if ts < self.last_output_timestamp:
                    self.timestamp_monotonic = False
                    self.failures.append("non-monotonic worker timestamp")
                self.last_output_timestamp = ts
            except Exception:
                pass
        self._check_stt_quality(output)
        self._check_intent_quality(output)

    def record_presenter(self) -> None:
        self._update_last("presenter")
        self.events_seen["presenter"] = True

    def duplicate_triggers(self) -> int:
        return len(self.output_ids) - len(set(self.output_ids))

    def missed_triggers(self) -> int:
        return max(self.triggers - len(self.outputs), 0)

    def latency_stats(self) -> Tuple[Optional[float], Optional[float]]:
        if not self.outputs:
            return None, None
        totals = [o.total_latency_ms for o in self.outputs]
        avg = statistics.mean(totals)
        p95 = statistics.quantiles(totals, n=100)[94] if len(totals) >= 20 else max(totals)
        return avg, p95

    def minutes_total_triggers(self) -> int:
        return self.triggers

    def _check_event_order(self, event_id: str, kind: str) -> None:
        ids = self.trigger_ids if kind == "sentinel" else self.output_ids
        if len(ids) >= 2 and ids[-1] <= ids[-2]:
            self.event_id_order_ok = False
            self.failures.append(f"non-increasing {kind} event_id")

    def _update_last(self, role: str) -> None:
        self.last_seen[role] = time.monotonic()

    def check_stalls(self, now: float) -> None:
        for role in self.last_seen:
            if not self.events_seen.get(role, False):
                continue
            stalled = (now - self.last_seen[role]) > 2.0
            self.stalls[role] = stalled
            if stalled:
                self.failures.append(f"stall detected: {role}")

    def _check_duplicate_interval(self, event: SilenceEvent, previous_ts: Optional[float]) -> None:
        if len(self.trigger_ids) >= 2:
            try:
                prev_ts = float(self.trigger_ids[-2].split("-")[-1])
                curr_ts = float(self.trigger_ids[-1].split("-")[-1])
                if (curr_ts - prev_ts) < 300:
                    self.duplicate_trigger_intervals += 1
            except Exception:
                return
        # missed trigger estimation via silence gap
        if previous_ts is not None:
            try:
                gap = event.timestamp - previous_ts
                if gap > 2.0:
                    self.missed_trigger_intervals += 1
            except Exception:
                return

    def _check_stt_quality(self, output: WorkerOutput) -> None:
        self.stt_checks += 1
        text = output.text or ""
        words = text.strip().split()
        if not text.strip():
            self.stt_quality_errors += 1
            self.stt_failures += 1
            return
        if len(words) <= 1:
            self.stt_quality_errors += 1
        non_alpha_ratio = 0.0
        if text:
            non_alpha = sum(1 for c in text if not c.isalpha() and not c.isspace())
            non_alpha_ratio = non_alpha / max(len(text), 1)
        if non_alpha_ratio > 0.5:
            self.stt_quality_errors += 1

    def _check_intent_quality(self, output: WorkerOutput) -> None:
        self.intent_checks += 1
        if not output.prompt_id:
            self.intent_quality_flags += 1
            self.intent_failures += 1
        if output.score < 0.05:
            self.intent_quality_flags += 1
        if output.intent_latency_ms > 100:
            self.intent_quality_flags += 1


class ResourceSampler(Process):
    def __init__(self, interval: float, stop_event: Event, metrics_queue: Queue):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_event = stop_event
        self.metrics_queue = metrics_queue

    def run(self) -> None:  # pragma: no cover - runtime only
        if psutil is None:
            return
        proc = psutil.Process()
        while not self.stop_event.is_set():
            try:
                cpu = proc.cpu_percent(interval=None)
                mem = proc.memory_info().rss / (1024 * 1024)
                self.metrics_queue.put(("resource_sample", {"cpu": cpu, "mem": mem}))
            except Exception:
                pass
            self.stop_event.wait(self.interval)


def _fail_fast(mode: str) -> None:
    try:
        root = CompositionRoot(mode=mode)
        sentinel, worker, presenter = root.build_pipeline()
    except Exception as exc:  # pragma: no cover - guard path
        raise SystemExit(f"[FAIL] composition build failed: {exc}")

    bindings = [
        (getattr(sentinel, "audio_source", None), AudioSource, "audio_source"),
        (getattr(sentinel, "silence_policy", None), SilencePolicy, "silence_policy"),
        (getattr(worker, "stt", None), STTEngine, "stt_engine"),
        (getattr(worker, "intent_classifier", None), IntentClassifier, "intent_classifier"),
        (getattr(worker, "governor", None), GovernorService, "governor"),
        (getattr(presenter, "formatter", None), ResultFormatter, "result_formatter"),
    ]
    telemetry_client = getattr(sentinel, "telemetry", None)
    if telemetry_client is not None:
        bindings.append((getattr(telemetry_client, "_client", None), TelemetryClient, "sentinel_telemetry_client"))
    worker_telemetry = getattr(worker, "telemetry", None)
    if worker_telemetry is not None:
        bindings.append((worker_telemetry, TelemetryClient, "worker_telemetry"))
    presenter_telemetry = getattr(presenter, "telemetry", None)
    if presenter_telemetry is not None:
        bindings.append((getattr(presenter_telemetry, "_client", None), TelemetryClient, "presenter_telemetry"))
    for impl, iface, name in bindings:
        if impl is None or not isinstance(impl, iface):
            raise SystemExit(f"[FAIL] missing or invalid binding: {name}")


def _run_sentinel(mode: str, event_queue: Queue, metrics_queue: Queue, stop_event: Event, duration_s: float) -> None:
    root = CompositionRoot(mode=mode)
    sentinel, _, _ = root.build_pipeline()
    start = time.monotonic()
    while not stop_event.is_set() and (time.monotonic() - start) < duration_s:
        try:
            sentinel.run(lambda ev: _emit_event(ev, event_queue, metrics_queue))
        except Exception:
            break


def _emit_event(event: SilenceEvent, event_queue: Queue, metrics_queue: Queue) -> None:
    metrics_queue.put(("trigger", event))
    event_queue.put(event)


def _run_worker(mode: str, event_queue: Queue, output_queue: Queue, metrics_queue: Queue, stop_event: Event) -> None:
    root = CompositionRoot(mode=mode)
    _, worker, _ = root.build_pipeline()

    def pending() -> int:
        try:
            return event_queue.qsize()
        except NotImplementedError:
            return 0

    def events() -> Iterable[SilenceEvent]:
        while not stop_event.is_set():
            try:
                event = event_queue.get(timeout=0.5)
                yield event
            except queue.Empty:
                continue

    def emit(output: WorkerOutput) -> None:
        metrics_queue.put(("worker_output", output))
        output_queue.put(output)

    worker.run(events(), emit, pending_events=pending)


def _run_presenter(mode: str, output_queue: Queue, metrics_queue: Queue, stop_event: Event) -> None:
    root = CompositionRoot(mode=mode)
    _, _, presenter = root.build_pipeline()

    def outputs() -> Iterable[WorkerOutput]:
        while not stop_event.is_set():
            try:
                yield output_queue.get(timeout=0.5)
            except queue.Empty:
                continue

    rendered = presenter.run(outputs())
    metrics_queue.put(("presenter_rendered", rendered))


def _collect_metrics(
    metrics_queue: Queue,
    collector: MetricsCollector,
    stop_event: Event,
    event_queue: Queue,
    output_queue: Queue,
) -> None:
    last_watchdog = time.monotonic()
    while not stop_event.is_set() or not metrics_queue.empty():
        try:
            kind, payload = metrics_queue.get(timeout=0.5)
        except queue.Empty:
            kind = None
        if kind == "trigger" and validate_silence_event(payload):
            collector.record_trigger(payload)
        elif kind == "worker_output":
            if validate_worker_output(payload):
                collector.record_output(payload)
        elif kind == "resource_sample":
            collector.cpu_samples.append(payload.get("cpu", 0.0))
            collector.mem_samples.append(payload.get("mem", 0.0))
        elif kind == "presenter_rendered":
            collector.record_presenter()

        now = time.monotonic()
        if now - last_watchdog >= 5.0:
            collector.check_stalls(now)
            collector.audio_overruns += _estimate_overruns(event_queue, output_queue)
            last_watchdog = now


def run_dry_live(
    minutes: int = 3,
    mode: str = "real",
    fail_on_warning: bool = False,
    output_file: Optional[str] = "dry_live_report.json",
    verbose: bool = False,
) -> HealthReport:
    duration_s = minutes * 60
    _fail_fast(mode)

    event_queue: Queue = Queue()
    output_queue: Queue = Queue()
    metrics_queue: Queue = Queue()
    stop_event = Event()

    collector = MetricsCollector()

    sentinel_proc = Process(target=_run_sentinel, args=(mode, event_queue, metrics_queue, stop_event, duration_s), daemon=True)
    worker_proc = Process(target=_run_worker, args=(mode, event_queue, output_queue, metrics_queue, stop_event), daemon=True)
    presenter_proc = Process(target=_run_presenter, args=(mode, output_queue, metrics_queue, stop_event), daemon=True)

    sampler = ResourceSampler(interval=5.0, stop_event=stop_event, metrics_queue=metrics_queue)

    collector_thread = Thread(
        target=_collect_metrics,
        args=(metrics_queue, collector, stop_event, event_queue, output_queue),
        daemon=True,
    )

    sampler.start()
    collector_thread.start()
    sentinel_proc.start()
    worker_proc.start()
    presenter_proc.start()

    start = time.monotonic()
    try:
        while (time.monotonic() - start) < duration_s:
            time.sleep(1)
    finally:
        stop_event.set()
        sentinel_proc.join(timeout=5)
        worker_proc.join(timeout=5)
        presenter_proc.join(timeout=5)
        sampler.join(timeout=5)
        collector_thread.join(timeout=5)

    avg_lat, p95_lat = collector.latency_stats()
    stt_error_rate = (collector.stt_quality_errors / max(collector.stt_checks, 1)) * 100
    stats = HealthStats(
        total_triggers=collector.triggers,
        avg_latency_ms=avg_lat,
        p95_latency_ms=p95_lat,
        duplicate_triggers=collector.duplicate_triggers(),
        missed_triggers=collector.missed_triggers(),
        audio_overruns=collector.audio_overruns,
        stt_failures=collector.stt_failures,
        intent_failures=collector.intent_failures,
        memory_drift=_memory_drift(collector.mem_samples),
        cpu_spikes=_cpu_spikes(collector.cpu_samples),
        stt_error_rate=stt_error_rate,
        intent_quality_flags=collector.intent_quality_flags,
        stall_flags=collector.stalls,
        event_id_order_ok=collector.event_id_order_ok,
        timestamp_monotonic=collector.timestamp_monotonic,
        missed_trigger_intervals=collector.missed_trigger_intervals,
        duplicate_trigger_intervals=collector.duplicate_trigger_intervals,
        failures=collector.failures,
    )
    report = HealthReport.from_stats(stats)
    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
        except Exception:
            pass
    if verbose:
        print(report.render())
    if fail_on_warning and not report.ready:
        raise SystemExit(1)
    return report


def _memory_drift(samples: List[float]) -> Optional[float]:
    if len(samples) < 2:
        return None
    return max(samples) - min(samples)


def _cpu_spikes(samples: List[float]) -> int:
    if not samples:
        return 0
    return sum(1 for s in samples if s > 65.0)


def _estimate_overruns(event_queue: Queue, output_queue: Queue) -> int:
    overrun = 0
    for q in (event_queue, output_queue):
        try:
            if q.qsize() > 10:
                overrun += 1
        except NotImplementedError:
            continue
    return overrun


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv or sys.argv[1:]
    minutes = 3
    mode = "real"
    fail_on_warning = False
    output_file: Optional[str] = "dry_live_report.json"
    verbose = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--minutes" and i + 1 < len(argv):
            try:
                minutes = int(argv[i + 1])
            except ValueError:
                pass
            i += 1
        elif arg == "--fail-on-warning":
            fail_on_warning = True
        elif arg == "--output-file" and i + 1 < len(argv):
            output_file = argv[i + 1]
            i += 1
        elif arg == "--verbose":
            verbose = True
        i += 1

    report = run_dry_live(
        minutes=minutes,
        mode=mode,
        fail_on_warning=fail_on_warning,
        output_file=output_file,
        verbose=verbose,
    )
    if not verbose:
        print(report.render())
    return 0 if report.ready or not fail_on_warning else 1


if __name__ == "__main__":
    raise SystemExit(main())
