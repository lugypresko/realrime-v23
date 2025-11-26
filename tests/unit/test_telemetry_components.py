import json
from pathlib import Path

from src.telemetry.error_state import ErrorStateManager
from src.telemetry.prompt_quality import PromptQualityMonitor
from src.telemetry.telemetry_aggregator import TelemetryAggregator
from src.telemetry.watchdog import PipelineWatchdog


def test_error_state_safe_mode():
    mgr = ErrorStateManager(window_sec=1)
    mgr.record_whisper_failure()
    mgr.record_whisper_failure()
    assert mgr.should_use_safe_mode() is True


def test_prompt_quality_monitor(tmp_path, monkeypatch):
    monitor = PromptQualityMonitor(threshold=0.5)
    assert monitor.evaluate("p1", 0.6) is True
    assert monitor.evaluate("p1", 0.1) is False


def test_telemetry_aggregator_percentiles():
    agg = TelemetryAggregator(max_events=5)
    agg.add_measurement(0.1, 0.01, 0.2, "SUCCESS")
    agg.add_measurement(0.2, 0.02, 0.3, "SUCCESS")
    summary = agg.summary()
    assert summary["whisper_p50"] > 0
    assert "suppression_rate" in summary


def test_watchdog_timeout(tmp_path):
    wd = PipelineWatchdog(timeout_sec=0.0)
    wd.start("e1")
    wd.check()
