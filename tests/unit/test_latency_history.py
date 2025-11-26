from src.cache.latency_history import LatencyHistory


def test_latency_history_metrics_and_warning(monkeypatch):
    lh = LatencyHistory(max_events=5)
    for _ in range(5):
        lh.add({"whisper_latency": 1.4, "intent_latency": 0.02, "event_age": 1.0, "decision": "SUCCESS"})
    metrics = lh.log_warnings()
    assert metrics["p95"]["whisper_ms"] >= 1400
    assert metrics["suppression_rate"] == 0.0
