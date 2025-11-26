import os
from pathlib import Path

from src.cache.warm_start import warm_start_whisper
from src.logging.structured_logger import LOG_PATH


class DummyModel:
    def __init__(self):
        self.calls = 0

    def transcribe(self, *_args, **_kwargs):
        self.calls += 1
        return [], None


def test_warm_start_logs_duration(tmp_path, monkeypatch):
    monkeypatch.setattr("src.logging.structured_logger.LOG_PATH", tmp_path / "events.jsonl", raising=False)
    model = DummyModel()
    duration = warm_start_whisper(model)
    assert model.calls == 1
    assert duration >= 0
    assert (tmp_path / "events.jsonl").exists()
