import json
from pathlib import Path
from typing import Dict

LOG_PATH = Path("logs/events.jsonl")


def log_event(event: Dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
