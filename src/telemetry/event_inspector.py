from typing import Dict, Iterable

from src.telemetry.telemetry_writer import write_event


def inspect_event(event: Dict, required_keys: Iterable[str], name: str) -> bool:
    missing = set(required_keys) - set(event.keys())
    if missing:
        write_event({"type": "EVENT_SCHEMA_ERROR", "name": name, "missing": sorted(missing)})
        return False
    return True
