from typing import List

from src.telemetry.telemetry_writer import write_event


def enumerate_microphones() -> List[str]:
    devices = []
    try:
        import sounddevice as sd

        devices = [str(device['name']) for device in sd.query_devices()]  # type: ignore[index]
    except Exception:
        devices = []
    write_event({"type": "DEVICE_ENUM", "devices": devices})
    return devices


def detect_sampling_drift(expected_rate: int, observed_rate: int) -> bool:
    drift = abs(expected_rate - observed_rate)
    if drift > 0:
        write_event({"type": "DEVICE_DRIFT", "expected": expected_rate, "observed": observed_rate})
    return drift > 0
