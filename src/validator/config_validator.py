from pathlib import Path
from typing import Dict

from src.logging.structured_logger import log_event


REQUIRED_FILES = ["embeddings.npy", "prompts.json"]
REQUIRED_DIRS = ["logs"]
SAMPLE_RATE = 16000


def validate_config(audio_device_index: int | None = None) -> Dict:
    results: Dict[str, bool] = {}

    for fname in REQUIRED_FILES:
        exists = Path(fname).exists()
        results[f"file:{fname}"] = exists
    for dname in REQUIRED_DIRS:
        exists = Path(dname).exists() or Path(dname).mkdir(parents=True, exist_ok=True) is None
        results[f"dir:{dname}"] = exists

    results["sample_rate"] = SAMPLE_RATE == 16000

    # Dependency probes
    for mod in ("numpy", "faster_whisper"):
        try:
            __import__(mod)
            results[f"import:{mod}"] = True
        except Exception:
            results[f"import:{mod}"] = False

    # Audio device presence check (best-effort)
    if audio_device_index is not None:
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            results["audio_device"] = 0 <= audio_device_index < len(devices)
        except Exception:
            results["audio_device"] = False
    else:
        results["audio_device"] = True

    log_event({"type": "CONFIG_VALIDATION", "results": results})
    return results
