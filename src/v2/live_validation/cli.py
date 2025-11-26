from __future__ import annotations

import sys
from typing import Callable

from src.v2.live_validation.test_live_validation import (
    validate_audio_device,
    validate_end_to_end_live,
    validate_intent_live,
    validate_stt_live,
    validate_vad_live,
)


def _run_and_print(name: str, fn: Callable[[], object]) -> bool:
    outcome = fn()
    status = "OK" if getattr(outcome, "success", False) else "FAIL"
    print(f"[{status}] {name}: {getattr(outcome, 'message', '')}")
    return bool(getattr(outcome, "success", False))


def main():
    checks = [
        ("Audio device", validate_audio_device),
        ("VAD", validate_vad_live),
        ("STT", validate_stt_live),
        ("Intent", validate_intent_live),
        ("End-to-end", validate_end_to_end_live),
    ]
    all_ok = True
    for name, fn in checks:
        if not _run_and_print(name, fn):
            all_ok = False
    if not all_ok:
        print("[WARN] Missing dependencies or validation failure detected. Fix before live pipeline run.")
        sys.exit(1)
    print("[OK] All live validations passed. Ready for live pipeline testing.")


if __name__ == "__main__":
    main()
