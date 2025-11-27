from __future__ import annotations

import io
import time

import pytest

rich = pytest.importorskip("rich.console")
from rich.console import Console  # type: ignore

from ui.minimal_hud import HUDState, MinimalHUD


def test_hud_render_is_fast_and_non_blocking():
    console = Console(file=io.StringIO(), force_terminal=False, width=80)
    hud = MinimalHUD(console=console)

    durations = []
    for i in range(100):
        state = HUDState(
            text=f"utterance {i}",
            intent="probe",
            state="discovery",
            suggestion="ask for concrete example",
            whisper_ms=40,
            intent_ms=10,
            brain_ms=5,
            total_ms=55,
        )
        start = time.perf_counter()
        hud.render(state)
        durations.append((time.perf_counter() - start) * 1000)

    durations.sort()
    avg = sum(durations) / len(durations)
    p99 = durations[int(0.99 * len(durations))]
    assert avg < 4.0, f"HUD average render too slow: {avg}ms"
    assert p99 < 10.0, f"HUD p99 render too slow: {p99}ms"

    start = time.perf_counter()
    for _ in range(100):
        hud.render()
    total = (time.perf_counter() - start) * 1000
    assert total < 200, f"HUD blocked pipeline for {total}ms"
