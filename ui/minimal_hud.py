from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from rich import box
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


@dataclass
class HUDState:
    text: str = ""
    intent: str = ""
    state: str = "opening"
    suggestion: str = ""
    whisper_ms: float = 0.0
    intent_ms: float = 0.0
    brain_ms: float = 0.0
    total_ms: float = 0.0
    suppressed: bool = False


class MinimalHUD:
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.state = HUDState()
        self._lock = threading.Lock()

    def _render_table(self) -> Table:
        table = Table.grid(expand=True)
        table.add_row("Input", f"[yellow]{self.state.text}[/yellow]")
        table.add_row("Intent", f"[green]{self.state.intent}[/green]")
        table.add_row("State", f"[blue]{self.state.state}[/blue]")
        sug_color = "magenta" if not self.state.suppressed else "red"
        table.add_row("Suggestion", f"[{sug_color}]{self.state.suggestion}[/{sug_color}]")
        timing = (
            f"Whisper={self.state.whisper_ms:.0f}ms | Intent={self.state.intent_ms:.0f}ms | "
            f"Brain={self.state.brain_ms:.0f}ms | Total={self.state.total_ms/1000:.2f}s"
        )
        timing_color = "green" if not self.state.suppressed else "red"
        table.add_row("Timing", f"[{timing_color}]{timing}[/{timing_color}]")
        return table

    def render(self, state: Optional[HUDState] = None) -> Panel:
        with self._lock:
            if state:
                self.state = state
            header = "" if not self.state.suppressed else "SUPPRESSED"
            return Panel(
                Align.left(self._render_table()),
                title="Dialogue HUD",
                subtitle=header,
                box=box.SQUARE,
                padding=(1, 2),
            )

    def refresh(self, state: HUDState, live: Optional[Live] = None) -> None:
        self.state = state
        if live:
            live.update(self.render())
        else:
            self.console.print(self.render())

    def run_demo(self, updates, refresh_rate: float = 0.25) -> None:
        with Live(self.render(), console=self.console, refresh_per_second=8) as live:
            for update in updates:
                self.refresh(update, live=live)
                time.sleep(refresh_rate)

