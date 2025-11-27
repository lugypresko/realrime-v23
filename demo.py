from __future__ import annotations

import random
import signal
import time

from src.dialogue_brain.brain import DialogueBrain
from ui.minimal_hud import HUDState, MinimalHUD

INSTRUCTIONS = """
Real-Time Cognitive Assistant Demo
Press Ctrl+C to exit.
Simulating audio -> intent -> brain -> HUD.
"""

SAMPLE_TEXT = [
    "Yeah, that timeline is tight.",
    "We're ready once procurement signs off.",
    "Price is still a sticking point.",
    "Who will own rollout internally?",
    "What happens if this slips again?",
]


class DemoRunner:
    def __init__(self) -> None:
        self.brain = DialogueBrain(debug=True)
        self.hud = MinimalHUD()
        self._running = True

    def _handle_exit(self, *_):
        self._running = False

    def _fake_intent(self, text: str) -> tuple[str, float]:
        start = time.perf_counter()
        lowered = text.lower()
        if "price" in lowered:
            intent = "Objection - Price"
        elif "timeline" in lowered:
            intent = "Objection - Timeline"
        elif "ready" in lowered:
            intent = "Decision Ready"
        elif "own" in lowered:
            intent = "Discovery - Stakeholders"
        else:
            intent = "General Inquiry"
        return intent, (time.perf_counter() - start) * 1000

    def run(self) -> None:
        signal.signal(signal.SIGINT, self._handle_exit)
        console = self.hud.console
        console.print(INSTRUCTIONS)
        console.print("Starting mock audio stream (simulated)...\n")

        while self._running:
            text = random.choice(SAMPLE_TEXT)
            whisper_ms = random.uniform(90, 160)
            intent, intent_ms = self._fake_intent(text)
            brain_start = time.perf_counter()
            result = self.brain.process(text=text, intent=intent)
            brain_ms = (time.perf_counter() - brain_start) * 1000
            total_ms = whisper_ms + intent_ms + brain_ms

            self.hud.refresh(
                HUDState(
                    text=text,
                    intent=intent,
                    state=result["state"],
                    suggestion=result["suggestion"],
                    whisper_ms=whisper_ms,
                    intent_ms=intent_ms,
                    brain_ms=brain_ms,
                    total_ms=total_ms,
                    suppressed=total_ms > 500,
                )
            )
            time.sleep(0.6)

        console.print("\nDemo stopped. Goodbye!\n")


if __name__ == "__main__":
    DemoRunner().run()

