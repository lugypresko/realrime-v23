from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.event_bus import EventBus


@dataclass
class StateTransition:
    old_state: str
    new_state: str
    reason: str


class StateMachine:
    """Deterministic conversation state machine with sticky transitions."""

    STATES: List[str] = ["opening", "discovery", "pain", "objection", "close"]
    CONFIRMATIONS_REQUIRED = 2

    INTENT_TO_STATE: Dict[str, str] = {
        "rapport": "opening",
        "probe": "discovery",
        "question": "discovery",
        "pain_signal": "pain",
        "pain": "pain",
        "stall_objection": "objection",
        "objection": "objection",
        "price": "objection",
        "timeline": "objection",
        "decision": "discovery",
        "closing_buy_indicator": "close",
        "ready": "close",
    }

    ALLOWED_TRANSITIONS: Dict[str, List[str]] = {
        "opening": ["opening", "discovery", "objection"],
        "discovery": ["discovery", "pain", "objection"],
        "pain": ["pain", "objection"],
        "objection": ["objection", "close"],
        "close": ["close"],
    }

    def __init__(self, event_bus: Optional[EventBus] = None, debug: bool = False, memory=None):
        self.current_state = "opening"
        self.event_bus = event_bus
        self._memory = memory
        self._noise_pattern = re.compile(r"[^a-zA-Z\s']+")
        self._pending_state: Optional[str] = None
        self._pending_count = 0
        self.reset_seconds = 10 if debug else 90
        self._last_update = time.monotonic()

    def _normalize(self, text: str) -> str:
        lowered = text.lower()
        return self._noise_pattern.sub("", lowered)

    def _intent_to_state(self, intent: str, text: str | None) -> str:
        normalized = self._normalize(f"{intent} {text or ''}")
        for keyword, target_state in self.INTENT_TO_STATE.items():
            if keyword in normalized:
                return target_state
        return self.current_state

    def _reset_if_inactive(self) -> None:
        now = time.monotonic()
        if now - self._last_update > self.reset_seconds:
            previous = self.current_state
            self.current_state = "opening"
            self._pending_state = None
            self._pending_count = 0
            self._last_update = now
            if self._memory:
                self._memory.reset()
            if self.event_bus:
                self.event_bus.publish(
                    "reset_event",
                    {"old_state": previous, "new_state": self.current_state, "reason": "timeout"},
                )

    def transition(self, intent: str, text: str | None = None) -> str:
        self._reset_if_inactive()
        target_state = self._intent_to_state(intent, text)
        allowed_targets = self.ALLOWED_TRANSITIONS.get(self.current_state, [self.current_state])
        if target_state not in allowed_targets and target_state != "close":
            target_state = self.current_state

        if target_state == self.current_state:
            self._pending_state = None
            self._pending_count = 0
            self._last_update = time.monotonic()
            return self.current_state

        if target_state != self._pending_state:
            self._pending_state = target_state
            self._pending_count = 1
            self._last_update = time.monotonic()
            return self.current_state

        self._pending_count += 1
        self._last_update = time.monotonic()
        if self._pending_count < self.CONFIRMATIONS_REQUIRED:
            return self.current_state

        old_state = self.current_state
        self.current_state = target_state
        self._pending_state = None
        self._pending_count = 0

        if self.event_bus:
            self.event_bus.publish(
                "state_transition_event",
                StateTransition(old_state=old_state, new_state=self.current_state, reason=intent).__dict__,
            )
        return self.current_state
