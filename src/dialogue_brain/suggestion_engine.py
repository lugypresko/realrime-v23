from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

from .cache import LastSuggestionRing
from .memory import RollingMemory


class SuggestionEngine:
    def __init__(self, rules_path: Path | None = None, memory: RollingMemory | None = None):
        self.rules_path = rules_path or Path(__file__).parent / "rules" / "suggestion_rules.json"
        self.rules: Dict[str, List[str]] = self._load_rules()
        self._state_indices: Dict[str, int] = {state: 0 for state in self.rules}
        self.memory = memory or RollingMemory()
        self._last_suggestion_ring = LastSuggestionRing()

    def _load_rules(self) -> Dict[str, List[str]]:
        with self.rules_path.open("r", encoding="utf-8") as f:
            data: Dict[str, List[str]] = json.load(f)
        return data

    def _next_suggestion(self, state: str) -> str:
        options = self.rules.get(state, [])
        if not options:
            return "Let me think about the best next step for you."

        start_idx = self._state_indices.get(state, 0)
        attempts = 0
        while attempts < len(options):
            candidate = options[(start_idx + attempts) % len(options)]
            if not self.memory.contains_suggestion(candidate) and not self._last_suggestion_ring.contains(candidate):
                self._state_indices[state] = (start_idx + attempts + 1) % len(options)
                return candidate
            attempts += 1

        # fallback if everything is in memory
        choice = random.choice(options)
        self._state_indices[state] = (options.index(choice) + 1) % len(options)
        return choice

    def suggest(self, intent: str, state: str) -> str:
        suggestion = self._next_suggestion(state)
        self.memory.record_intent(intent)
        self.memory.record_suggestion(suggestion)
        self._last_suggestion_ring.push(suggestion)
        return suggestion

