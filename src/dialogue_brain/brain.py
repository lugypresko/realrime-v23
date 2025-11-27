from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict

from src.event_bus import EventBus

from .cache import LRUCache
from .memory import RollingMemory
from .state_machine import StateMachine
from .suggestion_engine import SuggestionEngine


logger = logging.getLogger(__name__)


@dataclass
class BrainResult:
    state: str
    suggestion: str
    cache_hit: bool
    timestamp: float
    latency_ms: float


class DialogueBrain:
    def __init__(self, debug: bool = False, event_bus: EventBus | None = None):
        self.memory = RollingMemory(debug=debug)
        self.state_machine = StateMachine(event_bus=event_bus, debug=debug, memory=self.memory)
        self.intent_state_cache = LRUCache[str, str]()
        self.state_suggestion_cache = LRUCache[str, str]()
        self.suggestion_engine = SuggestionEngine(memory=self.memory)
        self.event_bus = event_bus

    def _log_cache(self, cache_hit: bool, intent: str, state: str) -> None:
        logger.debug("DialogueBrain cache %s for intent='%s', state='%s'", "hit" if cache_hit else "miss", intent, state)

    def process(self, text: str, intent: str) -> Dict[str, str | float | bool]:
        start = time.perf_counter()
        cache_hit = False

        cached_state = self.intent_state_cache.get(intent)
        if cached_state:
            state = cached_state
            cache_hit = True
        else:
            state = self.state_machine.transition(intent=intent, text=text)
            self.intent_state_cache.set(intent, state)

        suggestion_from_cache = False
        cached_suggestion = self.state_suggestion_cache.get(state)
        if cached_suggestion and not self.memory.contains_suggestion(cached_suggestion):
            suggestion = cached_suggestion
            suggestion_from_cache = True
            cache_hit = cache_hit or suggestion_from_cache
        else:
            suggestion = self.suggestion_engine.suggest(intent=intent, state=state)
            self.state_suggestion_cache.set(state, suggestion)

        if suggestion_from_cache:
            self.memory.record_intent(intent)
            self.memory.record_suggestion(suggestion)

        self._log_cache(cache_hit, intent, state)
        timestamp = time.time()
        brain_latency_ms = (time.perf_counter() - start) * 1000
        logger.debug("DialogueBrain latency_ms=%.3f", brain_latency_ms)
        result = BrainResult(
            state=state,
            suggestion=suggestion,
            cache_hit=cache_hit,
            timestamp=timestamp,
            latency_ms=brain_latency_ms,
        )

        if self.event_bus:
            self.event_bus.publish(
                "suggestion_event",
                {
                    "state": result.state,
                    "suggestion": result.suggestion,
                    "cache_hit": result.cache_hit,
                    "timestamp": result.timestamp,
                    "brain_ms": result.latency_ms,
                },
            )

        return result.__dict__

