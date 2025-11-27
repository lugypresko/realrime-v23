import time

from src.dialogue_brain.brain import DialogueBrain
from src.dialogue_brain.cache import LRUCache, LastSuggestionRing
from src.dialogue_brain.memory import RollingMemory
from src.dialogue_brain.state_machine import StateMachine
from src.dialogue_brain.suggestion_engine import SuggestionEngine


def test_cache_hit_on_second_call():
    brain = DialogueBrain(debug=True)
    first = brain.process("Testing price timeline", "Objection - Price")
    second = brain.process("Testing price timeline", "Objection - Price")

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["state"] == first["state"]


def test_state_machine_transitions():
    sm = StateMachine()
    sm.transition("probe", "Can we dig into this?")
    assert sm.transition("probe", "Dig in") == "discovery"
    sm.transition("pain_signal", "That hurt last time")
    assert sm.transition("pain_signal", "Still hurts") == "pain"
    sm.transition("stall_objection", "price is high")
    assert sm.transition("stall_objection", "price is still high") == "objection"
    sm.transition("ready", "we're ready to go")
    assert sm.transition("ready", "ready to move") == "close"


def test_suggestion_rotation_no_recent_repeats():
    memory = RollingMemory(debug=True)
    engine = SuggestionEngine(memory=memory)
    selections = [engine.suggest("intent", "opening") for _ in range(6)]
    last_five = selections[-5:]
    assert len(set(last_five)) == len(last_five)


def test_memory_resets_after_inactivity():
    memory = RollingMemory(debug=True)
    memory.record_suggestion("first")
    memory._last_access -= memory.reset_seconds + 1
    memory.record_suggestion("second")
    assert memory.recent_suggestions() == ["second"]


def test_state_machine_timeout_resets():
    memory = RollingMemory(debug=True)
    sm = StateMachine(debug=True, memory=memory)
    sm.transition("probe", "explore")
    sm._last_update -= sm.reset_seconds + 1
    assert sm.transition("probe", "explore") == "opening"


def test_lru_cache_bounded_and_evicts_oldest():
    cache: LRUCache[str, int] = LRUCache(max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)

    assert len(cache) == 2
    assert cache.get("a") is None  # evicted
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_last_suggestion_ring_caps_growth():
    ring = LastSuggestionRing(max_items=3)
    for item in ["one", "two", "three", "four"]:
        ring.push(item)

    assert ring.recent() == ["two", "three", "four"]
    assert len(ring) == 3
