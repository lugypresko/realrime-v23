from __future__ import annotations

import time

from src.dialogue_brain.brain import DialogueBrain


def test_state_does_not_regress_without_rule():
    brain = DialogueBrain(debug=True)
    intents = ["rapport", "probe", "probe", "pain_signal", "pain_signal", "rapport"]
    states = []
    for intent in intents:
        states.append(brain.process("", intent)["state"])
        brain.intent_state_cache.clear()
    assert states[-2] == "pain", f"State machine should confirm pain, got {states}"


def test_objection_and_ready_jump_logic():
    brain = DialogueBrain(debug=True)
    brain.process("", "stall_objection")
    brain.intent_state_cache.clear()
    second = brain.process("", "stall_objection")
    brain.intent_state_cache.clear()
    ready_first = brain.process("", "ready")
    brain.intent_state_cache.clear()
    ready_second = brain.process("", "ready")
    assert second["state"] == "objection", "Objection intents must force objection state"
    assert ready_second["state"] == "close", "Ready intent must jump to close state"


def test_budget_objection_sticky():
    brain = DialogueBrain(debug=True)
    states = []
    for intent in ["price", "price", "timeline", "timeline", "budget"]:
        states.append(brain.process("", intent)["state"])
        brain.intent_state_cache.clear()
    assert all(state == "objection" for state in states[-3:]), "Budget concerns should remain objection"


def test_suggestion_rotation_and_timeout_reset():
    brain = DialogueBrain(debug=True)
    seen = []
    for _ in range(10):
        res = brain.process("", "probe")
        assert res["suggestion"] not in seen[-5:], "Suggestion repeated within last five"
        seen.append(res["suggestion"])
        brain.intent_state_cache.clear()
    brain.memory.reset_seconds = 0.01
    brain.state_machine.reset_seconds = 0.01
    time.sleep(0.02)
    res = brain.process("", "rapport")
    assert res["state"] == "opening", "Brain should reset to opening after inactivity"
