DAY 5 – “Dialogue Brain” Implementation Instructions (Including Caching Layer)

Objective:
Turn your Day 4 pipeline into an intelligent, ultra-low-latency conversational engine that:

Interprets intent,

Tracks state,

Generates tactical suggestions,

Prevents repetition,

Responds in <50ms consistently
using strategic caching.

1. Create the Dialogue Brain Module

Create a new module at:

src/dialogue_brain/


It will contain three major components:

State Machine – determines where the conversation currently sits
(Opening → Discovery → Pain → Objection → Close).

Suggestion Engine – chooses the best tactical move for the detected intent.

Memory – stores last 5 interactions to avoid repetition.

Caching Layer – accelerates all repeated logic to near zero latency.

2. Add the Caching Layer (Mandatory)
Caching Goals

Completely eliminate repeated work.

Ensure the “Brain” always responds in 1–8ms, even when the pipeline is busy.

Prevent repeated suggestions or repeated state decisions.

Caching Components to Create

Create a unified caching module with:

Intent Cache

Key: normalized input text

Value: {intent_label, score}
Purpose: Skip computing embeddings for repeated phrases.

Suggestion Cache

Key: (intent_label, state)

Value: preloaded rotation list + current index
Purpose: Avoid repeating rule matching and ensure suggestion rotation.

State Transition Cache

Key: (current_state, intent)

Value: next_state
Purpose: Avoid recomputing transitions.

Recent Suggestion Memory

Tracks last 5 delivered suggestions
Purpose: Prevent duplicates.

Cache Policy

Use small LRU caches (max sizes ~200).

Log hits vs. misses.

Caches reset per session only.

3. Build the State Machine

Function: Convert intents into high-level phases.

Examples:

“timeline uncertainty” → Objection

“decision maker missing” → Discovery

“just checking options” → Opening

Requirements

Must support caching of repeated transitions.

Must allow forced overrides in the future (manual control).

Must log state changes (“Discovery → Objection”).

4. Build the Suggestion Engine

Function:
For each (intent, state) pair, recommend the next best tactical move.

Rules Source

JSON rules file at:

src/dialogue_brain/rules/suggestion_rules.json


Contains:

intent → state → suggestion list

rotation order

suppression rules (don’t repeat recent suggestions)

Suggestion Engine Requirements

Use Suggestion Cache to avoid reloading rules for the same pair.

Use Memory to avoid repeating recent suggestions.

Must return a single suggestion per event.

5. Build the Memory Component

Purpose:
Short-term conversational memory to avoid repeating the same suggestion.

Requirements

Stores last 5 suggestions

Suggestion Engine checks memory before selecting

Memory resets after 90 seconds of inactivity (configurable)

6. Integrate Dialogue Brain Into the Pipeline

Modify 04_pipeline_test.py (and rename to 05_dialogue_pipeline.py) so the flow becomes:

Step-by-step Processing

Audio detected by Sentinel

Transcription produced by Whisper

Intent determined

Check Intent Cache

If miss → compute embedding → store in cache

State updated

Check State Transition Cache

Apply transition rule

Suggestion generated

Check Suggestion Cache

Apply rotation

Check Memory to avoid repetition

Log suggestion

Return structured output
(Input → Intent → State → Suggestion → Timing)

7. Performance Requirements
Brain Performance Targets

Intent: 0–10ms (0ms after cache warm-up)

State Machine: <1ms

Suggestion Engine: <5ms

Memory Access: <1ms

Brain Total: 1–8ms

Total Pipeline Latency Requirement

Whisper CPU (distil-tiny): 350–450ms

Brain: <10ms

TOTAL: <500ms end-to-end

8. Validation Tests (Must Complete)

Before signing off Day 5, run:

Test 1 – Cache Warmup

Repeat the same phrase 3 times.

Intent should hit cache on 2nd & 3rd runs.

Suggestion rotation should advance correctly.

State should NOT recompute.

Test 2 – No Repetition Guarantee

Trigger the same intent 5 times in a row.
Memory should ensure:

No suggestion is repeated within last 5.

Test 3 – Drift Stability

Run pipeline for 2 minutes:

Latency should remain steady (<500ms).

No memory overflow.

No cache runaway growth.

Test 4 – State Progression

Give conversation in phases:

Opening phrase

Discovery question

Pain trigger

Objection

Close attempt
Ensure state transitions are correct and cached.

9. Day 5 Definition of Done (Final Checklist)
Architecture

✔ Dialogue Brain module created
✔ State Machine working
✔ Suggestion Engine working
✔ Short-Term Memory working
✔ Caching Layer working

Integration

✔ Integrated into pipeline
✔ Unified structured output

Performance

✔ Brain <10ms
✔ Pipeline <500ms end-to-end
✔ No drift after 2 minutes
✔ No repeated suggestions
✔ No repeated state calculations

Reliability

✔ Cache hit/miss logs
✔ Memory rotation correct
✔ Rules loaded cleanly
✔ No crashes or overflow warnings