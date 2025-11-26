PDATED MVP PLAN (After Day 3 Success)
DAY 4 — End-to-End Pipeline Test

Goal:
“SPEAK → SILENCE → BUFFER → WHISPER → INTENT → OUTPUT
ALL under 1.5 seconds.”

This is the most meaningful milestone in the entire project.

Success Criteria:

The output text is correct

The intent is correct

Total latency < 1.5s

No CPU spikes > 60% during normal operation

No drift over 1–2 minutes

Hard Checkpoint:

If the end-to-end loop works reliably →
We merge Days 5 and 6.

If there is jitter →
We keep the plan as-is and tune chunk size + silence threshold.

DAY 5 — HUD Overlay + Governor

Only after the pipeline works.

HUD requirements:

Always on top

Transparent

Bottom-right

Auto-hide

Minimum distraction

Governor requirements:

Kill events > 1.5s

Do not display stale prompts

Output suppression visible via yellow dot

Hard Checkpoint:

UI must appear within 20–30ms

No flicker

Application stays responsive

DAY 6 — Audio Setup Wizard + Stability

Wizard includes:

Select microphone

“Green bar moves” test

Confirm routing

Save config

Stability includes:

10–15 min load test

No VAD drift

No missed silence events

No threading stalls

No memory creep

Hard Checkpoint:

If the system runs 10 minutes without:

Drift

Lag buildup

Missed triggers

→ MVP is ready.

DAY 7 — MVP Demo

You can show it to a real salesperson

They speak

The screen pops the right prompt

Timely, accurate, and simple