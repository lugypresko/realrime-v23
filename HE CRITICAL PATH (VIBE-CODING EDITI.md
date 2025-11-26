HE CRITICAL PATH (VIBE-CODING EDITION)

Your goal:
Get the whole pipeline working end-to-end ONCE — ugly, messy, no refactor — and measure latency.

You are NOT building the app.
You are proving the app can be built.

This is the only thing that matters.

We will do it in 5 micro-milestones.

DAY 1 — Whisper Loop

Goal:
You speak → Whisper tiny.en (CPU) → Text appears in terminal → Under 1.4s.

TODO:

Create a new file called:
01_whisper_test.py

Use faster-whisper:

stt = WhisperModel("tiny.en", device="cpu", compute_type="int8")


Record exactly 1.2 seconds of audio using sounddevice:

Fixed 16000 Hz

Mono

Pass that raw PCM audio directly into Whisper.

Print the text and the total latency.

What NOT to do:

No GUI

No VAD

No multiprocessing

No embeddings

No queues

Success Criteria:

1.2s–1.4s consistent transcribe time on your slowest laptop

Text isn’t garbage

Script finishes without errors

DAY 2 — Embedding + Intent Loop

Goal:
Take Whisper text → MiniLM → dot product → Correct prompt returned.

TODO:

Create file:
02_intent_test.py

Load sentence-transformers MiniLM:

encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')


Embed once:

Your prompt list (20–25 prompts)

Save vectors to .npy

Load your Whisper text from yesterday into this file.

Embed it.

Compute dot products vs prompt_matrix.

Print top 1 match.

What NOT to do:

No UI

No VAD

No async

No queues

No batching

No normalization

Success Criteria:

Embedding time < 80ms

Dot product < 5ms

Correct prompt shown

DAY 3 — The Sentinel (VAD + Buffer Loop)

Goal:
Detect speech → Detect silence → Grab last 1.2s audio → Print “TRIGGER”.

TODO:

Create file:
03_sentinel_test.py

Use sounddevice stream:

16000 Hz

Blocksize = 320 samples (20ms)

Use Silero VAD:

model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')


Build a ring buffer holding last 1200ms of audio (60 frames).

Logic:

VAD > 0.5 → speaking

VAD < 0.3 for 0.6s → silence

When silence hits:

Dump the ring buffer to a WAV/PCM file

Print “SILENCE DETECTED – AUDIO SAVED”

What NOT to do:

No queues

No Whisper

No embeddings

Success Criteria:

Silence reliably detected

No jitter

No dropped frames

Ring buffer extraction works

DAY 4 — Integrate Sentinel → Worker (Single Process ONLY)

Goal:
Speak → Silence → Whisper → MiniLM → Prompt → Print result.

THIS IS THE MVP MOMENT.

TODO:

Create file:
04_pipeline_test.py

Copy your Day 3 sentinel code.

Inside the silence event:

Call Whisper immediately

Call embedding

Call dot product

Print resulting prompt

Add timer:

start = time.time()
...
print("Latency:", time.time() - start)


What NOT to do:

No multiprocessing

No UI

No governor

No background threads

No async

Success Criteria:

Speak

Pause

<1.5s later: prompt prints in terminal

Run 20 times → 90%+ success

This is your OMTM moment.

DAY 5 — Add the UI (Still Single Process)

Goal:
Replace “print prompt” with UI overlay text.

TODO:

Create file:
05_ui_overlay_test.py

Use PyQt6 with transparent window.

On silence event:

Run Whisper → MiniLM → prompt

Set label text

Show window

Hide after 3s

Test under heavy CPU load (Zoom ON).

What NOT to do:

No multiprocessing

No queues

No threading (except Qt)

No “governor” logic yet

Success Criteria:

UI shows the prompt

No stutter

No lag

Window stays on top

Does not steal focus

🚨 AFTER ALL 5 ARE WORKING — DEADLINE

Only THEN do you go:

→ Multiprocessing
→ Queues
→ Governor
→ UX polish
→ Installer
→ Auto-start
→ Wizard
→ Deployment

Right now, your ONLY job is to:

Build a single-process prototype that proves end-to-end latency:
SILENCE → TRANSCRIBE → EMBED → PROMPT under 1.5 seconds.

If that works, the MVP works.
If that fails, you fix Whisper OR your buffer — NOT the architecture.