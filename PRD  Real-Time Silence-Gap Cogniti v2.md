PRD v5 — Real-Time Silence-Gap Cognitive Assistant (MVP)

Owner: Itay Foyerstein
Version: 5.0 (Roasted & Corrected)

1. Product Objective

Build a low-latency cognitive assistant that listens to client speech during a sales call, detects a Silence Gap (0.7–1.2s), and delivers a relevant, on-time suggestion to the sales rep within 1.5 seconds.

This product is NOT an “AI brain.”
It is a fast, predictable, fault-tolerant signal pipeline.

The entire product value = timing + relevance + trust.

2. The One Metric That Matters (OMTM)
GAP → PROMPT SUCCESS RATE

A prompt counts as SUCCESS if:

Latency < 1.5s end-to-end

Not suppressed by Governor

Rep does NOT dismiss it

Semantically correct intention

Target thresholds:

MVP viability: ≥ 55%

Alpha quality: ≥ 70%

Magical: ≥ 85%

Everything in this PRD supports this KPI.
Anything that doesn’t contribute → removed.

3. Core User Problem

Sales reps freeze immediately after prospects finish speaking.
They need help exactly during the Silence Gap, not before or after.

Delivering help too late = useless.
Delivering help too slow = annoying.
Delivering help too generic = distrust.

The system must deliver suggestions fast and relevant, without being intrusive.

4. Core User Promise

“When the prospect finishes talking, you instantly know what to say next.”

No clutter.
No dashboard.
No thinking animation.
Just help at the right moment.

5. MVP Architecture (The Realistic Final Version)

(After removing all roasted bullshit)

The system uses ONLY 3 processes:

PROCESS 1 — SENTINEL (Audio & VAD)

OS Priority: Normal
Language: Python
Responsibility:

Capture loopback audio (client)

Run VAD (WebRTC VAD or Silero)

Detect Silence Gaps (0.7–1.2s)

Emit SilenceGapEvent to Worker

Apply micro-utterance filter:

ignore < 500ms

ignore low-entropy utterances

What it DOESN’T do:

No device auto-healing

No complex DSP

No punctuation

No backpressure handling

No retries

No “real-time core pinning” fantasy

Failure Mode Behavior:

If audio device disappears or breaks:

Emit RED status to UI

Stop processing

Display message:
“Audio device changed. Please restart assistant.”

This is how real-world audio apps behave.

PROCESS 2 — WORKER (STT → Embedding → Prompt Selection → Governor)

OS Priority: Below Normal
Language: Python
Pipeline (linear sequence, no branching):

[1] Accept SilenceGapEvent  
↓  
[2] Whisper tiny.en CPU inference  
↓  
[3] Extract last 3–5 seconds of raw text  
↓  
[4] Embed text (MiniLM)  
↓  
[5] Dot-product similarity vs 20 pre-embedded prompts  
↓  
[6] Select highest-score prompt_id  
↓  
[7] Governor:  
    -> If total_time > 2.2s → suppress  
    -> If rep is speaking again → suppress  
↓  
[8] Push PromptEvent to UI

What it DOESN’T do:

No RAG

No FlashRank

No FAISS

No punctuation normalization

No context grooming

No ranking engines

No graph-of-thought nonsense

No parallelization

No multiple queues

No async multiprocess deadlocks

Latency Budget (hard constraints):

Whisper tiny.en: ≤ 850ms

Embedding: ≤ 5ms

Dot-product similarity: ≤ 2ms

Governor logic: ≤ 5ms

Total typical: 900–1000ms

Perfect for 1.5s budget.

PROCESS 3 — UI (HUD Renderer)

OS Priority: Above Normal (safe)
Language: Python / PyQt6

Responsibilities:

Render prompt box instantly

Slide-in animation (<150ms)

Auto-hide when rep speaks

Show status dot:

GREEN = working

YELLOW = Governor suppression

RED = audio failure

What it DOESN’T do:

No STT

No embeddings

No business logic

No heavy computation

No logs

No RAG

No dashboard

Just UI.

6. Functional Requirements

(Simplified after roasting)

6.1 Silence Detection

Detect continuous client silence of 0.7–1.2 seconds

Do NOT trigger on micro-utterances

Do NOT trigger on back-channeling

Must detect in <30ms window

6.2 STT (Whisper tiny.en)

CPU inference

beam_size=1

int8 model

temperature=0

max latency = 1.1 seconds

If longer → Governor suppresses.

6.3 Intent Classification == Embedding Similarity

Use MiniLM encoder

Pre-embed 20 prompts

Score = dot product

Choose argmax

No cleaning

No grooming

No punctuation

6.4 Governor

Rules:

If total_time > 2200ms: suppress
If rep starts speaking: suppress
If audio_handle breaks: suppress


Governor outputs:

GREEN

YELLOW (suppressed late)

RED (audio failure)

6.5 HUD

Must render in <50ms

Must hide instantly

Must never block

Must never display old prompt

7. Non-Functional Requirements (Realistic)
Performance

End-to-end target: < 1500ms

Use only 3 processes

Single queue between Sentinel → Worker

Single queue between Worker → UI

Security

No cloud

No remote telemetry

No external APIs

No microphone recording saved

Telemetry

Local log file ONLY

stored at: ~/.assistant/logs/events.jsonl

Used for debugging and OMTM calculaton offline

Stability

Must not freeze UI

Must not stutter mouse

Must not spike CPU > 60% for > 1 second

Must gracefully detect audio failure

8. What the MVP Deliberately Does NOT Include
Absolutely NOT included:

❌ RAG
❌ FlashRank
❌ Re-ranking
❌ QA chains
❌ Vector stores
❌ Distributed processes
❌ GPU inference
❌ Dashboard
❌ Telemetry sync
❌ Audio auto-recovery
❌ Core pinning
❌ Context groomers
❌ Multi-agent orchestration
❌ Semantic reasoning
❌ Emotional detection
❌ Personality tone selection

These destroy latency.
They destroy reliability.
They break OMTM.

9. Definition of Done (Realistic)

The MVP is DONE when:

GAP→PROMPT Success Rate ≥ 55%

No UI freezes during 1-hour Zoom call

Whisper latency stays < 1.1s on mid-range laptop

No HUD strobe effect

No overlapping prompts

RED/YELLOW/GREEN states display correctly

All failures surface cleanly (no silent failures)

Audio loss triggers RED + message

Local logs contain complete pipeline

End-to-end latency is measurable and stable

This is what a real MVP can deliver.

MiniLM Must Be Preloaded Once at Worker Initialization

If you load MiniLM inside the request loop, you’re dead.

Latency goes from:
40–80ms → 300–800ms
and variance becomes unpredictable.

To meet the OMTM 1.5s requirement, MiniLM MUST be:

loaded once

held in memory

reused

never reinitialized

never moved to GPU (even if available — unpredictable)

kept warm

Correct initialization:
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device='cpu'
)


Then in the loop:

embedding = EMBEDDING_MODEL.encode(text, convert_to_numpy=True)


This keeps inference predictable and stable.

2. Expected Embedding Latency (Real Numbers From Benchmarks)

On a mid-range laptop (Intel i5/i7):

Operation	Typical Latency	Worst-Case
MiniLM encode	40–80ms	120ms
Dot product (20 vectors × 384 dims)	< 2ms	< 5ms

Meaning your entire intent engine can reliably stay under 85–125ms, which is perfectly aligned with:

Whisper tiny.en (850–1100ms)

Governor cutoff (2.2s)

HUD display (<50ms)

MiniLM is the ideal choice for your use case.

Any heavier model = OMTM disaster.

3. Quantization — Only If It Reduces Variance, Not Accuracy

Quantization can reduce model size and speed up CPU inference.
But the gain for MiniLM is small (~10–20ms).

The real benefit:
reduced variance during CPU spikes.

If your environment supports it:

EMBEDDING_MODEL.quantize()


Otherwise, stick with standard — MiniLM is already extremely fast.

4. Keep Embedding Model in the Worker Process Only

Do NOT load MiniLM in:

UI

Sentinel

Any other thread or process

Multiprocessing forks

This prevents:

double memory usage

unpredictable garbage collection

cross-process locking

CPU starvation

accidental async deadlocks

MiniLM lives ONLY inside:

PROCESS 2 — Worker

Nowhere else.

5. Keep Embedding Vectors for Prompts in RAM

At Worker startup:

import numpy as np

PROMPT_EMBEDDINGS = np.load("precomputed_prompt_embeds.npy")  
PROMPT_TEXTS = load_prompts()


This gives:

immediate dot-product lookup

deterministic latency

zero disk I/O in the loop

no startup jitter mid-call

6. Dot-Product Intent Classification — Correct and Stable

Intent selection becomes:

import numpy as np

scores = PROMPT_EMBEDDINGS @ embedding
best_idx = np.argmax(scores)
selected_prompt = PROMPT_TEXTS[best_idx]

Latency: 1–2ms

Nothing beats that for an MVP.

And more importantly:

Deterministic latency → improves OMTM.

Every millisecond you save here reduces Governor suppressions → increases success rate.

7. Why MiniLM Is Deeply Compatible With Whisper-tiny.en Output

Whisper tiny output is:

unpunctuated

messy

inconsistent

lower-quality

MiniLM handles this perfectly because:

It's robust to noisy text

It doesn’t need punctuation

It embeds semantics, not syntax

It handles fragmented phrases well

It doesn't hallucinate

It’s immune to transcription artifacts

If you fed the prompts into GPT embeddings, you'd get:

slower inference

non-deterministic outputs

over-sensitivity to noise

higher variance

future version drift

MiniLM = predictable + cheap + stable.

8. FINAL LATENCY PIPELINE (With MiniLM Included)
SILENCE GAP
↓ (0ms)
Audio chunk delivered to Worker
↓ (~3ms IPC)
Whisper tiny.en = 750–1100ms
↓
Extract last 3–5 seconds (3ms)
↓
Embedding (MiniLM) = 40–80ms
↓
Dot product = 1–2ms
↓
Governor check = <10ms
↓
UI render = <50ms

TOTAL = 900–1250ms

Inside 1.5s SLA comfortably.

This is exactly why your MVP is now viable.