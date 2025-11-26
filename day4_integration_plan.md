# Day 4 Integrated Pipeline Test Plan

## 1. Integration Objective
- Build a single continuous pipeline that wires the validated components end-to-end: Sentinel (silence detection) → buffer extraction → Whisper (tiny.en) speech-to-text → intent match (MiniLM embeddings) → console output.
- Demonstrate that the complete path from SILENCE_TRIGGER emission to intent result delivery stays within the 1.5 second budget defined in the PRD.
- Capture latency breakdowns for each stage to verify budget compliance and identify headroom.

## 2. Integration Architecture
### Process 1 — Sentinel
- Runs continuously, monitoring real-time audio using Silero VAD at 16 kHz with the existing 1.2s ring buffer and ~0.6–0.8s silence trigger.
- On detecting a qualifying silence gap, emits an event on **Queue S→W** with payload:
  - `type: "SILENCE_TRIGGER"`
  - `audio`: numpy array of the captured buffer (1.2s window)
  - `timestamp`: monotonic time at moment of emission
  - `id`: unique identifier for correlation

### Process 2 — Worker
- Blocks on **Queue S→W** to receive events.
- For each event:
  - Record `worker_start_ts` (monotonic) to measure transport overhead.
  - Start latency timer for the event ID.
  - Run Whisper tiny.en on the `audio` payload; capture `whisper_latency` and returned text.
  - Run intent classification using pre-embedded prompts; capture `intent_latency`, selected prompt ID, and score.
  - Compute `event_age = current_monotonic - event.timestamp`.
  - Governor rule: if `event_age > 1.5s`, mark result as `SUPPRESSED_LATE` and skip presenting; otherwise mark as `SUCCESS`.
  - Emit result on **Queue W→P** with fields: `id`, `event_timestamp`, `worker_start_ts`, `whisper_latency`, `intent_latency`, `event_age`, `decision` (SUCCESS|SUPPRESSED_LATE), `text`, `prompt_id`, `score`.

### Process 3 — Presenter
- Blocks on **Queue W→P**.
- For Day 4, uses console-only output to print a structured line per event:
  - Event ID, decision, Whisper text, prompt ID, score.
  - Latency breakdown: transport (worker_start_ts − event_timestamp), Whisper latency, intent latency, total age.
- No UI beyond console logging; timestamps printed in milliseconds for readability.

### Queues and Contracts
- **Queue S→W**: Sentinel → Worker; carries SILENCE_TRIGGER events as specified above. Must guarantee FIFO and preserve the audio payload.
- **Queue W→P**: Worker → Presenter; carries result objects with latency fields and governor decision. Presenter must not mutate payload.

## 3. Measurement Plan
Capture the following per event and periodically:
- Silence trigger timestamp (Sentinel emission).
- Worker start timestamp (upon dequeue) to measure transport overhead.
- Whisper latency measurement.
- Intent latency measurement.
- Total event age at decision time.
- Governor decision (SUCCESS or SUPPRESSED_LATE).
- CPU usage snapshots for each process (e.g., ps/top sampling every 10s during the 2-minute run).
- Memory usage to detect growth/drift (track RSS every 10s; expect stability within ±5 MB).

## 4. Acceptance Criteria
### A. Performance
- End-to-end pipeline latency (event age at decision) ≤ 1.5s.
- Whisper latency ≤ 1.0s average; no single run exceeding budget.
- Intent latency ≤ 85 ms.
- Event transport overhead (worker_start_ts − event_timestamp) ≤ 50 ms.

### B. Stability
- No missed triggers during a continuous 2-minute run.
- No duplicate triggers for the same silence gap.
- No accumulation delay (latency drift) over time.
- Memory usage stable within ±5 MB.

### C. Correctness
- Whisper returns non-empty text for each SUCCESS event.
- Intent classification returns a top prompt using cosine/dot-product selection with a valid score.
- Governor suppresses events where `event_age > 1.5s` and marks them as `SUPPRESSED_LATE`.
- Result objects conform to the defined schema and are printed by Presenter.

## 5. Test Procedure (Human Steps)
1. Start all three processes with logging enabled: Sentinel, Worker, Presenter.
2. Speak naturally for 3–4 seconds into the microphone.
3. Pause for ~0.7–1.0 seconds to trigger silence detection.
4. Observe the pipeline output:
   - SILENCE_TRIGGER appears from Sentinel.
   - Worker logs Whisper execution, intent selection, and governor decision.
   - Presenter prints the result with latency breakdown.
5. Repeat steps 2–4 for 2 minutes with varied speech and pause timing.
6. During the run, record:
   - Fastest, slowest, and average end-to-end latency.
   - CPU samples and memory snapshots every 10 seconds.
   - Any jitter, drift, missed or duplicate triggers.
7. At the end, declare PASS/FAIL based on acceptance criteria above; note any anomalies for follow-up.

## 6. Expected Output
- A repeatable Day 4 integration plan ready for immediate implementation by a senior engineer.
- Console logs showing per-event payloads and latency breakdowns.
- A short run report summarizing performance stats (fastest/slowest/average), stability observations, and governor actions.
- Telemetry/observability expectations including watchdogs, drift warnings, warm-start confirmation, replay buffers, and structured JSONL output for the Sentinel → Worker → Presenter chain.
