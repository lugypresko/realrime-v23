# v2 Live Validation Harness

This lightweight harness validates real-audio readiness **without** starting the Sentinel/Worker/Presenter processes. It performs sequential checks to confirm hardware, VAD, STT, intent, and end-to-end decision paths are viable before launching the live pipeline.

## What each validation does
- **Audio device**: Enumerates the default input device, records ~3 seconds at 16 kHz, and verifies non-zero samples are captured.
- **VAD**: Records ~3 seconds and applies the v2 silence policy to ensure both speech frames and a silence trigger are detected.
- **STT**: Captures ~3 seconds and runs Whisper tiny.en via the v2 STT interface, reporting transcription text and latency.
- **Intent**: Uses the real intent classifier (MiniLM embeddings + prompts) to score the STT result and report top prompt/score.
- **End-to-end**: Runs Audio → VAD → STT → Intent → Governor in-process (no multiprocessing) and enforces the 1.5s latency budget with schema validation.

## How to run
```
python -m v2.live_validation.cli --all
```
Outputs resemble:
```
[OK] Audio device: Captured 48000 samples; peak 0.1234
[OK] VAD: Speech frames: 10; silence events: 1
[OK] STT: Text: 'hello world', latency=820.5ms
[OK] Intent: Prompt=3, score=0.4123, intent_latency=12.4ms
[OK] End-to-end: decision=SUCCESS, total_latency=940.1ms
[OK] All live validations passed. Ready for live pipeline testing.
```

If any step fails, you will see `[FAIL] <reason>` and a non-zero exit code.

## Requirements
- Working microphone/input device
- `sounddevice`, `numpy`, `faster_whisper`, and `sentence_transformers` installed
- `prompts.json` and `embeddings.npy` present at the repository root
- Whisper tiny.en model will download on first use

Run these validations in a quiet environment and speak briefly when prompted to ensure VAD and STT produce meaningful results.
