# Real-Time Cognitive Assistant

This repository contains a 7-day MVP for a real-time "Silence-Gap Cognitive Assistant".

## Status
Day 1-4 complete ✓
Day 5 (Dialogue Brain) delivered
Day 6 (HUD) delivered
Day 7 (Demo) delivered

## Architecture
Audio → VAD → Whisper → Intent → Dialogue Brain → HUD

## Setup

- Python 3.10+
- System dependencies: ffmpeg, PortAudio (pyaudio), sounddevice. Install with your OS package manager.
- Python packages: `pip install -r requirements.txt`

## Running the demo

```bash
python demo.py
```

The demo starts a lightweight mock pipeline, shows console HUD updates, and stops with `Ctrl+C`.

## Running tests

```bash
pytest
```

## Tech stack
- Silero VAD
- Whisper tiny-distil (fast CPU)
- MiniLM-L6-v2 embeddings
- Real-time ring buffer
- Python 3.10
