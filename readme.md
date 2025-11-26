# Real-Time Cognitive Assistant

This repository contains a 7-day MVP for a real-time "Silence-Gap Cognitive Assistant".

## Status
Day 1-4 complete ✓
Day 5 (Dialogue Brain) planned
Day 6 (HUD) planned
Day 7 (Demo) planned

## Architecture
Audio → VAD → Whisper → Intent → Dialogue Brain → HUD

## Tech stack
- Silero VAD
- Whisper tiny-distil (fast CPU)
- MiniLM-L6-v2 embeddings
- Real-time ring buffer
- Python 3.10
