"""
DAY 1 — Whisper Loop Test
Goal: Speak → Whisper tiny.en (CPU) → Text in terminal → Under 1.4s

This is the CRITICAL feasibility test.
If this fails, the entire MVP is impossible on CPU.
"""

import sys
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# CRITICAL: Configure UTF-8 encoding for Windows console
# Without this, Whisper output with non-ASCII chars will crash the terminal
sys.stdout.reconfigure(encoding='utf-8')

# Configuration
SAMPLE_RATE = 16000
DURATION = 1.2  # seconds
CHANNELS = 1

print("=" * 60)
print("DAY 1: WHISPER FEASIBILITY TEST")
print("=" * 60)
print(f"Sample Rate: {SAMPLE_RATE} Hz")
print(f"Duration: {DURATION}s")
print(f"Channels: {CHANNELS} (mono)")
print()

# Load Whisper model (MUST be done before recording to exclude from latency)
print("Loading Whisper tiny.en (int8, CPU)...")
load_start = time.time()
stt = WhisperModel("tiny.en", device="cpu", compute_type="int8")
load_time = time.time() - load_start
print(f"[OK] Model loaded in {load_time:.2f}s")
print()

# CRITICAL: Microphone warmup for Windows WASAPI
# Windows returns blank first buffer unless device is pre-warmed
# This prevents transcribing silence
print("Warming up microphone (Windows WASAPI fix)...")
warmup = sd.rec(int(0.1 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
sd.wait()
print("[OK] Microphone ready")
print()

# Record audio
print(f"Recording {DURATION}s of audio...")
print("SPEAK NOW:")
print("-" * 60)

record_start = time.time()
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype='float32'
)
sd.wait()  # Wait until recording is finished
record_time = time.time() - record_start

print(f"[OK] Recording complete ({record_time:.2f}s)")
print()

# Convert to format Whisper expects
audio = audio.flatten()

# Transcribe
print("Transcribing...")
print("-" * 60)

transcribe_start = time.time()
segments, info = stt.transcribe(
    audio,
    beam_size=1,
    language="en",  # Explicit language prevents detection overhead
    temperature=0.0  # Deterministic output
)

# Extract text
text = " ".join([segment.text for segment in segments])
transcribe_time = time.time() - transcribe_start

print(f"[OK] Transcription complete")
print()

# Results
print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Transcribed Text: {text}")
print()
print(f"Whisper Latency: {transcribe_time:.3f}s")
print()

# Evaluation
print("=" * 60)
print("EVALUATION")
print("=" * 60)

if transcribe_time < 1.0:
    print("[EXCELLENT] Well under budget")
elif transcribe_time < 1.3:
    print("[GOOD] Within target")
elif transcribe_time < 1.5:
    print("[MARGINAL] At the edge of feasibility")
else:
    print("[FAILED] Too slow for MVP")
    print("   Consider: GPU, smaller model, or architecture change")

print()
print(f"Target: < 1.3s (ideal), < 1.5s (acceptable)")
print(f"Actual: {transcribe_time:.3f}s")
print()

# Next steps
if transcribe_time < 1.5:
    print("[OK] PROCEED TO DAY 2: Embedding + Intent Loop")
else:
    print("[STOP] Fix Whisper latency before continuing")

print("=" * 60)
