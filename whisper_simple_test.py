"""
Simple Whisper latency test - outputs only the critical metric
"""

import sys
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

sys.stdout.reconfigure(encoding='utf-8')

SAMPLE_RATE = 16000
DURATION = 1.2
CHANNELS = 1

print("Loading Whisper tiny.en...")
stt = WhisperModel("tiny.en", device="cpu", compute_type="int8")
print("Model loaded.\n")

# Warmup microphone
print("Warming up microphone...")
warmup = sd.rec(int(0.1 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
sd.wait()
print("Ready.\n")

# Record
print(f"Recording {DURATION}s - SPEAK NOW:")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
sd.wait()
audio = audio.flatten()
print("Recording complete.\n")

# Transcribe
print("Transcribing...")
start = time.time()
segments, info = stt.transcribe(audio, beam_size=1, language="en", temperature=0.0)
text = " ".join([segment.text for segment in segments])
latency = time.time() - start

print(f"\nText: {text}")
print(f"\nWHISPER LATENCY: {latency:.3f}s")

if latency < 1.0:
    print("Result: EXCELLENT")
elif latency < 1.3:
    print("Result: GOOD - Proceed to Day 2")
elif latency < 1.5:
    print("Result: MARGINAL")
else:
    print("Result: FAILED - Too slow")

# Write to file
with open("whisper_result.txt", "w") as f:
    f.write(f"Latency: {latency:.3f}s\n")
    f.write(f"Text: {text}\n")
