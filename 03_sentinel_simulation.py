"""
DAY 3 — Sentinel Logic Simulation
Goal: Verify VAD + Buffer logic using synthetic audio (no microphone needed)
"""

import sys
import time
import collections
import numpy as np
import torch

# Configuration
SAMPLE_RATE = 16000
BLOCK_SIZE = 512
BUFFER_DURATION = 1.2
BUFFER_FRAMES = int((SAMPLE_RATE * BUFFER_DURATION) / BLOCK_SIZE) + 1

# VAD Thresholds
SPEECH_THRESHOLD = 0.5
SILENCE_THRESHOLD = 0.3
SILENCE_DURATION = 0.6
SILENCE_FRAMES = int(SILENCE_DURATION / (BLOCK_SIZE / SAMPLE_RATE))

print("=" * 60)
print("DAY 3: SENTINEL LOGIC SIMULATION")
print("=" * 60)

# Load Silero VAD
print("Loading Silero VAD...")
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=False)
print("[OK] VAD Loaded")
print()

# State
ring_buffer = collections.deque(maxlen=BUFFER_FRAMES)
silence_counter = 0
speaking = False
triggered = False

def process_chunk(audio_chunk):
    global silence_counter, speaking, triggered
    
    # 1. Add to buffer
    ring_buffer.append(audio_chunk.copy())
    
    # 2. VAD Inference
    audio_tensor = torch.from_numpy(audio_chunk)
    speech_prob = model(audio_tensor, SAMPLE_RATE).item()
    
    # 3. Logic
    if speech_prob > SPEECH_THRESHOLD:
        if not speaking:
            print(f" [SPEECH START] (prob: {speech_prob:.2f})")
        speaking = True
        silence_counter = 0
    else:
        if speaking:
            silence_counter += 1
            if silence_counter >= SILENCE_FRAMES:
                print(f"\n>>> SILENCE DETECTED ({SILENCE_DURATION}s) <<<")
                full_audio = np.concatenate(ring_buffer)
                duration = len(full_audio) / SAMPLE_RATE
                print(f"    Extracted {duration:.2f}s buffer")
                speaking = False
                silence_counter = 0
                return True
    return False

# Generate synthetic audio
print("Generating synthetic audio...")
# 1. 0.5s Silence
silence1 = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
# 2. 1.0s Speech (White noise * 0.5 amplitude) - VAD usually detects high energy noise as speech or we can use a sine wave
# Silero is smart, it might reject pure noise. Let's use a mix of frequencies to simulate speech-like spectral content
t = np.linspace(0, 1.0, int(1.0 * SAMPLE_RATE))
speech = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 800 * t) + 0.1 * np.random.normal(0, 1, len(t))
speech = speech.astype(np.float32)
# 3. 1.0s Silence
silence2 = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)

full_signal = np.concatenate([silence1, speech, silence2])
print(f"Signal created: {len(full_signal)/SAMPLE_RATE}s")
print("Feeding signal to Sentinel logic...")
print("-" * 60)

# Process in chunks
chunk_size = BLOCK_SIZE
num_chunks = len(full_signal) // chunk_size

trigger_count = 0
for i in range(num_chunks):
    chunk = full_signal[i*chunk_size : (i+1)*chunk_size]
    if process_chunk(chunk):
        trigger_count += 1

print("-" * 60)
if trigger_count > 0:
    print("✅ SUCCESS: Silence detected and buffer extracted")
else:
    print("❌ FAILED: No trigger event")
    # Debug info
    print(f"Final state: Speaking={speaking}, SilenceCounter={silence_counter}/{SILENCE_FRAMES}")
