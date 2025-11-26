"""
DAY 3 — The Sentinel (VAD + Buffer)
Goal: Detect speech → Detect silence → Grab last 1.2s audio → Print “TRIGGER”

This is the "Physics Test" Phase 2 & 3 implementation.
"""

import sys
import time
import collections
import numpy as np
import sounddevice as sd
import torch

# Windows console fix
sys.stdout.reconfigure(encoding='utf-8')

# Configuration
SAMPLE_RATE = 16000
BLOCK_SIZE = 512  # 32ms frames (Silero likes 32ms or 64ms better than 20ms sometimes, but let's stick to spec if possible, or adjust)
# The Physics Test said 320 samples (20ms). Let's use 512 (32ms) as it's standard for Silero, 
# but the prompt asked for 320. Let's stick to 512 for stability with Silero unless forced.
# Actually, Silero VAD supports 512, 1024, 1536 samples for 16kHz. 
# 320 samples (20ms) is NOT natively supported by the new Silero VAD (v4+).
# It typically expects 512 (32ms) or 1024 (64ms).
# Let's use 512 (32ms).
BLOCK_SIZE = 512 

BUFFER_DURATION = 1.2  # seconds
BUFFER_FRAMES = int((SAMPLE_RATE * BUFFER_DURATION) / BLOCK_SIZE) + 1

# VAD Thresholds
SPEECH_THRESHOLD = 0.5
SILENCE_THRESHOLD = 0.3
SILENCE_DURATION = 0.6  # seconds
SILENCE_FRAMES = int(SILENCE_DURATION / (BLOCK_SIZE / SAMPLE_RATE))

print("=" * 60)
print("DAY 3: SENTINEL TEST (VAD + BUFFER)")
print("=" * 60)
print(f"Sample Rate: {SAMPLE_RATE}")
print(f"Block Size: {BLOCK_SIZE} ({BLOCK_SIZE/SAMPLE_RATE*1000:.1f}ms)")
print(f"Buffer: {BUFFER_DURATION}s ({BUFFER_FRAMES} frames)")
print(f"Silence Trigger: {SILENCE_DURATION}s ({SILENCE_FRAMES} frames)")
print("-" * 60)

# Load Silero VAD
print("Loading Silero VAD...")
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
print("[OK] VAD Loaded")
print()

# State
ring_buffer = collections.deque(maxlen=BUFFER_FRAMES)
silence_counter = 0
speaking = False
triggered = False

def audio_callback(indata, frames, time_info, status):
    global silence_counter, speaking, triggered
    
    if status:
        print(status, file=sys.stderr)
    
    # 1. Add to buffer
    # indata is (frames, channels), we want flat array
    audio_chunk = indata[:, 0]
    ring_buffer.append(audio_chunk.copy())
    
    # 2. VAD Inference
    # Convert to tensor
    audio_tensor = torch.from_numpy(audio_chunk)
    
    # Get speech probability
    # Silero expects (batch, samples) or just (samples)
    speech_prob = model(audio_tensor, SAMPLE_RATE).item()
    
    # 3. Logic
    if speech_prob > SPEECH_THRESHOLD:
        if not speaking:
            sys.stdout.write(" [SPEECH START] ")
            sys.stdout.flush()
        speaking = True
        silence_counter = 0
    else:
        if speaking:
            silence_counter += 1
            
            # Check for silence trigger
            if silence_counter >= SILENCE_FRAMES:
                sys.stdout.write(f"\n>>> SILENCE DETECTED ({SILENCE_DURATION}s) <<<\n")
                
                # Extract buffer
                full_audio = np.concatenate(ring_buffer)
                
                # Verify length
                duration = len(full_audio) / SAMPLE_RATE
                sys.stdout.write(f"    Extracted {duration:.2f}s buffer\n")
                sys.stdout.write("    (Ready for Whisper...)\n\n")
                sys.stdout.flush()
                
                # Reset
                speaking = False
                silence_counter = 0
                triggered = True

# Main loop
print("Starting audio stream...")
print("SPEAK NATURALLY. PAUSE TO TRIGGER.")
print("Press Ctrl+C to stop.")
print("-" * 60)

# Warmup
warmup = sd.rec(int(0.1 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()

try:
    with sd.InputStream(samplerate=SAMPLE_RATE, 
                        blocksize=BLOCK_SIZE,
                        channels=1, 
                        dtype='float32',
                        callback=audio_callback):
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\nStopped.")
except Exception as e:
    print(f"\nError: {e}")
