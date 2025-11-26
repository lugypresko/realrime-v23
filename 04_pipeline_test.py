"""
DAY 4 — Integrated Pipeline Test (Sentinel -> Worker)
Goal: Speak -> Silence -> Whisper -> Intent -> Output
Constraint: ALL under 1.5 seconds (The "Latency Guard")

This script integrates:
1. Silero VAD (Sentinel)
2. Faster-Whisper (tiny-distil)
3. Sentence-Transformers (MiniLM)
4. Latency Governor (> 1.5s suppression)
"""

import sys
import time
import collections
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from prompts import PROMPTS

# Windows console fix
sys.stdout.reconfigure(encoding='utf-8')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
SAMPLE_RATE = 16000
BLOCK_SIZE = 512  # 32ms
BUFFER_DURATION = 1.2  # seconds
BUFFER_FRAMES = int((SAMPLE_RATE * BUFFER_DURATION) / BLOCK_SIZE) + 1

# VAD Thresholds
SPEECH_THRESHOLD = 0.5
SILENCE_THRESHOLD = 0.3
SILENCE_DURATION = 0.6  # seconds
SILENCE_FRAMES = int(SILENCE_DURATION / (BLOCK_SIZE / SAMPLE_RATE))

# Latency Budget (The Governor)
MAX_LATENCY = 1.5  # seconds

print("=" * 60)
print("DAY 4: INTEGRATED PIPELINE TEST")
print("=" * 60)
print(f"Sample Rate: {SAMPLE_RATE}")
print(f"Buffer: {BUFFER_DURATION}s")
print(f"Silence Trigger: {SILENCE_DURATION}s")
print(f"Latency Budget: {MAX_LATENCY}s")
print("-" * 60)

# ==================================================================================
# INITIALIZATION (Heavy Lifting)
# ==================================================================================

# 1. Load Whisper
print("1. Loading Faster-Whisper tiny-distil (int8, CPU)...")
load_start = time.time()
# User requested "tiny-distil". Note: If this fails, it might be "distil-small.en" or similar.
# But following exact instructions.
try:
    stt = WhisperModel("tiny-distil", device="cpu", compute_type="int8")
except Exception as e:
    print(f"[WARNING] 'tiny-distil' failed ({e}). Falling back to 'tiny.en'...")
    stt = WhisperModel("tiny.en", device="cpu", compute_type="int8")

print(f"[OK] Whisper loaded in {time.time() - load_start:.2f}s")

# 2. Load MiniLM & Embed Prompts
print("2. Loading MiniLM & Pre-computing embeddings...")
load_start = time.time()
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
prompt_embeddings = encoder.encode(PROMPTS, convert_to_numpy=True, show_progress_bar=False)
print(f"[OK] MiniLM loaded & {len(PROMPTS)} prompts embedded in {time.time() - load_start:.2f}s")

# 3. Load Silero VAD
print("3. Loading Silero VAD...")
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=False)
print("[OK] VAD Loaded")
print()

# ==================================================================================
# STATE
# ==================================================================================
ring_buffer = collections.deque(maxlen=BUFFER_FRAMES)
silence_counter = 0
speaking = False
last_process_time = 0

# ==================================================================================
# WORKER FUNCTION
# ==================================================================================
def process_audio(audio_buffer, trigger_time):
    """
    The "Worker" logic: Transcribe -> Embed -> Match -> Govern
    """
    global last_process_time
    
    # 1. Transcribe (Whisper)
    t0 = time.time()
    
    # Flatten audio buffer
    audio_data = np.concatenate(audio_buffer)
    
    # Faster-Whisper transcribe
    segments, _ = stt.transcribe(
        audio_data,
        beam_size=1,
        language="en",
        temperature=0.0
    )
    text = " ".join([s.text for s in segments]).strip()
    
    t1 = time.time()
    whisper_latency = t1 - t0
    
    if not text:
        print("    [Ignored] No text transcribed")
        return

    # 2. Intent (MiniLM)
    t2 = time.time()
    utterance_embedding = encoder.encode(text, convert_to_numpy=True, show_progress_bar=False)
    scores = prompt_embeddings @ utterance_embedding
    best_idx = np.argmax(scores)
    best_prompt = PROMPTS[best_idx]
    best_score = scores[best_idx]
    
    t3 = time.time()
    intent_latency = t3 - t2
    
    # 3. Governor Check
    total_latency = t3 - trigger_time
    
    print("-" * 40)
    print(f"Input:  '{text}'")
    print(f"Intent: '{best_prompt}' ({best_score:.2f})")
    print(f"Timing: Whisper={whisper_latency*1000:.0f}ms | Intent={intent_latency*1000:.0f}ms")
    
    if total_latency > MAX_LATENCY:
        print(f"TOTAL:  {total_latency:.3f}s ❌ [SUPPRESSED] > {MAX_LATENCY}s")
    else:
        print(f"TOTAL:  {total_latency:.3f}s ✅ [SUCCESS]")
    print("-" * 40)
    print()

# ==================================================================================
# AUDIO CALLBACK (The Sentinel)
# ==================================================================================
def audio_callback(indata, frames, time_info, status):
    global silence_counter, speaking
    
    if status:
        print(status, file=sys.stderr)
    
    # 1. Add to buffer
    audio_chunk = indata[:, 0]
    ring_buffer.append(audio_chunk.copy())
    
    # 2. VAD Inference
    audio_tensor = torch.from_numpy(audio_chunk)
    speech_prob = model(audio_tensor, SAMPLE_RATE).item()
    
    # 3. Logic
    if speech_prob > SPEECH_THRESHOLD:
        if not speaking:
            sys.stdout.write(" [SPEECH] ")
            sys.stdout.flush()
        speaking = True
        silence_counter = 0
    else:
        if speaking:
            silence_counter += 1
            
            # Check for silence trigger
            if silence_counter >= SILENCE_FRAMES:
                trigger_time = time.time()
                sys.stdout.write(f"\n>>> SILENCE ({SILENCE_DURATION}s) -> PROCESSING...\n")
                
                # Snapshot buffer immediately
                buffer_snapshot = list(ring_buffer)
                process_audio(buffer_snapshot, trigger_time)
                
                # Reset
                speaking = False
                silence_counter = 0

                
                # Clear buffer to avoid re-triggering on same audio? 
                # Actually, ring buffer naturally flows. 
                # But we should probably avoid immediate re-trigger.
                # Let's just reset state.

# ==================================================================================
# MAIN LOOP
# ==================================================================================
print("Starting audio stream...")
print("INSTRUCTIONS:")
print("1. Speak naturally into the mic.")
print("2. Pause for ~0.6s to trigger.")
print("3. Watch the latency output.")
print("4. RUN FOR AT LEAST 2 MINUTES to check for drift.")
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
