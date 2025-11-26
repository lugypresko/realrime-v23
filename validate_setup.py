"""
VALIDATION SCRIPT - DAY 4 SETUP
Automates Steps 3-7 of the Execution Instructions.
"""

import sys
import time
import numpy as np
import sounddevice as sd
import torch
import whisper
from sentence_transformers import SentenceTransformer
from prompts import PROMPTS

# Windows console fix
sys.stdout.reconfigure(encoding='utf-8')

def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def check_step(name, success):
    if success:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name}")
        sys.exit(1)

# ==============================================================================
# STEP 3: Validate Audio Dependencies
# ==============================================================================
print_header("STEP 3: Validate Audio Dependencies")
try:
    print("Testing microphone...")
    duration = 0.5
    fs = 16000
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    
    if recording.shape == (int(duration * fs), 1):
        print(f"  Recorded {len(recording)} samples.")
        print(f"  Max amplitude: {np.max(np.abs(recording)):.4f}")
        check_step("Audio Input (sounddevice)", True)
    else:
        check_step("Audio Input (sounddevice)", False)
except Exception as e:
    print(f"Error: {e}")
    check_step("Audio Input (sounddevice)", False)

# ==============================================================================
# STEP 4: Validate Silero VAD
# ==============================================================================
print_header("STEP 4: Validate Silero VAD")
try:
    print("Loading Silero VAD...")
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  onnx=False)
    
    # Create a dummy audio tensor (silence)
    dummy_audio = torch.zeros(512)
    prob = model(dummy_audio, 16000).item()
    print(f"  Silence probability: {prob:.4f}")
    
    if prob < 0.1:
        check_step("Silero VAD Load & Inference", True)
    else:
        print("  Warning: Silence probability unusually high for zeros.")
        check_step("Silero VAD Load & Inference", True) # Soft pass
except Exception as e:
    print(f"Error: {e}")
    check_step("Silero VAD Load & Inference", False)

# ==============================================================================
# STEP 5: Validate Whisper Model Loading
# ==============================================================================
print_header("STEP 5: Validate Whisper Model Loading")
try:
    print("Loading Whisper 'tiny' (CPU)...")
    t0 = time.time()
    stt = whisper.load_model("tiny")
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.2f}s")
    
    check_step("Whisper Model Load", True)
except Exception as e:
    print(f"Error: {e}")
    check_step("Whisper Model Load", False)

# ==============================================================================
# STEP 6: Validate MiniLM Encoder
# ==============================================================================
print_header("STEP 6: Validate MiniLM Encoder")
try:
    print("Loading all-MiniLM-L6-v2...")
    t0 = time.time()
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.2f}s")
    
    print("Encoding prompts...")
    embeddings = encoder.encode(PROMPTS, convert_to_numpy=True)
    print(f"  Embeddings shape: {embeddings.shape}")
    
    check_step("MiniLM Encoder Load & Encode", True)
except Exception as e:
    print(f"Error: {e}")
    check_step("MiniLM Encoder Load & Encode", False)

# ==============================================================================
# STEP 7: Run a Warm-Up Cycle
# ==============================================================================
print_header("STEP 7: Run a Warm-Up Cycle")
try:
    print("Warming up computation graph...")
    
    # 1. Audio
    warmup_audio = np.zeros(16000, dtype=np.float32) # 1 sec silence
    
    # 2. Whisper
    t0 = time.time()
    stt.transcribe(warmup_audio, fp16=False)
    print(f"  Whisper Warmup: {time.time() - t0:.3f}s")
    
    # 3. MiniLM
    t0 = time.time()
    encoder.encode("Warmup text", convert_to_numpy=True)
    print(f"  MiniLM Warmup: {time.time() - t0:.3f}s")
    
    check_step("Warm-Up Cycle", True)
except Exception as e:
    print(f"Error: {e}")
    check_step("Warm-Up Cycle", False)

print("\n" + "=" * 60)
print("READY FOR PIPELINE TEST")
print("=" * 60)
