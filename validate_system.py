"""
COMPREHENSIVE SYSTEM VALIDATION
Tests all components and verifies they meet requirements
"""

import sys
import time
import numpy as np
import sounddevice as sd
import psutil
import gc
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from prompts import PROMPTS

sys.stdout.reconfigure(encoding='utf-8')

# Test configuration
SAMPLE_RATE = 16000
DURATION = 1.2
CHANNELS = 1

print("=" * 70)
print("COMPREHENSIVE SYSTEM VALIDATION")
print("=" * 70)
print()

# Track process for resource monitoring
process = psutil.Process()
initial_memory = process.memory_info().rss / 1024 / 1024  # MB

# ============================================================================
# TEST 1: Whisper Loading
# ============================================================================
print("[TEST 1] Whisper Model Loading")
print("-" * 70)

try:
    load_start = time.time()
    stt = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    load_time = time.time() - load_start
    print(f"✓ Whisper loaded successfully in {load_time:.2f}s")
    whisper_loaded = True
except Exception as e:
    print(f"✗ FAILED: {e}")
    whisper_loaded = False

print()

# ============================================================================
# TEST 2: Audio Recording
# ============================================================================
print("[TEST 2] Audio Recording")
print("-" * 70)

try:
    # Warmup
    warmup = sd.rec(int(0.1 * SAMPLE_RATE), samplerate=SAMPLE_RATE, 
                    channels=CHANNELS, dtype='float32')
    sd.wait()
    
    # Test recording
    print("Recording 1.2s of audio (you can stay silent for this test)...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=CHANNELS, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    
    # Validate
    assert len(audio) == int(DURATION * SAMPLE_RATE), "Audio length mismatch"
    assert audio.dtype == np.float32, "Audio dtype incorrect"
    assert not np.all(audio == 0), "Audio is all zeros (device issue)"
    
    print(f"✓ Audio recording works")
    print(f"  Shape: {audio.shape}, Dtype: {audio.dtype}")
    print(f"  Non-zero samples: {np.count_nonzero(audio)}/{len(audio)}")
    audio_works = True
except Exception as e:
    print(f"✗ FAILED: {e}")
    audio_works = False

print()

# ============================================================================
# TEST 3: Whisper Transcription
# ============================================================================
print("[TEST 3] Whisper Transcription")
print("-" * 70)

if whisper_loaded and audio_works:
    try:
        transcribe_start = time.time()
        segments, info = stt.transcribe(audio, beam_size=1, language="en", 
                                       temperature=0.0)
        text = " ".join([segment.text for segment in segments])
        transcribe_time = time.time() - transcribe_start
        
        print(f"✓ Transcription completed in {transcribe_time:.3f}s")
        print(f"  Text: \"{text}\"")
        print(f"  Language: {info.language} (prob: {info.language_probability:.2f})")
        
        # Latency check
        if transcribe_time < 1.0:
            print(f"  Latency: EXCELLENT (<1.0s)")
        elif transcribe_time < 1.3:
            print(f"  Latency: GOOD (<1.3s)")
        elif transcribe_time < 1.5:
            print(f"  Latency: ACCEPTABLE (<1.5s)")
        else:
            print(f"  Latency: ✗ TOO SLOW (>{1.5}s)")
        
        whisper_works = True
        whisper_latency = transcribe_time
    except Exception as e:
        print(f"✗ FAILED: {e}")
        whisper_works = False
        whisper_latency = 999
else:
    print("⊘ SKIPPED (dependencies failed)")
    whisper_works = False
    whisper_latency = 999

print()

# ============================================================================
# TEST 4: MiniLM Loading
# ============================================================================
print("[TEST 4] MiniLM Embedding Model Loading")
print("-" * 70)

try:
    load_start = time.time()
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', 
                                  device='cpu')
    load_time = time.time() - load_start
    print(f"✓ MiniLM loaded successfully in {load_time:.2f}s")
    minilm_loaded = True
except Exception as e:
    print(f"✗ FAILED: {e}")
    minilm_loaded = False

print()

# ============================================================================
# TEST 5: Embedding Computation
# ============================================================================
print("[TEST 5] Embedding Computation")
print("-" * 70)

if minilm_loaded:
    try:
        # Pre-compute prompt embeddings
        embed_start = time.time()
        prompt_embeddings = encoder.encode(PROMPTS, convert_to_numpy=True, 
                                          show_progress_bar=False)
        embed_time = time.time() - embed_start
        
        print(f"✓ Prompt embeddings computed in {embed_time:.3f}s")
        print(f"  Shape: {prompt_embeddings.shape}")
        print(f"  Prompts: {len(PROMPTS)}")
        
        # Test single utterance embedding
        test_text = "That's too expensive"
        single_start = time.time()
        test_embedding = encoder.encode(test_text, convert_to_numpy=True, 
                                       show_progress_bar=False)
        single_time = time.time() - single_start
        
        print(f"✓ Single embedding computed in {single_time*1000:.1f}ms")
        
        if single_time * 1000 < 50:
            print(f"  Latency: EXCELLENT (<50ms)")
        elif single_time * 1000 < 85:
            print(f"  Latency: GOOD (<85ms)")
        else:
            print(f"  Latency: ✗ TOO SLOW (>85ms)")
        
        embedding_works = True
        embedding_latency = single_time * 1000
    except Exception as e:
        print(f"✗ FAILED: {e}")
        embedding_works = False
        embedding_latency = 999
else:
    print("⊘ SKIPPED (MiniLM not loaded)")
    embedding_works = False
    embedding_latency = 999

print()

# ============================================================================
# TEST 6: Similarity Scoring
# ============================================================================
print("[TEST 6] Similarity Scoring (Dot Product)")
print("-" * 70)

if embedding_works:
    try:
        # Test similarity search
        dot_start = time.time()
        scores = prompt_embeddings @ test_embedding
        best_idx = np.argmax(scores)
        dot_time = time.time() - dot_start
        
        print(f"✓ Similarity scoring completed in {dot_time*1000:.2f}ms")
        print(f"  Best match: \"{PROMPTS[best_idx]}\"")
        print(f"  Score: {scores[best_idx]:.3f}")
        print(f"  Top 3 scores: {sorted(scores, reverse=True)[:3]}")
        
        # Verify reasonable score distribution
        assert scores[best_idx] > 0, "Best score should be positive"
        assert np.std(scores) > 0.01, "Scores should have variance"
        
        similarity_works = True
        similarity_latency = dot_time * 1000
    except Exception as e:
        print(f"✗ FAILED: {e}")
        similarity_works = False
        similarity_latency = 999
else:
    print("⊘ SKIPPED (embedding failed)")
    similarity_works = False
    similarity_latency = 999

print()

# ============================================================================
# TEST 7: Memory and CPU Usage
# ============================================================================
print("[TEST 7] Resource Usage (Memory & CPU)")
print("-" * 70)

# Current memory
current_memory = process.memory_info().rss / 1024 / 1024
memory_used = current_memory - initial_memory

print(f"Initial memory: {initial_memory:.1f} MB")
print(f"Current memory: {current_memory:.1f} MB")
print(f"Memory used: {memory_used:.1f} MB")

# Run multiple inferences to check for leaks
print("\nRunning 10 inference cycles to check for memory leaks...")
memory_samples = []

for i in range(10):
    # Simulate full pipeline
    test_audio = np.random.randn(int(DURATION * SAMPLE_RATE)).astype(np.float32)
    
    if whisper_works:
        segments, _ = stt.transcribe(test_audio, beam_size=1, language="en", 
                                    temperature=0.0)
        _ = " ".join([s.text for s in segments])
    
    if embedding_works:
        emb = encoder.encode("test utterance", convert_to_numpy=True, 
                           show_progress_bar=False)
        _ = prompt_embeddings @ emb
    
    memory_samples.append(process.memory_info().rss / 1024 / 1024)
    
    if i % 3 == 0:
        gc.collect()

memory_growth = memory_samples[-1] - memory_samples[0]
print(f"Memory after 10 cycles: {memory_samples[-1]:.1f} MB")
print(f"Memory growth: {memory_growth:.1f} MB")

if memory_growth < 50:
    print("✓ No significant memory leak detected")
    memory_ok = True
elif memory_growth < 100:
    print("⚠ Minor memory growth detected")
    memory_ok = True
else:
    print("✗ Significant memory leak detected")
    memory_ok = False

# CPU usage
cpu_percent = process.cpu_percent(interval=1.0)
print(f"\nCPU usage: {cpu_percent:.1f}%")

if cpu_percent < 60:
    print("✓ CPU usage acceptable")
    cpu_ok = True
else:
    print("⚠ High CPU usage")
    cpu_ok = True  # Not a failure, just a warning

print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

tests = [
    ("Whisper Loading", whisper_loaded),
    ("Audio Recording", audio_works),
    ("Whisper Transcription", whisper_works),
    ("MiniLM Loading", minilm_loaded),
    ("Embedding Computation", embedding_works),
    ("Similarity Scoring", similarity_works),
    ("Memory Usage", memory_ok),
    ("CPU Usage", cpu_ok),
]

passed = sum(1 for _, result in tests if result)
total = len(tests)

for test_name, result in tests:
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"{test_name:30s} {status}")

print()
print(f"Tests Passed: {passed}/{total}")

# Latency budget check
if whisper_works and embedding_works and similarity_works:
    total_latency = whisper_latency + embedding_latency + similarity_latency
    print()
    print("LATENCY BUDGET:")
    print(f"  Whisper:    {whisper_latency*1000:6.1f}ms")
    print(f"  Embedding:  {embedding_latency:6.1f}ms")
    print(f"  Similarity: {similarity_latency:6.1f}ms")
    print(f"  Total:      {total_latency*1000:6.1f}ms")
    print(f"  Target:     1500.0ms")
    print()
    
    if total_latency < 1.5:
        print(f"✓ LATENCY BUDGET: PASS ({total_latency*1000:.0f}ms < 1500ms)")
        budget_ok = True
    else:
        print(f"✗ LATENCY BUDGET: FAIL ({total_latency*1000:.0f}ms > 1500ms)")
        budget_ok = False
else:
    budget_ok = False

print()
print("=" * 70)

if passed == total and budget_ok:
    print("OVERALL: ✓ ALL SYSTEMS GO")
    print("Ready to proceed to Day 3: Sentinel (VAD + Buffer)")
    exit(0)
else:
    print("OVERALL: ✗ ISSUES DETECTED")
    print("Fix failing tests before proceeding")
    exit(1)
