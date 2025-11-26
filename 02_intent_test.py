"""
DAY 2 — Embedding + Intent Loop Test
Goal: Take Whisper text → MiniLM → dot product → Correct prompt returned

This validates the intent classification layer.
"""

import sys
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from prompts import PROMPTS

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 60)
print("DAY 2: EMBEDDING + INTENT CLASSIFICATION TEST")
print("=" * 60)
print()

# Load MiniLM model (MUST be done once and reused)
print("Loading MiniLM model...")
load_start = time.time()
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
load_time = time.time() - load_start
print(f"[OK] Model loaded in {load_time:.2f}s")
print()

# Pre-compute prompt embeddings (done once at startup)
print(f"Pre-computing embeddings for {len(PROMPTS)} prompts...")
embed_start = time.time()
prompt_embeddings = encoder.encode(PROMPTS, convert_to_numpy=True, show_progress_bar=False)
embed_time = time.time() - embed_start
print(f"[OK] Prompt embeddings computed in {embed_time:.3f}s")
print(f"    Embedding shape: {prompt_embeddings.shape}")
print()

# Save embeddings for later use
np.save("prompt_embeddings.npy", prompt_embeddings)
print("[OK] Embeddings saved to prompt_embeddings.npy")
print()

# Test with sample client utterances
test_utterances = [
    "That's way too expensive for our budget",
    "We need to think about it",
    "How long would implementation take?",
    "Can you send me the pricing breakdown?",
    "What kind of results have other companies seen?",
    "I'm not sure this is the right time for us",
]

print("=" * 60)
print("TESTING INTENT CLASSIFICATION")
print("=" * 60)
print()

total_inference_time = 0
for utterance in test_utterances:
    print(f"Client: \"{utterance}\"")
    print("-" * 60)
    
    # Embed the utterance
    embed_start = time.time()
    utterance_embedding = encoder.encode(utterance, convert_to_numpy=True, show_progress_bar=False)
    embed_time = time.time() - embed_start
    
    # Compute dot product similarity
    dot_start = time.time()
    scores = prompt_embeddings @ utterance_embedding
    best_idx = np.argmax(scores)
    dot_time = time.time() - dot_start
    
    total_time = embed_time + dot_time
    total_inference_time += total_time
    
    print(f"Matched Prompt: {PROMPTS[best_idx]}")
    print(f"Similarity Score: {scores[best_idx]:.3f}")
    print(f"Latency: Embed={embed_time*1000:.1f}ms, Dot={dot_time*1000:.1f}ms, Total={total_time*1000:.1f}ms")
    print()

# Results
print("=" * 60)
print("RESULTS")
print("=" * 60)
avg_latency = (total_inference_time / len(test_utterances)) * 1000
print(f"Average inference latency: {avg_latency:.1f}ms")
print(f"Target: < 85ms")
print()

# Evaluation
print("=" * 60)
print("EVALUATION")
print("=" * 60)

if avg_latency < 50:
    print("[EXCELLENT] Well under budget")
elif avg_latency < 85:
    print("[GOOD] Within target")
elif avg_latency < 120:
    print("[MARGINAL] Acceptable but tight")
else:
    print("[FAILED] Too slow - consider quantization or smaller model")

print()
print(f"Combined budget check:")
print(f"  Whisper: ~571ms (from Day 1)")
print(f"  Intent:  ~{avg_latency:.0f}ms (from Day 2)")
print(f"  Total:   ~{571 + avg_latency:.0f}ms")
print(f"  Target:  < 1500ms")
print()

if (571 + avg_latency) < 1500:
    print("[OK] PROCEED TO DAY 3: Sentinel (VAD + Buffer)")
else:
    print("[STOP] Combined latency too high - optimize before continuing")

print("=" * 60)
