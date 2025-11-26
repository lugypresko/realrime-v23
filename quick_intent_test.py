"""Quick test to get just the latency number"""
import sys
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from prompts import PROMPTS

sys.stdout.reconfigure(encoding='utf-8')

print("Loading MiniLM...")
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
print("Computing prompt embeddings...")
prompt_embeddings = encoder.encode(PROMPTS, convert_to_numpy=True, show_progress_bar=False)

# Test utterance
test = "That's way too expensive for our budget"

# Measure inference
times = []
for _ in range(10):
    start = time.time()
    emb = encoder.encode(test, convert_to_numpy=True, show_progress_bar=False)
    scores = prompt_embeddings @ emb
    best = np.argmax(scores)
    times.append((time.time() - start) * 1000)

avg = np.mean(times)
print(f"\nAverage latency: {avg:.1f}ms")
print(f"Target: < 85ms")
print(f"Result: {'PASS' if avg < 85 else 'FAIL'}")

with open("day2_result.txt", "w") as f:
    f.write(f"{avg:.1f}ms\n")
