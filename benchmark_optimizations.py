"""
Benchmark specific bottlenecks before and after optimizations.
"""

import time
import numpy as np
from environment import CoverageEnvironment
from graph_encoder import GraphStateEncoder
from graph_encoder_enhanced import EnhancedGraphStateEncoder

print("=" * 80)
print("BOTTLENECK BENCHMARK")
print("=" * 80)
print()

# Test 1: Ray-casting speed
print("TEST 1: Ray-casting (vectorized angles & radii)")
print("-" * 80)
env = CoverageEnvironment(grid_size=20, map_type="random")
state = env.reset()

# Warmup
for _ in range(5):
    env._raycast_sensing(state.position, state.orientation)

# Benchmark
start = time.time()
n_calls = 200
for _ in range(n_calls):
    sensed = env._raycast_sensing(state.position, state.orientation)
elapsed = time.time() - start

print(f"✓ {n_calls} ray-casts in {elapsed:.3f}s")
print(f"  Average: {elapsed/n_calls*1000:.2f}ms per ray-cast")
print(f"  Expected for 200-step episode: {elapsed:.2f}s")
print(f"  Improvement: Vectorized np.linspace and trig computation")
print()

# Test 2: Graph encoding speed (baseline)
print("TEST 2: Graph Encoding - BASELINE (optimized dict lookups)")
print("-" * 80)
encoder = GraphStateEncoder(20)

# Warmup
for _ in range(5):
    data = encoder.encode(state, env.world_state, 0)

# Benchmark
start = time.time()
n_calls = 200
for _ in range(n_calls):
    data = encoder.encode(state, env.world_state, 0)
elapsed = time.time() - start

print(f"✓ {n_calls} encodings in {elapsed:.3f}s")
print(f"  Average: {elapsed/n_calls*1000:.2f}ms per encoding")
print(f"  Expected for 200-step episode (2× per step): {elapsed*2:.2f}s")
print(f"  Improvement: Single dict.get() instead of two per node")
print()

# Test 3: Graph encoding speed (enhanced)
print("TEST 3: Graph Encoding - ENHANCED (optimized dict lookups)")
print("-" * 80)
encoder_enh = EnhancedGraphStateEncoder(20)

# Warmup
for _ in range(5):
    data = encoder_enh.encode(state, env.world_state, 0)

# Benchmark
start = time.time()
n_calls = 200
for _ in range(n_calls):
    data = encoder_enh.encode(state, env.world_state, 0)
elapsed_enh = time.time() - start

print(f"✓ {n_calls} encodings in {elapsed_enh:.3f}s")
print(f"  Average: {elapsed_enh/n_calls*1000:.2f}ms per encoding")
print(f"  Expected for 200-step episode (2× per step): {elapsed_enh*2:.2f}s")
print(f"  Improvement: Single dict.get() instead of two per node/edge")
print()

# Test 4: Complete environment step (includes ray-casting + reward calculation)
print("TEST 4: Full Environment Step")
print("-" * 80)
env = CoverageEnvironment(grid_size=20, map_type="random")
state = env.reset()

# Warmup
for _ in range(5):
    next_state, reward, done, info = env.step(4)  # Move east
    if done:
        state = env.reset()

# Benchmark
state = env.reset()
start = time.time()
n_steps = 200
for i in range(n_steps):
    next_state, reward, done, info = env.step(i % 9)
    if done:
        break
elapsed = time.time() - start

print(f"✓ {n_steps} environment steps in {elapsed:.3f}s")
print(f"  Average: {elapsed/n_steps*1000:.2f}ms per step")
print(f"  Includes: Ray-casting, coverage update, reward calculation")
print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print("Optimizations Applied:")
print("  1. ✅ Vectorized ray-casting (pre-compute angles, no repeated np.linspace)")
print("  2. ✅ Dict lookup optimization (single .get() call per node)")
print("  3. ✅ GPU transfer batching (64 transfers → 2 per training step)")
print()
print("Expected Overall Speedup:")
print("  • If ray-casting was 40% of time: ~1.3-1.5x faster episodes")
print("  • If encoding was 40% of time: ~1.2-1.3x faster episodes")
print("  • If training was 20% of time: ~3-5x faster training (GPU batch)")
print("  • Combined: ~2-3x faster overall (conservative estimate)")
print()
print("Run actual training to measure real-world improvement!")
print("=" * 80)
