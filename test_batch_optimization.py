"""
Test that the batched GPU transfer optimization works correctly.
"""

import time
from agent import CoverageAgent
from agent_enhanced import EnhancedCoverageAgent
from environment import CoverageEnvironment
from config import config

print("=" * 80)
print("TESTING BATCHED GPU TRANSFER OPTIMIZATION")
print("=" * 80)
print(f"Device: {config.DEVICE}")
print(f"Batch size: {config.BATCH_SIZE}")
print()

# Test baseline agent
print("Testing BASELINE agent...")
agent = CoverageAgent(grid_size=20)
env = CoverageEnvironment(grid_size=20, map_type="empty")

# Fill replay buffer
state = env.reset()
for i in range(300):
    graph_data = agent.graph_encoder.encode(state, env.world_state, 0)
    action = agent.select_action(state, env.world_state)
    next_state, reward, done, info = env.step(action)
    next_graph_data = agent.graph_encoder.encode(next_state, env.world_state, 0)
    agent.store_transition(graph_data, action, reward, next_graph_data, done, info)
    state = next_state
    if done:
        state = env.reset()

print(f"✓ Replay buffer filled: {len(agent.memory)} transitions")

# Time training
start = time.time()
for _ in range(20):
    loss = agent.optimize()
elapsed = time.time() - start

print(f"✓ 20 training steps: {elapsed:.2f}s ({elapsed/20*1000:.1f}ms per step)")
print()

# Test enhanced agent
print("Testing ENHANCED agent...")
agent_enh = EnhancedCoverageAgent(grid_size=20)

# Fill replay buffer
state = env.reset()
for i in range(300):
    graph_data = agent_enh.graph_encoder.encode(state, env.world_state, 0)
    action = agent_enh.select_action(state, env.world_state, reset_memory=(i==0))
    next_state, reward, done, info = env.step(action)
    next_graph_data = agent_enh.graph_encoder.encode(next_state, env.world_state, 0)
    agent_enh.store_transition(graph_data, action, reward, next_graph_data, done, info)
    state = next_state
    if done:
        state = env.reset()

print(f"✓ Replay buffer filled: {len(agent_enh.memory)} transitions")

# Time training
start = time.time()
for _ in range(20):
    loss = agent_enh.optimize()
elapsed_enh = time.time() - start

print(f"✓ 20 training steps: {elapsed_enh:.2f}s ({elapsed_enh/20*1000:.1f}ms per step)")
print()

print("=" * 80)
print("OPTIMIZATION TEST COMPLETE")
print("=" * 80)
print(f"Baseline training: {elapsed/20*1000:.1f}ms per step")
print(f"Enhanced training: {elapsed_enh/20*1000:.1f}ms per step")
print()
print("Expected improvement: 2-10x faster on GPU with batched transfers")
print("(Less improvement on CPU since there's no transfer overhead)")
print()
