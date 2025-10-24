"""
Quick test to verify reward scaling is reasonable for Q-learning.
"""

from environment import CoverageEnvironment
from environment_probabilistic import ProbabilisticCoverageEnvironment
from config import config
import numpy as np

print("=" * 80)
print("REWARD SCALE VERIFICATION TEST")
print("=" * 80)
print()

# Test binary environment
print("1. Binary Environment (10 steps):")
print("-" * 80)
env = CoverageEnvironment(grid_size=20, map_type="empty")
state = env.reset()

total_reward = 0
step_rewards = []
for i in range(10):
    action = np.random.randint(0, 9)
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    step_rewards.append(reward)
    
print(f"  Per-step rewards: {[f'{r:.3f}' for r in step_rewards[:5]]}...")
print(f"  Total reward (10 steps): {total_reward:.2f}")
print(f"  Average reward per step: {total_reward/10:.2f}")
print()

# Simulate full episode
print("2. Binary Environment (Full Episode - 100 steps):")
print("-" * 80)
env = CoverageEnvironment(grid_size=20, map_type="empty")
state = env.reset()

total_reward = 0
for i in range(100):
    action = np.random.randint(0, 9)
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    if done:
        print(f"  Episode ended at step {i+1}")
        break

print(f"  Total episode reward: {total_reward:.2f}")
print(f"  Coverage achieved: {info['coverage_pct']*100:.1f}%")
print()

# Test probabilistic environment
print("3. Probabilistic Environment (10 steps):")
print("-" * 80)
env = ProbabilisticCoverageEnvironment(grid_size=20, map_type="empty")
state = env.reset()

total_reward = 0
step_rewards = []
for i in range(10):
    action = np.random.randint(0, 9)
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    step_rewards.append(reward)

print(f"  Per-step rewards: {[f'{r:.3f}' for r in step_rewards[:5]]}...")
print(f"  Total reward (10 steps): {total_reward:.2f}")
print(f"  Average reward per step: {total_reward/10:.2f}")
print()

# Assessment
print("=" * 80)
print("ASSESSMENT")
print("=" * 80)
print()

# Calculate expected Q-values
binary_expected_episode_reward = total_reward * 10  # Scale to 100 steps
gamma = config.GAMMA

print(f"Configuration:")
print(f"  COVERAGE_REWARD: {config.COVERAGE_REWARD}")
print(f"  GAMMA: {gamma}")
print(f"  TARGET_UPDATE_FREQ: {config.TARGET_UPDATE_FREQ}")
print(f"  LEARNING_RATE: {config.LEARNING_RATE}")
print()

print(f"Expected Q-value magnitude (rough estimate):")
print(f"  Per-step reward ≈ {total_reward/10:.2f}")
print(f"  Q-value ≈ r / (1 - γ) ≈ {(total_reward/10) / (1 - gamma):.2f}")
print()

# Good ranges
print("Target Ranges for Stable Q-Learning:")
print("  ✅ Per-step reward: 0.1 - 2.0 (good)")
print("  ✅ Episode reward: 10 - 200 (good)")
print("  ✅ Q-values: 1 - 100 (good)")
print()

# Check if in range
per_step = total_reward / 10
episode_est = per_step * 100
q_est = per_step / (1 - gamma)

status = "✅ GOOD" if 0.1 <= per_step <= 2.0 else "❌ BAD"
print(f"Current per-step reward: {per_step:.2f} {status}")

status = "✅ GOOD" if 10 <= episode_est <= 200 else "❌ BAD"
print(f"Current episode reward (est): {episode_est:.2f} {status}")

status = "✅ GOOD" if 1 <= q_est <= 100 else "❌ BAD"
print(f"Current Q-value (est): {q_est:.2f} {status}")

print()
print("=" * 80)

if 0.1 <= per_step <= 2.0:
    print("✅ REWARD SCALE IS APPROPRIATE FOR STABLE TRAINING!")
else:
    print("❌ REWARD SCALE NEEDS ADJUSTMENT!")
    if per_step > 2.0:
        print("   → Rewards TOO HIGH (reduce COVERAGE_REWARD)")
    else:
        print("   → Rewards TOO LOW (increase COVERAGE_REWARD)")

print("=" * 80)
