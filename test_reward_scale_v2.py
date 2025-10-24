"""
Test script to verify reward magnitudes are appropriate for Q-learning.

Target ranges for stable DQN (UPDATED):
- Per-step reward: 0.3 - 1.5 (good range for learning)
- Episode reward: 30 - 150 (for 100 step episodes)
- Expected Q-values: 30 - 150 (Q ≈ r / (1 - γ) when γ = 0.99)
"""

import sys
import numpy as np
from config import Config
from environment import CoverageEnvironment
from environment_probabilistic import ProbabilisticCoverageEnvironment

def test_reward_scale():
    print("\n" + "="*60)
    print("REWARD SCALE VERIFICATION TEST (UPDATED)")
    print("="*60)
    
    cfg = Config()
    
    # Test binary environment
    print("\n1. Binary Environment (10 steps):")
    env = CoverageEnvironment(grid_size=cfg.GRID_SIZE)
    env.reset()
    
    rewards = []
    for i in range(10):
        action = np.random.randint(0, 9)
        _, reward, _, _ = env.step(action)
        rewards.append(reward)
    
    print(f"   Per-step rewards: {[f'{r:.3f}' for r in rewards[:5]]}...")
    print(f"   Total reward (10 steps): {sum(rewards):.2f}")
    print(f"   Average reward per step: {np.mean(rewards):.2f}")
    
    # Test full binary episode
    print("\n2. Binary Environment (Full Episode - 100 steps):")
    env = CoverageEnvironment(grid_size=cfg.GRID_SIZE)
    env.reset()
    
    total_reward = 0
    for i in range(100):
        action = np.random.randint(0, 9)
        _, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    
    print(f"   Total episode reward: {total_reward:.2f}")
    print(f"   Coverage achieved: {env.get_coverage_percentage():.1f}%")
    
    # Test probabilistic environment
    print("\n3. Probabilistic Environment (10 steps):")
    env_prob = ProbabilisticCoverageEnvironment(grid_size=cfg.GRID_SIZE)
    env_prob.reset()
    
    rewards_prob = []
    for i in range(10):
        action = np.random.randint(0, 9)
        _, reward, _, _ = env_prob.step(action)
        rewards_prob.append(reward)
    
    print(f"   Per-step rewards: {[f'{r:.3f}' for r in rewards_prob[:5]]}...")
    print(f"   Total reward (10 steps): {sum(rewards_prob):.2f}")
    print(f"   Average reward per step: {np.mean(rewards_prob):.2f}")
    
    # Test full probabilistic episode
    print("\n4. Probabilistic Environment (Full Episode - 100 steps):")
    env_prob = ProbabilisticCoverageEnvironment(grid_size=cfg.GRID_SIZE)
    env_prob.reset()
    
    total_reward_prob = 0
    for i in range(100):
        action = np.random.randint(0, 9)
        _, reward, done, _ = env_prob.step(action)
        total_reward_prob += reward
        if done:
            break
    
    print(f"   Total episode reward: {total_reward_prob:.2f}")
    print(f"   Coverage achieved: {env_prob.get_coverage_percentage():.1f}%")
    
    # Assessment
    print("\n" + "="*60)
    print("ASSESSMENT:")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  COVERAGE_REWARD: {cfg.COVERAGE_REWARD}")
    print(f"  GAMMA: {cfg.GAMMA}")
    print(f"  TARGET_UPDATE_FREQ: {cfg.TARGET_UPDATE_FREQ}")
    print(f"  LEARNING_RATE: {cfg.LEARNING_RATE}")
    print(f"  PROBABILISTIC_REWARD_SCALE: {cfg.PROBABILISTIC_REWARD_SCALE}")
    
    binary_per_step = np.mean(rewards)
    prob_per_step = np.mean(rewards_prob)
    
    # Rough Q-value estimate: Q ≈ r / (1 - γ)
    expected_q_binary = binary_per_step / (1 - cfg.GAMMA)
    expected_q_prob = prob_per_step / (1 - cfg.GAMMA)
    
    print(f"\nExpected Q-value magnitude (rough estimate):")
    print(f"  Binary: Q ≈ {expected_q_binary:.2f}")
    print(f"  Probabilistic: Q ≈ {expected_q_prob:.2f}")
    
    print(f"\nTarget Ranges for Stable Q-Learning:")
    print(f"  {'✅' if 0.3 <= binary_per_step <= 1.5 else '❌'} Per-step reward: 0.3 - 1.5")
    print(f"  {'✅' if 30 <= total_reward <= 150 else '❌'} Episode reward: 30 - 150")
    print(f"  {'✅' if 30 <= expected_q_binary <= 150 else '❌'} Q-values: 30 - 150")
    
    print(f"\nBinary Results:")
    print(f"  Current per-step reward: {binary_per_step:.2f} {'✅ GOOD' if 0.3 <= binary_per_step <= 1.5 else '⚠️  CHECK'}")
    print(f"  Current episode reward (est): {total_reward:.2f} {'✅ GOOD' if 30 <= total_reward <= 150 else '⚠️  CHECK'}")
    print(f"  Current Q-value (est): {expected_q_binary:.2f} {'✅ GOOD' if 30 <= expected_q_binary <= 150 else '⚠️  CHECK'}")
    
    print(f"\nProbabilistic Results:")
    print(f"  Current per-step reward: {prob_per_step:.2f} {'✅ GOOD' if 0.3 <= prob_per_step <= 1.5 else '⚠️  CHECK'}")
    print(f"  Current episode reward (est): {total_reward_prob:.2f} {'✅ GOOD' if 30 <= total_reward_prob <= 150 else '⚠️  CHECK'}")
    print(f"  Current Q-value (est): {expected_q_prob:.2f} {'✅ GOOD' if 30 <= expected_q_prob <= 150 else '⚠️  CHECK'}")
    
    # Final verdict
    binary_ok = (0.3 <= binary_per_step <= 1.5 and 30 <= total_reward <= 150 and 30 <= expected_q_binary <= 150)
    prob_ok = (0.3 <= prob_per_step <= 1.5 and 30 <= total_reward_prob <= 150 and 30 <= expected_q_prob <= 150)
    
    print("\n" + "="*60)
    if binary_ok and prob_ok:
        print("✅ REWARD SCALE IS APPROPRIATE FOR STABLE TRAINING!")
        print("   Both environments should now learn effectively.")
    elif binary_ok:
        print("⚠️  BINARY GOOD, PROBABILISTIC NEEDS ADJUSTMENT")
    elif prob_ok:
        print("⚠️  PROBABILISTIC GOOD, BINARY NEEDS ADJUSTMENT")
    else:
        print("❌ BOTH ENVIRONMENTS NEED REWARD ADJUSTMENT")
        if binary_per_step < 0.3:
            print("   → Rewards too small, increase COVERAGE_REWARD")
        elif binary_per_step > 1.5:
            print("   → Rewards too large, decrease COVERAGE_REWARD")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_reward_scale()
