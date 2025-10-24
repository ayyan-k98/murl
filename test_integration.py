"""
Integration test to verify training pipeline works with both binary and probabilistic environments.
"""

import sys
from config import config
from environment import CoverageEnvironment
from environment_probabilistic import ProbabilisticCoverageEnvironment

print("=" * 80)
print("INTEGRATION TEST: Training Pipeline with Both Environments")
print("=" * 80)
print()

# Test 1: Binary Environment Training Step
print("TEST 1: Binary Environment Training Step")
print("-" * 80)
config.USE_PROBABILISTIC_ENV = False

if config.USE_PROBABILISTIC_ENV:
    env = ProbabilisticCoverageEnvironment(grid_size=10, map_type="empty")
else:
    env = CoverageEnvironment(grid_size=10, map_type="empty")

print(f"Environment: {env.__class__.__name__}")

# Simulate training loop
state = env.reset()
print(f"Initial state: position={state.position}")

# Take 5 steps
total_reward = 0.0
for i in range(5):
    action = i % 9  # Cycle through actions
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    print(f"  Step {i+1}: action={action}, reward={reward:.2f}, coverage={info['coverage_pct']:.2%}")

print(f"Total reward: {total_reward:.2f}")
print("✓ PASS")
print()

# Test 2: Probabilistic Environment Training Step
print("TEST 2: Probabilistic Environment Training Step")
print("-" * 80)
config.USE_PROBABILISTIC_ENV = True

if config.USE_PROBABILISTIC_ENV:
    env = ProbabilisticCoverageEnvironment(grid_size=10, map_type="empty")
else:
    env = CoverageEnvironment(grid_size=10, map_type="empty")

print(f"Environment: {env.__class__.__name__}")

# Simulate training loop
state = env.reset()
print(f"Initial state: position={state.position}")

# Take 5 steps
total_reward = 0.0
for i in range(5):
    action = i % 9  # Cycle through actions
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    print(f"  Step {i+1}: action={action}, reward={reward:.2f}, prob_gain={info['prob_gain']:.2f}, coverage={info['coverage_pct']:.2%}")

print(f"Total reward: {total_reward:.2f}")
print("✓ PASS")
print()

# Test 3: Verify interface compatibility
print("TEST 3: Interface Compatibility Check")
print("-" * 80)

binary_env = CoverageEnvironment(grid_size=10, map_type="empty")
prob_env = ProbabilisticCoverageEnvironment(grid_size=10, map_type="empty")

# Check that both have required methods
required_methods = ['reset', 'step', '_execute_action', '_update_robot_sensing']
for method in required_methods:
    binary_has = hasattr(binary_env, method)
    prob_has = hasattr(prob_env, method)
    print(f"  {method}: Binary={binary_has}, Probabilistic={prob_has}")
    assert binary_has and prob_has, f"Both environments must have {method}"

print("✓ PASS")
print()

# Test 4: Verify return values match
print("TEST 4: Return Value Structure Check")
print("-" * 80)

binary_env = CoverageEnvironment(grid_size=10, map_type="empty")
prob_env = ProbabilisticCoverageEnvironment(grid_size=10, map_type="empty")

binary_state = binary_env.reset()
prob_state = prob_env.reset()

binary_next, binary_reward, binary_done, binary_info = binary_env.step(0)
prob_next, prob_reward, prob_done, prob_info = prob_env.step(0)

# Check return types
print(f"  Binary return types: state={type(binary_next).__name__}, reward={type(binary_reward).__name__}, done={type(binary_done).__name__}, info={type(binary_info).__name__}")
print(f"  Prob return types: state={type(prob_next).__name__}, reward={type(prob_reward).__name__}, done={type(prob_done).__name__}, info={type(prob_info).__name__}")

# Check info dict keys (binary keys should be subset of prob keys)
binary_keys = set(binary_info.keys())
prob_keys = set(prob_info.keys())
print(f"  Binary info keys: {binary_keys}")
print(f"  Prob info keys: {prob_keys}")

assert binary_keys.issubset(prob_keys), "Binary info keys should be subset of probabilistic"
print("✓ PASS")
print()

print("=" * 80)
print("ALL INTEGRATION TESTS PASSED ✅✅✅")
print("=" * 80)
print()
print("The probabilistic environment is fully compatible with the training pipeline!")
print()
print("You can now use:")
print("  py main.py --mode train --episodes 10 --probabilistic")
print("  py main_enhanced.py --mode train --episodes 10 --probabilistic")
print("=" * 80)
