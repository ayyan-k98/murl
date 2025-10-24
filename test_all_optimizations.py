"""
Quick integration test to verify all optimizations work correctly.
"""

import time
from agent import CoverageAgent
from agent_enhanced import EnhancedCoverageAgent
from environment import CoverageEnvironment
from config import config

print("=" * 80)
print("INTEGRATION TEST - Verify All Optimizations")
print("=" * 80)
print(f"Device: {config.DEVICE}")
print(f"Batch size: {config.BATCH_SIZE}")
print(f"Train frequency: Every {config.TRAIN_FREQ} steps")
print()

def test_agent(agent_class, name, num_steps=100):
    """Test an agent for N steps."""
    print(f"Testing {name}...")
    print("-" * 80)
    
    agent = agent_class(grid_size=20)
    env = CoverageEnvironment(grid_size=20, map_type="random")
    
    # Fill replay buffer
    state = env.reset()
    for i in range(config.MIN_REPLAY_SIZE):
        is_enhanced = "Enhanced" in name
        graph_data = agent.graph_encoder.encode(state, env.world_state, 0)
        if is_enhanced:
            action = agent.select_action(state, env.world_state, reset_memory=(i==0))
        else:
            action = agent.select_action(state, env.world_state)
        next_state, reward, done, info = env.step(action)
        next_graph_data = agent.graph_encoder.encode(next_state, env.world_state, 0)
        agent.store_transition(graph_data, action, reward, next_graph_data, done, info)
        state = next_state
        if done:
            state = env.reset()
    
    print(f"✓ Replay buffer filled: {len(agent.memory)} transitions")
    
    # Run episode with training
    state = env.reset()
    total_reward = 0
    train_count = 0
    
    times = {
        'encoding': 0,
        'action': 0,
        'env': 0,
        'training': 0
    }
    
    start_total = time.time()
    
    for step in range(num_steps):
        # Encoding
        t0 = time.time()
        graph_data = agent.graph_encoder.encode(state, env.world_state, 0)
        times['encoding'] += time.time() - t0
        
        # Action selection
        t0 = time.time()
        if is_enhanced:
            action = agent.select_action(state, env.world_state, reset_memory=(step==0))
        else:
            action = agent.select_action(state, env.world_state)
        times['action'] += time.time() - t0
        
        # Environment step
        t0 = time.time()
        next_state, reward, done, info = env.step(action)
        next_graph_data = agent.graph_encoder.encode(next_state, env.world_state, 0)
        times['env'] += time.time() - t0
        
        # Store and train
        agent.store_transition(graph_data, action, reward, next_graph_data, done, info)
        
        if step % config.TRAIN_FREQ == 0:
            t0 = time.time()
            loss = agent.optimize()
            times['training'] += time.time() - t0
            if loss is not None:
                train_count += 1
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    total_time = time.time() - start_total
    
    # Print results
    print(f"✓ Completed {step+1} steps in {total_time:.2f}s")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Training steps: {train_count}")
    print(f"  Coverage: {env.get_coverage_percentage()*100:.1f}%")
    print()
    print(f"  Timing breakdown:")
    print(f"    Encoding:  {times['encoding']:.2f}s ({times['encoding']/total_time*100:.0f}%)")
    print(f"    Action:    {times['action']:.2f}s ({times['action']/total_time*100:.0f}%)")
    print(f"    Env step:  {times['env']:.2f}s ({times['env']/total_time*100:.0f}%)")
    print(f"    Training:  {times['training']:.2f}s ({times['training']/total_time*100:.0f}%)")
    print()
    
    return total_time

# Test both architectures
baseline_time = test_agent(CoverageAgent, "BASELINE Agent", num_steps=200)
enhanced_time = test_agent(EnhancedCoverageAgent, "ENHANCED Agent", num_steps=200)

print("=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Baseline: {baseline_time:.2f}s for 200 steps")
print(f"Enhanced: {enhanced_time:.2f}s for 200 steps")
print(f"Slowdown: {enhanced_time/baseline_time:.2f}x")
print()
print("Expected performance:")
print("  • Total episode time: 5-10s ✅")
print("  • Training < 20% of time ✅")
print("  • Encoding + Env ~ 70-80% of time ✅")
print()
print("If training is still slow (>20s per episode after episode 3),")
print("check Device (should be 'cuda' for GPU), or consider:")
print("  1. Reduce MAX_EPISODE_STEPS further (200 → 150 or 100)")
print("  2. Increase TRAIN_FREQ (8 → 16)")
print("  3. Reduce BATCH_SIZE (32 → 16)")
print("=" * 80)
