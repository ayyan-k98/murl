"""
Performance Diagnostic Test

Run this to identify exactly where the bottleneck is.
Compares both BASELINE and ENHANCED architectures.
"""

import time
import torch


def compare_architectures():
    """Compare baseline vs enhanced architecture performance."""
    print("=" * 80)
    print("ARCHITECTURE COMPARISON")
    print("=" * 80)
    
    from agent import CoverageAgent
    from agent_enhanced import EnhancedCoverageAgent
    
    # Count parameters
    baseline_agent = CoverageAgent(grid_size=20)
    enhanced_agent = EnhancedCoverageAgent(grid_size=20)
    
    baseline_params = sum(p.numel() for p in baseline_agent.policy_net.parameters())
    enhanced_params = sum(p.numel() for p in enhanced_agent.policy_net.parameters())
    
    print(f"Baseline:  {baseline_params:,} parameters")
    print(f"Enhanced:  {enhanced_params:,} parameters")
    print(f"Increase:  +{enhanced_params-baseline_params:,} ({(enhanced_params/baseline_params-1)*100:.1f}%)")
    print()


def test_graph_encoding_speed():
    """Test how fast graph encoding is."""
    print("=" * 80)
    print("TEST 1: Graph Encoding Speed")
    print("=" * 80)
    
    from graph_encoder_enhanced import EnhancedGraphStateEncoder
    from map_generator import MapGenerator
    from data_structures import RobotState, WorldState
    import numpy as np
    
    # Setup
    gen = MapGenerator(20)
    graph, obstacles = gen.generate("empty")
    world_state = WorldState(
        grid_size=20,
        graph=graph,
        obstacles=obstacles,
        coverage_map=np.zeros((20, 20)),
        map_type="empty"
    )
    
    robot_state = RobotState(position=(10, 10), orientation=0.0)
    # Add some cells to local map
    for i in range(20):
        for j in range(20):
            if abs(i-10) + abs(j-10) < 5:
                robot_state.local_map[(i, j)] = (0.0, "free")
    
    encoder = EnhancedGraphStateEncoder(20)
    
    # Warmup
    for _ in range(5):
        data = encoder.encode(robot_state, world_state, 0)
    
    # Time it
    num_runs = 100
    start = time.time()
    for _ in range(num_runs):
        data = encoder.encode(robot_state, world_state, 0)
    elapsed = time.time() - start
    
    print(f"✓ Encoded {num_runs} times in {elapsed:.2f}s")
    print(f"  Average: {elapsed/num_runs*1000:.1f}ms per encoding")
    print(f"  Nodes per encoding: {data.num_nodes}")
    print(f"  Expected time for 200-step episode: {elapsed/num_runs*200:.1f}s")
    print()


def test_network_forward_speed():
    """Test how fast the network forward pass is for both architectures."""
    print("=" * 80)
    print("TEST 2: Network Forward Pass Speed")
    print("=" * 80)
    
    from agent import CoverageAgent
    from agent_enhanced import EnhancedCoverageAgent
    from environment import CoverageEnvironment
    
    env = CoverageEnvironment(grid_size=20, map_type="empty")
    
    # Test baseline
    print("\nBASELINE Agent:")
    baseline_agent = CoverageAgent(grid_size=20)
    state = env.reset()
    
    # Warmup
    for _ in range(5):
        action = baseline_agent.select_action(state, env.world_state)
    
    # Time it
    num_runs = 100
    start = time.time()
    for _ in range(num_runs):
        action = baseline_agent.select_action(state, env.world_state)
    baseline_time = time.time() - start
    
    print(f"✓ Selected action {num_runs} times in {baseline_time:.2f}s")
    print(f"  Average: {baseline_time/num_runs*1000:.1f}ms per action")
    print(f"  Expected time for 200-step episode: {baseline_time/num_runs*200:.1f}s")
    
    # Test enhanced
    print("\nENHANCED Agent:")
    enhanced_agent = EnhancedCoverageAgent(grid_size=20)
    state = env.reset()
    
    # Warmup
    for _ in range(5):
        action = enhanced_agent.select_action(state, env.world_state, reset_memory=True)
    
    # Time it
    start = time.time()
    for _ in range(num_runs):
        action = enhanced_agent.select_action(state, env.world_state, reset_memory=False)
    enhanced_time = time.time() - start
    
    print(f"✓ Selected action {num_runs} times in {enhanced_time:.2f}s")
    print(f"  Average: {enhanced_time/num_runs*1000:.1f}ms per action")
    print(f"  Expected time for 200-step episode: {enhanced_time/num_runs*200:.1f}s")
    print(f"  Slowdown vs baseline: {enhanced_time/baseline_time:.2f}x")
    print()


def test_training_step_speed():
    """Test how fast a single training step is for both architectures."""
    print("=" * 80)
    print("TEST 3: Training Step Speed")
    print("=" * 80)
    
    from agent import CoverageAgent
    from agent_enhanced import EnhancedCoverageAgent
    from environment import CoverageEnvironment
    
    env = CoverageEnvironment(grid_size=20, map_type="empty")
    
    # Test baseline
    print("\nBASELINE Agent:")
    baseline_agent = CoverageAgent(grid_size=20)
    
    # Fill replay buffer
    print("Filling replay buffer...")
    state = env.reset()
    for i in range(300):
        graph_data = baseline_agent.graph_encoder.encode(state, env.world_state, 0)
        action = baseline_agent.select_action(state, env.world_state)
        next_state, reward, done, info = env.step(action)
        next_graph_data = baseline_agent.graph_encoder.encode(next_state, env.world_state, 0)
        baseline_agent.store_transition(graph_data, action, reward, next_graph_data, done, info)
        state = next_state
        if done:
            state = env.reset()
    
    print(f"Buffer filled: {len(baseline_agent.memory)} transitions")
    
    # Warmup
    for _ in range(5):
        baseline_agent.optimize()
    
    # Time it
    num_runs = 50
    start = time.time()
    for _ in range(num_runs):
        loss = baseline_agent.optimize()
    baseline_time = time.time() - start
    
    print(f"✓ Trained {num_runs} times in {baseline_time:.2f}s")
    print(f"  Average: {baseline_time/num_runs*1000:.1f}ms per training step")
    print(f"  At TRAIN_FREQ=8, expect ~{(200/8)*(baseline_time/num_runs):.1f}s training per episode")
    
    # Test enhanced
    print("\nENHANCED Agent:")
    enhanced_agent = EnhancedCoverageAgent(grid_size=20)
    
    # Fill replay buffer
    print("Filling replay buffer...")
    state = env.reset()
    for i in range(300):
        graph_data = enhanced_agent.graph_encoder.encode(state, env.world_state, 0)
        action = enhanced_agent.select_action(state, env.world_state, reset_memory=(i==0))
        next_state, reward, done, info = env.step(action)
        next_graph_data = enhanced_agent.graph_encoder.encode(next_state, env.world_state, 0)
        enhanced_agent.store_transition(graph_data, action, reward, next_graph_data, done, info)
        state = next_state
        if done:
            state = env.reset()
    
    print(f"Buffer filled: {len(enhanced_agent.memory)} transitions")
    
    # Warmup
    for _ in range(5):
        enhanced_agent.optimize()
    
    # Time it
    start = time.time()
    for _ in range(num_runs):
        loss = enhanced_agent.optimize()
    enhanced_time = time.time() - start
    
    print(f"✓ Trained {num_runs} times in {enhanced_time:.2f}s")
    print(f"  Average: {enhanced_time/num_runs*1000:.1f}ms per training step")
    print(f"  At TRAIN_FREQ=8, expect ~{(200/8)*(enhanced_time/num_runs):.1f}s training per episode")
    print(f"  Slowdown vs baseline: {enhanced_time/baseline_time:.2f}x")
    print()


def test_full_episode():
    """Test a complete episode to measure real performance."""
    print("=" * 60)
    print("TEST 4: Complete Episode Performance")
    print("=" * 60)
    
    from agent_enhanced import EnhancedCoverageAgent
    from environment import CoverageEnvironment
    from config import config
    
    agent = EnhancedCoverageAgent(grid_size=20)
    env = CoverageEnvironment(grid_size=20, map_type="empty")
    
    # Run 3 episodes
    times = []
    for ep in range(3):
        state = env.reset()
        start = time.time()
        
        for step in range(config.MAX_EPISODE_STEPS):
            graph_data = agent.graph_encoder.encode(state, env.world_state, 0)
            action = agent.select_action(state, env.world_state, reset_memory=(step==0))
            next_state, reward, done, info = env.step(action)
            next_graph_data = agent.graph_encoder.encode(next_state, env.world_state, 0)
            agent.store_transition(graph_data, action, reward, next_graph_data, done, info)
            
            if len(agent.memory) >= config.MIN_REPLAY_SIZE and step % config.TRAIN_FREQ == 0:
                loss = agent.optimize()
            
            state = next_state
            if done:
                break
        
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Episode {ep+1}: {elapsed:.2f}s ({step+1} steps)")
    
    avg_time = sum(times) / len(times)
    print(f"\n✓ Average episode time: {avg_time:.2f}s")
    print(f"  Estimated 100 episodes: {avg_time*100/60:.1f} minutes")
    print(f"  Estimated 1600 episodes: {avg_time*1600/60:.1f} minutes ({avg_time*1600/3600:.1f} hours)")
    print()


def main():
    """Run all performance tests."""
    print("\n" + "=" * 80)
    print("PERFORMANCE DIAGNOSTIC SUITE - BASELINE vs ENHANCED")
    print("=" * 80)
    print("This will identify bottlenecks in both architectures.")
    print()
    
    compare_architectures()
    test_graph_encoding_speed()
    test_network_forward_speed()
    test_training_step_speed()
    test_full_episode()
    
    print("=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)
    print()
    print("Look for the slowest component above.")
    print("Common bottlenecks:")
    print("  • Graph encoding > 50ms: Too slow")
    print("  • Action selection > 50ms: Network too large")
    print("  • Training step > 100ms: Batch size too large or network too complex")
    print("  • Episode time > 10s: Check all above")
    print()
    print("RECOMMENDATIONS:")
    print("  • If baseline is fast, enhanced architecture needs optimization")
    print("  • If both are slow, the issue is in shared components (encoding/env)")
    print("  • Compare baseline vs enhanced to see if Phase 1 improvements are worth it")
    print()


if __name__ == "__main__":
    main()