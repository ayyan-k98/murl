"""
Quick Component Test Script

Tests individual components quickly.
"""

def test_environment():
    """Test basic environment."""
    print("Testing CoverageEnvironment...")
    try:
        from environment import CoverageEnvironment
        
        env = CoverageEnvironment(grid_size=10, sensor_range=2.0, map_type="empty")
        state = env.reset()
        
        for i in range(3):
            action = i % 9  # Simple action sequence
            next_state, reward, done, info = env.step(action)
            if done:
                break
        
        print(f"   ✅ Environment works!")
        print(f"      Coverage: {env.get_coverage_percentage()*100:.1f}%")
        print(f"      Steps: {env.steps}")
        return True
    except Exception as e:
        print(f"   ❌ Environment failed: {e}")
        return False

def test_baseline_agent():
    """Test baseline agent."""
    print("\nTesting CoverageAgent (baseline)...")
    try:
        from agent import CoverageAgent
        from environment import CoverageEnvironment
        
        agent = CoverageAgent(grid_size=10)
        env = CoverageEnvironment(grid_size=10, map_type="empty")
        state = env.reset()
        
        action = agent.select_action(state, env.world_state)
        
        print(f"   ✅ Baseline agent works!")
        print(f"      Parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}")
        print(f"      Action: {action}")
        return True
    except Exception as e:
        print(f"   ❌ Baseline agent failed: {e}")
        return False

def test_curriculum():
    """Test curriculum manager."""
    print("\nTesting CurriculumManager...")
    try:
        from curriculum import CurriculumManager
        
        curriculum = CurriculumManager()
        
        # Test different episodes
        for ep in [0, 300, 600, 1000, 1500]:
            phase = curriculum.get_current_phase(ep)
            map_type = curriculum.get_map_type(ep)
            
        print(f"   ✅ Curriculum works!")
        print(f"      Phases: {len(curriculum.phases)}")
        return True
    except Exception as e:
        print(f"   ❌ Curriculum failed: {e}")
        return False

def test_replay_memory():
    """Test stratified replay memory."""
    print("\nTesting StratifiedReplayMemory...")
    try:
        from replay_memory import StratifiedReplayMemory
        
        memory = StratifiedReplayMemory(capacity=100)
        
        # Add some transitions
        for i in range(10):
            state = f"state_{i}"
            action = i % 9
            reward = float(i)
            next_state = f"next_state_{i}"
            done = (i == 9)
            
            if i % 3 == 0:
                info = {'coverage_gain': 1, 'knowledge_gain': 0, 'collision': False}
            else:
                info = {'coverage_gain': 0, 'knowledge_gain': 1, 'collision': False}
            
            memory.push(state, action, reward, next_state, done, info)
        
        if len(memory) >= 5:
            batch = memory.sample(5)
            
        stats = memory.get_stats()
        
        print(f"   ✅ Replay memory works!")
        print(f"      Total: {stats['total']}")
        print(f"      Coverage: {stats['coverage']}")
        return True
    except Exception as e:
        print(f"   ❌ Replay memory failed: {e}")
        return False

def main():
    """Run all component tests."""
    print("=" * 60)
    print("COMPONENT TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_environment,
        test_baseline_agent,
        test_curriculum,
        test_replay_memory
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        print("\nBaseline architecture components are working!")
        print("\nNext: Test enhanced architecture:")
        print("py test_enhanced.py")
    else:
        print(f"❌ {total - passed} tests failed ({passed}/{total} passed)")
    
    print("=" * 60)

if __name__ == "__main__":
    main()