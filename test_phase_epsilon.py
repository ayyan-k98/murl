"""
Test phase-specific epsilon decay

Verifies that each curriculum phase has its own decay strategy.
"""

from curriculum import CurriculumManager
from config import config

def test_phase_epsilon():
    """Test that each phase has proper epsilon parameters."""
    
    print("=" * 80)
    print("TESTING PHASE-SPECIFIC EPSILON DECAY")
    print("=" * 80)
    
    manager = CurriculumManager()
    
    # Test key episode transitions
    test_episodes = [
        0,      # Phase 1 start
        100,    # Phase 1 middle
        200,    # Phase 2 start
        500,    # Phase 4 (consolidation 1)
        550,    # Phase 5 start
        850,    # Phase 7 (consolidation 2)
        1225,   # Phase 10 (consolidation 3)
        1550,   # Phase 13 (final polish)
    ]
    
    print(f"\n{'Episode':<10} {'Phase':<35} {'ε Decay':<12} {'ε Floor':<10}")
    print("-" * 80)
    
    for ep in test_episodes:
        phase = manager.get_current_phase(ep)
        decay = manager.get_epsilon_decay(ep)
        floor = manager.get_epsilon_floor(ep)
        
        print(f"{ep:<10} {phase.name:<35} {decay:<12.4f} {floor:<10.3f}")
    
    # Verify specific values
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Phase 1 should have fast decay (0.99)
    tests_total += 1
    decay_phase1 = manager.get_epsilon_decay(0)
    if abs(decay_phase1 - 0.99) < 0.001:
        print(f"✅ Phase 1 decay = {decay_phase1:.4f} (expected 0.99)")
        tests_passed += 1
    else:
        print(f"❌ Phase 1 decay = {decay_phase1:.4f} (expected 0.99)")
    
    # Test 2: Consolidation phases should have slow decay (0.998)
    tests_total += 1
    decay_phase4 = manager.get_epsilon_decay(500)  # Consolidation 1
    if abs(decay_phase4 - 0.998) < 0.001:
        print(f"✅ Consolidation decay = {decay_phase4:.4f} (expected 0.998)")
        tests_passed += 1
    else:
        print(f"❌ Consolidation decay = {decay_phase4:.4f} (expected 0.998)")
    
    # Test 3: Epsilon floor should decrease over phases
    tests_total += 1
    floor_phase1 = manager.get_epsilon_floor(0)
    floor_phase13 = manager.get_epsilon_floor(1550)
    if floor_phase1 > floor_phase13:
        print(f"✅ Epsilon floor decreases: {floor_phase1:.3f} → {floor_phase13:.3f}")
        tests_passed += 1
    else:
        print(f"❌ Epsilon floor should decrease across phases")
    
    # Test 4: Config values exist
    tests_total += 1
    try:
        assert hasattr(config, 'EPSILON_DECAY_PHASE1')
        assert hasattr(config, 'EPSILON_DECAY_PHASE13')
        print(f"✅ Config has all phase-specific epsilon decay values")
        tests_passed += 1
    except AssertionError:
        print(f"❌ Config missing phase-specific epsilon decay values")
    
    # Simulate epsilon decay across phases
    print("\n" + "=" * 80)
    print("EPSILON TRAJECTORY SIMULATION")
    print("=" * 80)
    print("Simulating how epsilon decays across all 13 phases...\n")
    
    epsilon = 1.0
    key_episodes = [0, 50, 100, 150, 200, 350, 500, 550, 850, 1225, 1550, 1600]
    
    print(f"{'Episode':<10} {'Phase':<8} {'ε Decay':<10} {'ε Value':<10} {'Note'}")
    print("-" * 80)
    
    for i, ep in enumerate(key_episodes):
        phase = manager.get_current_phase(ep)
        decay = manager.get_epsilon_decay(ep)
        floor = manager.get_epsilon_floor(ep)
        
        # Simulate decay from previous episode
        if i > 0:
            steps = ep - key_episodes[i-1]
            prev_decay = manager.get_epsilon_decay(key_episodes[i-1])
            epsilon *= (prev_decay ** steps)
            epsilon = max(epsilon, floor)
        
        # Extract phase number
        import re
        match = re.search(r'Phase(\d+)', phase.name)
        phase_num = int(match.group(1)) if match else 1
        
        note = ""
        if "Foundation" in phase.name:
            note = "Fast learning"
        elif "Consolidation" in phase.name:
            note = "Refinement"
        elif "Intro" in phase.name:
            note = "New environment"
        elif "Final" in phase.name:
            note = "Pure exploitation"
        
        print(f"{ep:<10} {phase_num:<8} {decay:<10.4f} {epsilon:<10.4f} {note}")
    
    # Final summary
    print("\n" + "=" * 80)
    print(f"SUMMARY: {tests_passed}/{tests_total} tests passed")
    print("=" * 80)
    
    if tests_passed == tests_total:
        print("✅ All tests passed! Phase-specific epsilon decay is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check configuration.")
        return False


if __name__ == "__main__":
    success = test_phase_epsilon()
    exit(0 if success else 1)
