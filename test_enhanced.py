"""
Test Enhanced Architecture

Quick test to verify the enhanced architecture components work correctly.
"""

def test_enhanced_components():
    """Test all enhanced components individually."""
    
    print("=" * 80)
    print("TESTING ENHANCED ARCHITECTURE COMPONENTS")
    print("=" * 80)
    print()
    
    # Test 1: Enhanced Graph Encoder
    print("1. Testing EnhancedGraphStateEncoder...")
    try:
        from graph_encoder_enhanced import EnhancedGraphStateEncoder
        from map_generator import MapGenerator
        from data_structures import RobotState, WorldState
        import numpy as np
        
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
        robot_state.local_map[(10, 10)] = (0.0, "free")
        robot_state.local_map[(11, 10)] = (0.5, "free")
        robot_state.local_map[(10, 11)] = (0.3, "free")
        
        encoder = EnhancedGraphStateEncoder(20)
        data = encoder.encode(robot_state, world_state, 0)
        
        print(f"   ✅ Enhanced encoder works!")
        print(f"      Nodes: {data.num_nodes}")
        print(f"      Node features: {data.x.shape} (should be [n, 10])")
        print(f"      Edge features: {data.edge_attr.shape} (should be [m, 3])")
        
    except Exception as e:
        print(f"   ❌ Enhanced encoder failed: {e}")
        return False
    
    # Test 2: Enhanced GAT Network
    print("\n2. Testing EnhancedGATCoverageDQN...")
    try:
        from gat_network_enhanced import EnhancedGATCoverageDQN
        import torch
        from torch_geometric.data import Data
        
        net = EnhancedGATCoverageDQN(
            node_feature_dim=10,
            agent_feature_dim=10,
            edge_feature_dim=3,
            hidden_dim=128
        )
        
        # Create test data
        num_nodes = 5
        x = torch.randn(num_nodes, 10)
        edge_index = torch.tensor([[0, 1, 2, 1, 2, 3], [1, 2, 3, 0, 1, 2]], dtype=torch.long)
        edge_attr = torch.randn(6, 3)
        agent_features = torch.randn(1, 10)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            agent_features=agent_features,
            num_nodes=num_nodes
        )
        
        q_values = net(data)
        
        print(f"   ✅ Enhanced GAT works!")
        print(f"      Q-values shape: {q_values.shape}")
        print(f"      Parameters: {sum(p.numel() for p in net.parameters()):,}")
        
    except Exception as e:
        print(f"   ❌ Enhanced GAT failed: {e}")
        return False
    
    # Test 3: Recurrent Encoder
    print("\n3. Testing RecurrentStateEncoder...")
    try:
        from recurrent_encoder import RecurrentStateEncoder, RecurrentGATCoverageDQN
        
        gat_net = EnhancedGATCoverageDQN(
            node_feature_dim=10,
            agent_feature_dim=10,
            edge_feature_dim=3,
            hidden_dim=128
        )
        
        recurrent_enc = RecurrentStateEncoder(
            input_dim=128 * 4,  # JK concatenation
            hidden_dim=128
        )
        
        full_net = RecurrentGATCoverageDQN(gat_net, recurrent_enc)
        
        # Test sequence
        q1 = full_net(data, reset_recurrent=True)
        q2 = full_net(data, reset_recurrent=False)
        
        print(f"   ✅ Recurrent encoder works!")
        print(f"      Has memory: {not torch.allclose(q1, q2)}")
        print(f"      Total params: {sum(p.numel() for p in full_net.parameters()):,}")
        
    except Exception as e:
        print(f"   ❌ Recurrent encoder failed: {e}")
        return False
    
    # Test 4: Enhanced Agent
    print("\n4. Testing EnhancedCoverageAgent...")
    try:
        from agent_enhanced import EnhancedCoverageAgent
        from environment import CoverageEnvironment
        
        agent = EnhancedCoverageAgent(grid_size=20)
        env = CoverageEnvironment(grid_size=20, map_type="empty")
        
        state = env.reset()
        action = agent.select_action(state, env.world_state, reset_memory=True)
        
        print(f"   ✅ Enhanced agent works!")
        print(f"      Agent type: {type(agent).__name__}")
        print(f"      Network type: {type(agent.policy_net).__name__}")
        print(f"      Has reset_memory: {hasattr(agent, 'reset_memory')}")
        print(f"      Selected action: {action}")
        
    except Exception as e:
        print(f"   ❌ Enhanced agent failed: {e}")
        return False
    
    print("\n✅ ALL ENHANCED COMPONENTS WORKING!")
    return True


def test_enhanced_training():
    """Test a short enhanced training run."""
    
    print("\n" + "=" * 80)
    print("TESTING ENHANCED TRAINING")
    print("=" * 80)
    print()
    
    try:
        from train_enhanced import train_stage1_enhanced
        
        print("Running 5-episode enhanced training test...")
        agent, metrics = train_stage1_enhanced(
            num_episodes=5,
            grid_size=15,  # Smaller for speed
            verbose=True
        )
        
        print(f"\n✅ Enhanced training works!")
        print(f"   Episodes completed: {len(metrics.episode_rewards)}")
        print(f"   Final reward: {metrics.episode_rewards[-1]:.2f}")
        print(f"   Final coverage: {metrics.episode_coverages[-1]*100:.1f}%")
        print(f"   Memory size: {len(agent.memory)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    
    print("Testing Enhanced GAT-MARL Architecture...")
    print(f"Date: October 22, 2025")
    print()
    
    # Test components
    components_ok = test_enhanced_components()
    
    if components_ok:
        # Test training
        training_ok = test_enhanced_training()
        
        if training_ok:
            print("\n" + "=" * 80)
            print("🎉 ENHANCED ARCHITECTURE FULLY FUNCTIONAL!")
            print("=" * 80)
            print()
            print("Ready to use enhanced architecture:")
            print()
            print("# Quick test:")
            print("python main_enhanced.py --mode train --episodes 50")
            print()
            print("# Full training:")
            print("python main_enhanced.py --mode train --episodes 1600")
            print()
            print("The enhanced architecture includes:")
            print("✅ 10D node features + 3D edge features")
            print("✅ Adaptive virtual node")
            print("✅ Recurrent memory for POMDP")
            print("✅ Enhanced spatial representations")
            print("=" * 80)
        else:
            print("\n❌ Components work but training failed")
    else:
        print("\n❌ Component tests failed")


if __name__ == "__main__":
    main()