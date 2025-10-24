"""
Enhanced DQN Agent with Phase 1 Improvements

Combines:
- Enhanced graph encoder (edge features + 10D nodes)
- Enhanced GAT (adaptive virtual node)
- Recurrent state encoder (POMDP memory)
- AGC gradient clipping

This is the COMPLETE Phase 1 enhanced architecture.
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional

from config import config
from data_structures import RobotState, WorldState
from gat_network_enhanced import EnhancedGATCoverageDQN
from graph_encoder_enhanced import EnhancedGraphStateEncoder
from recurrent_encoder import RecurrentStateEncoder, RecurrentGATCoverageDQN
from replay_memory import StratifiedReplayMemory


class EnhancedCoverageAgent:
    """
    Enhanced DQN Agent with Phase 1 improvements.

    Enhancements:
    1. Edge features in graph encoding
    2. 10D node features (vs 8D baseline)
    3. Adaptive virtual node
    4. Recurrent state for POMDP memory
    5. AGC gradient clipping
    """

    def __init__(self,
                 grid_size: int = 20,
                 learning_rate: float = None,
                 gamma: float = None,
                 device: str = None):

        self.grid_size = grid_size
        
        # Use config values if not provided
        if learning_rate is None:
            learning_rate = config.LEARNING_RATE
        if gamma is None:
            gamma = config.GAMMA
        
        self.gamma = gamma
        self.device = device or config.DEVICE

        # Enhanced graph encoder
        self.graph_encoder = EnhancedGraphStateEncoder(grid_size)

        # Build enhanced network
        # 1. GAT network
        gat_network = EnhancedGATCoverageDQN(
            node_feature_dim=10,  # Enhanced
            agent_feature_dim=10,
            edge_feature_dim=3,   # NEW
            hidden_dim=config.GAT_HIDDEN_DIM,
            n_actions=config.N_ACTIONS,
            n_gat_layers=config.GAT_N_LAYERS,
            n_heads=config.GAT_N_HEADS,
            dropout=config.GAT_DROPOUT
        ).to(self.device)

        # 2. Recurrent encoder
        jk_dim = config.GAT_HIDDEN_DIM * (config.GAT_N_LAYERS + 1)
        recurrent_encoder = RecurrentStateEncoder(
            input_dim=jk_dim,
            hidden_dim=config.GAT_HIDDEN_DIM,
            num_layers=1,
            dropout=config.GAT_DROPOUT
        ).to(self.device)

        # 3. Combine into full network
        self.policy_net = RecurrentGATCoverageDQN(
            gat_network=gat_network,
            recurrent_encoder=recurrent_encoder
        ).to(self.device)

        # Target network
        gat_network_target = EnhancedGATCoverageDQN(
            node_feature_dim=10,
            agent_feature_dim=10,
            edge_feature_dim=3,
            hidden_dim=config.GAT_HIDDEN_DIM,
            n_actions=config.N_ACTIONS,
            n_gat_layers=config.GAT_N_LAYERS,
            n_heads=config.GAT_N_HEADS,
            dropout=config.GAT_DROPOUT
        ).to(self.device)

        recurrent_encoder_target = RecurrentStateEncoder(
            input_dim=jk_dim,
            hidden_dim=config.GAT_HIDDEN_DIM,
            num_layers=1,
            dropout=config.GAT_DROPOUT
        ).to(self.device)

        self.target_net = RecurrentGATCoverageDQN(
            gat_network=gat_network_target,
            recurrent_encoder=recurrent_encoder_target
        ).to(self.device)

        # Initialize target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay memory
        self.memory = StratifiedReplayMemory(capacity=config.REPLAY_BUFFER_SIZE)

        # Epsilon (exploration)
        self.epsilon = config.EPSILON_START

        # Gradient tracking
        self.grad_norm_history = []

    def select_action(self,
                     robot_state: RobotState,
                     world_state: WorldState,
                     epsilon: Optional[float] = None,
                     reset_memory: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            robot_state: Current robot state
            world_state: World state (for graph encoding)
            epsilon: Override default epsilon
            reset_memory: Reset recurrent memory (episode start)

        Returns:
            action: Integer action [0-8]
        """
        if epsilon is None:
            epsilon = self.epsilon

        # Epsilon-greedy
        if random.random() < epsilon:
            return random.randint(0, config.N_ACTIONS - 1)

        # Greedy action
        with torch.no_grad():
            # Encode state to graph
            graph_data = self.graph_encoder.encode(robot_state, world_state, agent_idx=0)
            graph_data = graph_data.to(self.device)

            # Forward pass (with recurrent state)
            q_values = self.policy_net(graph_data, reset_recurrent=reset_memory)

            # Select best action
            action = q_values.argmax(dim=1).item()

        return action

    def select_action_from_graph(self, graph_data, epsilon: Optional[float] = None,
                                  reset_memory: bool = False) -> int:
        """
        OPTIMIZED: Select action from pre-encoded graph (avoids redundant encoding).

        Args:
            graph_data: Pre-encoded graph data
            epsilon: Override default epsilon
            reset_memory: Reset recurrent memory (episode start)

        Returns:
            action: Integer action [0-8]
        """
        if epsilon is None:
            epsilon = self.epsilon

        # Epsilon-greedy
        if random.random() < epsilon:
            return random.randint(0, config.N_ACTIONS - 1)

        # Greedy action
        with torch.no_grad():
            # Move to device if not already
            if graph_data.x.device != self.device:
                graph_data = graph_data.to(self.device)

            # Forward pass (with recurrent state)
            q_values = self.policy_net(graph_data, reset_recurrent=reset_memory)

            # Select best action
            action = q_values.argmax(dim=1).item()

        return action

    def store_transition(self, state, action, reward, next_state, done, info):
        """Store transition in replay memory."""
        self.memory.push(state, action, reward, next_state, done, info)

    def optimize(self) -> Optional[float]:
        """
        Perform one optimization step.

        Returns:
            loss: DQN loss value, or None if not enough samples
        """
        # Check if enough samples
        if len(self.memory) < config.MIN_REPLAY_SIZE:
            return None

        # Sample batch
        batch = self.memory.sample(config.BATCH_SIZE)

        if len(batch) == 0:
            return None

        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Compute Q(s, a) for current states
        # Note: For recurrent network, we process each transition independently
        # (not maintaining temporal continuity in batch - approximation)
        # OPTIMIZED: Batch graphs before GPU transfer for speed
        from torch_geometric.data import Batch

        # FIX: Ensure all graphs are on CPU before batching to avoid device mismatch
        state_graphs = list(states)
        next_state_graphs = list(next_states)
        state_graphs_cpu = [s.to('cpu') if s.x.device.type != 'cpu' else s for s in state_graphs]
        next_state_graphs_cpu = [s.to('cpu') if s.x.device.type != 'cpu' else s for s in next_state_graphs]

        batched_states = Batch.from_data_list(state_graphs_cpu).to(self.device)
        # Reset recurrent state for each sample (conservative)
        q_values = self.policy_net(batched_states, reset_recurrent=True)  # [batch_size, n_actions]
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [batch_size]

        # Compute V(s') for next states using target network
        with torch.no_grad():
            batched_next_states = Batch.from_data_list(next_state_graphs_cpu).to(self.device)
            next_q_values = self.target_net(batched_next_states, reset_recurrent=True)  # [batch_size, n_actions]
            next_q_values = next_q_values.max(dim=1)[0]  # [batch_size]

        # Compute target: r + gamma * max_a' Q(s', a')
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss (Huber loss for stability)
        loss = F.smooth_l1_loss(q_values, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Apply Adaptive Gradient Clipping (AGC)
        grad_norm = self._apply_agc(
            self.policy_net.parameters(),
            clip_ratio=config.AGC_CLIP_RATIO,
            eps=config.AGC_EPS
        )

        # Global gradient clipping (backup)
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            config.GRAD_CLIP_THRESHOLD
        )

        self.optimizer.step()

        # Track gradient norm
        self.grad_norm_history.append(grad_norm)

        return loss.item()

    def _apply_agc(self, parameters, clip_ratio: float = 0.02, eps: float = 1e-3) -> float:
        """Apply Adaptive Gradient Clipping (AGC)."""
        total_norm = 0.0

        for param in parameters:
            if param.grad is not None:
                param_norm = param.data.norm()
                grad_norm = param.grad.data.norm()

                # Compute max allowed gradient norm
                max_norm = max(param_norm * clip_ratio, eps)

                # Clip if necessary
                if grad_norm > max_norm:
                    param.grad.data.mul_(max_norm / (grad_norm + 1e-8))

                total_norm += grad_norm.item() ** 2

        total_norm = total_norm ** 0.5

        return total_norm

    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self, decay_rate: float = None, min_epsilon: float = None):
        """Decay epsilon for exploration."""
        if decay_rate is None:
            decay_rate = config.EPSILON_DECAY_RATE
        if min_epsilon is None:
            min_epsilon = config.EPSILON_MIN

        self.epsilon = max(self.epsilon * decay_rate, min_epsilon)

    def update_learning_rate(self, decay_rate: float = None, min_lr: float = None):
        """
        Decay learning rate for training stability.

        Args:
            decay_rate: Decay rate (default from config)
            min_lr: Minimum learning rate (default from config)
        """
        if decay_rate is None:
            decay_rate = config.LR_DECAY_RATE
        if min_lr is None:
            min_lr = config.LEARNING_RATE_MIN

        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
            new_lr = max(current_lr * decay_rate, min_lr)
            param_group['lr'] = new_lr
    
    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def set_epsilon(self, epsilon: float):
        """Manually set epsilon."""
        self.epsilon = epsilon

    def reset_memory(self):
        """Reset recurrent memory (call at episode start)."""
        self.policy_net.reset_memory()
        self.target_net.reset_memory()

    def save_checkpoint(self, path: str, episode: int, metrics: dict):
        """Save agent checkpoint."""
        checkpoint = {
            'episode': episode,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'grad_norm_history': self.grad_norm_history,
            'metrics': metrics,
            'config': {
                'grid_size': self.grid_size,
                'gamma': self.gamma,
                'device': self.device
            },
            'architecture': 'enhanced_phase1'  # Mark as enhanced
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.grad_norm_history = checkpoint['grad_norm_history']

        return checkpoint['episode'], checkpoint['metrics']

    def get_grad_stats(self) -> dict:
        """Get gradient statistics."""
        if len(self.grad_norm_history) == 0:
            return {'mean': 0, 'max': 0, 'min': 0, 'explosions': 0}

        recent_norms = self.grad_norm_history[-100:]
        explosions = sum(1 for g in recent_norms if g > config.EXPLOSION_THRESHOLD)

        return {
            'mean': sum(recent_norms) / len(recent_norms),
            'max': max(recent_norms),
            'min': min(recent_norms),
            'explosions': explosions
        }


if __name__ == "__main__":
    # Test enhanced agent
    print("Testing EnhancedCoverageAgent...")

    from environment import CoverageEnvironment

    agent = EnhancedCoverageAgent(grid_size=20)
    env = CoverageEnvironment(grid_size=20, map_type="empty")

    num_params = sum(p.numel() for p in agent.policy_net.parameters())

    print(f"\n✓ Enhanced agent initialized")
    print(f"  Device: {agent.device}")
    print(f"  Policy params: {num_params:,}")
    print(f"  Baseline params: ~200,000")
    print(f"  Increase: {num_params - 200000:,} (+{(num_params/200000 - 1)*100:.1f}%)")
    print(f"  Epsilon: {agent.epsilon}")

    # Test episode
    state = env.reset()
    total_reward = 0

    print(f"\n✓ Testing episode (10 steps)...")

    for step in range(10):
        # Reset memory at episode start
        reset_mem = (step == 0)

        # Encode state
        graph_data = agent.graph_encoder.encode(state, env.world_state, 0)

        # Select action
        action = agent.select_action(state, env.world_state, epsilon=0.5, reset_memory=reset_mem)

        # Step environment
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        # Store transition
        next_graph_data = agent.graph_encoder.encode(next_state, env.world_state, 0)
        agent.store_transition(graph_data, action, reward, next_graph_data, done, info)

        state = next_state

        if step == 0:
            print(f"  Step 1: action={config.ACTION_NAMES[action]}, reward={reward:.2f}")

        if done:
            break

    print(f"\n✓ Enhanced agent test complete")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Memory size: {len(agent.memory)}")
    print(f"\n  Phase 1 Features:")
    print(f"    ✓ Edge features (3D)")
    print(f"    ✓ Enhanced node features (10D)")
    print(f"    ✓ Adaptive virtual node")
    print(f"    ✓ Recurrent state encoder")
    print(f"    ✓ AGC gradient clipping")
