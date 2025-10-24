"""
DQN Agent with Adaptive Gradient Clipping

Agent for coverage planning using GAT-based DQN.
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional

from config import config
from data_structures import RobotState, WorldState
from gat_network import GATCoverageDQN
from graph_encoder import GraphStateEncoder
from replay_memory import StratifiedReplayMemory


class CoverageAgent:
    """
    DQN Agent with GAT policy network and Adaptive Gradient Clipping.
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

        # Graph encoder
        self.graph_encoder = GraphStateEncoder(grid_size)

        # Policy network (online)
        self.policy_net = GATCoverageDQN(
            node_feature_dim=config.NODE_FEATURE_DIM,
            agent_feature_dim=config.AGENT_FEATURE_DIM,
            hidden_dim=config.GAT_HIDDEN_DIM,
            n_actions=config.N_ACTIONS,
            n_gat_layers=config.GAT_N_LAYERS,
            n_heads=config.GAT_N_HEADS,
            dropout=config.GAT_DROPOUT
        ).to(self.device)

        # Target network (for stability)
        self.target_net = GATCoverageDQN(
            node_feature_dim=config.NODE_FEATURE_DIM,
            agent_feature_dim=config.AGENT_FEATURE_DIM,
            hidden_dim=config.GAT_HIDDEN_DIM,
            n_actions=config.N_ACTIONS,
            n_gat_layers=config.GAT_N_LAYERS,
            n_heads=config.GAT_N_HEADS,
            dropout=config.GAT_DROPOUT
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
                     epsilon: Optional[float] = None) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            robot_state: Current robot state
            world_state: World state (for graph encoding)
            epsilon: Override default epsilon

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

            # Forward pass
            q_values = self.policy_net(graph_data)

            # Select best action
            action = q_values.argmax(dim=1).item()

        return action

    def select_action_from_graph(self, graph_data, epsilon: Optional[float] = None) -> int:
        """
        OPTIMIZED: Select action from pre-encoded graph (avoids redundant encoding).

        Args:
            graph_data: Pre-encoded graph data
            epsilon: Override default epsilon

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

            # Forward pass
            q_values = self.policy_net(graph_data)

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

        # Encode states to graphs
        state_graphs = [s for s in states]  # Already graph data from environment
        next_state_graphs = [s for s in next_states]

        # OPTIMIZED: Batch graphs before GPU transfer (much faster than individual transfers)
        from torch_geometric.data import Batch

        # FIX: Ensure all graphs are on CPU before batching to avoid device mismatch
        state_graphs_cpu = [s.to('cpu') if s.x.device.type != 'cpu' else s for s in state_graphs]
        next_state_graphs_cpu = [s.to('cpu') if s.x.device.type != 'cpu' else s for s in next_state_graphs]

        # Compute Q(s, a) for current states
        # Batch all graphs together then transfer to device
        batched_states = Batch.from_data_list(state_graphs_cpu).to(self.device)
        q_values = self.policy_net(batched_states)  # [batch_size, n_actions]
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [batch_size]

        # Compute V(s') for next states using target network
        with torch.no_grad():
            # Batch all next state graphs then transfer to device
            batched_next_states = Batch.from_data_list(next_state_graphs_cpu).to(self.device)
            next_q_values = self.target_net(batched_next_states)  # [batch_size, n_actions]
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
        """
        Apply Adaptive Gradient Clipping (AGC).

        AGC clips gradients based on the ratio of gradient norm to parameter norm.
        This prevents gradient explosion while allowing larger gradients for larger parameters.

        Args:
            parameters: Model parameters
            clip_ratio: Clipping ratio (default 0.02)
            eps: Epsilon for numerical stability

        Returns:
            total_grad_norm: Total gradient norm before clipping
        """
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
        """
        Decay epsilon for exploration.

        Args:
            decay_rate: Decay rate (default from config)
            min_epsilon: Minimum epsilon (default from config)
        """
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

    def save_checkpoint(self, path: str, episode: int, metrics: dict):
        """
        Save agent checkpoint.

        Args:
            path: Save path
            episode: Current episode
            metrics: Training metrics
        """
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
            }
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """
        Load agent checkpoint.

        Args:
            path: Checkpoint path

        Returns:
            episode: Episode number
            metrics: Training metrics
        """
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
    # Test agent
    print("Testing CoverageAgent...")

    from environment import CoverageEnvironment

    agent = CoverageAgent(grid_size=20)
    env = CoverageEnvironment(grid_size=20, map_type="empty")

    print(f"\n✓ Agent initialized")
    print(f"  Device: {agent.device}")
    print(f"  Policy params: {sum(p.numel() for p in agent.policy_net.parameters()):,}")
    print(f"  Epsilon: {agent.epsilon}")

    # Test episode
    state = env.reset()
    total_reward = 0

    for step in range(10):
        # Encode state
        graph_data = agent.graph_encoder.encode(state, env.world_state, 0)

        # Select action
        action = agent.select_action(state, env.world_state, epsilon=0.5)

        # Step environment
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        # Store transition (graph_data instead of raw state)
        next_graph_data = agent.graph_encoder.encode(next_state, env.world_state, 0)
        agent.store_transition(graph_data, action, reward, next_graph_data, done, info)

        state = next_state

        if step == 0:
            print(f"\n✓ First action complete")
            print(f"  Action: {config.ACTION_NAMES[action]}")
            print(f"  Reward: {reward:.2f}")

        if done:
            break

    print(f"\n✓ Agent test complete")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Memory size: {len(agent.memory)}")

    # Test optimization (if enough samples)
    if len(agent.memory) >= config.BATCH_SIZE:
        loss = agent.optimize()
        if loss is not None:
            print(f"  Optimization loss: {loss:.4f}")
    else:
        print(f"  Not enough samples for optimization ({len(agent.memory)}/{config.BATCH_SIZE})")
