"""
Probabilistic Coverage Environment

Based on Bouman et al. (2023) "Adaptive Coverage Path Planning"
Uses probabilistic coverage model where coverage probability decreases with distance.

Key Features:
- Sigmoid coverage probability: Pcov(cell | robot) = 1 / (1 + e^(k*(r - r0)))
- Marginal probability gain for rewards
- Dense reward signal (better for learning)
- Binary metrics for evaluation (interpretability)
"""

import numpy as np
import math
from typing import Tuple, Dict, Set
from environment import CoverageEnvironment
from config import config
from data_structures import RobotState, WorldState


class ProbabilisticCoverageEnvironment(CoverageEnvironment):
    """
    Coverage environment with probabilistic coverage model.

    Differences from base environment:
    1. Coverage probability map (in addition to binary)
    2. Sigmoid-based coverage probability
    3. Marginal probability gain for rewards
    """

    def __init__(self, grid_size: int = 20, map_type: str = "empty", seed: int = None):
        super().__init__(grid_size=grid_size, sensor_range=config.SENSOR_RANGE, map_type=map_type)

        # Probabilistic coverage map (0.0-1.0 probabilities)
        self.coverage_map_prob = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Sigmoid parameters for Pcov(r)
        self.sigmoid_r0 = config.SENSOR_RANGE / 2.0  # Midpoint (where Pcov = 0.5)
        self.sigmoid_k = 2.0  # Steepness (higher = sharper transition)

    def reset(self) -> 'RobotState':
        """Reset environment including probabilistic coverage map."""
        state = super().reset()

        # Reset probabilistic coverage
        self.coverage_map_prob = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        return state

    def _calculate_coverage_probability(self, distance: float) -> float:
        """
        Calculate coverage probability based on distance from robot.

        Uses sigmoid function: Pcov(r) = 1 / (1 + e^(k*(r - r0)))

        Args:
            distance: Euclidean distance from robot to cell

        Returns:
            Coverage probability (0.0-1.0)
        """
        # Sigmoid function
        exponent = self.sigmoid_k * (distance - self.sigmoid_r0)
        Pcov = 1.0 / (1.0 + math.exp(exponent))

        return Pcov

    def _update_robot_sensing(self):
        """
        Update coverage maps (both binary and probabilistic) - OPTIMIZED.

        Binary map: Used for ground truth metrics
        Probabilistic map: Used for reward calculation
        """
        # Update robot's local map via ray-cast sensing (uses optimized parent method)
        super()._update_robot_sensing()

        # OPTIMIZATION: Vectorize probabilistic coverage calculation
        robot_pos = self.robot_state.position

        # Extract free cells efficiently
        free_cells = [(cell, data) for cell, data in self.robot_state.local_map.items()
                      if data[1] == "free"]

        if len(free_cells) == 0:
            return

        # Vectorize distance calculation using NumPy
        cell_positions = np.array([cell for cell, _ in free_cells])
        dx = cell_positions[:, 0] - robot_pos[0]
        dy = cell_positions[:, 1] - robot_pos[1]
        distances = np.sqrt(dx**2 + dy**2)

        # Vectorize sigmoid calculation
        exponents = self.sigmoid_k * (distances - self.sigmoid_r0)
        Pcov_values = 1.0 / (1.0 + np.exp(exponents))

        # Update probabilistic coverage (vectorized)
        for i, (cell, _) in enumerate(free_cells):
            current_prob = self.coverage_map_prob[cell[0], cell[1]]
            self.coverage_map_prob[cell[0], cell[1]] = max(current_prob, Pcov_values[i])

    def _calculate_coverage_gain(self, prev_coverage: np.ndarray) -> int:
        """
        Calculate coverage gain (binary, for metrics).

        This is the same as base environment (binary coverage).
        """
        return super()._calculate_coverage_gain(prev_coverage)

    def _calculate_probabilistic_coverage_gain(self, prev_coverage_prob: np.ndarray) -> float:
        """
        Calculate probabilistic coverage gain for reward.

        Returns sum of marginal probability increases.

        Args:
            prev_coverage_prob: Previous probabilistic coverage map

        Returns:
            Sum of probability increases
        """
        # Marginal probability gain
        current_prob = self.coverage_map_prob
        prob_gain = np.sum(np.maximum(0, current_prob - prev_coverage_prob))

        return prob_gain

    def step(self, action: int) -> Tuple['RobotState', float, bool, Dict]:
        """
        Execute action and return next state, reward, done, info.

        Uses probabilistic coverage gain for reward calculation.
        """
        self.steps += 1

        # Store previous states
        prev_coverage = self.world_state.coverage_map.copy()
        prev_coverage_prob = self.coverage_map_prob.copy()
        prev_local_map_size = len(self.robot_state.local_map)

        # Execute action (movement)
        collision = self._execute_action(action)

        # Update sensing (both binary and probabilistic)
        self._update_robot_sensing()

        # Calculate gains
        coverage_gain = self._calculate_coverage_gain(prev_coverage)  # Binary (for metrics)
        prob_gain = self._calculate_probabilistic_coverage_gain(prev_coverage_prob)  # Probabilistic (for reward)
        knowledge_gain = len(self.robot_state.local_map) - prev_local_map_size

        # Calculate reward (using probabilistic gain!)
        reward = self._calculate_reward_probabilistic(
            action=action,
            prob_gain=prob_gain,  # Use probabilistic gain
            knowledge_gain=knowledge_gain,
            collision=collision
        )

        # Check if done
        done = self._check_done()

        # Info dict
        info = {
            'coverage_gain': coverage_gain,  # Binary (for metrics)
            'prob_gain': prob_gain,  # Probabilistic (for debugging)
            'knowledge_gain': knowledge_gain,
            'collision': collision,
            'coverage_pct': self._get_coverage_percentage(),  # Binary metric
            'steps': self.steps
        }

        return self.robot_state, reward, done, info

    def _calculate_reward_probabilistic(self,
                                       action: int,
                                       prob_gain: float,
                                       knowledge_gain: int,
                                       collision: bool) -> float:
        """
        Calculate reward using probabilistic coverage gain.

        Key difference: Uses prob_gain instead of discrete coverage_gain.
        This provides denser reward signal, but is scaled to match binary reward magnitudes.

        Args:
            action: Action taken
            prob_gain: Probabilistic coverage gain (0.0+)
            knowledge_gain: Number of newly sensed cells
            collision: Whether collision occurred

        Returns:
            Reward value (scaled to match binary environment)
        """
        reward = 0.0

        # Probabilistic coverage reward (DENSE SIGNAL!)
        reward += prob_gain * config.COVERAGE_REWARD

        # Exploration reward
        reward += knowledge_gain * config.EXPLORATION_REWARD

        # Frontier bonus (encourages exploring boundaries)
        frontier_cells = self._count_frontier_cells()
        frontier_bonus = min(frontier_cells * config.FRONTIER_BONUS, config.FRONTIER_CAP)
        reward += frontier_bonus

        # Collision penalty
        if collision:
            reward += config.COLLISION_PENALTY

        # Step penalty (encourages efficiency)
        reward += config.STEP_PENALTY

        # Stay penalty (discourage staying in place)
        if action == 8:  # STAY action
            reward += config.STAY_PENALTY

        # Scale down to match binary environment reward magnitudes
        # This prevents gradient explosion and training instability
        reward *= config.PROBABILISTIC_REWARD_SCALE

        return reward


# Test code
if __name__ == "__main__":
    print("Testing Probabilistic Coverage Environment")
    print("=" * 80)

    # Create environment
    env = ProbabilisticCoverageEnvironment(grid_size=20, map_type="random", seed=42)

    # Test sigmoid function
    print("\n1. Coverage Probability Sigmoid Function:")
    print(f"   r0 (midpoint) = {env.sigmoid_r0:.2f}")
    print(f"   k (steepness) = {env.sigmoid_k:.2f}")
    print("\n   Distance | Coverage Probability")
    print("   " + "-" * 35)
    for d in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        Pcov = env._calculate_coverage_probability(d)
        print(f"   {d:4.1f}m    | {Pcov:.4f}")

    # Reset and run episode
    print("\n2. Running Test Episode (10 steps):")
    state = env.reset()

    total_reward = 0.0
    for step in range(10):
        # Random action
        action = np.random.randint(0, 9)

        next_state, reward, done, info = env.step(action)
        total_reward += reward

        print(f"\n   Step {step + 1}:")
        print(f"   - Action: {action}")
        print(f"   - Prob Gain: {info['prob_gain']:.4f}")
        print(f"   - Coverage Gain: {info['coverage_gain']}")
        print(f"   - Reward: {reward:.4f}")
        print(f"   - Coverage: {info['coverage_pct']:.2%}")

        if done:
            break

    print(f"\n   Total Reward: {total_reward:.4f}")

    # Compare coverage maps
    print("\n3. Coverage Map Comparison:")
    binary_coverage = env.world_state.coverage_map.sum()
    prob_coverage = env.coverage_map_prob.sum()
    print(f"   Binary Coverage Sum: {binary_coverage:.1f}")
    print(f"   Probabilistic Coverage Sum: {prob_coverage:.4f}")
    print(f"   Ratio: {prob_coverage / max(binary_coverage, 1):.4f}")

    print("\n" + "=" * 80)
    print("✅ Probabilistic Coverage Environment Test Complete!")
