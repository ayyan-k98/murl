"""
Improved Coverage Environment

Key improvements over baseline:
1. Better reward shaping (denser, more balanced)
2. Coverage progress bonus (always positive signal)
3. Terminal bonus for high coverage
4. Diagonal action correction (fair rewards)
5. Adaptive step penalty (less harsh late-game)

Based on Bouman et al. (2023) and reward analysis.
"""

import numpy as np
import math
from typing import Tuple, Dict
from environment import CoverageEnvironment
from config import config


class ImprovedCoverageEnvironment(CoverageEnvironment):
    """
    Coverage environment with improved reward shaping.

    Uses baseline architecture but with MUCH better rewards.
    """

    def __init__(self, grid_size: int = 20, map_type: str = "empty"):
        super().__init__(grid_size=grid_size, map_type=map_type)

        # Track coverage percentage for progress bonus
        self.prev_coverage_pct = 0.0

    def reset(self):
        """Reset environment including coverage tracking."""
        state = super().reset()
        self.prev_coverage_pct = 0.0
        return state

    def step(self, action: int) -> Tuple:
        """Execute action with improved rewards."""
        self.steps += 1

        # Store previous states
        prev_coverage = self.world_state.coverage_map.copy()
        prev_local_map_size = len(self.robot_state.local_map)
        prev_coverage_pct = self._get_coverage_percentage()

        # Execute action (movement)
        collision = self._execute_action(action)

        # Update sensing
        self._update_coverage()

        # Calculate gains
        coverage_gain = self._calculate_coverage_gain(prev_coverage)
        knowledge_gain = len(self.robot_state.local_map) - prev_local_map_size
        current_coverage_pct = self._get_coverage_percentage()
        coverage_pct_gain = current_coverage_pct - prev_coverage_pct

        # Calculate IMPROVED reward
        reward = self._calculate_improved_reward(
            action=action,
            coverage_gain=coverage_gain,
            knowledge_gain=knowledge_gain,
            collision=collision,
            coverage_pct_gain=coverage_pct_gain,
            current_coverage_pct=current_coverage_pct
        )

        # Update prev coverage pct
        self.prev_coverage_pct = current_coverage_pct

        # Check if done
        done = self._check_done()

        # Add terminal bonus if done with high coverage
        if done:
            reward += self._calculate_terminal_bonus(current_coverage_pct)

        # Info dict
        info = {
            'coverage_gain': coverage_gain,
            'knowledge_gain': knowledge_gain,
            'collision': collision,
            'coverage_pct': current_coverage_pct,
            'coverage_pct_gain': coverage_pct_gain,
            'steps': self.steps,
            'reward_components': self._get_reward_breakdown(
                coverage_gain, knowledge_gain, collision,
                coverage_pct_gain, current_coverage_pct, action
            )
        }

        return self.robot_state, reward, done, info

    def _calculate_improved_reward(self,
                                   action: int,
                                   coverage_gain: int,
                                   knowledge_gain: int,
                                   collision: bool,
                                   coverage_pct_gain: float,
                                   current_coverage_pct: float) -> float:
        """
        IMPROVED reward function with better shaping.

        Key improvements:
        1. Balanced rewards (coverage : exploration ≈ 5:1, not 20:1)
        2. Coverage progress bonus (dense signal!)
        3. Diagonal action correction (fair rewards)
        4. Adaptive step penalty (easier late-game)
        5. Stronger frontier bonus
        """
        reward = 0.0

        # 1. Coverage reward (REDUCED from 10.0 to 5.0)
        # Why: Make exploration more valuable relative to coverage
        reward += coverage_gain * 5.0

        # 2. Exploration reward (INCREASED from 0.5 to 1.0)
        # Why: Encourage exploring even without immediate coverage
        reward += knowledge_gain * 1.0

        # 3. Coverage PROGRESS bonus (NEW!)
        # Why: Provide dense signal even when not discovering new cells
        # This is CRITICAL for late-game learning
        if coverage_pct_gain > 0:
            # Scale: 1% coverage gain = +2.0 reward
            progress_bonus = coverage_pct_gain * 100 * 2.0
            reward += progress_bonus

        # 4. Frontier bonus (INCREASED from 0.05 to 0.15)
        # Why: Encourage boundary exploration more strongly
        frontier_cells = self._count_frontier_cells()
        frontier_bonus = min(frontier_cells * 0.15, 3.0)  # Cap at 3.0 (was 1.5)
        reward += frontier_bonus

        # 5. Collision penalty (keep as is)
        if collision:
            reward += -2.0

        # 6. ADAPTIVE step penalty (NEW!)
        # Why: Don't punish agent as harshly when coverage is high
        if current_coverage_pct < 0.7:
            step_penalty = -0.01  # Standard penalty early-game
        else:
            step_penalty = -0.005  # Reduced penalty late-game (harder to find cells)
        reward += step_penalty

        # 7. Stay penalty (keep as is)
        if action == 8:  # STAY action
            reward += -0.1

        # 8. DIAGONAL action correction (NEW!)
        # Why: Diagonal moves travel √2 distance but get same reward
        # This makes them unfairly penalized
        if action in [1, 3, 5, 7]:  # NE, SE, SW, NW
            # Compensate for extra distance (roughly)
            diagonal_bonus = 0.005  # Small bonus to make diagonals fair
            reward += diagonal_bonus

        return reward

    def _calculate_terminal_bonus(self, final_coverage_pct: float) -> float:
        """
        Terminal bonus for high coverage at episode end.

        Encourages agent to maximize final coverage.
        """
        bonus = 0.0

        if final_coverage_pct >= 0.95:
            bonus += 100.0  # Exceptional!
        elif final_coverage_pct >= 0.90:
            bonus += 50.0  # Excellent
        elif final_coverage_pct >= 0.85:
            bonus += 25.0  # Very good
        elif final_coverage_pct >= 0.80:
            bonus += 10.0  # Good
        elif final_coverage_pct >= 0.75:
            bonus += 5.0  # Decent

        return bonus

    def _get_reward_breakdown(self, coverage_gain, knowledge_gain,
                             collision, coverage_pct_gain,
                             current_coverage_pct, action) -> Dict:
        """
        Return reward breakdown for debugging/analysis.
        """
        breakdown = {
            'coverage': coverage_gain * 5.0,
            'exploration': knowledge_gain * 1.0,
            'progress': coverage_pct_gain * 100 * 2.0 if coverage_pct_gain > 0 else 0.0,
            'frontier': min(self._count_frontier_cells() * 0.15, 3.0),
            'collision': -2.0 if collision else 0.0,
            'step': -0.01 if current_coverage_pct < 0.7 else -0.005,
            'stay': -0.1 if action == 8 else 0.0,
            'diagonal': 0.005 if action in [1, 3, 5, 7] else 0.0
        }
        breakdown['total'] = sum(breakdown.values())
        return breakdown


# Test code
if __name__ == "__main__":
    print("Testing Improved Coverage Environment")
    print("=" * 80)

    # Create environment
    env = ImprovedCoverageEnvironment(grid_size=20, map_type="empty")

    # Run test episode
    print("\n1. Running Test Episode (20 steps):")
    state = env.reset()

    total_reward = 0.0
    for step in range(20):
        # Random action
        action = np.random.randint(0, 9)

        next_state, reward, done, info = env.step(action)
        total_reward += reward

        if step < 5:  # Show first 5 steps in detail
            print(f"\n   Step {step + 1}:")
            print(f"   - Action: {action}")
            print(f"   - Coverage Gain: {info['coverage_gain']}")
            print(f"   - Knowledge Gain: {info['knowledge_gain']}")
            print(f"   - Coverage: {info['coverage_pct']:.2%}")
            print(f"   - Reward: {reward:.4f}")
            print(f"   - Reward Breakdown: {info['reward_components']}")

        if done:
            break

    print(f"\n   Total Reward: {total_reward:.4f}")
    print(f"   Final Coverage: {info['coverage_pct']:.2%}")

    # Compare reward scales
    print("\n2. Reward Scale Comparison:")
    print("   " + "-" * 60)
    print("   Scenario                    | Old Reward | New Reward")
    print("   " + "-" * 60)
    print(f"   Cover 1 cell                |  +10.00    |  +5.00")
    print(f"   Sense 5 new cells           |  +2.50     |  +5.00")
    print(f"   0.5% coverage progress      |  +0.00     |  +1.00")
    print(f"   20 frontier cells           |  +1.00     |  +3.00")
    print(f"   Collision                   |  -2.00     |  -2.00")
    print(f"   Normal step                 |  -0.01     |  -0.01")
    print(f"   Diagonal move bonus         |  +0.00     |  +0.005")
    print(f"   95% coverage terminal       |  +0.00     |  +100.00")
    print("   " + "-" * 60)

    print("\n" + "=" * 80)
    print("✅ Improved Coverage Environment Test Complete!")