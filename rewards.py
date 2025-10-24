"""
Modular Reward Calculation System

Provides pluggable reward components for easy experimentation and ablation studies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from constants import (
    STAY_ACTION_INDEX,
    DIAGONAL_ACTIONS,
    LATE_GAME_COVERAGE,
    HIGH_COVERAGE_THRESHOLD
)


class RewardComponent(ABC):
    """Abstract base class for reward components."""

    @abstractmethod
    def compute(self, state: Any, action: int, info: Dict) -> float:
        """
        Compute reward contribution.

        Args:
            state: Current state
            action: Action taken
            info: Step information dictionary

        Returns:
            Reward value
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get component name for logging."""
        pass


class CoverageReward(RewardComponent):
    """Reward for covering new cells."""

    def __init__(self, weight: float = 10.0):
        self.weight = weight

    def compute(self, state, action, info):
        coverage_gain = info.get('coverage_gain', 0)
        return coverage_gain * self.weight

    def get_name(self):
        return "coverage"


class ExplorationReward(RewardComponent):
    """Reward for sensing new cells."""

    def __init__(self, weight: float = 0.5):
        self.weight = weight

    def compute(self, state, action, info):
        knowledge_gain = info.get('knowledge_gain', 0)
        return knowledge_gain * self.weight

    def get_name(self):
        return "exploration"


class FrontierBonus(RewardComponent):
    """Bonus for being near unexplored areas."""

    def __init__(self, weight: float = 0.05, cap: float = 1.5):
        self.weight = weight
        self.cap = cap

    def compute(self, state, action, info):
        frontier_cells = info.get('frontier_cells', 0)
        bonus = frontier_cells * self.weight
        return min(bonus, self.cap)

    def get_name(self):
        return "frontier"


class CollisionPenalty(RewardComponent):
    """Penalty for collisions."""

    def __init__(self, weight: float = -2.0):
        self.weight = weight

    def compute(self, state, action, info):
        if info.get('collision', False):
            return self.weight
        return 0.0

    def get_name(self):
        return "collision"


class EfficiencyPenalty(RewardComponent):
    """Step and stay penalties for efficiency."""

    def __init__(self, step_penalty: float = -0.01, stay_penalty: float = -0.1):
        self.step_penalty = step_penalty
        self.stay_penalty = stay_penalty

    def compute(self, state, action, info):
        reward = self.step_penalty
        if action == STAY_ACTION_INDEX:
            reward += self.stay_penalty
        return reward

    def get_name(self):
        return "efficiency"


class AdaptiveStepPenalty(RewardComponent):
    """Adaptive step penalty based on coverage progress."""

    def __init__(self, early_penalty: float = -0.01, late_penalty: float = -0.005):
        self.early_penalty = early_penalty
        self.late_penalty = late_penalty

    def compute(self, state, action, info):
        coverage_pct = info.get('coverage_pct', 0.0)
        if coverage_pct < LATE_GAME_COVERAGE:
            return self.early_penalty
        return self.late_penalty

    def get_name(self):
        return "adaptive_step"


class DiagonalBonus(RewardComponent):
    """Compensate diagonal moves for extra distance."""

    def __init__(self, bonus: float = 0.005):
        self.bonus = bonus

    def compute(self, state, action, info):
        if action in DIAGONAL_ACTIONS:
            return self.bonus
        return 0.0

    def get_name(self):
        return "diagonal"


class ProgressBonus(RewardComponent):
    """Dense reward for coverage progress."""

    def __init__(self, scale: float = 2.0):
        self.scale = scale

    def compute(self, state, action, info):
        coverage_pct_gain = info.get('coverage_pct_gain', 0.0)
        if coverage_pct_gain > 0:
            return coverage_pct_gain * 100 * self.scale
        return 0.0

    def get_name(self):
        return "progress"


class TerminalBonus(RewardComponent):
    """Bonus for high final coverage."""

    def __init__(self):
        self.thresholds = [
            (0.95, 100.0),
            (0.90, 50.0),
            (0.85, 25.0),
            (0.80, 10.0),
            (0.75, 5.0)
        ]

    def compute(self, state, action, info):
        if not info.get('done', False):
            return 0.0

        coverage_pct = info.get('coverage_pct', 0.0)
        for threshold, bonus in self.thresholds:
            if coverage_pct >= threshold:
                return bonus
        return 0.0

    def get_name(self):
        return "terminal"


class ProbabilisticCoverageReward(RewardComponent):
    """Reward based on probabilistic coverage gain."""

    def __init__(self, weight: float = 10.0, scale: float = 0.15):
        self.weight = weight
        self.scale = scale

    def compute(self, state, action, info):
        prob_gain = info.get('prob_gain', 0.0)
        return prob_gain * self.weight * self.scale

    def get_name(self):
        return "prob_coverage"


class RewardCalculator:
    """
    Modular reward calculator with pluggable components.

    Example:
        calc = RewardCalculator([
            CoverageReward(weight=10.0),
            ExplorationReward(weight=0.5),
            CollisionPenalty(weight=-2.0)
        ])
        reward = calc.compute(state, action, info)
    """

    def __init__(self, components: list = None):
        if components is None:
            # Default baseline configuration
            components = [
                CoverageReward(weight=10.0),
                ExplorationReward(weight=0.5),
                FrontierBonus(weight=0.05, cap=1.5),
                CollisionPenalty(weight=-2.0),
                EfficiencyPenalty(step_penalty=-0.01, stay_penalty=-0.1)
            ]
        self.components = components

    def compute(self, state, action, info) -> float:
        """Compute total reward from all components."""
        return sum(comp.compute(state, action, info) for comp in self.components)

    def get_breakdown(self, state, action, info) -> Dict[str, float]:
        """Get detailed reward breakdown for debugging."""
        breakdown = {}
        for comp in self.components:
            breakdown[comp.get_name()] = comp.compute(state, action, info)
        breakdown['total'] = sum(breakdown.values())
        return breakdown

    def add_component(self, component: RewardComponent):
        """Add a reward component."""
        self.components.append(component)

    def remove_component(self, name: str):
        """Remove a reward component by name."""
        self.components = [c for c in self.components if c.get_name() != name]


# Preset configurations
def get_baseline_calculator():
    """Get baseline reward configuration."""
    return RewardCalculator([
        CoverageReward(weight=10.0),
        ExplorationReward(weight=0.5),
        FrontierBonus(weight=0.05, cap=1.5),
        CollisionPenalty(weight=-2.0),
        EfficiencyPenalty(step_penalty=-0.01, stay_penalty=-0.1)
    ])


def get_improved_calculator():
    """Get improved reward configuration with better shaping."""
    return RewardCalculator([
        CoverageReward(weight=5.0),  # Reduced
        ExplorationReward(weight=1.0),  # Increased
        ProgressBonus(scale=2.0),  # NEW: Dense signal
        FrontierBonus(weight=0.15, cap=3.0),  # Increased
        CollisionPenalty(weight=-2.0),
        AdaptiveStepPenalty(early_penalty=-0.01, late_penalty=-0.005),  # Adaptive
        DiagonalBonus(bonus=0.005),  # NEW: Fair diagonal moves
        TerminalBonus()  # NEW: Encourage high coverage
    ])


def get_probabilistic_calculator():
    """Get probabilistic reward configuration."""
    return RewardCalculator([
        ProbabilisticCoverageReward(weight=10.0, scale=0.15),
        ExplorationReward(weight=0.5),
        FrontierBonus(weight=0.05, cap=1.5),
        CollisionPenalty(weight=-2.0),
        EfficiencyPenalty(step_penalty=-0.01, stay_penalty=-0.1)
    ])


if __name__ == "__main__":
    print("Testing Reward Calculator System")
    print("=" * 80)

    # Test baseline calculator
    calc = get_baseline_calculator()

    test_info = {
        'coverage_gain': 3,
        'knowledge_gain': 10,
        'frontier_cells': 15,
        'collision': False,
        'coverage_pct': 0.65,
        'done': False
    }

    reward = calc.compute(None, 2, test_info)
    breakdown = calc.get_breakdown(None, 2, test_info)

    print("\n1. Baseline Calculator Test:")
    print(f"   Total Reward: {reward:.4f}")
    print(f"   Breakdown:")
    for name, value in breakdown.items():
        if name != 'total':
            print(f"     {name:15s}: {value:+7.4f}")

    # Test improved calculator
    calc_improved = get_improved_calculator()
    test_info['coverage_pct_gain'] = 0.02  # 2% gain

    reward_improved = calc_improved.compute(None, 2, test_info)
    breakdown_improved = calc_improved.get_breakdown(None, 2, test_info)

    print("\n2. Improved Calculator Test:")
    print(f"   Total Reward: {reward_improved:.4f}")
    print(f"   Breakdown:")
    for name, value in breakdown_improved.items():
        if name != 'total':
            print(f"     {name:15s}: {value:+7.4f}")

    print("\n3. Component Ablation Test:")
    calc_ablation = get_baseline_calculator()
    print(f"   Full reward: {calc_ablation.compute(None, 2, test_info):.4f}")

    calc_ablation.remove_component('exploration')
    print(f"   Without exploration: {calc_ablation.compute(None, 2, test_info):.4f}")

    calc_ablation.remove_component('frontier')
    print(f"   Without exploration & frontier: {calc_ablation.compute(None, 2, test_info):.4f}")

    print("\n" + "=" * 80)
    print("✅ Reward Calculator Test Complete!")
