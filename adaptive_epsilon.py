"""
Adaptive Epsilon Strategy

Automatically adjusts epsilon decay based on learning progress.
If agent is learning well → decay faster
If agent is struggling → decay slower (more exploration)

This prevents premature exploitation and ensures adequate exploration.
"""

from typing import List
import numpy as np


class AdaptiveEpsilonScheduler:
    """
    Adaptive epsilon decay based on performance.

    Monitors recent coverage and adjusts decay rate:
    - High coverage growth → can exploit more (faster decay)
    - Low coverage growth → needs exploration (slower decay)
    """

    def __init__(self,
                 epsilon_start: float = 1.0,
                 epsilon_min: float = 0.10,
                 base_decay: float = 0.997,
                 adaptation_window: int = 20,
                 coverage_threshold_high: float = 0.60,
                 coverage_threshold_low: float = 0.40):
        """
        Initialize adaptive epsilon scheduler.

        Args:
            epsilon_start: Starting epsilon
            epsilon_min: Minimum epsilon
            base_decay: Base decay rate (used when performance is average)
            adaptation_window: Window for measuring performance
            coverage_threshold_high: High performance threshold
            coverage_threshold_low: Low performance threshold
        """
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.base_decay = base_decay
        self.adaptation_window = adaptation_window
        self.coverage_threshold_high = coverage_threshold_high
        self.coverage_threshold_low = coverage_threshold_low

        # Performance tracking
        self.coverage_history: List[float] = []
        self.phase_min_epsilon = epsilon_min  # Can be overridden per phase

    def update(self, current_coverage: float, phase_decay: float = None) -> float:
        """
        Update epsilon based on recent performance.

        Args:
            current_coverage: Current episode coverage
            phase_decay: Phase-specific decay rate (optional)

        Returns:
            Updated epsilon value
        """
        # Record coverage
        self.coverage_history.append(current_coverage)

        # Determine decay rate based on performance
        if len(self.coverage_history) >= self.adaptation_window:
            recent_coverage = self.coverage_history[-self.adaptation_window:]
            avg_coverage = np.mean(recent_coverage)
            coverage_trend = self._compute_trend(recent_coverage)

            # Adaptive decay
            if avg_coverage >= self.coverage_threshold_high and coverage_trend > 0:
                # Learning well → can decay faster
                decay_rate = min(phase_decay or self.base_decay, 0.99) if phase_decay else 0.99
                mode = "FAST (learning well)"
            elif avg_coverage <= self.coverage_threshold_low or coverage_trend < 0:
                # Struggling → decay much slower
                decay_rate = max(phase_decay or self.base_decay, 0.999) if phase_decay else 0.999
                mode = "SLOW (need exploration)"
            else:
                # Average → use phase or base decay
                decay_rate = phase_decay if phase_decay is not None else self.base_decay
                mode = "NORMAL"

        else:
            # Not enough data → use phase or base decay
            decay_rate = phase_decay if phase_decay is not None else self.base_decay
            mode = "WARMUP"

        # Apply decay
        self.epsilon = max(self.epsilon * decay_rate, self.phase_min_epsilon)

        return self.epsilon

    def _compute_trend(self, coverage_values: List[float]) -> float:
        """
        Compute coverage trend (slope).

        Positive = improving, Negative = declining, Zero = flat
        """
        if len(coverage_values) < 2:
            return 0.0

        x = np.arange(len(coverage_values))
        y = np.array(coverage_values)

        # Linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return slope

    def set_phase_minimum(self, phase_min: float):
        """Set minimum epsilon for current phase."""
        self.phase_min_epsilon = max(phase_min, self.epsilon_min)

    def get_epsilon(self) -> float:
        """Get current epsilon."""
        return self.epsilon

    def set_epsilon(self, epsilon: float):
        """Manually set epsilon."""
        self.epsilon = max(epsilon, self.epsilon_min)

    def get_stats(self) -> dict:
        """Get statistics about epsilon adaptation."""
        if len(self.coverage_history) < self.adaptation_window:
            return {
                'epsilon': self.epsilon,
                'avg_coverage': 0.0,
                'trend': 0.0,
                'mode': 'WARMUP'
            }

        recent_coverage = self.coverage_history[-self.adaptation_window:]
        avg_coverage = np.mean(recent_coverage)
        trend = self._compute_trend(recent_coverage)

        if avg_coverage >= self.coverage_threshold_high and trend > 0:
            mode = 'FAST'
        elif avg_coverage <= self.coverage_threshold_low or trend < 0:
            mode = 'SLOW'
        else:
            mode = 'NORMAL'

        return {
            'epsilon': self.epsilon,
            'avg_coverage': avg_coverage,
            'trend': trend,
            'mode': mode
        }


class CurriculumAwareEpsilonScheduler(AdaptiveEpsilonScheduler):
    """
    Epsilon scheduler that combines curriculum phases with adaptive decay.

    Each phase has:
    - Target coverage threshold
    - Base decay rate
    - Minimum epsilon

    Adaptation happens within each phase's constraints.
    """

    def __init__(self,
                 epsilon_start: float = 1.0,
                 epsilon_min: float = 0.10,
                 base_decay: float = 0.997):
        super().__init__(
            epsilon_start=epsilon_start,
            epsilon_min=epsilon_min,
            base_decay=base_decay
        )

        # Phase tracking
        self.current_phase = None
        self.phase_start_epsilon = epsilon_start

    def start_phase(self, phase_name: str, phase_decay: float, phase_min: float):
        """
        Start a new curriculum phase.

        Args:
            phase_name: Phase name
            phase_decay: Base decay for this phase
            phase_min: Minimum epsilon for this phase
        """
        self.current_phase = phase_name
        self.base_decay = phase_decay
        self.phase_min_epsilon = max(phase_min, self.epsilon_min)
        self.phase_start_epsilon = self.epsilon

        # Clear history for new phase (fresh adaptation)
        self.coverage_history = []

    def update_with_phase(self,
                         current_coverage: float,
                         phase_name: str,
                         phase_decay: float,
                         phase_min: float) -> float:
        """
        Update epsilon within curriculum phase constraints.

        Args:
            current_coverage: Current episode coverage
            phase_name: Current phase name
            phase_decay: Phase decay rate
            phase_min: Phase minimum epsilon

        Returns:
            Updated epsilon
        """
        # Check if phase changed
        if self.current_phase != phase_name:
            self.start_phase(phase_name, phase_decay, phase_min)

        # Use adaptive decay within phase constraints
        return self.update(current_coverage, phase_decay)


if __name__ == "__main__":
    print("Testing Adaptive Epsilon Scheduler")
    print("=" * 80)

    # Test 1: Basic adaptive scheduler
    print("\n1. Basic Adaptive Scheduler:")
    scheduler = AdaptiveEpsilonScheduler(
        epsilon_start=1.0,
        epsilon_min=0.10,
        base_decay=0.997
    )

    # Simulate learning with good progress
    print("\n   Good Learning Scenario:")
    for ep in range(50):
        # Simulated improving coverage
        coverage = 0.3 + ep * 0.01  # Improving
        epsilon = scheduler.update(coverage)

        if ep % 10 == 0:
            stats = scheduler.get_stats()
            print(f"   Ep {ep:3d}: ε={epsilon:.4f}, "
                  f"Cov={coverage:.2%}, Mode={stats['mode']}")

    # Test 2: Struggling scenario
    print("\n2. Struggling Learning Scenario:")
    scheduler2 = AdaptiveEpsilonScheduler(
        epsilon_start=1.0,
        epsilon_min=0.10,
        base_decay=0.997
    )

    for ep in range(50):
        # Simulated poor/declining coverage
        coverage = 0.25 + np.random.uniform(-0.02, 0.01)  # Noisy, not improving
        epsilon = scheduler2.update(coverage)

        if ep % 10 == 0:
            stats = scheduler2.get_stats()
            print(f"   Ep {ep:3d}: ε={epsilon:.4f}, "
                  f"Cov={coverage:.2%}, Mode={stats['mode']}")

    # Test 3: Curriculum-aware scheduler
    print("\n3. Curriculum-Aware Scheduler:")
    curriculum_scheduler = CurriculumAwareEpsilonScheduler(
        epsilon_start=1.0,
        epsilon_min=0.10
    )

    # Phase 1
    print("\n   Phase 1 (Empty grids):")
    for ep in range(30):
        coverage = 0.4 + ep * 0.02  # Good progress
        epsilon = curriculum_scheduler.update_with_phase(
            coverage,
            phase_name="Phase1",
            phase_decay=0.993,
            phase_min=0.15
        )

        if ep % 10 == 0:
            stats = curriculum_scheduler.get_stats()
            print(f"   Ep {ep:3d}: ε={epsilon:.4f}, "
                  f"Cov={coverage:.2%}, Mode={stats['mode']}")

    # Phase 2
    print("\n   Phase 2 (Obstacles introduced):")
    for ep in range(30):
        coverage = 0.35 + np.random.uniform(-0.03, 0.01)  # Harder, struggling
        epsilon = curriculum_scheduler.update_with_phase(
            coverage,
            phase_name="Phase2",
            phase_decay=0.997,
            phase_min=0.20
        )

        if ep % 10 == 0:
            stats = curriculum_scheduler.get_stats()
            print(f"   Ep {ep:3d}: ε={epsilon:.4f}, "
                  f"Cov={coverage:.2%}, Mode={stats['mode']}")

    print("\n" + "=" * 80)
    print("✅ Adaptive Epsilon Scheduler Test Complete!")
    print("\nKey Features:")
    print("  • Automatically speeds up decay when learning well")
    print("  • Slows down decay when struggling (more exploration)")
    print("  • Integrates with curriculum phases")
    print("  • Prevents premature exploitation")
