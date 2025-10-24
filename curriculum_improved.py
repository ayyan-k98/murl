"""
Improved Curriculum Learning

Key improvements:
1. Easier starting phase (50% target, not 70%)
2. More gradual progression
3. Longer phases (more time to master)
4. Adaptive phase advancement (based on actual performance)
5. Grace period (don't advance if struggling)
"""

from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class CurriculumPhase:
    """Definition of a curriculum phase."""
    name: str
    episode_start: int
    episode_end: int
    map_distribution: dict  # Map type -> probability
    expected_coverage: float  # Target coverage to advance
    epsilon_floor: float  # Minimum epsilon for this phase
    description: str


class ImprovedCurriculumScheduler:
    """
    Improved curriculum with easier start and adaptive advancement.
    """

    def __init__(self, total_episodes: int = 2000):
        self.total_episodes = total_episodes
        self.current_phase_idx = 0
        self.phases = self._create_improved_curriculum()

        # Adaptive advancement tracking
        self.phase_entry_episode = 0
        self.grace_period = 100  # Don't advance in first 100 episodes of phase
        self.advancement_threshold = 0.8  # Need 80% of target to advance

    def _create_improved_curriculum(self) -> List[CurriculumPhase]:
        """
        Create IMPROVED 15-phase curriculum (was 13).

        Key changes:
        - Start at 50% target (not 70%)
        - More gradual progression
        - Longer early phases
        - Consolidation phases every 3 phases (not 4)
        """
        phases = []

        # ============ PHASE 1-3: Foundation (Easier!) ============
        # IMPROVEMENT: Much easier start!
        phases.append(CurriculumPhase(
            name="Phase 1: Empty Maps (Easy)",
            episode_start=0,
            episode_end=200,  # Longer!
            map_distribution={"empty": 1.0},
            expected_coverage=0.50,  # Was 0.70! Much easier
            epsilon_floor=0.3,
            description="Learn basic movement and sensing"
        ))

        phases.append(CurriculumPhase(
            name="Phase 2: Sparse Obstacles",
            episode_start=200,
            episode_end=350,  # Longer!
            map_distribution={"empty": 0.5, "random": 0.5},
            expected_coverage=0.55,  # Was 0.72
            epsilon_floor=0.25,
            description="Introduce simple obstacles (10-15%)"
        ))

        phases.append(CurriculumPhase(
            name="Phase 3: More Obstacles",
            episode_start=350,
            episode_end=500,
            map_distribution={"random": 1.0},
            expected_coverage=0.60,  # Was 0.73
            epsilon_floor=0.2,
            description="More obstacles (20-25%)"
        ))

        # ============ PHASE 4: Consolidation 1 ============
        phases.append(CurriculumPhase(
            name="Phase 4: Consolidation I",
            episode_start=500,
            episode_end=600,
            map_distribution={"empty": 0.3, "random": 0.7},
            expected_coverage=0.63,
            epsilon_floor=0.18,
            description="Practice on mixed difficulty"
        ))

        # ============ PHASE 5-7: Intermediate ============
        phases.append(CurriculumPhase(
            name="Phase 5: Structured Environments",
            episode_start=600,
            episode_end=750,
            map_distribution={"random": 0.5, "room": 0.5},
            expected_coverage=0.65,
            epsilon_floor=0.15,
            description="Introduce rooms and structure"
        ))

        phases.append(CurriculumPhase(
            name="Phase 6: Corridors",
            episode_start=750,
            episode_end=900,
            map_distribution={"room": 0.4, "corridor": 0.6},
            expected_coverage=0.67,
            epsilon_floor=0.12,
            description="Learn corridor navigation"
        ))

        phases.append(CurriculumPhase(
            name="Phase 7: Complex Rooms",
            episode_start=900,
            episode_end=1050,
            map_distribution={"room": 0.5, "corridor": 0.3, "lshape": 0.2},
            expected_coverage=0.68,
            epsilon_floor=0.10,
            description="Multi-room environments"
        ))

        # ============ PHASE 8: Consolidation 2 ============
        phases.append(CurriculumPhase(
            name="Phase 8: Consolidation II",
            episode_start=1050,
            episode_end=1150,
            map_distribution={"random": 0.3, "room": 0.4, "corridor": 0.3},
            expected_coverage=0.69,
            epsilon_floor=0.10,
            description="Practice structured environments"
        ))

        # ============ PHASE 9-11: Advanced ============
        phases.append(CurriculumPhase(
            name="Phase 9: Caves",
            episode_start=1150,
            episode_end=1300,
            map_distribution={"corridor": 0.3, "cave": 0.4, "lshape": 0.3},
            expected_coverage=0.70,
            epsilon_floor=0.08,
            description="Irregular cave environments"
        ))

        phases.append(CurriculumPhase(
            name="Phase 10: L-Shapes",
            episode_start=1300,
            episode_end=1450,
            map_distribution={"lshape": 0.5, "cave": 0.3, "corridor": 0.2},
            expected_coverage=0.71,
            epsilon_floor=0.07,
            description="Complex L-shaped maps"
        ))

        phases.append(CurriculumPhase(
            name="Phase 11: Mixed Complex",
            episode_start=1450,
            episode_end=1600,
            map_distribution={"cave": 0.3, "lshape": 0.3, "corridor": 0.4},
            expected_coverage=0.72,
            epsilon_floor=0.06,
            description="All complex environments"
        ))

        # ============ PHASE 12: Consolidation 3 ============
        phases.append(CurriculumPhase(
            name="Phase 12: Consolidation III",
            episode_start=1600,
            episode_end=1700,
            map_distribution={
                "random": 0.2, "room": 0.2, "corridor": 0.2,
                "cave": 0.2, "lshape": 0.2
            },
            expected_coverage=0.73,
            epsilon_floor=0.05,
            description="All environments mixed"
        ))

        # ============ PHASE 13-15: Mastery ============
        phases.append(CurriculumPhase(
            name="Phase 13: Hard Mix I",
            episode_start=1700,
            episode_end=1800,
            map_distribution={
                "corridor": 0.25, "cave": 0.25,
                "lshape": 0.25, "room": 0.25
            },
            expected_coverage=0.74,
            epsilon_floor=0.05,
            description="Focus on harder maps"
        ))

        phases.append(CurriculumPhase(
            name="Phase 14: Hard Mix II",
            episode_start=1800,
            episode_end=1900,
            map_distribution={
                "cave": 0.33, "lshape": 0.33, "corridor": 0.34
            },
            expected_coverage=0.75,
            epsilon_floor=0.05,
            description="Only complex environments"
        ))

        phases.append(CurriculumPhase(
            name="Phase 15: Final Challenge",
            episode_start=1900,
            episode_end=2000,
            map_distribution={
                "empty": 0.1, "random": 0.1, "room": 0.1,
                "corridor": 0.2, "cave": 0.25, "lshape": 0.25
            },
            expected_coverage=0.76,
            epsilon_floor=0.05,
            description="Full curriculum, all difficulty levels"
        ))

        return phases

    def get_current_phase(self, episode: int) -> CurriculumPhase:
        """Get current curriculum phase."""
        for phase in self.phases:
            if phase.episode_start <= episode < phase.episode_end:
                return phase
        # If beyond last phase, return last phase
        return self.phases[-1]

    def should_advance_phase(self,
                            episode: int,
                            recent_coverage: float,
                            phase: CurriculumPhase) -> bool:
        """
        ADAPTIVE phase advancement based on actual performance.

        Don't advance if:
        1. Still in grace period (first 100 episodes of phase)
        2. Performance below threshold (< 80% of target)
        """
        # Check if in grace period
        episodes_in_phase = episode - phase.episode_start
        if episodes_in_phase < self.grace_period:
            return False

        # Check if performance is good enough
        performance_ratio = recent_coverage / phase.expected_coverage
        if performance_ratio < self.advancement_threshold:
            return False

        return True

    def get_phase_info(self, episode: int) -> dict:
        """Get detailed info about current phase."""
        phase = self.get_current_phase(episode)
        episodes_in_phase = episode - phase.episode_start
        phase_progress = episodes_in_phase / (phase.episode_end - phase.episode_start)

        return {
            'phase_name': phase.name,
            'phase_number': self.phases.index(phase) + 1,
            'total_phases': len(self.phases),
            'episode_range': (phase.episode_start, phase.episode_end),
            'episodes_in_phase': episodes_in_phase,
            'phase_progress': phase_progress,
            'target_coverage': phase.expected_coverage,
            'epsilon_floor': phase.epsilon_floor,
            'map_distribution': phase.map_distribution,
            'description': phase.description
        }


# Test code
if __name__ == "__main__":
    print("Improved Curriculum Learning")
    print("=" * 80)

    scheduler = ImprovedCurriculumScheduler(total_episodes=2000)

    print("\n1. Curriculum Overview:")
    print("-" * 80)
    for i, phase in enumerate(scheduler.phases):
        print(f"\n{phase.name}")
        print(f"  Episodes: {phase.episode_start}-{phase.episode_end} ({phase.episode_end - phase.episode_start} eps)")
        print(f"  Target: {phase.expected_coverage:.1%}")
        print(f"  Maps: {phase.map_distribution}")
        print(f"  {phase.description}")

    print("\n" + "=" * 80)
    print("\n2. Key Improvements:")
    print("   ✓ Phase 1: 50% target (was 70%) - MUCH easier start!")
    print("   ✓ Longer early phases (200 eps vs 100)")
    print("   ✓ More gradual progression (5% steps, not 10%)")
    print("   ✓ 15 phases (was 13) - more granular")
    print("   ✓ Adaptive advancement (don't move up if struggling)")
    print("   ✓ Grace period (100 eps to master each phase)")

    print("\n3. Expected Learning Curve:")
    print("-" * 60)
    print(f"{'Episode':<15} | {'Phase':<10} | {'Target':<15}")
    print("-" * 60)
    for ep in [0, 200, 500, 750, 1000, 1300, 1600, 1900]:
        phase = scheduler.get_current_phase(ep)
        phase_num = scheduler.phases.index(phase) + 1
        print(f"{ep:<15} | {phase_num:<10} | {phase.expected_coverage:.1%}")

    print("\n" + "=" * 80)
    print("✅ Improved Curriculum Complete!")