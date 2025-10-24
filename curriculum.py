"""
Curriculum Manager

Manages 13-phase curriculum learning with mastery gates.
Each phase has its own epsilon decay strategy for optimal exploration.
"""

from typing import List, Optional
from data_structures import CurriculumPhase
from config import config


class CurriculumManager:
    """
    Manages curriculum learning phases.

    13 progressive phases + 3 consolidation cycles for overlearning.
    """

    def __init__(self):
        self.phases = self._initialize_phases()
        self.current_phase_idx = 0
        self.phase_transitions = []

    def _initialize_phases(self) -> List[CurriculumPhase]:
        """
        Initialize all 13 curriculum phases.

        Design principles:
            - Gradual difficulty increase
            - Overlearning on early phases
            - Interleaving of map types
            - Mastery gates (expected_coverage thresholds)
            - Phase-specific epsilon decay (fast for simple, slow for complex)
        """
        phases = [
            # Phase 1: Foundation - Pure open environments
            # Balanced decay (0.985) - slower to reach 70% coverage on empty grids
            CurriculumPhase(
                name="Phase1_Foundation_PureOpen",
                start_ep=0,
                end_ep=200,
                map_distribution={"empty": 1.0},
                expected_coverage=0.70,
                epsilon_floor=0.15,  # Higher floor - keep exploring to reach 70%
                epsilon_decay=config.EPSILON_DECAY_PHASE1  # 0.985 - balanced
            ),

            # Phase 2: Introduce random obstacles
            # Moderate decay (0.995) - need exploration for new dynamics
            CurriculumPhase(
                name="Phase2_IntroObstacles",
                start_ep=200,
                end_ep=350,
                map_distribution={"empty": 0.6, "random": 0.4},
                expected_coverage=0.70,
                epsilon_floor=0.23,
                epsilon_decay=config.EPSILON_DECAY_PHASE2  # 0.995 - moderate
            ),

            # Phase 3: More random obstacles
            # Slower decay (0.996) - more complexity requires more exploration
            CurriculumPhase(
                name="Phase3_MoreRandom",
                start_ep=350,
                end_ep=500,
                map_distribution={"random": 0.6, "empty": 0.4},
                expected_coverage=0.72,
                epsilon_floor=0.20,
                epsilon_decay=config.EPSILON_DECAY_PHASE3  # 0.996 - slower
            ),

            # Phase 4: Consolidation 1
            # Very slow decay (0.998) - focus on exploitation and refinement
            CurriculumPhase(
                name="Phase4_Consolidation1",
                start_ep=500,
                end_ep=550,
                map_distribution={"empty": 0.5, "random": 0.5},
                expected_coverage=0.75,
                epsilon_floor=0.18,
                epsilon_decay=config.EPSILON_DECAY_PHASE4  # 0.998 - very slow (consolidation)
            ),

            # Phase 5: Introduce rooms
            # Moderate decay (0.995) - new structure type needs exploration
            CurriculumPhase(
                name="Phase5_IntroRooms",
                start_ep=550,
                end_ep=700,
                map_distribution={"room": 0.4, "empty": 0.3, "random": 0.3},
                expected_coverage=0.72,
                epsilon_floor=0.18,
                epsilon_decay=config.EPSILON_DECAY_PHASE5  # 0.995 - moderate
            ),

            # Phase 6: More rooms
            # Slower decay (0.996) - room variations need exploration
            CurriculumPhase(
                name="Phase6_MoreRooms",
                start_ep=700,
                end_ep=850,
                map_distribution={"room": 0.55, "random": 0.25, "empty": 0.20},
                expected_coverage=0.75,
                epsilon_floor=0.17,
                epsilon_decay=config.EPSILON_DECAY_PHASE6  # 0.996 - slower
            ),

            # Phase 7: Consolidation 2
            # Very slow decay (0.998) - refine room-handling skills
            CurriculumPhase(
                name="Phase7_Consolidation2",
                start_ep=850,
                end_ep=925,
                map_distribution={"empty": 0.35, "random": 0.35, "room": 0.30},
                expected_coverage=0.78,
                epsilon_floor=0.17,
                epsilon_decay=config.EPSILON_DECAY_PHASE7  # 0.998 - very slow (consolidation)
            ),

            # Phase 8: Introduce corridors
            # Moderate decay (0.995) - narrow spaces need learning
            CurriculumPhase(
                name="Phase8_IntroCorridor",
                start_ep=925,
                end_ep=1075,
                map_distribution={"room": 0.45, "corridor": 0.25, "random": 0.20, "empty": 0.10},
                expected_coverage=0.73,
                epsilon_floor=0.16,
                epsilon_decay=config.EPSILON_DECAY_PHASE8  # 0.995 - moderate
            ),

            # Phase 9: Introduce caves
            # Moderate decay (0.995) - irregular shapes need exploration
            CurriculumPhase(
                name="Phase9_IntroCave",
                start_ep=1075,
                end_ep=1225,
                map_distribution={"room": 0.35, "cave": 0.25, "corridor": 0.20, "random": 0.20},
                expected_coverage=0.70,
                epsilon_floor=0.16,
                epsilon_decay=config.EPSILON_DECAY_PHASE9  # 0.995 - moderate
            ),

            # Phase 10: Consolidation 3
            # Very slow decay (0.998) - polish complex environment skills
            CurriculumPhase(
                name="Phase10_Consolidation3",
                start_ep=1225,
                end_ep=1300,
                map_distribution={"room": 0.40, "corridor": 0.25, "cave": 0.20, "random": 0.15},
                expected_coverage=0.72,
                epsilon_floor=0.16,
                epsilon_decay=config.EPSILON_DECAY_PHASE10  # 0.998 - very slow (consolidation)
            ),

            # Phase 11: Introduce L-shapes
            # Slower decay (0.996) - complex geometry needs careful exploration
            CurriculumPhase(
                name="Phase11_IntroLShape",
                start_ep=1300,
                end_ep=1450,
                map_distribution={"room": 0.30, "cave": 0.20, "lshape": 0.20, "corridor": 0.15, "random": 0.15},
                expected_coverage=0.68,
                epsilon_floor=0.15,
                epsilon_decay=config.EPSILON_DECAY_PHASE11  # 0.996 - slower
            ),

            # Phase 12: Complex mix
            # Very slow decay (0.997) - variety requires broad exploration
            CurriculumPhase(
                name="Phase12_ComplexMix",
                start_ep=1450,
                end_ep=1550,
                map_distribution={"room": 0.25, "cave": 0.20, "lshape": 0.20, "corridor": 0.20, "random": 0.15},
                expected_coverage=0.70,
                epsilon_floor=0.15,
                epsilon_decay=config.EPSILON_DECAY_PHASE12  # 0.997 - very slow
            ),

            # Phase 13: Final polish
            # Very slow decay (0.998) - pure refinement and exploitation
            CurriculumPhase(
                name="Phase13_FinalPolish",
                start_ep=1550,
                end_ep=1650,
                map_distribution={"room": 0.25, "empty": 0.20, "random": 0.20, "cave": 0.15, "corridor": 0.10, "lshape": 0.10},
                expected_coverage=0.72,
                epsilon_floor=0.15,
                epsilon_decay=config.EPSILON_DECAY_PHASE13  # 0.998 - very slow (final polish)
            ),
        ]

        return phases

    def get_current_phase(self, episode: int) -> CurriculumPhase:
        """Get the active phase for given episode."""
        for phase in self.phases:
            if phase.is_active(episode):
                return phase

        # If past all phases, return last phase
        return self.phases[-1]

    def get_map_type(self, episode: int) -> str:
        """Sample a map type for current episode."""
        phase = self.get_current_phase(episode)
        return phase.sample_map_type()

    def get_epsilon_floor(self, episode: int) -> float:
        """Get minimum epsilon for current phase."""
        phase = self.get_current_phase(episode)
        return phase.epsilon_floor

    def get_epsilon_decay(self, episode: int) -> float:
        """Get epsilon decay rate for current phase."""
        phase = self.get_current_phase(episode)
        return phase.epsilon_decay

    def check_mastery(self, episode: int, avg_coverage: float) -> bool:
        """
        Check if agent has achieved mastery for current phase.

        Args:
            episode: Current episode
            avg_coverage: Average coverage over recent episodes

        Returns:
            True if mastery achieved
        """
        phase = self.get_current_phase(episode)
        return avg_coverage >= phase.expected_coverage

    def should_advance(self, episode: int, avg_coverage: float) -> bool:
        """Check if should advance to next phase."""
        # Can only advance if at end of current phase and mastery achieved
        phase = self.get_current_phase(episode)
        at_phase_end = episode >= phase.end_ep - 1
        has_mastery = self.check_mastery(episode, avg_coverage)

        return at_phase_end and has_mastery

    def get_summary(self) -> str:
        """Get curriculum summary."""
        summary = "=" * 80 + "\n"
        summary += "CURRICULUM OVERVIEW (Phase-Specific Epsilon Decay)\n"
        summary += "=" * 80 + "\n"
        summary += f"{'Phase':<6} {'Episodes':<15} {'ε Decay':<10} {'Map Mix':<30} {'Target'}\n"
        summary += "-" * 80 + "\n"

        for i, phase in enumerate(self.phases, 1):
            ep_range = f"{phase.start_ep}-{phase.end_ep}"
            decay_str = f"{phase.epsilon_decay:.4f}"

            # Format map distribution (abbreviated for space)
            map_mix = ", ".join([f"{k[:3]}:{int(v*100)}%" for k, v in sorted(phase.map_distribution.items())])

            target = f"{int(phase.expected_coverage*100)}%+"

            summary += f"{i:<6} {ep_range:<15} {decay_str:<10} {map_mix:<30} {target}\n"

        summary += "=" * 80 + "\n"
        summary += "Strategy:\n"
        summary += "  - Fast decay (0.99): Phase 1 - Learn empty grid quickly\n"
        summary += "  - Moderate decay (0.995-0.996): New environment types - Explore dynamics\n"
        summary += "  - Slow decay (0.997-0.998): Consolidation phases - Refine & exploit\n"
        summary += "=" * 80

        return summary


if __name__ == "__main__":
    # Test curriculum manager
    manager = CurriculumManager()

    print(manager.get_summary())
    print("\n\nPhase transitions:")

    for ep in [0, 200, 500, 1000, 1500, 1600]:
        phase = manager.get_current_phase(ep)
        map_type = manager.get_map_type(ep)
        print(f"  Episode {ep:4d}: {phase.name:30s} -> {map_type}")
