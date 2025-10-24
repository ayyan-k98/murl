"""
Configuration Presets for GAT-MARL Coverage

Provides predefined configurations for different use cases:
- baseline: Original proven configuration
- fast: Quick training for debugging
- stable: Conservative settings for reliability
- aggressive: High performance but may be unstable
- probabilistic: For probabilistic coverage environment
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any
import torch


@dataclass
class ConfigPreset:
    """Base configuration dataclass."""

    # Environment
    GRID_SIZE: int = 20
    SENSOR_RANGE: float = 5.0
    COMM_RANGE: float = 10.0
    NUM_RAYS: int = 12
    SAMPLES_PER_RAY: int = 8
    MAX_EPISODE_STEPS: int = 350
    USE_PROBABILISTIC_ENV: bool = False

    # Agent
    N_ACTIONS: int = 9

    # Learning
    LEARNING_RATE: float = 5e-4
    LEARNING_RATE_MIN: float = 1e-5
    LR_DECAY_RATE: float = 0.9995
    GAMMA: float = 0.99
    BATCH_SIZE: int = 128
    REPLAY_BUFFER_SIZE: int = 50000
    TARGET_UPDATE_FREQ: int = 100
    MIN_REPLAY_SIZE: int = 500
    TRAIN_FREQ: int = 2
    N_STEP: int = 3
    N_STEP_ENABLED: bool = True

    # Exploration
    EPSILON_START: float = 1.0
    EPSILON_MIN: float = 0.05
    EPSILON_DECAY_RATE: float = 0.99

    # Phase-specific epsilon decay
    EPSILON_DECAY_PHASE1: float = 0.985
    EPSILON_DECAY_PHASE2: float = 0.995
    EPSILON_DECAY_PHASE3: float = 0.996
    EPSILON_DECAY_PHASE4: float = 0.998
    EPSILON_DECAY_PHASE5: float = 0.995
    EPSILON_DECAY_PHASE6: float = 0.996
    EPSILON_DECAY_PHASE7: float = 0.998
    EPSILON_DECAY_PHASE8: float = 0.995
    EPSILON_DECAY_PHASE9: float = 0.995
    EPSILON_DECAY_PHASE10: float = 0.998
    EPSILON_DECAY_PHASE11: float = 0.996
    EPSILON_DECAY_PHASE12: float = 0.997
    EPSILON_DECAY_PHASE13: float = 0.998

    # GAT Architecture
    GAT_HIDDEN_DIM: int = 128
    GAT_N_LAYERS: int = 3
    GAT_N_HEADS: int = 4
    GAT_DROPOUT: float = 0.1
    NODE_FEATURE_DIM: int = 12
    AGENT_FEATURE_DIM: int = 10

    # Rewards
    COVERAGE_REWARD: float = 10.0
    EXPLORATION_REWARD: float = 0.5
    FRONTIER_BONUS: float = 0.05
    FRONTIER_CAP: float = 1.5
    COLLISION_PENALTY: float = -2.0
    STEP_PENALTY: float = -0.01
    STAY_PENALTY: float = -0.1
    PROBABILISTIC_REWARD_SCALE: float = 0.15

    # Gradient Stability
    GRAD_CLIP_THRESHOLD: float = 1.0
    AGC_CLIP_RATIO: float = 0.01
    AGC_EPS: float = 1e-3
    EXPLOSION_THRESHOLD: float = 500.0
    MAX_GRAD_NORM: float = 200.0

    # Training
    STAGE1_EPISODES: int = 1600
    VALIDATION_INTERVAL: int = 50
    VALIDATION_EPISODES: int = 10
    CHECKPOINT_INTERVAL: int = 100

    # Performance
    ENABLE_TIMING_BREAKDOWN: bool = False
    GRADIENT_ACCUMULATION_STEPS: int = 1
    FAST_VALIDATION: bool = True
    VALIDATION_MAX_STEPS: int = 200
    USE_AMP: bool = False
    PIN_MEMORY: bool = True
    NUM_WORKERS: int = 0
    PERSISTENT_WORKERS: bool = False
    COMPILE_MODEL: bool = False
    COMPILE_MODE: str = "default"

    # Paths
    CHECKPOINT_DIR: str = "./checkpoints"
    RESULTS_DIR: str = "./results"

    # Device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    VERBOSE: bool = True
    LOG_INTERVAL: int = 10
    LOG_INVALID_ACTIONS: bool = True
    LOG_STAY_RATE: bool = True
    LOG_SPATIAL_STATS: bool = True

    # Action deltas
    ACTION_DELTAS: tuple = (
        (0, -1),   # N
        (1, -1),   # NE
        (1, 0),    # E
        (1, 1),    # SE
        (0, 1),    # S
        (-1, 1),   # SW
        (-1, 0),   # W
        (-1, -1),  # NW
        (0, 0)     # STAY
    )

    # Action names
    ACTION_NAMES: tuple = ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'STAY')

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class BaselineConfig(ConfigPreset):
    """Baseline configuration - proven stable settings."""
    pass


class FastConfig(ConfigPreset):
    """Fast training for debugging and quick iteration."""

    MAX_EPISODE_STEPS: int = 200  # Shorter episodes
    STAGE1_EPISODES: int = 500  # Fewer episodes
    BATCH_SIZE: int = 64  # Smaller batches
    REPLAY_BUFFER_SIZE: int = 10000  # Smaller buffer
    TARGET_UPDATE_FREQ: int = 50  # More frequent updates
    VALIDATION_INTERVAL: int = 100  # Less frequent validation
    GAT_HIDDEN_DIM: int = 96  # Smaller network
    GAT_N_LAYERS: int = 2  # Fewer layers
    ENABLE_TIMING_BREAKDOWN: bool = True


class StableConfig(ConfigPreset):
    """Stable configuration - conservative settings for reliability."""

    LEARNING_RATE: float = 3e-4  # Lower LR
    EPSILON_DECAY_RATE: float = 0.998  # Slower exploration decay
    TARGET_UPDATE_FREQ: int = 200  # Less frequent target updates
    BATCH_SIZE: int = 64  # Smaller batches for stability
    GRAD_CLIP_THRESHOLD: float = 0.5  # Tighter clipping
    AGC_CLIP_RATIO: float = 0.01  # Tighter AGC
    STAGE1_EPISODES: int = 2000  # More training time
    EPSILON_MIN: float = 0.10  # Higher minimum exploration


class AggressiveConfig(ConfigPreset):
    """Aggressive configuration - high performance but may be unstable."""

    LEARNING_RATE: float = 1e-3  # Higher LR
    BATCH_SIZE: int = 256  # Larger batches
    TARGET_UPDATE_FREQ: int = 50  # More frequent updates
    TRAIN_FREQ: int = 1  # Train every step
    GAT_HIDDEN_DIM: int = 192  # Larger network
    GAT_N_LAYERS: int = 4  # More layers
    N_STEP: int = 5  # Longer n-step returns
    GRAD_CLIP_THRESHOLD: float = 2.0  # Looser clipping
    MAX_EPISODE_STEPS: int = 500  # Longer episodes


class ProbabilisticConfig(ConfigPreset):
    """Configuration optimized for probabilistic coverage environment."""

    USE_PROBABILISTIC_ENV: bool = True
    PROBABILISTIC_REWARD_SCALE: float = 0.15
    COVERAGE_REWARD: float = 10.0
    EPSILON_DECAY_RATE: float = 0.997  # Slower decay for dense rewards
    LEARNING_RATE: float = 3e-4  # More conservative
    STAGE1_EPISODES: int = 2000  # More training time


class ImprovedConfig(ConfigPreset):
    """Improved configuration based on performance analysis."""

    # Smaller network for faster training
    GAT_HIDDEN_DIM: int = 96
    GAT_N_LAYERS: int = 2

    # Slower epsilon decay for better exploration
    EPSILON_DECAY_RATE: float = 0.998
    EPSILON_DECAY_PHASE1: float = 0.990  # Faster initial learning
    EPSILON_DECAY_PHASE2: float = 0.996
    EPSILON_DECAY_PHASE3: float = 0.997
    EPSILON_DECAY_PHASE4: float = 0.998

    # Larger replay buffer
    REPLAY_BUFFER_SIZE: int = 75000

    # More training episodes
    STAGE1_EPISODES: int = 2000

    # Better reward balance
    COVERAGE_REWARD: float = 5.0  # Reduced
    EXPLORATION_REWARD: float = 1.0  # Increased
    FRONTIER_BONUS: float = 0.15  # Increased
    FRONTIER_CAP: float = 3.0


# Preset registry
PRESETS = {
    "baseline": BaselineConfig,
    "fast": FastConfig,
    "stable": StableConfig,
    "aggressive": AggressiveConfig,
    "probabilistic": ProbabilisticConfig,
    "improved": ImprovedConfig
}


def get_config(preset_name: str = "baseline") -> ConfigPreset:
    """
    Get configuration by preset name.

    Args:
        preset_name: Name of preset ("baseline", "fast", "stable", etc.)

    Returns:
        Configuration instance

    Example:
        config = get_config("fast")
        print(config.LEARNING_RATE)
    """
    if preset_name not in PRESETS:
        raise ValueError(
            f"Unknown preset '{preset_name}'. "
            f"Available presets: {list(PRESETS.keys())}"
        )
    return PRESETS[preset_name]()


def print_preset_comparison():
    """Print comparison of all presets."""
    print("=" * 100)
    print("CONFIGURATION PRESET COMPARISON")
    print("=" * 100)

    # Key parameters to compare
    keys = [
        "LEARNING_RATE",
        "BATCH_SIZE",
        "GAT_HIDDEN_DIM",
        "GAT_N_LAYERS",
        "EPSILON_DECAY_RATE",
        "TARGET_UPDATE_FREQ",
        "STAGE1_EPISODES",
        "MAX_EPISODE_STEPS",
        "REPLAY_BUFFER_SIZE"
    ]

    # Get all configs
    configs = {name: cls() for name, cls in PRESETS.items()}

    # Print header
    print(f"{'Parameter':<25}", end="")
    for name in configs.keys():
        print(f"{name:<15}", end="")
    print()
    print("-" * 100)

    # Print each parameter
    for key in keys:
        print(f"{key:<25}", end="")
        for config in configs.values():
            value = getattr(config, key)
            if isinstance(value, float):
                print(f"{value:<15.2e}", end="")
            else:
                print(f"{value:<15}", end="")
        print()

    print("=" * 100)

    # Print use cases
    print("\nUse Cases:")
    print("  baseline:       Proven stable configuration (default)")
    print("  fast:           Quick iteration and debugging")
    print("  stable:         Maximum reliability, slower learning")
    print("  aggressive:     High performance, may be unstable")
    print("  probabilistic:  Optimized for probabilistic coverage")
    print("  improved:       Based on performance analysis")
    print("=" * 100)


if __name__ == "__main__":
    # Test presets
    print("Testing Configuration Presets\n")

    # Test loading
    baseline = get_config("baseline")
    fast = get_config("fast")

    print(f"✓ Baseline LR: {baseline.LEARNING_RATE}")
    print(f"✓ Fast LR: {fast.LEARNING_RATE}")
    print(f"✓ Fast episodes: {fast.STAGE1_EPISODES} (vs baseline {baseline.STAGE1_EPISODES})")

    # Print comparison
    print("\n")
    print_preset_comparison()

    # Test dictionary conversion
    print("\n✓ Dictionary conversion test:")
    config_dict = baseline.to_dict()
    print(f"  Keys: {len(config_dict)}")
    print(f"  Sample: LR={config_dict['LEARNING_RATE']}")

    print("\n✅ Configuration Presets Test Complete!")
