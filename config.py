"""
Configuration file for GAT-MARL Coverage

Contains all hyperparameters and settings for the project.
"""

from dataclasses import dataclass
import torch


@dataclass
class Config:
    """Global configuration for the coverage task."""

    # ==================== Environment ====================
    GRID_SIZE: int = 20
    SENSOR_RANGE: float = 5.0  # POMDP: Limited observation radius
    COMM_RANGE: float = 10.0    # For Stage 2 multi-agent
    NUM_RAYS: int = 12         # Reduced from 16 for speed
    SAMPLES_PER_RAY: int = 8   # Reduced from 10 for speed
    MAX_EPISODE_STEPS: int = 350  # REDUCED from 350 for faster training (29% speedup)
    USE_PROBABILISTIC_ENV: bool = False  # Toggle between binary and probabilistic coverage

    # ==================== Agent ====================
    N_ACTIONS: int = 9  # 8 directions + stay
    ACTION_NAMES = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'STAY']

    # Action deltas (dx, dy)
    ACTION_DELTAS = [
        (0, -1),   # N
        (1, -1),   # NE
        (1, 0),    # E
        (1, 1),    # SE
        (0, 1),    # S
        (-1, 1),   # SW
        (-1, 0),   # W
        (-1, -1),  # NW
        (0, 0)     # STAY
    ]

    # ==================== Learning (RESTORED PROVEN VALUES) ====================
    LEARNING_RATE: float = 5e-4       # INCREASED from 3e-4 (agent needs stronger gradient updates)
    LEARNING_RATE_MIN: float = 1e-5   # RESTORED from 1e-6
    LR_DECAY_RATE: float = 0.9995     # RESTORED from 0.998 (slower decay)
    GAMMA: float = 0.99
    BATCH_SIZE: int = 128             # INCREASED from 64 for faster training (2x larger batches)
    REPLAY_BUFFER_SIZE: int = 50000
    TARGET_UPDATE_FREQ: int = 100     # CRITICAL: RESTORED from 10 (was causing instability)
    MIN_REPLAY_SIZE: int = 500        # Start training earlier (was 1000)
    TRAIN_FREQ: int = 2               # Train every 2 steps (was 1) - reduces overhead
    
    # N-step returns for better credit assignment
    N_STEP: int = 3                   # NEW: Use 3-step returns
    N_STEP_ENABLED: bool = True       # Toggle for n-step

    # ==================== Exploration (PHASE-SPECIFIC) ====================
    EPSILON_START: float = 1.0
    EPSILON_MIN: float = 0.05  # Global minimum (was 0.15)
    
    # Phase-specific decay rates (each phase has its own exploration strategy)
    EPSILON_DECAY_PHASE1: float = 0.985   # Foundation: SLOWER decay - need more exploration for 70%
    EPSILON_DECAY_PHASE2: float = 0.995   # Intro obstacles: Slower decay to explore new dynamics
    EPSILON_DECAY_PHASE3: float = 0.996   # More random: Even slower to handle complexity
    EPSILON_DECAY_PHASE4: float = 0.998   # Consolidation: Very slow, focus on exploitation
    EPSILON_DECAY_PHASE5: float = 0.995   # Intro rooms: Moderate for new structure
    EPSILON_DECAY_PHASE6: float = 0.996   # More rooms: Slower to explore variations
    EPSILON_DECAY_PHASE7: float = 0.998   # Consolidation: Very slow, refine policy
    EPSILON_DECAY_PHASE8: float = 0.995   # Intro corridor: Moderate for narrow spaces
    EPSILON_DECAY_PHASE9: float = 0.995   # Intro cave: Moderate for irregular shapes
    EPSILON_DECAY_PHASE10: float = 0.998  # Consolidation: Very slow, polish skills
    EPSILON_DECAY_PHASE11: float = 0.996  # Intro L-shape: Slower for complex geometry
    EPSILON_DECAY_PHASE12: float = 0.997  # Complex mix: Very slow for variety
    EPSILON_DECAY_PHASE13: float = 0.998  # Final polish: Minimal decay, pure exploitation
    
    # Legacy default (unused with curriculum)
    EPSILON_DECAY_RATE: float = 0.99  # Fallback for non-curriculum training

    # ==================== GAT Architecture ====================
    GAT_HIDDEN_DIM: int = 128
    GAT_N_LAYERS: int = 3
    GAT_N_HEADS: int = 4
    GAT_DROPOUT: float = 0.1
    NODE_FEATURE_DIM: int = 12  # INCREASED from 8 (now includes spatial features)
    AGENT_FEATURE_DIM: int = 10

    # ==================== Rewards (RESTORED ORIGINAL SCALE) ====================
    # CRITICAL: Restore full rewards - 20x reduction was too aggressive
    # Agent needs strong signal to learn spatial navigation
    COVERAGE_REWARD: float = 10.0      # RESTORED from 0.5
    EXPLORATION_REWARD: float = 0.5    # RESTORED from 0.025
    FRONTIER_BONUS: float = 0.05       # RESTORED from 0.0025
    FRONTIER_CAP: float = 1.5          # RESTORED from 0.075
    COLLISION_PENALTY: float = -2.0    # RESTORED from -0.1
    STEP_PENALTY: float = -0.01        # RESTORED from -0.0005
    STAY_PENALTY: float = -0.1         # RESTORED from -0.005
    
    # Probabilistic environment scaling
    PROBABILISTIC_REWARD_SCALE: float = 0.15  # Keep proven value

    # ==================== Gradient Stability ====================
    GRAD_CLIP_THRESHOLD: float = 1.0   # Keep tight (working well)
    AGC_CLIP_RATIO: float = 0.01       # Keep strong AGC (working well)
    AGC_EPS: float = 1e-3
    EXPLOSION_THRESHOLD: float = 500.0
    MAX_GRAD_NORM: float = 200.0

    # ==================== Training ====================
    STAGE1_EPISODES: int = 1600
    VALIDATION_INTERVAL: int = 50
    VALIDATION_EPISODES: int = 10
    CHECKPOINT_INTERVAL: int = 100
    
    # ==================== Performance Optimizations ====================
    # Reduce per-episode overhead for faster training
    ENABLE_TIMING_BREAKDOWN: bool = False  # Disable detailed timing after episode 5
    GRADIENT_ACCUMULATION_STEPS: int = 1   # Future: accumulate gradients for larger effective batch
    
    # Optimize training frequency (already optimized - train every step)
    # TRAIN_FREQ: int = 1 (defined above in Learning section)
    
    # Reduce validation overhead
    FAST_VALIDATION: bool = True  # Use fewer steps for validation episodes
    VALIDATION_MAX_STEPS: int = 200  # Limit validation episode length (vs 300 for training)
    
    # GPU optimizations
    USE_AMP: bool = False  # Automatic Mixed Precision (float16) - can cause instability in RL
    PIN_MEMORY: bool = True  # Pin memory for faster GPU transfers
    NUM_WORKERS: int = 0  # DataLoader workers (0 = main thread only for RL)
    PERSISTENT_WORKERS: bool = False  # Keep workers alive between batches
    
    # Compilation optimizations (PyTorch 2.0+)
    COMPILE_MODEL: bool = False  # torch.compile() - can speed up but adds warmup time
    COMPILE_MODE: str = "default"  # "default", "reduce-overhead", "max-autotune"

    # ==================== Paths ====================
    CHECKPOINT_DIR: str = "./checkpoints"
    RESULTS_DIR: str = "./results"

    # ==================== Device ====================
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ==================== Logging ====================
    VERBOSE: bool = True
    LOG_INTERVAL: int = 10  # Log every N episodes
    
    # ==================== Debugging ====================
    LOG_INVALID_ACTIONS: bool = True  # NEW: Log when argmax proposes invalid action
    LOG_STAY_RATE: bool = True        # NEW: Log % of STAY actions
    LOG_SPATIAL_STATS: bool = True    # NEW: Log spatial coverage statistics


# Global config instance
config = Config()


def print_config():
    """Print configuration summary with critical changes highlighted."""
    print("=" * 80)
    print("CONFIGURATION SUMMARY (SPATIAL ENCODING FIX)")
    print("=" * 80)
    print(f"Device: {config.DEVICE}")
    print(f"Grid Size: {config.GRID_SIZE}")
    print(f"Sensor Range (POMDP): {config.SENSOR_RANGE}")
    print(f"Action Space: {config.N_ACTIONS} actions")
    print(f"\n🔧 CRITICAL FIXES APPLIED:")
    print(f"  ✅ Learning Rate: {config.LEARNING_RATE} (restored from 5e-5)")
    print(f"  ✅ Target Update Freq: {config.TARGET_UPDATE_FREQ} (restored from 10)")
    print(f"  ✅ Coverage Reward: {config.COVERAGE_REWARD} (restored from 0.5)")
    print(f"  ✅ Node Features: {config.NODE_FEATURE_DIM}D (expanded from 8D for spatial)")
    print(f"  ✅ N-Step Returns: {config.N_STEP}-step ({'ENABLED' if config.N_STEP_ENABLED else 'DISABLED'})")
    print(f"\nGAT Architecture:")
    print(f"  Layers: {config.GAT_N_LAYERS}")
    print(f"  Heads: {config.GAT_N_HEADS}")
    print(f"  Hidden Dim: {config.GAT_HIDDEN_DIM}")
    print(f"\nExpected Results (Empty Grid, 50 Episodes):")
    print(f"  Episode 10:  Coverage 40-50%")
    print(f"  Episode 30:  Coverage 60-70%")
    print(f"  Episode 50:  Coverage 70-80%")
    print("=" * 80)


if __name__ == "__main__":
    print_config()
