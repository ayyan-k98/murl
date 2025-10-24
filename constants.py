"""
Constants for GAT-MARL Coverage System

Extracted magic numbers and common constants for better maintainability.
"""

# Grid and Sensing
DEFAULT_GRID_SIZE = 20
NEIGHBOR_COUNT = 8  # 8-connected grid
MAX_DIAGONAL_CELLS = 28  # For 20x20 grid (sqrt(20^2 + 20^2))

# Gradient Thresholds
GRADIENT_EXPLOSION_THRESHOLD = 500.0
MAX_GRADIENT_NORM = 200.0

# Normalization Constants
MAX_VISIT_COUNT_NORM = 10.0  # Cap visits at 10 for normalization
MAX_RECENCY_STEPS = 100.0  # Normalize recency by 100 steps

# Coverage Thresholds
BINARY_COVERAGE_THRESHOLD = 0.5  # Threshold for binary coverage determination
HIGH_COVERAGE_THRESHOLD = 0.95  # Early termination threshold
LATE_GAME_COVERAGE = 0.7  # Threshold for adaptive penalties

# Episode Milestones
TIMING_BREAKDOWN_EPISODES = 5  # Only show timing for first N episodes
WARMUP_EPISODES = 10  # Initial episodes for warmup

# Action Constants
STAY_ACTION_INDEX = 8
DIAGONAL_ACTIONS = [1, 3, 5, 7]  # NE, SE, SW, NW

# Probabilistic Coverage
SIGMOID_STEEPNESS_DEFAULT = 2.0
