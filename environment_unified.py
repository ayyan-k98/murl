"""
Unified Coverage Environment

Consolidates baseline, improved, and probabilistic environments into a single class
with feature flags for different modes.

Modes:
- Binary coverage (baseline)
- Improved reward shaping
- Probabilistic coverage

Usage:
    # Baseline mode
    env = CoverageEnvironment(mode="baseline")

    # Improved rewards
    env = CoverageEnvironment(mode="improved")

    # Probabilistic coverage
    env = CoverageEnvironment(mode="probabilistic")
"""

import math
import random
from typing import Tuple, Dict, Set, Optional
import numpy as np

from config import config
from data_structures import RobotState, WorldState
from map_generator import MapGenerator
from rewards import (
    RewardCalculator,
    get_baseline_calculator,
    get_improved_calculator,
    get_probabilistic_calculator
)
from constants import (
    BINARY_COVERAGE_THRESHOLD,
    HIGH_COVERAGE_THRESHOLD,
    SIGMOID_STEEPNESS_DEFAULT
)


class CoverageEnvironment:
    """
    Unified coverage environment with multiple modes.

    Supports partial observability (POMDP) via ray-casting sensors.
    """

    def __init__(self,
                 grid_size: int = 20,
                 sensor_range: float = 5.0,
                 map_type: str = "empty",
                 mode: str = "baseline"):
        """
        Initialize environment.

        Args:
            grid_size: Grid dimensions
            sensor_range: POMDP sensor range
            map_type: Map type (empty, random, room, etc.)
            mode: Environment mode ("baseline", "improved", "probabilistic")
        """
        self.grid_size = grid_size
        self.sensor_range = sensor_range
        self.map_type = map_type
        self.mode = mode

        # Validate mode
        if mode not in ["baseline", "improved", "probabilistic"]:
            raise ValueError(
                f"Unknown mode '{mode}'. "
                f"Available: 'baseline', 'improved', 'probabilistic'"
            )

        self.map_generator = MapGenerator(grid_size)

        # Reward calculator (mode-specific)
        if mode == "baseline":
            self.reward_calculator = get_baseline_calculator()
        elif mode == "improved":
            self.reward_calculator = get_improved_calculator()
        elif mode == "probabilistic":
            self.reward_calculator = get_probabilistic_calculator()

        # World state (ground truth)
        self.world_state: Optional[WorldState] = None

        # Robot state (POMDP)
        self.robot_state: Optional[RobotState] = None

        # Episode tracking
        self.steps = 0
        self.max_steps = config.MAX_EPISODE_STEPS

        # Probabilistic coverage (if enabled)
        self.coverage_map_prob = None
        if mode == "probabilistic":
            self.coverage_map_prob = np.zeros((grid_size, grid_size), dtype=np.float32)
            self.sigmoid_r0 = sensor_range / 2.0
            self.sigmoid_k = SIGMOID_STEEPNESS_DEFAULT

        # Tracking for rewards
        self.prev_coverage_pct = 0.0

    def reset(self, map_type: Optional[str] = None) -> RobotState:
        """
        Reset environment for new episode.

        Args:
            map_type: Override default map type

        Returns:
            Initial robot state
        """
        if map_type is not None:
            self.map_type = map_type

        # Generate map
        graph, obstacles = self.map_generator.generate(self.map_type)

        # Initialize world state
        coverage_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.world_state = WorldState(
            grid_size=self.grid_size,
            graph=graph,
            obstacles=obstacles,
            coverage_map=coverage_map,
            map_type=self.map_type
        )

        # Find valid starting position
        start_pos = self._find_valid_start_position()

        # Initialize robot state
        self.robot_state = RobotState(
            position=start_pos,
            orientation=random.uniform(0, 2 * math.pi)
        )

        # Reset tracking
        self.steps = 0
        self.prev_coverage_pct = 0.0

        # Reset probabilistic coverage if applicable
        if self.mode == "probabilistic":
            self.coverage_map_prob = np.zeros(
                (self.grid_size, self.grid_size), dtype=np.float32
            )

        # Perform initial sensing
        self._update_robot_sensing()

        return self.robot_state

    def step(self, action: int) -> Tuple[RobotState, float, bool, Dict]:
        """
        Execute action and return next state.

        Args:
            action: Integer action [0-8]

        Returns:
            next_state: Updated robot state
            reward: Reward for this step
            done: Episode termination flag
            info: Additional information dict
        """
        self.steps += 1

        # Store previous state
        prev_coverage = self.world_state.coverage_map.copy()
        prev_local_map_size = len(self.robot_state.local_map)
        prev_coverage_pct = self._get_coverage_percentage()

        if self.mode == "probabilistic":
            prev_coverage_prob = self.coverage_map_prob.copy()

        # Execute action
        collision = self._execute_action(action)

        # Update sensing (POMDP)
        self._update_robot_sensing()

        # Calculate gains
        coverage_gain = self._calculate_coverage_gain(prev_coverage)
        knowledge_gain = len(self.robot_state.local_map) - prev_local_map_size
        current_coverage_pct = self._get_coverage_percentage()
        coverage_pct_gain = current_coverage_pct - prev_coverage_pct

        # Build info dict for reward calculation
        info = {
            'coverage_gain': coverage_gain,
            'knowledge_gain': knowledge_gain,
            'collision': collision,
            'coverage_pct': current_coverage_pct,
            'coverage_pct_gain': coverage_pct_gain,
            'frontier_cells': self._count_frontier_cells(),
            'steps': self.steps,
            'done': False  # Will be updated below
        }

        # Add probabilistic gain if applicable
        if self.mode == "probabilistic":
            prob_gain = self._calculate_probabilistic_coverage_gain(prev_coverage_prob)
            info['prob_gain'] = prob_gain

        # Calculate reward using modular calculator
        reward = self.reward_calculator.compute(self.robot_state, action, info)

        # Check termination
        done = self._check_done()
        info['done'] = done

        # Add terminal bonus if applicable (updated info with done=True)
        if done:
            terminal_reward = self.reward_calculator.compute(self.robot_state, action, info)
            reward = terminal_reward  # Includes terminal bonus if in calculator

        # Update prev coverage pct
        self.prev_coverage_pct = current_coverage_pct

        # Add reward breakdown for debugging
        if config.VERBOSE and self.steps <= 5:
            info['reward_breakdown'] = self.reward_calculator.get_breakdown(
                self.robot_state, action, info
            )

        return self.robot_state, reward, done, info

    def _find_valid_start_position(self) -> Tuple[int, int]:
        """Find a valid (non-obstacle) starting position."""
        max_attempts = 100
        for _ in range(max_attempts):
            x = random.randint(1, self.grid_size - 2)
            y = random.randint(1, self.grid_size - 2)
            if (x, y) not in self.world_state.obstacles:
                return (x, y)

        # Fallback: systematic search
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in self.world_state.obstacles:
                    return (x, y)

        # Last resort: center
        return (self.grid_size // 2, self.grid_size // 2)

    def _execute_action(self, action: int) -> bool:
        """
        Execute movement action.

        Args:
            action: Action index [0-8]

        Returns:
            collision: True if collided with obstacle or boundary
        """
        # Update last action
        self.robot_state.last_action = action

        # Get action delta
        dx, dy = config.ACTION_DELTAS[action]

        # Calculate new position
        new_x = self.robot_state.position[0] + dx
        new_y = self.robot_state.position[1] + dy
        new_pos = (new_x, new_y)

        # Check boundaries
        if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size):
            return True  # Collision with boundary

        # Check obstacles
        if new_pos in self.world_state.obstacles:
            return True  # Collision with obstacle

        # Valid move - update position
        self.robot_state.position = new_pos

        # Update orientation (point towards movement direction)
        if dx != 0 or dy != 0:
            self.robot_state.orientation = math.atan2(dy, dx)

        # Mark as visited
        self.robot_state.visited_positions.add(new_pos)
        self.robot_state.visit_heat[new_x, new_y] += 1

        return False  # No collision

    def _update_robot_sensing(self):
        """
        Update robot's local map via ray-cast sensing (POMDP).
        """
        sensed_cells = self._raycast_sensing(
            self.robot_state.position,
            self.robot_state.orientation
        )

        # Update local map with sensed cells
        for cell in sensed_cells:
            if cell in self.world_state.obstacles:
                # Sensed obstacle
                self.robot_state.local_map[cell] = (0.0, "obstacle")
            else:
                # Sensed free cell - update coverage
                coverage = self.world_state.coverage_map[cell[0], cell[1]]
                self.robot_state.local_map[cell] = (coverage, "free")

                # Update coverage (agent presence covers cell)
                if cell == self.robot_state.position:
                    self.world_state.coverage_map[cell[0], cell[1]] = 1.0
                    self.robot_state.coverage_history[cell[0], cell[1]] = 1.0

        # Update probabilistic coverage if applicable
        if self.mode == "probabilistic":
            self._update_probabilistic_coverage(sensed_cells)

    def _update_probabilistic_coverage(self, sensed_cells: Set[Tuple[int, int]]):
        """Update probabilistic coverage map."""
        robot_pos = self.robot_state.position

        for cell in sensed_cells:
            if cell not in self.world_state.obstacles:
                # Calculate distance
                dx = cell[0] - robot_pos[0]
                dy = cell[1] - robot_pos[1]
                distance = math.sqrt(dx**2 + dy**2)

                # Calculate coverage probability
                Pcov = self._calculate_coverage_probability(distance)

                # Update (take maximum)
                current_prob = self.coverage_map_prob[cell[0], cell[1]]
                self.coverage_map_prob[cell[0], cell[1]] = max(current_prob, Pcov)

    def _calculate_coverage_probability(self, distance: float) -> float:
        """Calculate coverage probability using sigmoid function."""
        exponent = self.sigmoid_k * (distance - self.sigmoid_r0)
        Pcov = 1.0 / (1.0 + math.exp(exponent))
        return Pcov

    def _raycast_sensing(self,
                         position: Tuple[int, int],
                         orientation: float) -> Set[Tuple[int, int]]:
        """
        Ray-cast sensing from position (OPTIMIZED with NumPy vectorization).

        Returns:
            Set of sensed cell positions
        """
        sensed = set()
        px, py = position

        # Always sense current position
        sensed.add(position)

        # Pre-compute all angles (vectorized)
        angles = np.linspace(0, 2 * np.pi, config.NUM_RAYS, endpoint=False)
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        # Pre-compute all radii (vectorized)
        radii = np.linspace(0, self.sensor_range, config.SAMPLES_PER_RAY)[1:]  # Skip r=0

        # Cast all rays
        for i in range(config.NUM_RAYS):
            cos_a = cos_angles[i]
            sin_a = sin_angles[i]

            # Sample points along this ray
            for r in radii:
                cx = int(round(px + r * cos_a))
                cy = int(round(py + r * sin_a))

                # Check bounds
                if not (0 <= cx < self.grid_size and 0 <= cy < self.grid_size):
                    break  # Ray leaves grid

                cell = (cx, cy)
                sensed.add(cell)

                # Ray stops at obstacles
                if cell in self.world_state.obstacles:
                    break

        return sensed

    def _calculate_coverage_gain(self, prev_coverage: np.ndarray) -> int:
        """Calculate number of newly covered cells."""
        current_coverage = self.world_state.coverage_map
        newly_covered = np.sum(
            (current_coverage > BINARY_COVERAGE_THRESHOLD) &
            (prev_coverage < BINARY_COVERAGE_THRESHOLD)
        )
        return int(newly_covered)

    def _calculate_probabilistic_coverage_gain(self, prev_coverage_prob: np.ndarray) -> float:
        """Calculate probabilistic coverage gain."""
        prob_gain = np.sum(np.maximum(0, self.coverage_map_prob - prev_coverage_prob))
        return prob_gain

    def _count_frontier_cells(self) -> int:
        """Count frontier cells (known cells adjacent to unknown)."""
        frontier_count = 0

        for cell in self.robot_state.local_map.keys():
            x, y = cell
            # Check 8 neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (x + dx, y + dy)
                    if neighbor not in self.robot_state.local_map:
                        # This is a frontier cell
                        frontier_count += 1
                        break

        return frontier_count

    def _check_done(self) -> bool:
        """Check if episode should terminate."""
        # Max steps reached
        if self.steps >= self.max_steps:
            return True

        # Optional: Early termination if high coverage achieved
        coverage_pct = self._get_coverage_percentage()
        if coverage_pct > HIGH_COVERAGE_THRESHOLD:
            return True

        return False

    def _get_coverage_percentage(self) -> float:
        """Calculate coverage percentage."""
        total_free_cells = self.grid_size * self.grid_size - len(self.world_state.obstacles)
        if total_free_cells == 0:
            return 0.0

        covered_cells = np.sum(self.world_state.coverage_map > BINARY_COVERAGE_THRESHOLD)
        return covered_cells / total_free_cells

    def get_state(self) -> RobotState:
        """Get current robot state."""
        return self.robot_state

    def get_coverage_percentage(self) -> float:
        """Public method to get coverage percentage."""
        return self._get_coverage_percentage()

    def render(self):
        """Render environment (text-based)."""
        coverage_pct = self._get_coverage_percentage()
        sensed_pct = len(self.robot_state.local_map) / (self.grid_size * self.grid_size)

        print(f"Step {self.steps}/{self.max_steps}")
        print(f"Position: {self.robot_state.position}")
        print(f"Coverage: {coverage_pct*100:.1f}%")
        print(f"Sensed: {sensed_pct*100:.1f}%")
        print(f"Visited: {len(self.robot_state.visited_positions)} cells")
        print(f"Mode: {self.mode}")


if __name__ == "__main__":
    print("Testing Unified Coverage Environment")
    print("=" * 80)

    # Test baseline mode
    print("\n1. Testing Baseline Mode:")
    env_baseline = CoverageEnvironment(grid_size=20, map_type="empty", mode="baseline")
    state = env_baseline.reset()
    total_reward = 0

    for step in range(10):
        action = random.randint(0, 8)
        next_state, reward, done, info = env_baseline.step(action)
        total_reward += reward

        if step == 0:
            print(f"   First step: action={action}, reward={reward:.4f}, coverage={info['coverage_pct']:.2%}")

        if done:
            break

    print(f"   Total reward: {total_reward:.4f}")
    print(f"   Final coverage: {env_baseline.get_coverage_percentage():.2%}")

    # Test improved mode
    print("\n2. Testing Improved Mode:")
    env_improved = CoverageEnvironment(grid_size=20, map_type="empty", mode="improved")
    state = env_improved.reset()
    total_reward = 0

    for step in range(10):
        action = random.randint(0, 8)
        next_state, reward, done, info = env_improved.step(action)
        total_reward += reward

        if step == 0:
            print(f"   First step: action={action}, reward={reward:.4f}, coverage={info['coverage_pct']:.2%}")
            if 'reward_breakdown' in info:
                print(f"   Breakdown: {info['reward_breakdown']}")

        if done:
            break

    print(f"   Total reward: {total_reward:.4f}")
    print(f"   Final coverage: {env_improved.get_coverage_percentage():.2%}")

    # Test probabilistic mode
    print("\n3. Testing Probabilistic Mode:")
    env_prob = CoverageEnvironment(grid_size=20, map_type="empty", mode="probabilistic")
    state = env_prob.reset()
    total_reward = 0

    for step in range(10):
        action = random.randint(0, 8)
        next_state, reward, done, info = env_prob.step(action)
        total_reward += reward

        if step == 0:
            print(f"   First step: action={action}, reward={reward:.4f}, prob_gain={info.get('prob_gain', 0):.4f}")

        if done:
            break

    print(f"   Total reward: {total_reward:.4f}")
    print(f"   Final coverage: {env_prob.get_coverage_percentage():.2%}")

    print("\n" + "=" * 80)
    print("✅ Unified Environment Test Complete!")
