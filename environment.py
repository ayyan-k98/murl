"""
Coverage Environment with POMDP Sensing

Partially observable environment for coverage tasks.

NOTE: This file is maintained for backward compatibility.
For new code, consider using `environment_unified.py` which provides:
- Unified API for all environment modes (baseline, improved, probabilistic)
- Better integration with modular reward system (rewards.py)
- Cleaner code organization

Example migration:
    # Old (still works)
    from environment import CoverageEnvironment
    env = CoverageEnvironment(grid_size=20)

    # New (recommended)
    from environment_unified import CoverageEnvironment
    env = CoverageEnvironment(mode="baseline")  # or "improved", "probabilistic"
"""

import math
import random
from typing import Tuple, Dict, Set
import numpy as np
import networkx as nx

from config import config
from data_structures import RobotState, WorldState
from map_generator import MapGenerator


class CoverageEnvironment:
    """
    Coverage environment with partial observability.

    Agent can only sense cells within sensor range via ray-casting.
    """

    def __init__(self,
                 grid_size: int = 20,
                 sensor_range: float = 3.0,
                 map_type: str = "empty"):
        self.grid_size = grid_size
        self.sensor_range = sensor_range
        self.map_type = map_type

        self.map_generator = MapGenerator(grid_size)

        # World state (ground truth, not fully observable)
        self.world_state: WorldState = None

        # Robot state (POMDP)
        self.robot_state: RobotState = None

        # Episode tracking
        self.steps = 0
        self.max_steps = config.MAX_EPISODE_STEPS

        # Previous coverage for reward calculation
        self.prev_sensed_cells = set()

    def reset(self, map_type: str = None) -> RobotState:
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

        # Find valid starting position (not obstacle)
        start_pos = self._find_valid_start_position()

        # Initialize robot state
        self.robot_state = RobotState(
            position=start_pos,
            orientation=random.uniform(0, 2 * math.pi)
        )

        # Reset tracking
        self.steps = 0
        self.prev_sensed_cells = set()

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

        # Store previous state for reward calculation
        prev_coverage = self.world_state.coverage_map.copy()
        prev_local_map_size = len(self.robot_state.local_map)

        # Execute action
        collision = self._execute_action(action)

        # Update sensing (POMDP)
        self._update_robot_sensing()

        # Calculate coverage gain and knowledge gain
        coverage_gain = self._calculate_coverage_gain(prev_coverage)
        knowledge_gain = len(self.robot_state.local_map) - prev_local_map_size

        # Calculate reward
        reward = self._calculate_reward(action, coverage_gain, knowledge_gain, collision)

        # Check termination
        done = self._check_done()

        # Info for stratified replay
        info = {
            'coverage_gain': coverage_gain,
            'knowledge_gain': knowledge_gain,
            'collision': collision,
            'coverage_pct': self._get_coverage_percentage(),
            'steps': self.steps
        }

        return self.robot_state, reward, done, info

    def _find_valid_start_position(self) -> Tuple[int, int]:
        """Find a valid (non-obstacle) starting position."""
        max_attempts = 100
        for _ in range(max_attempts):
            x = random.randint(1, self.grid_size - 2)
            y = random.randint(1, self.grid_size - 2)
            if (x, y) not in self.world_state.obstacles:
                return (x, y)

        # Fallback: search systematically
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

        This is CRITICAL for partial observability!
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

    def _raycast_sensing(self, position: Tuple[int, int], orientation: float) -> Set[Tuple[int, int]]:
        """
        Ray-cast sensing from position (OPTIMIZED with NumPy vectorization).

        Casts NUM_RAYS rays in all directions up to SENSOR_RANGE.

        Returns:
            Set of sensed cell positions
        """
        sensed = set()
        px, py = position

        # Always sense current position
        sensed.add(position)

        # OPTIMIZED: Pre-compute all angles (vectorized)
        angles = np.linspace(0, 2 * np.pi, config.NUM_RAYS, endpoint=False)
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        
        # Pre-compute all radii (vectorized)
        radii = np.linspace(0, self.sensor_range, config.SAMPLES_PER_RAY)[1:]  # Skip r=0
        
        # Cast all rays (still need loop for obstacle detection)
        for i in range(config.NUM_RAYS):
            cos_a = cos_angles[i]
            sin_a = sin_angles[i]
            
            # Sample points along this ray (vectorized)
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
        newly_covered = np.sum((current_coverage > 0.5) & (prev_coverage < 0.5))
        return int(newly_covered)

    def _calculate_reward(self,
                         action: int,
                         coverage_gain: int,
                         knowledge_gain: int,
                         collision: bool) -> float:
        """
        Calculate reward for current step.

        Reward components:
            - Coverage gain: +10 per cell
            - Exploration: +0.5 per new sensed cell
            - Frontier bonus: +0.05 per frontier cell (capped)
            - Collision: -2.0
            - Step penalty: -0.01
            - Stay penalty: -0.1 (if action = STAY)
        """
        reward = 0.0

        # Coverage reward
        reward += coverage_gain * config.COVERAGE_REWARD

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

        return reward

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
        if coverage_pct > 0.95:  # 95% coverage
            return True

        return False

    def _get_coverage_percentage(self) -> float:
        """Calculate coverage percentage."""
        total_free_cells = self.grid_size * self.grid_size - len(self.world_state.obstacles)
        if total_free_cells == 0:
            return 0.0

        covered_cells = np.sum(self.world_state.coverage_map > 0.5)
        return covered_cells / total_free_cells

    def get_state(self) -> RobotState:
        """Get current robot state."""
        return self.robot_state

    def get_coverage_percentage(self) -> float:
        """Public method to get coverage percentage."""
        return self._get_coverage_percentage()

    def render(self):
        """
        Render environment (text-based).
        For visualization, use utils.py functions.
        """
        coverage_pct = self._get_coverage_percentage()
        sensed_pct = len(self.robot_state.local_map) / (self.grid_size * self.grid_size)

        print(f"Step {self.steps}/{self.max_steps}")
        print(f"Position: {self.robot_state.position}")
        print(f"Coverage: {coverage_pct*100:.1f}%")
        print(f"Sensed: {sensed_pct*100:.1f}%")
        print(f"Visited: {len(self.robot_state.visited_positions)} cells")


if __name__ == "__main__":
    # Test environment
    print("Testing CoverageEnvironment...")

    env = CoverageEnvironment(grid_size=20, sensor_range=3.0, map_type="empty")

    # Test reset
    state = env.reset()
    print(f"\n✓ Reset complete")
    print(f"  Start position: {state.position}")
    print(f"  Sensed cells: {len(state.local_map)}")

    # Test episode
    total_reward = 0
    for step in range(10):
        action = random.randint(0, 8)
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        if step == 0:
            print(f"\n✓ First step complete")
            print(f"  Action: {config.ACTION_NAMES[action]}")
            print(f"  Reward: {reward:.2f}")
            print(f"  Coverage gain: {info['coverage_gain']}")
            print(f"  Knowledge gain: {info['knowledge_gain']}")

        if done:
            print(f"\n✓ Episode terminated at step {step}")
            break

    print(f"\n✓ Environment test complete")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final coverage: {env.get_coverage_percentage()*100:.1f}%")
    print(f"  Sensed cells: {len(env.robot_state.local_map)}")
