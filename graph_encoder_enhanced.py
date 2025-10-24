"""
Enhanced Graph State Encoder (POMDP-Compliant)

Phase 1 Improvements:
- Edge features (distance, traversability, coverage gradient)
- Position encoding
- Better frontier representation

Converts agent's partial observations into graph representation.
Only uses sensed cells to preserve partial observability.
"""

import math
from typing import Tuple, List, Dict
import torch
from torch_geometric.data import Data

from config import config
from data_structures import RobotState, WorldState


class EnhancedGraphStateEncoder:
    """
    Enhanced encoder with edge features and position encoding.

    Improvements over baseline:
    1. Edge features (3D): distance, is_diagonal, coverage_gradient
    2. Position encoding for better spatial reasoning
    3. Enhanced node features (10D instead of 8D)
    """

    def __init__(self, grid_size: int = 20):
        self.grid_size = grid_size

    def encode(self,
               robot_state: RobotState,
               world_state: WorldState,
               agent_idx: int = 0) -> Data:
        """
        Build graph from agent's SENSED cells only (preserves partial observability).

        Returns:
            PyG Data object with:
                - x: Node features [num_nodes, 10]  # Enhanced: 10D
                - edge_index: Graph connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, 3]  # NEW!
                - agent_features: Agent-specific features [1, 10]
        """
        # Get sensed cells (POMDP-compliant)
        sensed_cells = list(robot_state.local_map.keys())

        if len(sensed_cells) == 0:
            # Initial state - create dummy graph at agent position
            sensed_cells = [robot_state.position]

        # Build node features (10-dimensional per node) - ENHANCED
        node_features = []
        for pos in sensed_cells:
            features = self._encode_node_features(
                pos, robot_state, world_state, sensed_cells
            )
            node_features.append(features)

        x = torch.stack(node_features)  # [num_nodes, 10]

        # Build edges WITH FEATURES - NEW!
        edge_index, edge_attr = self._build_edges_with_features(
            sensed_cells, robot_state.local_map, robot_state.position
        )

        # Agent features (10-dimensional)
        agent_features = self._encode_agent_features(
            robot_state, world_state, sensed_cells
        )

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,  # NEW!
            agent_features=agent_features.unsqueeze(0),  # [1, 10]
            num_nodes=len(sensed_cells)
        )

        return data

    def _encode_node_features(self,
                              pos: Tuple[int, int],
                              robot_state: RobotState,
                              world_state: WorldState,
                              all_sensed: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Encode 10D features for a single node (ENHANCED from 8D).

        Features:
            1. Coverage value (0-1)
            2. Is obstacle (0/1)
            3. Is agent position (0/1)
            4. Normalized x position
            5. Normalized y position
            6. Distance to agent
            7. Visit count (normalized)
            8. Frontier score (unknown neighbors / 8)
            9. NEW: Temporal decay (1 / (1 + time_since_visit))
            10. NEW: Coverage density (avg coverage of neighbors)
        """
        x, y = pos
        ax, ay = robot_state.position

        # OPTIMIZED: Single dict lookup instead of two
        cell_data = robot_state.local_map.get(pos, (0.0, "unknown"))
        coverage = cell_data[0]
        node_type = cell_data[1]
        is_obstacle = 1.0 if node_type == "obstacle" else 0.0

        # Is agent
        is_agent = 1.0 if pos == robot_state.position else 0.0

        # Normalized position
        norm_x = x / self.grid_size
        norm_y = y / self.grid_size

        # Distance to agent
        dist = math.sqrt((x - ax)**2 + (y - ay)**2)
        norm_dist = dist / (self.grid_size * 1.414)  # Max dist = diagonal

        # Visit count
        visit_count = robot_state.visit_heat[x, y]
        norm_visits = min(visit_count / 10.0, 1.0)  # Cap at 10 visits

        # Frontier score (how many unknown neighbors)
        frontier_score = self._compute_frontier_score(pos, robot_state.local_map)

        # NEW: Temporal decay (recency of visit)
        # Higher if visited recently, lower if visited long ago
        temporal_decay = 1.0 / (1.0 + visit_count) if visit_count > 0 else 1.0

        # NEW: Coverage density (local coverage context)
        coverage_density = self._compute_coverage_density(
            pos, robot_state.local_map, all_sensed
        )

        features = [
            coverage,
            is_obstacle,
            is_agent,
            norm_x,
            norm_y,
            norm_dist,
            norm_visits,
            frontier_score,
            temporal_decay,      # NEW
            coverage_density     # NEW
        ]

        return torch.tensor(features, dtype=torch.float32)

    def _compute_frontier_score(self,
                                pos: Tuple[int, int],
                                local_map: Dict) -> float:
        """Count unknown neighbors (frontier detection)."""
        x, y = pos
        unknown_count = 0

        # Check 8 neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (x + dx, y + dy)
                if neighbor not in local_map:
                    unknown_count += 1

        return unknown_count / 8.0  # Normalize to [0, 1]

    def _compute_coverage_density(self,
                                  pos: Tuple[int, int],
                                  local_map: Dict,
                                  all_sensed: List[Tuple[int, int]]) -> float:
        """
        NEW: Compute average coverage of neighboring cells.
        Helps network understand local coverage context.
        """
        x, y = pos
        neighbor_coverages = []

        # Check 8 neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (x + dx, y + dy)
                if neighbor in local_map:
                    coverage, _ = local_map[neighbor]
                    neighbor_coverages.append(coverage)

        if len(neighbor_coverages) == 0:
            return 0.0

        return sum(neighbor_coverages) / len(neighbor_coverages)

    def _build_edges_with_features(self,
                                   node_positions: List[Tuple[int, int]],
                                   local_map: Dict,
                                   agent_pos: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        NEW: Build edges WITH features.

        Edge features (3D):
            1. Distance (normalized)
            2. Is diagonal (0/1)
            3. Coverage gradient (destination - source coverage)

        Returns:
            edge_index: [2, num_edges]
            edge_attr: [num_edges, 3]
        """
        pos_to_idx = {pos: idx for idx, pos in enumerate(node_positions)}

        edges_src = []
        edges_dst = []
        edge_features = []

        for pos_src, idx_src in pos_to_idx.items():
            x_src, y_src = pos_src
            coverage_src = local_map.get(pos_src, (0.0, "unknown"))[0]

            # Check 8 neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue

                    pos_dst = (x_src + dx, y_src + dy)

                    # Only connect if neighbor is also in sensed cells
                    if pos_dst in pos_to_idx:
                        # Check not blocked by obstacle
                        node_type = local_map.get(pos_dst, (0.0, "unknown"))[1]
                        if node_type != "obstacle":
                            idx_dst = pos_to_idx[pos_dst]
                            coverage_dst = local_map.get(pos_dst, (0.0, "unknown"))[0]

                            # Add edge
                            edges_src.append(idx_src)
                            edges_dst.append(idx_dst)

                            # Compute edge features
                            # 1. Distance
                            dist = math.sqrt(dx**2 + dy**2)
                            norm_dist = dist / 1.414  # Max = sqrt(2) for diagonal

                            # 2. Is diagonal
                            is_diagonal = 1.0 if (dx != 0 and dy != 0) else 0.0

                            # 3. Coverage gradient (directional info)
                            coverage_gradient = coverage_dst - coverage_src

                            edge_features.append([norm_dist, is_diagonal, coverage_gradient])

        if len(edges_src) == 0:
            # No edges - return self-loop with dummy features
            edges_src = [0]
            edges_dst = [0]
            edge_features = [[0.0, 0.0, 0.0]]

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)

        return edge_index, edge_attr

    def _encode_agent_features(self,
                               robot_state: RobotState,
                               world_state: WorldState,
                               sensed_cells: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Encode 10D agent-specific features.

        Features:
            1-2. Normalized agent position (x, y)
            3-4. Orientation (cos, sin)
            5. Last action (normalized)
            6. Number of sensed cells (normalized)
            7. Coverage percentage (of sensed cells)
            8. Frontier cell count (normalized)
            9-10. Reserved for Stage 2 (neighbor count, avg neighbor coverage)
        """
        ax, ay = robot_state.position

        # Normalized position
        norm_x = ax / self.grid_size
        norm_y = ay / self.grid_size

        # Orientation
        cos_orient = math.cos(robot_state.orientation)
        sin_orient = math.sin(robot_state.orientation)

        # Last action (normalized to [0, 1])
        norm_action = robot_state.last_action / (config.N_ACTIONS - 1)

        # Number of sensed cells
        norm_sensed = len(sensed_cells) / (self.grid_size * self.grid_size)

        # Coverage percentage of sensed area
        if len(sensed_cells) > 0:
            covered_count = sum(1 for pos in sensed_cells
                              if robot_state.local_map.get(pos, (0.0, ""))[0] > 0.5)
            coverage_pct = covered_count / len(sensed_cells)
        else:
            coverage_pct = 0.0

        # Frontier cell count
        frontier_count = sum(1 for pos in sensed_cells
                           if self._compute_frontier_score(pos, robot_state.local_map) > 0.1)
        norm_frontier = frontier_count / max(len(sensed_cells), 1)

        # Reserved for Stage 2
        neighbor_count = 0.0  # Will be number of nearby agents
        neighbor_coverage = 0.0  # Will be average coverage of nearby agents

        features = [
            norm_x,
            norm_y,
            cos_orient,
            sin_orient,
            norm_action,
            norm_sensed,
            coverage_pct,
            norm_frontier,
            neighbor_count,      # Stage 2
            neighbor_coverage    # Stage 2
        ]

        return torch.tensor(features, dtype=torch.float32)


if __name__ == "__main__":
    # Test enhanced graph encoder
    from map_generator import MapGenerator
    import numpy as np

    gen = MapGenerator(20)
    graph, obstacles = gen.generate("empty")

    world_state = WorldState(
        grid_size=20,
        graph=graph,
        obstacles=obstacles,
        coverage_map=np.zeros((20, 20)),
        map_type="empty"
    )

    robot_state = RobotState(position=(10, 10), orientation=0.0)
    robot_state.local_map[(10, 10)] = (0.0, "free")
    robot_state.local_map[(11, 10)] = (0.5, "free")
    robot_state.local_map[(10, 11)] = (0.3, "free")
    robot_state.visit_heat[10, 10] = 2

    encoder = EnhancedGraphStateEncoder(20)
    data = encoder.encode(robot_state, world_state, 0)

    print(f"✓ EnhancedGraphStateEncoder test:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Node features shape: {data.x.shape}")  # Should be [n, 10]
    print(f"  Edge index shape: {data.edge_index.shape}")
    print(f"  Edge features shape: {data.edge_attr.shape}")  # NEW!
    print(f"  Agent features shape: {data.agent_features.shape}")
    print(f"\n  Sample node features (first node):")
    print(f"    {data.x[0]}")
    print(f"\n  Sample edge features (first edge):")
    print(f"    Distance: {data.edge_attr[0, 0]:.3f}")
    print(f"    Is diagonal: {data.edge_attr[0, 1]:.3f}")
    print(f"    Coverage gradient: {data.edge_attr[0, 2]:.3f}")
