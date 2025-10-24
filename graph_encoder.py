"""
Graph State Encoder with Spatial Features (POMDP-Compliant)

CRITICAL FIX: Added explicit spatial encoding for navigation:
- Absolute position (x, y normalized)
- Relative position to agent (dx, dy)
- Polar coordinates (distance, angle)
- Original coverage/frontier features

This gives the GNN spatial understanding it was missing.
"""

import math
from typing import Tuple, List, Dict
import torch
from torch_geometric.data import Data

from config import config
from data_structures import RobotState, WorldState


class GraphStateEncoder:
    """
    Encodes agent's local map into a graph with SPATIAL FEATURES (POMDP-compliant).
    Only uses cells that the agent has sensed.
    
    Key improvement: Node features now include explicit spatial information
    so the GNN can learn spatial navigation strategies.
    """

    def __init__(self, grid_size: int = 20):
        self.grid_size = grid_size

    def encode(self,
               robot_state: RobotState,
               world_state: WorldState,
               agent_idx: int = 0) -> Data:
        """
        Build graph from agent's SENSED cells with 12D spatial features.

        Returns:
            PyG Data object with:
                - x: Node features [num_nodes, 12] (expanded from 8D)
                - edge_index: Graph connectivity [2, num_edges]
                - agent_features: Agent-specific features [1, 10]
                - positions: Node positions [num_nodes, 2] (for debugging)
        """
        # Get sensed cells (POMDP-compliant)
        sensed_cells = list(robot_state.local_map.keys())

        if len(sensed_cells) == 0:
            # Initial state - create dummy graph at agent position
            sensed_cells = [robot_state.position]

        # Build node features (12-dimensional per node with spatial info)
        node_features = []
        positions = []
        for pos in sensed_cells:
            features = self._encode_node_features(
                pos, robot_state, world_state, sensed_cells
            )
            node_features.append(features)
            positions.append(pos)

        x = torch.stack(node_features)  # [num_nodes, 12]

        # Build edges (8-connected grid, only between known cells)
        edge_index = self._build_edges(sensed_cells, robot_state.local_map)

        # Agent features (10-dimensional)
        agent_features = self._encode_agent_features(
            robot_state, world_state, sensed_cells
        )

        # Store positions as tensor for potential use
        positions_tensor = torch.tensor(positions, dtype=torch.long)

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            agent_features=agent_features.unsqueeze(0),  # [1, 10]
            positions=positions_tensor,  # [num_nodes, 2]
            num_nodes=len(sensed_cells)
        )

        return data

    def _encode_node_features(self,
                              pos: Tuple[int, int],
                              robot_state: RobotState,
                              world_state: WorldState,
                              all_sensed: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Encode 12D features for a single node (expanded from 8D).

        New spatial features:
        [0-1]:  Absolute position (x, y) normalized
        [2-3]:  Relative position to agent (dx, dy) normalized  
        [4]:    Distance to agent (normalized by max diagonal)
        [5]:    Angle to agent (radians / pi, range [-1, 1])
        [6]:    Coverage value (0-1)
        [7]:    Is obstacle (0/1)
        [8]:    Is agent position (0/1)
        [9]:    Visit count (normalized)
        [10]:   Frontier score (unknown neighbors / 8)
        [11]:   Recency (steps since last visit, normalized)
        """
        x, y = pos
        ax, ay = robot_state.position
        
        # === NEW SPATIAL FEATURES ===
        
        # [0-1] Absolute position (normalized to [0, 1])
        x_norm = x / self.grid_size
        y_norm = y / self.grid_size
        
        # [2-3] Relative position to agent (normalized to [-1, 1])
        dx = (x - ax) / self.grid_size  
        dy = (y - ay) / self.grid_size
        
        # [4] Euclidean distance to agent (normalized)
        max_distance = self.grid_size * math.sqrt(2)  # Max diagonal
        distance = math.sqrt((x - ax)**2 + (y - ay)**2) / max_distance
        
        # [5] Angle to agent (radians, normalized to [-1, 1])
        angle = math.atan2(y - ay, x - ax) / math.pi
        
        # === ORIGINAL COVERAGE FEATURES ===

        # OPTIMIZED: Single dict lookup instead of two
        cell_data = robot_state.local_map.get(pos, (0.0, "unknown"))
        coverage = cell_data[0]  # [6]
        node_type = cell_data[1]
        is_obstacle = 1.0 if node_type == "obstacle" else 0.0  # [7]

        # Is agent [8]
        is_agent = 1.0 if pos == robot_state.position else 0.0

        # Visit count [9]
        visit_count = robot_state.visit_heat[x, y]
        norm_visits = min(visit_count / 10.0, 1.0)  # Cap at 10 visits

        # Frontier score [10] (how many unknown neighbors)
        frontier_score = self._compute_frontier_score(pos, robot_state.local_map)
        
        # Recency [11] (normalized steps since last visit)
        current_step = getattr(robot_state, 'steps', 0)
        last_visit = getattr(robot_state, 'last_visit_step', {}).get(pos, 0)
        recency = min((current_step - last_visit) / 100.0, 1.0) if current_step > 0 else 0.0

        # Assemble 12D feature vector
        features = [
            x_norm,           # [0] Absolute x (NEW)
            y_norm,           # [1] Absolute y (NEW)
            dx,               # [2] Relative dx (NEW - CRITICAL!)
            dy,               # [3] Relative dy (NEW - CRITICAL!)
            distance,         # [4] Distance to agent (NEW)
            angle,            # [5] Angle to agent (NEW)
            coverage,         # [6] Coverage value
            is_obstacle,      # [7] Is obstacle
            is_agent,         # [8] Is agent
            norm_visits,      # [9] Visit count
            frontier_score,   # [10] Frontier score
            recency          # [11] Recency (NEW)
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

    def _build_edges(self,
                     node_positions: List[Tuple[int, int]],
                     local_map: Dict) -> torch.Tensor:
        """
        Build 8-connected edges between KNOWN cells only.
        This preserves POMDP: edges only exist between sensed cells.
        """
        pos_to_idx = {pos: idx for idx, pos in enumerate(node_positions)}

        edges_src = []
        edges_dst = []

        for pos, idx in pos_to_idx.items():
            x, y = pos
            # Check 8 neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue

                    neighbor = (x + dx, y + dy)

                    # Only connect if neighbor is also in sensed cells
                    if neighbor in pos_to_idx:
                        # Check not blocked by obstacle (OPTIMIZED: single lookup)
                        neighbor_data = local_map.get(neighbor, (0.0, "unknown"))
                        if neighbor_data[1] != "obstacle":
                            neighbor_idx = pos_to_idx[neighbor]
                            edges_src.append(idx)
                            edges_dst.append(neighbor_idx)

        if len(edges_src) == 0:
            # No edges - return self-loop
            edges_src = [0]
            edges_dst = [0]

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        return edge_index

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
    
    def get_feature_names(self) -> list:
        """Return names of all 12 node features for debugging."""
        return [
            "x_absolute",      # [0] NEW
            "y_absolute",      # [1] NEW
            "dx_relative",     # [2] NEW - CRITICAL for spatial navigation
            "dy_relative",     # [3] NEW - CRITICAL for spatial navigation
            "distance_agent",  # [4] NEW
            "angle_agent",     # [5] NEW
            "coverage",        # [6] Original
            "is_obstacle",     # [7] Original
            "is_agent",        # [8] Original
            "visit_count",     # [9] Original
            "frontier_score",  # [10] Original
            "recency"          # [11] NEW
        ]


if __name__ == "__main__":
    # Test graph encoder
    from map_generator import MapGenerator

    gen = MapGenerator(20)
    graph, obstacles = gen.generate("empty")

    world_state = WorldState(
        grid_size=20,
        graph=graph,
        obstacles=obstacles,
        coverage_map=torch.zeros((20, 20)),
        map_type="empty"
    )

    robot_state = RobotState(position=(10, 10), orientation=0.0)
    robot_state.local_map[(10, 10)] = (0.0, "free")
    robot_state.local_map[(11, 10)] = (0.5, "free")

    encoder = GraphStateEncoder(20)
    data = encoder.encode(robot_state, world_state, 0)

    print(f"✓ GraphStateEncoder test:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Node features shape: {data.x.shape}")
    print(f"  Edge index shape: {data.edge_index.shape}")
    print(f"  Agent features shape: {data.agent_features.shape}")
