"""
Enhanced GAT Coverage DQN Network

Phase 1 Improvements:
- Edge features (distance, diagonal, coverage gradient)
- Adaptive virtual node (context-dependent)
- Enhanced node features (10D)

Graph Attention Network with dueling architecture for coverage planning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

from config import config


class AdaptiveVirtualNode(nn.Module):
    """
    Adaptive virtual node that adjusts based on graph context.

    Instead of a single learnable parameter, the virtual node is computed
    from the current graph state, making it adaptive to different map structures.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.base_virtual = nn.Parameter(torch.zeros(hidden_dim))

        # Context adapter network
        self.context_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()  # Bound the adaptation
        )

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive virtual node from current graph.

        Args:
            node_embeddings: [num_nodes, hidden_dim]

        Returns:
            virtual_node: [hidden_dim]
        """
        # Aggregate all node features to get context
        context = node_embeddings.mean(dim=0)  # [hidden_dim]

        # Adapt base virtual node
        adaptation = self.context_net(context)
        virtual_node = self.base_virtual + adaptation

        return virtual_node


class EnhancedGATCoverageDQN(nn.Module):
    """
    Enhanced Graph Attention Network for coverage planning.

    Improvements over baseline:
        1. Edge features (3D: distance, diagonal, coverage gradient)
        2. Adaptive virtual node (context-dependent)
        3. Support for 10D node features (vs 8D baseline)
        4. Improved residual connections

    Architecture:
        - Node encoder: Linear(10) -> LayerNorm -> ReLU -> Hidden
        - Adaptive virtual node (context-dependent)
        - GAT layers with edge features + residual connections + LayerNorm
        - Jumping Knowledge (concatenate all layers)
        - Dueling head: Value + Advantage streams
    """

    def __init__(self,
                 node_feature_dim: int = 10,  # Enhanced: 10D
                 agent_feature_dim: int = 10,
                 edge_feature_dim: int = 3,   # NEW: edge features
                 hidden_dim: int = 128,
                 n_actions: int = 9,
                 n_gat_layers: int = 3,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.n_gat_layers = n_gat_layers

        # Node feature encoder (10D input now)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Adaptive virtual node (NEW!)
        self.adaptive_virtual_node = AdaptiveVirtualNode(hidden_dim)

        # GAT layers with edge features (ENHANCED!)
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(n_gat_layers):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // n_heads,
                    heads=n_heads,
                    dropout=dropout,
                    concat=True,
                    edge_dim=edge_feature_dim  # NEW: edge features!
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Agent feature encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Jumping Knowledge dimension
        jk_dim = hidden_dim * (n_gat_layers + 1) + hidden_dim  # +1 for initial embedding

        # Dueling architecture
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(jk_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(jk_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_actions)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Xavier initialization for stability."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: PyG Data object with:
                - x: Node features [num_nodes, 10]
                - edge_index: [2, num_edges]
                - edge_attr: Edge features [num_edges, 3]  # NEW!
                - agent_features: [1, 10]

        Returns:
            Q-values [batch_size, n_actions]
        """
        # Encode nodes
        x = self.node_encoder(data.x)  # [num_nodes, hidden_dim]

        # Compute adaptive virtual node (NEW!)
        num_nodes = x.size(0)
        virtual_node = self.adaptive_virtual_node(x)  # [hidden_dim]
        virtual_node_expanded = virtual_node.unsqueeze(0)  # [1, hidden_dim]

        # Add virtual node to graph
        x_with_virtual = torch.cat([x, virtual_node_expanded], dim=0)  # [num_nodes+1, hidden_dim]

        # Create edges to/from virtual node (fully connected)
        virtual_idx = num_nodes
        virtual_edges_src = [virtual_idx] * num_nodes + list(range(num_nodes))
        virtual_edges_dst = list(range(num_nodes)) + [virtual_idx] * num_nodes
        virtual_edge_index = torch.tensor(
            [virtual_edges_src, virtual_edges_dst],
            dtype=torch.long,
            device=data.edge_index.device
        )

        # Edge features for virtual edges (neutral: [0.5, 0, 0])
        num_virtual_edges = len(virtual_edges_src)
        virtual_edge_attr = torch.zeros(num_virtual_edges, 3, device=data.x.device)
        virtual_edge_attr[:, 0] = 0.5  # Neutral distance

        # Combine original edges + virtual edges
        edge_index_full = torch.cat([data.edge_index, virtual_edge_index], dim=1)
        edge_attr_full = torch.cat([data.edge_attr, virtual_edge_attr], dim=0)

        # Store layer outputs for Jumping Knowledge
        layer_outputs = [x_with_virtual]

        # Apply GAT layers with edge features and residuals
        h = x_with_virtual
        for gat, norm in zip(self.gat_layers, self.layer_norms):
            # GAT with edge features (ENHANCED!)
            h_new = gat(h, edge_index_full, edge_attr=edge_attr_full)
            h_new = norm(h_new)

            # Residual connection with ReLU
            h = h_new + h
            h = F.relu(h)

            layer_outputs.append(h)

        # Jumping Knowledge: concatenate all layers
        jk_output = torch.cat(layer_outputs, dim=-1)  # [num_nodes+num_graphs, hidden_dim * (n_layers+1)]

        # Extract virtual node representations (one per graph in batch)
        if hasattr(data, 'batch') and data.batch is not None:
            # Batched data: extract one virtual node per graph
            batch_size = data.batch.max().item() + 1
            # Virtual nodes are at the end, one per graph
            virtual_indices = []
            for i in range(batch_size):
                # The virtual node for this graph is added after all real nodes
                virtual_idx = len(jk_output) - batch_size + i
                virtual_indices.append(virtual_idx)
            global_repr = jk_output[virtual_indices]  # [batch_size, hidden_dim * (n_layers+1)]
            
            # Encode agent features for each graph
            agent_repr = self.agent_encoder(data.agent_features)  # [batch_size, hidden_dim]
        else:
            # Single graph: original behavior
            global_repr = jk_output[-1].unsqueeze(0)  # [1, hidden_dim * (n_layers+1)]
            agent_repr = self.agent_encoder(data.agent_features)  # [1, hidden_dim]

        # Concatenate global graph representation + agent features
        combined = torch.cat([global_repr, agent_repr], dim=-1)  # [batch_size, jk_dim]

        # Dueling architecture
        value = self.value_stream(combined)  # [batch_size, 1]
        advantages = self.advantage_stream(combined)  # [batch_size, n_actions]

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values  # [batch_size, n_actions]

        return q_values  # [1, n_actions]


if __name__ == "__main__":
    # Test enhanced GAT network
    print("Testing EnhancedGATCoverageDQN...")

    net = EnhancedGATCoverageDQN(
        node_feature_dim=10,  # Enhanced
        agent_feature_dim=10,
        edge_feature_dim=3,   # NEW
        hidden_dim=128,
        n_actions=9,
        n_gat_layers=3,
        n_heads=4,
        dropout=0.1
    )

    # Create dummy data with edge features
    num_nodes = 10
    x = torch.randn(num_nodes, 10)  # 10D node features
    edge_index = torch.tensor([[0, 1, 2, 1, 2, 3], [1, 2, 3, 0, 1, 2]], dtype=torch.long)
    edge_attr = torch.randn(6, 3)  # 3D edge features
    agent_features = torch.randn(1, 10)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        agent_features=agent_features,
        num_nodes=num_nodes
    )

    # Forward pass
    q_values = net(data)

    num_params = sum(p.numel() for p in net.parameters())
    baseline_params = 200000  # Approximate baseline

    print(f"\n✓ EnhancedGATCoverageDQN test:")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Baseline parameters: ~{baseline_params:,}")
    print(f"  Difference: {num_params - baseline_params:,} (+{(num_params/baseline_params - 1)*100:.1f}%)")
    print(f"  Q-values shape: {q_values.shape}")
    print(f"  Q-values sample: {q_values[0, :3]}")
    print(f"\n  Edge features used: ✓")
    print(f"  Adaptive virtual node: ✓")
    print(f"  Enhanced node features (10D): ✓")
