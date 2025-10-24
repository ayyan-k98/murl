"""
GAT Coverage DQN Network

Graph Attention Network with dueling architecture for coverage planning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

from config import config


class GATCoverageDQN(nn.Module):
    """
    Graph Attention Network for coverage planning with dueling architecture.

    Architecture:
        - Node encoder: Linear(8) -> LayerNorm -> ReLU -> Hidden
        - Virtual node for O(n) global communication
        - GAT layers with residual connections + LayerNorm
        - Jumping Knowledge (concatenate all layers)
        - Dueling head: Value + Advantage streams
    """

    def __init__(self,
                 node_feature_dim: int = 8,
                 agent_feature_dim: int = 10,
                 hidden_dim: int = 128,
                 n_actions: int = 9,
                 n_gat_layers: int = 3,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.n_gat_layers = n_gat_layers

        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Virtual node for global context (learnable)
        self.virtual_node = nn.Parameter(torch.zeros(hidden_dim))

        # GAT layers with residual connections
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(n_gat_layers):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // n_heads,
                    heads=n_heads,
                    dropout=dropout,
                    concat=True
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Agent feature encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Jumping Knowledge dimension: sum of all layer outputs + agent
        jk_dim = hidden_dim * (n_gat_layers + 1) + hidden_dim  # +1 for initial embedding

        # Dueling architecture
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(jk_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(jk_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
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
            data: PyG Data object with x, edge_index, agent_features

        Returns:
            Q-values [batch_size, n_actions]
        """
        # Encode nodes
        x = self.node_encoder(data.x)  # [num_nodes, hidden_dim]

        # Add virtual node
        num_nodes = x.size(0)
        virtual_node_expanded = self.virtual_node.unsqueeze(0)  # [1, hidden_dim]
        x_with_virtual = torch.cat([x, virtual_node_expanded], dim=0)  # [num_nodes+1, hidden_dim]

        # Create edges to/from virtual node (fully connected to all nodes)
        virtual_idx = num_nodes
        virtual_edges_src = [virtual_idx] * num_nodes + list(range(num_nodes))
        virtual_edges_dst = list(range(num_nodes)) + [virtual_idx] * num_nodes
        virtual_edge_index = torch.tensor(
            [virtual_edges_src, virtual_edges_dst],
            dtype=torch.long,
            device=data.edge_index.device
        )

        # Combine original edges with virtual edges
        edge_index_full = torch.cat([data.edge_index, virtual_edge_index], dim=1)

        # Store layer outputs for Jumping Knowledge
        layer_outputs = [x_with_virtual]

        # Apply GAT layers with residuals
        h = x_with_virtual
        for gat, norm in zip(self.gat_layers, self.layer_norms):
            h_new = gat(h, edge_index_full)
            h_new = norm(h_new)
            h = h_new + h  # Residual connection
            h = F.relu(h)
            layer_outputs.append(h)

        # Jumping Knowledge: concatenate all layers
        jk_output = torch.cat(layer_outputs, dim=-1)  # [num_nodes+num_graphs, hidden_dim * (n_layers+1)]

        # Extract virtual node representations (one per graph in batch)
        # Virtual nodes are at the end, one per graph
        if hasattr(data, 'batch') and data.batch is not None:
            # Batched data: extract one virtual node per graph
            batch_size = data.batch.max().item() + 1
            # Virtual nodes are the last node for each graph
            virtual_indices = []
            for i in range(batch_size):
                # Find last node index for graph i
                mask = (data.batch == i)
                graph_nodes = torch.where(mask)[0]
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


if __name__ == "__main__":
    # Test GAT network
    net = GATCoverageDQN(
        node_feature_dim=config.NODE_FEATURE_DIM,
        agent_feature_dim=config.AGENT_FEATURE_DIM,
        hidden_dim=config.GAT_HIDDEN_DIM,
        n_actions=config.N_ACTIONS,
        n_gat_layers=config.GAT_N_LAYERS,
        n_heads=config.GAT_N_HEADS,
        dropout=config.GAT_DROPOUT
    )

    # Create dummy data
    num_nodes = 10
    x = torch.randn(num_nodes, config.NODE_FEATURE_DIM)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    agent_features = torch.randn(1, config.AGENT_FEATURE_DIM)

    data = Data(x=x, edge_index=edge_index, agent_features=agent_features, num_nodes=num_nodes)

    # Forward pass
    q_values = net(data)

    num_params = sum(p.numel() for p in net.parameters())
    print(f"✓ GATCoverageDQN test:")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Q-values shape: {q_values.shape}")
    print(f"  Q-values: {q_values}")
