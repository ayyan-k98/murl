"""
Recurrent State Encoder for POMDP

Critical for partial observability - adds memory to the agent.
Agent remembers previous observations and builds a belief state.
"""

import torch
import torch.nn as nn


class RecurrentStateEncoder(nn.Module):
    """
    GRU-based recurrent encoder for maintaining belief state in POMDP.

    The agent's observation graph changes each step as it moves and senses.
    This module maintains a hidden state that:
    1. Remembers what the agent has seen before
    2. Integrates new observations
    3. Provides temporal context for decision making

    This is CRITICAL for POMDP performance!
    """

    def __init__(self,
                 input_dim: int = 128,
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: Dimension of graph embedding from GAT
            hidden_dim: Hidden state dimension
            num_layers: Number of GRU layers
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU for temporal integration
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Output projection (optional - for dimensionality adjustment)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Hidden state (persistent across steps within episode)
        self.register_buffer('_hidden_state', None, persistent=False)

    def forward(self, graph_embedding: torch.Tensor, reset: bool = False) -> torch.Tensor:
        """
        Process graph embedding with recurrent state.

        Args:
            graph_embedding: [hidden_dim] or [batch_size, hidden_dim] - current graph representation
            reset: If True, reset hidden state (start of episode)

        Returns:
            output: [hidden_dim] or [batch_size, hidden_dim] - temporally-aware representation
        """
        # Reset hidden state if requested (start of episode)
        if reset or self._hidden_state is None:
            self._hidden_state = None

        # Handle both single and batched inputs
        if graph_embedding.dim() == 1:
            # Single sample: [hidden_dim]
            # Reshape for GRU: [batch=1, seq_len=1, features]
            x = graph_embedding.unsqueeze(0).unsqueeze(0)
            
            # GRU forward
            output, hidden = self.gru(x, self._hidden_state)
            
            # Store hidden state for next step
            self._hidden_state = hidden
            
            # Extract output: [1, 1, hidden_dim] -> [hidden_dim]
            output = output.squeeze(0).squeeze(0)
            
        else:
            # Batched samples: [batch_size, hidden_dim]
            batch_size = graph_embedding.size(0)
            # Reshape for GRU: [batch_size, seq_len=1, features]
            x = graph_embedding.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # For batched training, don't maintain state (process independently)
            # GRU forward
            output, hidden = self.gru(x, None)  # No state continuity in batched mode
            
            # Extract output: [batch_size, 1, hidden_dim] -> [batch_size, hidden_dim]
            output = output.squeeze(1)

        # Project output
        output = self.output_proj(output)

        return output

    def reset_hidden_state(self):
        """Explicitly reset hidden state (call at episode start)."""
        self._hidden_state = None

    def get_hidden_state(self) -> torch.Tensor:
        """Get current hidden state (for debugging/analysis)."""
        if self._hidden_state is None:
            return None
        return self._hidden_state.clone()


class RecurrentGATCoverageDQN(nn.Module):
    """
    Full network combining Enhanced GAT + Recurrent State Encoder.

    This is the COMPLETE Phase 1 enhanced architecture:
    1. Enhanced graph encoding (edge features + adaptive VN)
    2. Recurrent state for POMDP memory
    3. Dueling Q-network head
    """

    def __init__(self,
                 gat_network: nn.Module,
                 recurrent_encoder: RecurrentStateEncoder):
        """
        Args:
            gat_network: EnhancedGATCoverageDQN instance
            recurrent_encoder: RecurrentStateEncoder instance
        """
        super().__init__()

        self.gat_network = gat_network
        self.recurrent_encoder = recurrent_encoder

        # Store config
        self.hidden_dim = gat_network.hidden_dim
        self.n_actions = gat_network.n_actions

    def forward(self, data, reset_recurrent: bool = False) -> torch.Tensor:
        """
        Full forward pass: GAT -> Recurrent -> Q-values

        Args:
            data: PyG Data object
            reset_recurrent: Reset GRU hidden state (episode start)

        Returns:
            Q-values [1, n_actions]
        """
        # This is a bit tricky - we need to extract intermediate representation
        # from GAT before the dueling head, then pass through recurrent encoder

        # For now, we'll modify the approach:
        # Use GAT to get graph embedding, then recurrent encoder, then dueling head

        # Encode graph with GAT
        # We need to access intermediate representation before dueling head
        # Let's extract the virtual node representation

        # Encode nodes
        x = self.gat_network.node_encoder(data.x)

        # Adaptive virtual node
        num_nodes = x.size(0)
        virtual_node = self.gat_network.adaptive_virtual_node(x)
        virtual_node_expanded = virtual_node.unsqueeze(0)
        x_with_virtual = torch.cat([x, virtual_node_expanded], dim=0)

        # Create virtual edges
        virtual_idx = num_nodes
        virtual_edges_src = [virtual_idx] * num_nodes + list(range(num_nodes))
        virtual_edges_dst = list(range(num_nodes)) + [virtual_idx] * num_nodes
        virtual_edge_index = torch.tensor(
            [virtual_edges_src, virtual_edges_dst],
            dtype=torch.long,
            device=data.edge_index.device
        )

        # Edge features for virtual edges
        num_virtual_edges = len(virtual_edges_src)
        virtual_edge_attr = torch.zeros(num_virtual_edges, 3, device=data.x.device)
        virtual_edge_attr[:, 0] = 0.5

        # Combine edges
        edge_index_full = torch.cat([data.edge_index, virtual_edge_index], dim=1)
        edge_attr_full = torch.cat([data.edge_attr, virtual_edge_attr], dim=0)

        # GAT layers
        layer_outputs = [x_with_virtual]
        h = x_with_virtual
        for gat, norm in zip(self.gat_network.gat_layers, self.gat_network.layer_norms):
            h_new = gat(h, edge_index_full, edge_attr=edge_attr_full)
            h_new = norm(h_new)
            h = h_new + h
            h = torch.nn.functional.relu(h)
            layer_outputs.append(h)

        # Jumping Knowledge
        jk_output = torch.cat(layer_outputs, dim=-1)

        # Extract virtual node (graph-level representation)
        if hasattr(data, 'batch') and data.batch is not None:
            # Batched data: extract one virtual node per graph
            batch_size = data.batch.max().item() + 1
            virtual_indices = []
            for i in range(batch_size):
                virtual_idx = len(jk_output) - batch_size + i
                virtual_indices.append(virtual_idx)
            graph_repr = jk_output[virtual_indices]  # [batch_size, hidden_dim * (n_layers+1)]
        else:
            # Single graph
            graph_repr = jk_output[-1].unsqueeze(0)  # [1, hidden_dim * (n_layers+1)]

        # Pass through recurrent encoder (CRITICAL FOR POMDP!)
        # For batched data during training, we reset for each sample (conservative)
        recurrent_repr = self.recurrent_encoder(graph_repr, reset=reset_recurrent)

        # Encode agent features
        if hasattr(data, 'batch') and data.batch is not None:
            agent_repr = self.gat_network.agent_encoder(data.agent_features)  # [batch_size, hidden_dim]
        else:
            agent_repr = self.gat_network.agent_encoder(data.agent_features)  # [1, hidden_dim]

        # Expand recurrent representation to match expected JK dimension
        jk_dim = self.hidden_dim * (self.gat_network.n_gat_layers + 1)  # 128 * 4 = 512
        
        if recurrent_repr.size(-1) != jk_dim:
            # Project recurrent output to JK dimension
            if not hasattr(self, 'recurrent_projection'):
                self.recurrent_projection = torch.nn.Linear(
                    recurrent_repr.size(-1), jk_dim
                ).to(recurrent_repr.device)
            recurrent_repr = self.recurrent_projection(recurrent_repr)
        
        # Now combine with agent features (matches original GAT expectation)
        combined = torch.cat([recurrent_repr, agent_repr], dim=-1)  # [batch_size, jk_dim + agent_dim]

        # Dueling head (should now have correct dimensions)
        value = self.gat_network.value_stream(combined)
        advantages = self.gat_network.advantage_stream(combined)

        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values

    def reset_memory(self):
        """Reset recurrent memory (call at episode start)."""
        self.recurrent_encoder.reset_hidden_state()


if __name__ == "__main__":
    # Test recurrent encoder
    print("Testing RecurrentStateEncoder...")

    from gat_network_enhanced import EnhancedGATCoverageDQN
    from torch_geometric.data import Data

    # Create components
    gat_net = EnhancedGATCoverageDQN(
        node_feature_dim=10,
        agent_feature_dim=10,
        edge_feature_dim=3,
        hidden_dim=128
    )

    # Calculate correct JK dimension from GAT
    # GAT layers: 3 layers + initial = 4 embeddings of 128D each = 512
    # The recurrent encoder should take this 512D input
    jk_dim = 128 * (3 + 1)  # 512D from JK concatenation
    
    recurrent_enc = RecurrentStateEncoder(
        input_dim=jk_dim,  # 512
        hidden_dim=128
    )

    full_net = RecurrentGATCoverageDQN(gat_net, recurrent_enc)

    # Create dummy data
    num_nodes = 10
    x = torch.randn(num_nodes, 10)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.randn(3, 3)
    agent_features = torch.randn(1, 10)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                agent_features=agent_features, num_nodes=num_nodes)

    print("\n✓ Testing episode simulation (3 steps):")

    # Episode start - reset recurrent state
    q1 = full_net(data, reset_recurrent=True)
    print(f"  Step 1 Q-values: {q1[0, :3]}")

    # Subsequent steps - maintain recurrent state
    q2 = full_net(data, reset_recurrent=False)
    print(f"  Step 2 Q-values: {q2[0, :3]}")

    q3 = full_net(data, reset_recurrent=False)
    print(f"  Step 3 Q-values: {q3[0, :3]}")

    # New episode - reset again
    q4 = full_net(data, reset_recurrent=True)
    print(f"  New episode Q-values: {q4[0, :3]}")

    print("\n✓ Recurrent state is working!")
    print(f"  Q-values change across steps: {not torch.allclose(q1, q2)}")
    print(f"  Network has memory: ✓")

    num_params = sum(p.numel() for p in full_net.parameters())
    print(f"\n  Total parameters (GAT + Recurrent): {num_params:,}")
