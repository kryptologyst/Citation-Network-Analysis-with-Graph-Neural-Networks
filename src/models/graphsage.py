"""GraphSAGE implementation."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    """
    GraphSAGE for inductive node classification.
    
    Based on: "Inductive Representation Learning on Large Graphs"
    by Hamilton et al. (NeurIPS 2017).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        activation: str = "relu",
        batch_norm: bool = True,
        aggregator: str = "mean",
    ):
        """
        Initialize GraphSAGE model.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension (number of classes).
            num_layers: Number of SAGE layers.
            dropout: Dropout rate.
            activation: Activation function ("relu", "elu", "leaky_relu").
            batch_norm: Whether to use batch normalization.
            aggregator: Aggregation method ("mean", "max", "lstm").
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.aggregator = aggregator
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "elu":
            self.activation = F.elu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(input_dim, hidden_dim, aggr=aggregator))
        if batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, output_dim, aggr=aggregator))
        else:
            self.convs.append(SAGEConv(input_dim, output_dim, aggr=aggregator))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim].
            edge_index: Edge indices [2, num_edges].
            
        Returns:
            torch.Tensor: Node predictions [num_nodes, output_dim].
        """
        # Hidden layers
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            
            if self.batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.convs[-1](x, edge_index)
        
        return x
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get node embeddings from the last hidden layer.
        
        Args:
            x: Node features [num_nodes, input_dim].
            edge_index: Edge indices [2, num_edges].
            
        Returns:
            torch.Tensor: Node embeddings [num_nodes, hidden_dim].
        """
        # Forward through all layers except the last one
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            
            if self.batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
