"""Graph Convolutional Network (GCN) implementation."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    Graph Convolutional Network for node classification.
    
    Based on: "Semi-Supervised Classification with Graph Convolutional Networks"
    by Kipf & Welling (ICLR 2017).
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
    ):
        """
        Initialize GCN model.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension (number of classes).
            num_layers: Number of GCN layers.
            dropout: Dropout rate.
            activation: Activation function ("relu", "elu", "leaky_relu").
            batch_norm: Whether to use batch normalization.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        
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
        self.convs.append(GCNConv(input_dim, hidden_dim))
        if batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
        else:
            self.convs.append(GCNConv(input_dim, output_dim))
    
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
