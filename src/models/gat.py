"""Graph Attention Network (GAT) implementation."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """
    Graph Attention Network for node classification.
    
    Based on: "Graph Attention Networks" by Veličković et al. (ICLR 2018).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.5,
        activation: str = "elu",
        batch_norm: bool = True,
        attention_dropout: float = 0.0,
    ):
        """
        Initialize GAT model.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension (number of classes).
            num_layers: Number of GAT layers.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            activation: Activation function ("elu", "relu", "leaky_relu").
            batch_norm: Whether to use batch normalization.
            attention_dropout: Attention dropout rate.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.attention_dropout = attention_dropout
        
        # Activation function
        if activation == "elu":
            self.activation = F.elu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(
            GATConv(
                input_dim,
                hidden_dim,
                heads=num_heads,
                dropout=attention_dropout,
                concat=True,
            )
        )
        if batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_dim * num_heads,
                    hidden_dim,
                    heads=num_heads,
                    dropout=attention_dropout,
                    concat=True,
                )
            )
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(
                GATConv(
                    hidden_dim * num_heads,
                    output_dim,
                    heads=1,
                    dropout=attention_dropout,
                    concat=False,
                )
            )
        else:
            self.convs.append(
                GATConv(
                    input_dim,
                    output_dim,
                    heads=num_heads,
                    dropout=attention_dropout,
                    concat=False,
                )
            )
    
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
            torch.Tensor: Node embeddings [num_nodes, hidden_dim * num_heads].
        """
        # Forward through all layers except the last one
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            
            if self.batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def get_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor) -> list:
        """
        Get attention weights for visualization.
        
        Args:
            x: Node features [num_nodes, input_dim].
            edge_index: Edge indices [2, num_edges].
            
        Returns:
            list: Attention weights for each layer.
        """
        attention_weights = []
        
        # Forward through all layers and collect attention weights
        for i, conv in enumerate(self.convs):
            x, attention = conv(x, edge_index, return_attention_weights=True)
            attention_weights.append(attention)
            
            if self.batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            if i < self.num_layers - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return attention_weights
