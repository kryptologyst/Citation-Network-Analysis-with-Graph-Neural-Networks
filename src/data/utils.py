"""Data loading and preprocessing utilities."""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import Cora, CiteSeer, PubMed
from torch_geometric.transforms import NormalizeFeatures, AddSelfLoops
from torch_geometric.utils import to_networkx, from_networkx


def load_citation_dataset(
    name: str,
    root: str = "data",
    normalize_features: bool = True,
    add_self_loops: bool = True,
) -> Data:
    """
    Load a citation network dataset.
    
    Args:
        name: Dataset name ("cora", "citeseer", "pubmed").
        root: Root directory for data storage.
        normalize_features: Whether to normalize node features.
        add_self_loops: Whether to add self-loops to the graph.
        
    Returns:
        Data: PyTorch Geometric Data object.
    """
    transforms = []
    if normalize_features:
        transforms.append(NormalizeFeatures())
    if add_self_loops:
        transforms.append(AddSelfLoops())
    
    transform = transforms[0] if len(transforms) == 1 else transforms
    
    if name.lower() == "cora":
        dataset = Cora(root=root, transform=transform)
    elif name.lower() == "citeseer":
        dataset = CiteSeer(root=root, transform=transform)
    elif name.lower() == "pubmed":
        dataset = PubMed(root=root, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    return dataset[0]


def create_synthetic_citation_graph(
    num_nodes: int = 100,
    num_edges: int = 200,
    num_features: int = 50,
    num_classes: int = 7,
    seed: int = 42,
) -> Data:
    """
    Create a synthetic citation network for testing.
    
    Args:
        num_nodes: Number of nodes (papers).
        num_edges: Number of edges (citations).
        num_features: Number of node features.
        num_classes: Number of paper classes.
        seed: Random seed.
        
    Returns:
        Data: Synthetic citation graph.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate random features
    x = torch.randn(num_nodes, num_features)
    
    # Generate random edge indices
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Generate random labels
    y = torch.randint(0, num_classes, (num_nodes,))
    
    # Create train/val/test splits
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # Random split
    indices = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def get_data_info(data: Data) -> Dict[str, Union[int, float]]:
    """
    Get information about a graph dataset.
    
    Args:
        data: PyTorch Geometric Data object.
        
    Returns:
        Dict: Dataset information.
    """
    info = {
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "num_features": data.num_node_features,
        "num_classes": data.y.max().item() + 1 if data.y is not None else 0,
        "is_directed": not data.is_undirected(),
        "has_isolated_nodes": data.has_isolated_nodes(),
        "has_self_loops": data.has_self_loops(),
    }
    
    if hasattr(data, "train_mask") and data.train_mask is not None:
        info["train_nodes"] = data.train_mask.sum().item()
        info["val_nodes"] = data.val_mask.sum().item()
        info["test_nodes"] = data.test_mask.sum().item()
    
    return info


def save_graph_data(data: Data, filepath: str) -> None:
    """
    Save graph data to files.
    
    Args:
        data: PyTorch Geometric Data object.
        filepath: Base filepath (without extension).
    """
    # Save node features
    if data.x is not None:
        pd.DataFrame(data.x.numpy()).to_csv(f"{filepath}_nodes.csv", index=False)
    
    # Save edges
    if data.edge_index is not None:
        edges_df = pd.DataFrame({
            "src": data.edge_index[0].numpy(),
            "dst": data.edge_index[1].numpy(),
        })
        edges_df.to_csv(f"{filepath}_edges.csv", index=False)
    
    # Save labels
    if data.y is not None:
        pd.DataFrame({"label": data.y.numpy()}).to_csv(f"{filepath}_labels.csv", index=False)
    
    # Save masks
    if hasattr(data, "train_mask") and data.train_mask is not None:
        masks_df = pd.DataFrame({
            "train": data.train_mask.numpy(),
            "val": data.val_mask.numpy(),
            "test": data.test_mask.numpy(),
        })
        masks_df.to_csv(f"{filepath}_masks.csv", index=False)
