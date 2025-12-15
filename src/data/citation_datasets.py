"""Citation dataset loading and preprocessing."""

from typing import Dict, Optional, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Cora, CiteSeer, PubMed
from torch_geometric.transforms import NormalizeFeatures, AddSelfLoops

from .utils import create_synthetic_citation_graph


class CitationDataset:
    """
    Citation network dataset loader.
    
    Supports Cora, CiteSeer, PubMed, and synthetic datasets.
    """
    
    def __init__(
        self,
        dataset_name: str = "cora",
        root: str = "data",
        normalize_features: bool = True,
        add_self_loops: bool = True,
        train_split: float = 0.6,
        val_split: float = 0.2,
        test_split: float = 0.2,
        transform: Optional[object] = None,
    ):
        """
        Initialize citation dataset.
        
        Args:
            dataset_name: Name of the dataset ("cora", "citeseer", "pubmed", "synthetic").
            root: Root directory for data storage.
            normalize_features: Whether to normalize node features.
            add_self_loops: Whether to add self-loops to the graph.
            train_split: Training set ratio.
            val_split: Validation set ratio.
            test_split: Test set ratio.
            transform: Additional transforms to apply.
        """
        self.dataset_name = dataset_name.lower()
        self.root = root
        self.normalize_features = normalize_features
        self.add_self_loops = add_self_loops
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.transform = transform
        
        self.data = self._load_data()
    
    def _load_data(self) -> Data:
        """Load the dataset."""
        if self.dataset_name == "synthetic":
            return create_synthetic_citation_graph(
                num_nodes=100,
                num_edges=200,
                num_features=50,
                num_classes=7,
                seed=42,
            )
        
        # Prepare transforms
        transforms = []
        if self.normalize_features:
            transforms.append(NormalizeFeatures())
        if self.add_self_loops:
            transforms.append(AddSelfLoops())
        if self.transform is not None:
            transforms.append(self.transform)
        
        transform = transforms[0] if len(transforms) == 1 else transforms
        
        # Load dataset
        if self.dataset_name == "cora":
            dataset = Cora(root=self.root, transform=transform)
        elif self.dataset_name == "citeseer":
            dataset = CiteSeer(root=self.root, transform=transform)
        elif self.dataset_name == "pubmed":
            dataset = PubMed(root=self.root, transform=transform)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        return dataset[0]
    
    def get_data(self) -> Data:
        """Get the loaded data."""
        return self.data
    
    def get_info(self) -> Dict[str, int]:
        """Get dataset information."""
        return {
            "num_nodes": self.data.num_nodes,
            "num_edges": self.data.num_edges,
            "num_features": self.data.num_node_features,
            "num_classes": self.data.y.max().item() + 1,
            "train_nodes": self.data.train_mask.sum().item(),
            "val_nodes": self.data.val_mask.sum().item(),
            "test_nodes": self.data.test_mask.sum().item(),
        }
    
    def get_splits(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get train/val/test masks."""
        return self.data.train_mask, self.data.val_mask, self.data.test_mask
