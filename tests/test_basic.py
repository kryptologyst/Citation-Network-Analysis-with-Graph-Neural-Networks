"""Basic tests for the citation network analysis package."""

import pytest
import torch
import torch.nn as nn

from src.data.citation_datasets import CitationDataset
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE
from src.models.gat import GAT
from src.utils.device import get_device, set_seed, count_parameters


def test_device_detection():
    """Test device detection."""
    device = get_device("auto")
    assert isinstance(device, torch.device)


def test_seed_setting():
    """Test random seed setting."""
    set_seed(42)
    # This should not raise an exception
    assert True


def test_synthetic_dataset():
    """Test synthetic dataset creation."""
    dataset = CitationDataset(
        dataset_name="synthetic",
        normalize_features=True,
        add_self_loops=True,
    )
    
    data = dataset.get_data()
    assert data.num_nodes > 0
    assert data.num_edges > 0
    assert data.x.shape[1] > 0
    assert data.y.max().item() >= 0


def test_gcn_model():
    """Test GCN model initialization and forward pass."""
    model = GCN(
        input_dim=50,
        hidden_dim=64,
        output_dim=7,
        num_layers=2,
        dropout=0.5,
    )
    
    # Test forward pass
    x = torch.randn(100, 50)
    edge_index = torch.randint(0, 100, (2, 200))
    
    out = model(x, edge_index)
    assert out.shape == (100, 7)
    
    # Test embeddings
    embeddings = model.get_embeddings(x, edge_index)
    assert embeddings.shape == (100, 64)


def test_graphsage_model():
    """Test GraphSAGE model initialization and forward pass."""
    model = GraphSAGE(
        input_dim=50,
        hidden_dim=64,
        output_dim=7,
        num_layers=2,
        dropout=0.5,
        aggregator="mean",
    )
    
    # Test forward pass
    x = torch.randn(100, 50)
    edge_index = torch.randint(0, 100, (2, 200))
    
    out = model(x, edge_index)
    assert out.shape == (100, 7)
    
    # Test embeddings
    embeddings = model.get_embeddings(x, edge_index)
    assert embeddings.shape == (100, 64)


def test_gat_model():
    """Test GAT model initialization and forward pass."""
    model = GAT(
        input_dim=50,
        hidden_dim=64,
        output_dim=7,
        num_layers=2,
        num_heads=8,
        dropout=0.5,
    )
    
    # Test forward pass
    x = torch.randn(100, 50)
    edge_index = torch.randint(0, 100, (2, 200))
    
    out = model(x, edge_index)
    assert out.shape == (100, 7)
    
    # Test embeddings
    embeddings = model.get_embeddings(x, edge_index)
    assert embeddings.shape == (100, 64 * 8)  # hidden_dim * num_heads
    
    # Test attention weights
    attention_weights = model.get_attention_weights(x, edge_index)
    assert len(attention_weights) == 2  # num_layers


def test_parameter_counting():
    """Test parameter counting utility."""
    model = GCN(
        input_dim=50,
        hidden_dim=64,
        output_dim=7,
        num_layers=2,
    )
    
    param_count = count_parameters(model)
    assert param_count > 0
    assert isinstance(param_count, int)


if __name__ == "__main__":
    pytest.main([__file__])
