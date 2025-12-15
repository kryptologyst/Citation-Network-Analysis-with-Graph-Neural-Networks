"""Simple example script demonstrating the modern citation network analysis."""

import torch
from src.data.citation_datasets import CitationDataset
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE
from src.models.gat import GAT
from src.train.trainer import Trainer
from src.eval.evaluator import Evaluator
from src.utils.device import set_seed, get_device


def main():
    """Simple example of training and evaluating a GNN model."""
    print("Citation Network Analysis - Simple Example")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load synthetic dataset (for quick testing)
    print("Loading synthetic dataset...")
    dataset = CitationDataset(
        dataset_name="synthetic",
        normalize_features=True,
        add_self_loops=True,
    )
    
    data = dataset.get_data()
    data_info = dataset.get_info()
    
    print(f"Dataset loaded:")
    print(f"  Nodes: {data_info['num_nodes']}")
    print(f"  Edges: {data_info['num_edges']}")
    print(f"  Features: {data_info['num_features']}")
    print(f"  Classes: {data_info['num_classes']}")
    print()
    
    # Initialize GCN model
    print("Initializing GCN model...")
    model = GCN(
        input_dim=data_info["num_features"],
        hidden_dim=32,  # Smaller for quick training
        output_dim=data_info["num_classes"],
        num_layers=2,
        dropout=0.5,
        activation="relu",
        batch_norm=True,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        device="auto",
        optimizer="adam",
        learning_rate=0.01,
        weight_decay=5e-4,
        scheduler="cosine",
        warmup_epochs=5,
        grad_clip_norm=1.0,
        save_best=True,
        save_last=False,
        checkpoint_dir="assets/models/example",
    )
    
    # Train model (short training for demo)
    print("Training model...")
    history = trainer.train(
        data=data,
        epochs=50,  # Short training for demo
        patience=20,
        verbose=True,
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluator = Evaluator(
        num_classes=data_info["num_classes"],
        device="auto",
    )
    
    results = evaluator.evaluate_all_splits(model, data)
    
    print("\nResults:")
    for split, metrics in results.items():
        print(f"\n{split.upper()}:")
        for metric, value in metrics.items():
            if metric != "loss":
                print(f"  {metric}: {value:.4f}")
    
    # Create classification report
    print("\nClassification Report:")
    report = evaluator.create_classification_report(model, data, data.test_mask)
    print(report)
    
    print("\nExample completed successfully!")
    print("For more advanced features, see:")
    print("  - scripts/train.py for full training")
    print("  - scripts/compare_models.py for model comparison")
    print("  - demo/app.py for interactive visualization")


if __name__ == "__main__":
    main()
