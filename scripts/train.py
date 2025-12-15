"""Main training script for citation network analysis."""

import argparse
import os
from typing import Dict, Any

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from src.data.citation_datasets import CitationDataset
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE
from src.models.gat import GAT
from src.train.trainer import Trainer
from src.eval.evaluator import Evaluator
from src.utils.device import set_seed, get_device, count_parameters, get_model_size


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Set random seed for reproducibility
    set_seed(cfg.seed)
    
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("-" * 50)
    
    # Load dataset
    print("Loading dataset...")
    dataset = CitationDataset(
        dataset_name=cfg.data.dataset_name,
        root=cfg.paths.data_dir,
        normalize_features=cfg.data.normalize_features,
        add_self_loops=cfg.data.add_self_loops,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
    )
    
    data = dataset.get_data()
    data_info = dataset.get_info()
    
    print(f"Dataset: {cfg.data.dataset_name}")
    print(f"Nodes: {data_info['num_nodes']}")
    print(f"Edges: {data_info['num_edges']}")
    print(f"Features: {data_info['num_features']}")
    print(f"Classes: {data_info['num_classes']}")
    print(f"Train/Val/Test: {data_info['train_nodes']}/{data_info['val_nodes']}/{data_info['test_nodes']}")
    print("-" * 50)
    
    # Initialize model
    print("Initializing model...")
    if cfg.model._target_.endswith("GCN"):
        model = GCN(
            input_dim=data_info["num_features"],
            hidden_dim=cfg.model.hidden_dim,
            output_dim=data_info["num_classes"],
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            activation=cfg.model.activation,
            batch_norm=cfg.model.batch_norm,
        )
    elif cfg.model._target_.endswith("GraphSAGE"):
        model = GraphSAGE(
            input_dim=data_info["num_features"],
            hidden_dim=cfg.model.hidden_dim,
            output_dim=data_info["num_classes"],
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            activation=cfg.model.activation,
            batch_norm=cfg.model.batch_norm,
            aggregator=cfg.model.aggregator,
        )
    elif cfg.model._target_.endswith("GAT"):
        model = GAT(
            input_dim=data_info["num_features"],
            hidden_dim=cfg.model.hidden_dim,
            output_dim=data_info["num_classes"],
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            dropout=cfg.model.dropout,
            activation=cfg.model.activation,
            batch_norm=cfg.model.batch_norm,
            attention_dropout=cfg.model.attention_dropout,
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model._target_}")
    
    print(f"Model: {cfg.model._target_}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Size: {get_model_size(model)}")
    print("-" * 50)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        device=cfg.device,
        optimizer=cfg.train.optimizer,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        scheduler=cfg.train.scheduler,
        warmup_epochs=cfg.train.warmup_epochs,
        grad_clip_norm=cfg.train.grad_clip_norm,
        save_best=cfg.train.save_best,
        save_last=cfg.train.save_last,
        checkpoint_dir=cfg.paths.checkpoints_dir,
    )
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        data=data,
        epochs=cfg.train.epochs,
        patience=cfg.train.patience,
        verbose=True,
    )
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = Evaluator(
        num_classes=data_info["num_classes"],
        device=cfg.device,
    )
    
    # Evaluate on all splits
    results = evaluator.evaluate_all_splits(model, data)
    
    print("\nResults:")
    for split, metrics in results.items():
        print(f"\n{split.upper()}:")
        for metric, value in metrics.items():
            if metric != "loss":
                print(f"  {metric}: {value:.4f}")
    
    # Save results
    os.makedirs(cfg.paths.assets_dir, exist_ok=True)
    
    # Save embeddings
    if cfg.eval.save_embeddings:
        evaluator.save_embeddings(
            model,
            data,
            os.path.join(cfg.paths.assets_dir, "embeddings.csv"),
        )
    
    # Plot embeddings
    evaluator.plot_embeddings(
        model,
        data,
        method="tsne",
        save_path=os.path.join(cfg.paths.assets_dir, "embeddings_tsne.png"),
    )
    
    # Plot training history
    evaluator.plot_training_history(
        history["train"],
        history["val"],
        metrics=["loss", "accuracy", "f1_macro"],
        save_path=os.path.join(cfg.paths.assets_dir, "training_history.png"),
    )
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        model,
        data,
        data.test_mask,
        save_path=os.path.join(cfg.paths.assets_dir, "confusion_matrix.png"),
    )
    
    # Create classification report
    report = evaluator.create_classification_report(model, data, data.test_mask)
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open(os.path.join(cfg.paths.assets_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    
    print(f"\nResults saved to {cfg.paths.assets_dir}")


if __name__ == "__main__":
    main()
