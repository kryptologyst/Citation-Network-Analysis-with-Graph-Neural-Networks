"""Model comparison script for citation network analysis."""

import os
from typing import Dict, Any

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch

from src.data.citation_datasets import CitationDataset
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE
from src.models.gat import GAT
from src.train.trainer import Trainer
from src.eval.evaluator import Evaluator
from src.utils.device import set_seed, get_device, count_parameters


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Compare multiple models."""
    # Set random seed for reproducibility
    set_seed(cfg.seed)
    
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
    print(f"Nodes: {data_info['num_nodes']}, Edges: {data_info['num_edges']}")
    print(f"Features: {data_info['num_features']}, Classes: {data_info['num_classes']}")
    print("-" * 50)
    
    # Define models to compare
    models_config = {
        "GCN": {
            "class": GCN,
            "params": {
                "input_dim": data_info["num_features"],
                "hidden_dim": cfg.model.hidden_dim,
                "output_dim": data_info["num_classes"],
                "num_layers": cfg.model.num_layers,
                "dropout": cfg.model.dropout,
                "activation": "relu",
                "batch_norm": True,
            }
        },
        "GraphSAGE": {
            "class": GraphSAGE,
            "params": {
                "input_dim": data_info["num_features"],
                "hidden_dim": cfg.model.hidden_dim,
                "output_dim": data_info["num_classes"],
                "num_layers": cfg.model.num_layers,
                "dropout": cfg.model.dropout,
                "activation": "relu",
                "batch_norm": True,
                "aggregator": "mean",
            }
        },
        "GAT": {
            "class": GAT,
            "params": {
                "input_dim": data_info["num_features"],
                "hidden_dim": cfg.model.hidden_dim,
                "output_dim": data_info["num_classes"],
                "num_layers": cfg.model.num_layers,
                "num_heads": 8,
                "dropout": cfg.model.dropout,
                "activation": "elu",
                "batch_norm": True,
                "attention_dropout": 0.0,
            }
        },
    }
    
    # Train and evaluate each model
    results = {}
    
    for model_name, model_config in models_config.items():
        print(f"\nTraining {model_name}...")
        
        # Initialize model
        model = model_config["class"](**model_config["params"])
        print(f"Parameters: {count_parameters(model):,}")
        
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
            save_best=True,
            save_last=False,
            checkpoint_dir=os.path.join(cfg.paths.checkpoints_dir, model_name.lower()),
        )
        
        # Train model
        history = trainer.train(
            data=data,
            epochs=cfg.train.epochs,
            patience=cfg.train.patience,
            verbose=False,
        )
        
        # Evaluate model
        evaluator = Evaluator(
            num_classes=data_info["num_classes"],
            device=cfg.device,
        )
        
        model_results = evaluator.evaluate_all_splits(model, data)
        results[model_name] = model_results
        
        # Print results
        print(f"{model_name} Results:")
        for split, metrics in model_results.items():
            print(f"  {split.upper()}:")
            for metric, value in metrics.items():
                if metric != "loss":
                    print(f"    {metric}: {value:.4f}")
    
    # Create leaderboard
    print("\n" + "=" * 50)
    print("MODEL COMPARISON LEADERBOARD")
    print("=" * 50)
    
    evaluator = Evaluator(
        num_classes=data_info["num_classes"],
        device=cfg.device,
    )
    
    leaderboard = evaluator.create_leaderboard(results)
    print(leaderboard.to_string(index=False))
    
    # Save leaderboard
    os.makedirs(cfg.paths.assets_dir, exist_ok=True)
    leaderboard_path = os.path.join(cfg.paths.assets_dir, "model_leaderboard.csv")
    leaderboard.to_csv(leaderboard_path, index=False)
    print(f"\nLeaderboard saved to {leaderboard_path}")
    
    # Create detailed comparison plot
    import matplotlib.pyplot as plt
    import numpy as np
    
    metrics = ["accuracy", "f1_macro", "f1_micro", "auroc"]
    models = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        test_scores = [results[model]["test"][metric] for model in models]
        
        bars = axes[i].bar(models, test_scores, alpha=0.7)
        axes[i].set_title(f"Test {metric.capitalize()}")
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, test_scores):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{score:.3f}", ha="center", va="bottom")
    
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.paths.assets_dir, "model_comparison.png"), 
                dpi=300, bbox_inches="tight")
    plt.show()
    
    print(f"Comparison plot saved to {cfg.paths.assets_dir}/model_comparison.png")


if __name__ == "__main__":
    main()
