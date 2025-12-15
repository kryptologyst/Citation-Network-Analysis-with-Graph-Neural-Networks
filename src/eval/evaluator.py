"""Evaluation utilities and metrics."""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torchmetrics import Accuracy, F1Score, AUROC, Precision, Recall
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.device import get_device


class Evaluator:
    """
    Comprehensive evaluator for GNN models.
    """
    
    def __init__(self, num_classes: int, device: str = "auto"):
        """
        Initialize evaluator.
        
        Args:
            num_classes: Number of classes.
            device: Device to use.
        """
        self.num_classes = num_classes
        self.device = get_device(device)
        
        # Initialize metrics
        self.metrics = {
            "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
            "f1_macro": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
            "f1_micro": F1Score(task="multiclass", num_classes=num_classes, average="micro"),
            "precision_macro": Precision(task="multiclass", num_classes=num_classes, average="macro"),
            "precision_micro": Precision(task="multiclass", num_classes=num_classes, average="micro"),
            "recall_macro": Recall(task="multiclass", num_classes=num_classes, average="macro"),
            "recall_micro": Recall(task="multiclass", num_classes=num_classes, average="micro"),
            "auroc": AUROC(task="multiclass", num_classes=num_classes),
        }
        
        # Move metrics to device
        for metric in self.metrics.values():
            metric.to(self.device)
    
    def evaluate(
        self,
        model: nn.Module,
        data: torch.Tensor,
        mask: torch.Tensor,
        return_predictions: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate model on given data and mask.
        
        Args:
            model: Trained model.
            data: Graph data.
            mask: Node mask for evaluation.
            return_predictions: Whether to return predictions.
            
        Returns:
            Dict: Evaluation metrics and optionally predictions.
        """
        model.eval()
        
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out[mask].argmax(dim=1)
            target = data.y[mask]
            probs = torch.softmax(out[mask], dim=1)
            
            # Compute metrics
            results = {}
            for name, metric in self.metrics.items():
                results[name] = metric(pred, target).item()
            
            # Compute loss
            loss = nn.CrossEntropyLoss()(out[mask], target)
            results["loss"] = loss.item()
            
            if return_predictions:
                results["predictions"] = pred.cpu().numpy()
                results["probabilities"] = probs.cpu().numpy()
                results["targets"] = target.cpu().numpy()
        
        return results
    
    def evaluate_all_splits(
        self,
        model: nn.Module,
        data: torch.Tensor,
        return_predictions: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on all splits (train, val, test).
        
        Args:
            model: Trained model.
            data: Graph data.
            return_predictions: Whether to return predictions.
            
        Returns:
            Dict: Evaluation results for each split.
        """
        results = {}
        
        for split in ["train", "val", "test"]:
            mask = getattr(data, f"{split}_mask")
            results[split] = self.evaluate(model, data, mask, return_predictions)
        
        return results
    
    def create_classification_report(
        self,
        model: nn.Module,
        data: torch.Tensor,
        mask: torch.Tensor,
        class_names: Optional[List[str]] = None,
    ) -> str:
        """
        Create detailed classification report.
        
        Args:
            model: Trained model.
            data: Graph data.
            mask: Node mask for evaluation.
            class_names: Names of classes.
            
        Returns:
            str: Classification report.
        """
        model.eval()
        
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out[mask].argmax(dim=1)
            target = data.y[mask]
        
        return classification_report(
            target.cpu().numpy(),
            pred.cpu().numpy(),
            target_names=class_names,
        )
    
    def plot_confusion_matrix(
        self,
        model: nn.Module,
        data: torch.Tensor,
        mask: torch.Tensor,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            model: Trained model.
            data: Graph data.
            mask: Node mask for evaluation.
            class_names: Names of classes.
            save_path: Path to save the plot.
        """
        model.eval()
        
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out[mask].argmax(dim=1)
            target = data.y[mask]
        
        cm = confusion_matrix(target.cpu().numpy(), pred.cpu().numpy())
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
    
    def plot_training_history(
        self,
        train_history: List[Dict[str, float]],
        val_history: List[Dict[str, float]],
        metrics: List[str] = ["loss", "accuracy", "f1_macro"],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot training history.
        
        Args:
            train_history: Training history.
            val_history: Validation history.
            metrics: Metrics to plot.
            save_path: Path to save the plot.
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            train_values = [h[metric] for h in train_history]
            val_values = [h[metric] for h in val_history]
            
            axes[i].plot(train_values, label=f"Train {metric}", alpha=0.8)
            axes[i].plot(val_values, label=f"Val {metric}", alpha=0.8)
            axes[i].set_xlabel("Epoch")
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_title(f"{metric.capitalize()} vs Epoch")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
    
    def create_leaderboard(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Create model comparison leaderboard.
        
        Args:
            results: Results from multiple models.
            save_path: Path to save the leaderboard.
            
        Returns:
            pd.DataFrame: Leaderboard.
        """
        leaderboard_data = []
        
        for model_name, model_results in results.items():
            row = {"Model": model_name}
            
            # Add test set metrics
            test_results = model_results.get("test", {})
            for metric in ["accuracy", "f1_macro", "f1_micro", "auroc"]:
                row[f"Test {metric}"] = test_results.get(metric, 0.0)
            
            leaderboard_data.append(row)
        
        leaderboard = pd.DataFrame(leaderboard_data)
        leaderboard = leaderboard.sort_values("Test accuracy", ascending=False)
        
        if save_path:
            leaderboard.to_csv(save_path, index=False)
        
        return leaderboard
    
    def save_embeddings(
        self,
        model: nn.Module,
        data: torch.Tensor,
        save_path: str,
    ) -> None:
        """
        Save node embeddings.
        
        Args:
            model: Trained model.
            data: Graph data.
            save_path: Path to save embeddings.
        """
        model.eval()
        
        with torch.no_grad():
            embeddings = model.get_embeddings(data.x, data.edge_index)
        
        # Save as CSV
        embeddings_df = pd.DataFrame(embeddings.cpu().numpy())
        embeddings_df.to_csv(save_path, index=False)
        
        print(f"Embeddings saved to {save_path}")
        print(f"Embedding shape: {embeddings.shape}")
    
    def plot_embeddings(
        self,
        model: nn.Module,
        data: torch.Tensor,
        method: str = "tsne",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot node embeddings using dimensionality reduction.
        
        Args:
            model: Trained model.
            data: Graph data.
            method: Dimensionality reduction method ("tsne", "umap", "pca").
            save_path: Path to save the plot.
        """
        model.eval()
        
        with torch.no_grad():
            embeddings = model.get_embeddings(data.x, data.edge_index)
        
        embeddings_np = embeddings.cpu().numpy()
        labels = data.y.cpu().numpy()
        
        if method.lower() == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
        elif method.lower() == "umap":
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
            except ImportError:
                print("UMAP not available, using PCA instead")
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2)
        elif method.lower() == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        embeddings_2d = reducer.fit_transform(embeddings_np)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap="tab10",
            alpha=0.7,
        )
        plt.colorbar(scatter)
        plt.title(f"Node Embeddings ({method.upper()})")
        plt.xlabel(f"{method.upper()} 1")
        plt.ylabel(f"{method.upper()} 2")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
