"""Training utilities and trainer class."""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchmetrics import Accuracy, F1Score, AUROC
from tqdm import tqdm

from ..utils.device import get_device


class Trainer:
    """
    Trainer class for GNN models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        optimizer: str = "adam",
        learning_rate: float = 0.01,
        weight_decay: float = 5e-4,
        scheduler: str = "cosine",
        warmup_epochs: int = 10,
        grad_clip_norm: float = 1.0,
        save_best: bool = True,
        save_last: bool = True,
        checkpoint_dir: str = "assets/models",
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train.
            device: Device to use ("auto", "cuda", "mps", "cpu").
            optimizer: Optimizer type ("adam", "sgd").
            learning_rate: Learning rate.
            weight_decay: Weight decay for regularization.
            scheduler: Learning rate scheduler ("cosine", "step", "none").
            warmup_epochs: Number of warmup epochs.
            grad_clip_norm: Gradient clipping norm.
            save_best: Whether to save the best model.
            save_last: Whether to save the last model.
            checkpoint_dir: Directory to save checkpoints.
        """
        self.model = model
        self.device = get_device(device)
        self.model.to(self.device)
        
        self.optimizer = self._get_optimizer(optimizer, learning_rate, weight_decay)
        self.scheduler = self._get_scheduler(scheduler, warmup_epochs)
        self.grad_clip_norm = grad_clip_norm
        
        self.save_best = save_best
        self.save_last = save_last
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Metrics
        self.train_metrics = {
            "accuracy": Accuracy(task="multiclass", num_classes=model.output_dim),
            "f1_macro": F1Score(task="multiclass", num_classes=model.output_dim, average="macro"),
            "f1_micro": F1Score(task="multiclass", num_classes=model.output_dim, average="micro"),
            "auroc": AUROC(task="multiclass", num_classes=model.output_dim),
        }
        
        self.val_metrics = {
            "accuracy": Accuracy(task="multiclass", num_classes=model.output_dim),
            "f1_macro": F1Score(task="multiclass", num_classes=model.output_dim, average="macro"),
            "f1_micro": F1Score(task="multiclass", num_classes=model.output_dim, average="micro"),
            "auroc": AUROC(task="multiclass", num_classes=model.output_dim),
        }
        
        # Move metrics to device
        for metrics in [self.train_metrics, self.val_metrics]:
            for metric in metrics.values():
                metric.to(self.device)
        
        self.best_val_loss = float("inf")
        self.train_history = []
        self.val_history = []
    
    def _get_optimizer(self, optimizer: str, lr: float, weight_decay: float) -> optim.Optimizer:
        """Get optimizer."""
        if optimizer.lower() == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer.lower() == "sgd":
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
    
    def _get_scheduler(self, scheduler: str, warmup_epochs: int) -> Optional[object]:
        """Get learning rate scheduler."""
        if scheduler.lower() == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-6)
        elif scheduler.lower() == "step":
            return StepLR(self.optimizer, step_size=50, gamma=0.5)
        else:
            return None
    
    def train_epoch(self, data: torch.Tensor) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        # Forward pass
        out = self.model(data.x, data.edge_index)
        loss = nn.CrossEntropyLoss()(out[data.train_mask], data.y[data.train_mask])
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            pred = out[data.train_mask].argmax(dim=1)
            target = data.y[data.train_mask]
            
            metrics = {"loss": loss.item()}
            for name, metric in self.train_metrics.items():
                metrics[name] = metric(pred, target).item()
        
        return metrics
    
    def validate(self, data: torch.Tensor) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            loss = nn.CrossEntropyLoss()(out[data.val_mask], data.y[data.val_mask])
            
            pred = out[data.val_mask].argmax(dim=1)
            target = data.y[data.val_mask]
            
            metrics = {"loss": loss.item()}
            for name, metric in self.val_metrics.items():
                metrics[name] = metric(pred, target).item()
        
        return metrics
    
    def train(
        self,
        data: torch.Tensor,
        epochs: int = 200,
        patience: int = 50,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            data: Graph data.
            epochs: Number of training epochs.
            patience: Early stopping patience.
            verbose: Whether to show progress bar.
            
        Returns:
            Dict: Training history.
        """
        data = data.to(self.device)
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        if verbose:
            pbar = tqdm(range(epochs), desc="Training")
        else:
            pbar = range(epochs)
        
        for epoch in pbar:
            # Training
            train_metrics = self.train_epoch(data)
            self.train_history.append(train_metrics)
            
            # Validation
            val_metrics = self.validate(data)
            self.val_history.append(val_metrics)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Early stopping
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                
                if self.save_best:
                    self.save_checkpoint("best_model.pt")
            else:
                patience_counter += 1
            
            # Update progress bar
            if verbose:
                pbar.set_postfix({
                    "train_loss": f"{train_metrics['loss']:.4f}",
                    "val_loss": f"{val_metrics['loss']:.4f}",
                    "val_acc": f"{val_metrics['accuracy']:.4f}",
                })
            
            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        # Save last model
        if self.save_last:
            self.save_checkpoint("last_model.pt")
        
        return {
            "train": self.train_history,
            "val": self.val_history,
        }
    
    def evaluate(self, data: torch.Tensor) -> Dict[str, float]:
        """Evaluate the model on test set."""
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            loss = nn.CrossEntropyLoss()(out[data.test_mask], data.y[data.test_mask])
            
            pred = out[data.test_mask].argmax(dim=1)
            target = data.y[data.test_mask]
            
            metrics = {"loss": loss.item()}
            for name, metric in self.val_metrics.items():
                metrics[name] = metric(pred, target).item()
        
        return metrics
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_history": self.train_history,
            "val_history": self.val_history,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, filename), map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_history = checkpoint["train_history"]
        self.val_history = checkpoint["val_history"]
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
