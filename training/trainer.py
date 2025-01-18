import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from tqdm import tqdm
from pathlib import Path
import json
import numpy as np
from datetime import datetime

class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience: int = 7, min_delta: float = 0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.min_validation_loss = np.inf if mode == 'min' else -np.inf
        
    def __call__(self, validation_loss: float) -> bool:
        if self.mode == 'min':
            if validation_loss < self.min_validation_loss - self.min_delta:
                self.min_validation_loss = validation_loss
                self.counter = 0
            else:
                self.counter += 1
        else:
            if validation_loss > self.min_validation_loss + self.min_delta:
                self.min_validation_loss = validation_loss
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device: str,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        metrics: Dict[str, callable] = None,
        early_stopping_patience: int = 7,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.metrics = metrics or {}
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'metrics': {}
        }
        
    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
        metrics: Dict = None
    ) -> str:
        """Save model checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        # Save checkpoint
        prefix = "best_model" if is_best else f"checkpoint_epoch_{epoch}"
        checkpoint_path = self.checkpoint_dir / f"{prefix}_{timestamp}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata
        metadata_path = checkpoint_path.with_suffix('.json')
        metadata = {
            'epoch': epoch,
            'val_loss': val_loss,
            'metrics': metrics,
            'timestamp': timestamp
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.history = checkpoint['history']
        return checkpoint['epoch']

    def train_epoch(self, train_loader: DataLoader):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            self.optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            if self.scheduler is not None:
                current_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.2e}'})
            else:
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            total_loss += loss.item()

        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader):
        self.model.eval()
        total_loss = 0
        metrics_values = {name: 0 for name in self.metrics}
        
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Calculate metrics
            for name, metric_fn in self.metrics.items():
                metrics_values[name] += metric_fn(outputs, labels)

        # Average metrics
        metrics_values = {
            name: value / len(eval_loader) for name, value in metrics_values.items()
        }
        metrics_values["loss"] = total_loss / len(eval_loader)
        
        return metrics_values

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        log_interval: int = 1
    ):
        """Full training loop with early stopping and checkpoints"""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validation phase
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['loss']
            self.history['val_loss'].append(val_loss)
            
            # Store metrics
            for metric_name, value in val_metrics.items():
                if metric_name not in self.history['metrics']:
                    self.history['metrics'][metric_name] = []
                self.history['metrics'][metric_name].append(value)
            
            # Store learning rate
            if self.scheduler is not None:
                current_lr = self.scheduler.get_last_lr()[0]
                self.history['learning_rates'].append(current_lr)
            
            # Log progress
            if (epoch + 1) % log_interval == 0:
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f}")
                for metric_name, value in val_metrics.items():
                    if metric_name != 'loss':
                        print(f"Val {metric_name}: {value:.4f}")
            
            # Save checkpoint if best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(
                    epoch=epoch,
                    val_loss=val_loss,
                    is_best=True,
                    metrics=val_metrics
                )
            
            # Regular checkpoint
            if (epoch + 1) % 5 == 0:  # Save every 5 epochs
                self.save_checkpoint(
                    epoch=epoch,
                    val_loss=val_loss,
                    metrics=val_metrics
                )
            
            # Early stopping check
            if self.early_stopping(val_loss):
                print("Early stopping triggered!")
                break
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(metrics=val_loss)
        
        return self.history