"""Training module for hate speech classification models."""

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict
from pathlib import Path
import json
import yaml
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.config import Config
from .metrics import accuracy_metric, f1_metric
from .early_stopping import EarlyStopping
from .checkpoint import CheckpointManager

class Trainer:
    """Handles model training and evaluation."""

    def __init__(self, model, config: Config, device, run_dir=None):
        """Initialize trainer with model and configuration.
        
        Args:
            model: Neural network model to train
            config: Configuration object
            device: Device to use for training
            run_dir: Optional directory path for this run. If provided, will use this
                    instead of creating a new directory.
        """
        self.device = device
        self.model = model.to(self.device)
        self.config = config
        
        # Use provided run directory if available, otherwise create one
        if run_dir is not None:
            self.run_dir = Path(run_dir)
        else:
            # Create run directory with YYYY-MM-DD-HH:MM:SS format
            date_format = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            self.run_dir = Path(config.paths['runs_dir']) / date_format
            # If the directory already exists, add a suffix
            if self.run_dir.exists():
                # Find all directories with the same date prefix and add a suffix
                existing_dirs = list(Path(config.paths['runs_dir']).glob(f"{date_format}*"))
                suffix = len(existing_dirs) + 1
                self.run_dir = Path(config.paths['runs_dir']) / f"{date_format}-{suffix}"
            
            self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.optimizer = AdamW(model.parameters(), lr=float(config.training_config['learning_rate']))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.metrics = {'accuracy': accuracy_metric, 'f1_score': f1_metric}
        
        # Save config (only if we created the directory)
        if run_dir is None:
            self._save_config()
        
        # Initialize training utilities
        self.checkpoint_manager = CheckpointManager(self.run_dir / "checkpoints")
        self.early_stopping = EarlyStopping(
            patience=config.training_config['early_stopping_patience']
        )
        
        # Setup scheduler
        self._setup_scheduler()
        
        # Initialize history and create log file
        self.history = self._initialize_history()
        self._setup_logging()

    def _setup_scheduler(self):
        """Initialize learning rate scheduler if configured."""
        scheduler_config = self.config.training_config.get('scheduler', {})
        self.scheduler = None
        if scheduler_config.get('type') == 'reduce_lr_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_config['mode'],
                factor=scheduler_config['factor'],
                patience=scheduler_config['patience']
            )

    def _save_config(self):
        """Save configuration used for this run."""
        config_path = self.run_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False)

    def _initialize_history(self):
        """Initialize training history dictionary."""
        return {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'metrics': {},
            'epoch_times': [],
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }

    def _setup_logging(self):
        """Initialize training log file."""
        self.log_file = self.run_dir / "training.log"
        self._log("Training started")
        self._log(f"Model parameters: {self.history['total_parameters']:,} total, "
                 f"{self.history['trainable_parameters']:,} trainable")

    def _log(self, message: str):
        """Add message to log file with timestamp."""
        print(message)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")

    def _save_plots(self):
        """Generate and save training visualization plots."""
        # Loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.run_dir / "loss_plot.png")
        plt.close()

        # Metrics plot
        plt.figure(figsize=(10, 6))
        for metric_name, values in self.history['metrics'].items():
            if metric_name != 'loss':
                plt.plot(values, label=metric_name)
        plt.title('Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig(self.run_dir / "metrics_plot.png")
        plt.close()

    def _save_history(self):
        """Save training history to CSV and JSON."""
        # Save detailed history as JSON
        with open(self.run_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4)

        # Save metrics history as CSV for easy analysis
        history_df = pd.DataFrame({
            'epoch': range(1, len(self.history['train_loss']) + 1),
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            **{f'val_{k}': v for k, v in self.history['metrics'].items() if k != 'loss'}
        })
        history_df.to_csv(self.run_dir / "metrics.csv", index=False)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average training loss for the epoch
        """
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
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation/test data.
        
        Args:
            eval_loader: DataLoader for evaluation data
            
        Returns:
            Dictionary containing loss and metric values
        """
        self.model.eval()
        total_loss = 0
        metrics_values = {name: 0.0 for name in self.metrics}
        
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            
            for name, metric_fn in self.metrics.items():
                metrics_values[name] += metric_fn(outputs, labels)

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
        log_interval: int = 1,
        start_epoch: int = 0
    ) -> Dict:
        """Execute complete training loop with validation and checkpointing.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Maximum number of training epochs
            log_interval: Number of epochs between logging updates
            start_epoch: Starting epoch (for resuming training)
            
        Returns:
            Dictionary containing training history
        """
        best_val_loss = float('inf')
        start_time = datetime.now()
        
        # If resuming training, find the best validation loss from history
        if start_epoch > 0 and self.history['val_loss']:
            best_val_loss = min(self.history['val_loss'])
            self._log(f"Resuming training from epoch {start_epoch}, best val_loss: {best_val_loss:.4f}")
        else:
            self._log(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start = datetime.now()
            
            # Training and validation
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['loss']
            
            # Update history
            if epoch >= len(self.history['train_loss']):
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['epoch_times'].append(
                    (datetime.now() - epoch_start).total_seconds()
                )
            else:
                # Overwrite existing history if resuming and redoing an epoch
                self.history['train_loss'][epoch] = train_loss
                self.history['val_loss'][epoch] = val_loss
                self.history['epoch_times'][epoch] = (datetime.now() - epoch_start).total_seconds()
            
            # Update metrics and save plots
            for metric_name, value in val_metrics.items():
                if metric_name not in self.history['metrics']:
                    self.history['metrics'][metric_name] = []
                
                if epoch >= len(self.history['metrics'][metric_name]):
                    self.history['metrics'][metric_name].append(value)
                else:
                    # Overwrite existing metrics if resuming and redoing an epoch
                    self.history['metrics'][metric_name][epoch] = value
            
            self._save_plots()
            self._save_history()
            
            # Log progress
            if (epoch + 1) % log_interval == 0:
                self._log(f"Epoch {epoch + 1}/{num_epochs}")
                self._log(f"Train Loss: {train_loss:.4f}")
                self._log(f"Val Loss: {val_loss:.4f}")
                for metric_name, value in val_metrics.items():
                    if metric_name != 'loss':
                        self._log(f"Val {metric_name}: {value:.4f}")
            
            # Create checkpoint data
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
                'history': self.history
            }
            if self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
            # Save latest checkpoint (overwriting previous one)
            latest_checkpoint_path = self.run_dir / "checkpoints" / "latest_checkpoint.pt"
            latest_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, latest_checkpoint_path)
            
            # Save best model if improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._log(f"New best model (val_loss: {val_loss:.4f})")
                
                # Save best model in the run root directory
                best_model_path = self.run_dir / "best-model.pt"
                torch.save(checkpoint, best_model_path)
                self._log(f"Saved best model to {best_model_path}")
            
            # Early stopping check
            if self.early_stopping(val_loss):
                self._log("Early stopping triggered!")
                break
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(metrics=val_loss)
        
        # Final logging
        training_time = datetime.now() - start_time
        self._log(f"Training completed in {training_time}")
        self._log(f"Best validation loss: {best_val_loss:.4f}")
        
        return self.history