"""Training module for hate speech classification models."""

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional
from pathlib import Path
import json
import yaml
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
            run_dir: Optional directory path for this run
        """
        self.device = device
        self.model = model.to(self.device)
        self.config = config
        
        # Use provided run directory if available, otherwise create one
        if run_dir is not None:
            self.run_dir = Path(run_dir)
        else:
            date_format = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            self.run_dir = Path(config.paths['runs_dir']) / date_format
            if self.run_dir.exists():
                existing_dirs = list(Path(config.paths['runs_dir']).glob(f"{date_format}*"))
                suffix = len(existing_dirs) + 1
                self.run_dir = Path(config.paths['runs_dir']) / f"{date_format}-{suffix}"
            
            self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.optimizer = AdamW(model.parameters(), lr=float(config.training_config['learning_rate']))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.metrics = {'accuracy': accuracy_metric, 'f1_score': f1_metric}
        
        if run_dir is None:
            self._save_config()
        
        # Initialize training utilities
        self.checkpoint_manager = CheckpointManager(self.run_dir / "checkpoints")
        self.early_stopping = EarlyStopping(
            patience=config.training_config['early_stopping_patience']
        )
        
        self._setup_scheduler()
        self.history = self._initialize_history()
        self._setup_logging()
        
        # Cross-validation specific attributes
        self.fold_histories: List[Dict] = []
        self.current_fold: Optional[int] = None
        self.n_folds = config.data_config.get('n_folds', None)

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
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'fold_idx': None  # Track which fold this history belongs to
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

    def _save_plots(self, save_dir: Path = None):
        """Generate and save training visualization plots."""
        save_dir = save_dir or self.run_dir
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(save_dir / "loss_plot.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        for metric_name, values in self.history['metrics'].items():
            if metric_name != 'loss':
                plt.plot(values, label=metric_name)
        plt.title('Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig(save_dir / "metrics_plot.png")
        plt.close()

    def _save_history(self, save_dir: Path = None):
        """Save training history to CSV and JSON."""
        save_dir = save_dir or self.run_dir
        
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4)

        history_df = pd.DataFrame({
            'epoch': range(1, len(self.history['train_loss']) + 1),
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            **{f'val_{k}': v for k, v in self.history['metrics'].items() if k != 'loss'}
        })
        history_df.to_csv(save_dir / "metrics.csv", index=False)

    def _save_cv_results(self):
        """Save cross-validation results and aggregate metrics."""
        if not self.fold_histories:
            return

        # Create CV results directory
        cv_dir = self.run_dir / "cross_validation"
        cv_dir.mkdir(exist_ok=True)

        # Calculate mean and std of metrics across folds
        cv_metrics = {
            'val_loss': {'values': [], 'mean': 0.0, 'std': 0.0},
            'val_accuracy': {'values': [], 'mean': 0.0, 'std': 0.0},
            'val_f1_score': {'values': [], 'mean': 0.0, 'std': 0.0}
        }

        # Collect final metrics from each fold
        for fold_idx, history in enumerate(self.fold_histories):
            cv_metrics['val_loss']['values'].append(min(history['val_loss']))
            cv_metrics['val_accuracy']['values'].append(max(history['metrics']['accuracy']))
            cv_metrics['val_f1_score']['values'].append(max(history['metrics']['f1_score']))

        # Calculate statistics
        for metric_name, metric_data in cv_metrics.items():
            values = np.array(metric_data['values'])
            metric_data['mean'] = float(np.mean(values))
            metric_data['std'] = float(np.std(values))

        # Save CV metrics
        with open(cv_dir / "cv_results.json", "w", encoding="utf-8") as f:
            json.dump(cv_metrics, f, indent=4)

        # Create CV metrics plot
        plt.figure(figsize=(12, 6))
        metrics_df = pd.DataFrame({
            'Fold': range(1, len(self.fold_histories) + 1),
            'Validation Loss': cv_metrics['val_loss']['values'],
            'Accuracy': cv_metrics['val_accuracy']['values'],
            'F1 Score': cv_metrics['val_f1_score']['values']
        })
        
        metrics_df_melted = pd.melt(metrics_df, id_vars=['Fold'], 
                                  var_name='Metric', value_name='Value')
        
        sns.boxplot(data=metrics_df_melted, x='Metric', y='Value')
        plt.title('Cross-Validation Metrics Distribution')
        plt.savefig(cv_dir / "cv_metrics_boxplot.png")
        plt.close()

        # Log CV summary
        self._log("\nCross-Validation Results Summary:")
        for metric_name, metric_data in cv_metrics.items():
            self._log(f"{metric_name}:")
            self._log(f"  Mean: {metric_data['mean']:.4f}")
            self._log(f"  Std:  {metric_data['std']:.4f}")

    def train_fold(self, fold_idx: int, data_module, num_epochs: int) -> Dict:
        """Train model on a specific cross-validation fold.
        
        Args:
            fold_idx: Index of the current fold
            data_module: DataModule instance with cross-validation support
            num_epochs: Number of epochs to train
            
        Returns:
            Dictionary containing training history for this fold
        """
        self._log(f"\nStarting training for fold {fold_idx + 1}/{self.n_folds}")
        
        # Reset model and optimizer state
        self.model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        self.optimizer = AdamW(self.model.parameters(), lr=float(self.config.training_config['learning_rate']))
        self._setup_scheduler()
        self.early_stopping = EarlyStopping(patience=self.config.training_config['early_stopping_patience'])
        
        # Set up fold-specific directory and history
        fold_dir = self.run_dir / f"fold_{fold_idx + 1}"
        fold_dir.mkdir(exist_ok=True)
        
        # Initialize new history for this fold
        self.history = self._initialize_history()
        self.history['fold_idx'] = fold_idx
        
        # Set the current fold in the data module and get dataloaders
        data_module.set_fold(fold_idx)
        dataloaders = data_module.get_dataloaders()
        
        # Train the model for this fold
        self.train(
            train_loader=dataloaders['train'],
            val_loader=dataloaders['test'],
            num_epochs=num_epochs,
            fold_dir=fold_dir
        )
        
        # Store fold history
        self.fold_histories.append(self.history)
        
        return self.history

    def train_cv(self, data_module, num_epochs: int, folds: List[int] = None) -> List[Dict]:
        """Execute complete cross-validation training.
        
        Args:
            data_module: DataModule instance with cross-validation support
            num_epochs: Number of epochs to train each fold
            folds: Optional list of specific fold indices to run, if None runs all folds
            
        Returns:
            List of training histories for each fold
        """
        if self.n_folds is None:
            raise ValueError("Cross-validation not enabled. Initialize trainer with n_folds parameter.")

        # Determine which folds to run
        if folds is None:
            folds_to_run = list(range(self.n_folds))
        else:
            folds_to_run = folds
            # Validate fold indices
            invalid_folds = [f for f in folds_to_run if f < 0 or f >= self.n_folds]
            if invalid_folds:
                raise ValueError(f"Invalid fold indices: {invalid_folds}. Must be between 0 and {self.n_folds-1}")

        self._log(f"\nStarting cross-validation on {len(folds_to_run)} of {self.n_folds} folds")
        self._log(f"Running folds: {folds_to_run}")
        
        # Load existing fold histories if any
        if not hasattr(self, 'fold_histories') or not self.fold_histories:
            self.fold_histories = []
        
        # Ensure fold_histories has entries for all folds
        if len(self.fold_histories) < self.n_folds:
            # Initialize with empty entries for all folds
            self.fold_histories = [None] * self.n_folds

        for fold_idx in folds_to_run:
            self.current_fold = fold_idx
            fold_history = self.train_fold(fold_idx, data_module, num_epochs)
            self.fold_histories[fold_idx] = fold_history
            self._log(f"Completed fold {fold_idx + 1}/{self.n_folds}")

        # Filter out any None entries from fold_histories (folds that weren't run)
        self.fold_histories = [h for h in self.fold_histories if h is not None]
        
        # Save and visualize cross-validation results
        self._save_cv_results()
        
        return self.fold_histories

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
        start_epoch: int = 0,
        fold_dir: Optional[Path] = None
    ) -> Dict:
        """Execute complete training loop with validation and checkpointing.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Maximum number of training epochs
            log_interval: Number of epochs between logging updates
            start_epoch: Starting epoch (for resuming training)
            fold_dir: Optional directory for fold-specific files
            
        Returns:
            Dictionary containing training history
        """
        best_val_loss = float('inf')
        start_time = datetime.now()
        
        # Use fold directory if provided, otherwise use run directory
        save_dir = fold_dir if fold_dir is not None else self.run_dir
        
        if start_epoch > 0 and self.history['val_loss']:
            best_val_loss = min(self.history['val_loss'])
            self._log(f"Resuming training from epoch {start_epoch}, best val_loss: {best_val_loss:.4f}")
        else:
            self._log(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start = datetime.now()
            
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['loss']
            
            if epoch >= len(self.history['train_loss']):
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['epoch_times'].append(
                    (datetime.now() - epoch_start).total_seconds()
                )
            else:
                self.history['train_loss'][epoch] = train_loss
                self.history['val_loss'][epoch] = val_loss
                self.history['epoch_times'][epoch] = (datetime.now() - epoch_start).total_seconds()
            
            for metric_name, value in val_metrics.items():
                if metric_name not in self.history['metrics']:
                    self.history['metrics'][metric_name] = []
                
                if epoch >= len(self.history['metrics'][metric_name]):
                    self.history['metrics'][metric_name].append(value)
                else:
                    self.history['metrics'][metric_name][epoch] = value
            
            self._save_plots(save_dir)
            self._save_history(save_dir)
            
            if (epoch + 1) % log_interval == 0:
                self._log(f"Epoch {epoch + 1}/{num_epochs}")
                self._log(f"Train Loss: {train_loss:.4f}")
                self._log(f"Val Loss: {val_loss:.4f}")
                for metric_name, value in val_metrics.items():
                    if metric_name != 'loss':
                        self._log(f"Val {metric_name}: {value:.4f}")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
                'history': self.history,
                'fold_idx': self.current_fold
            }
            if self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
            latest_checkpoint_path = save_dir / "checkpoints" / "latest_checkpoint.pt"
            latest_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, latest_checkpoint_path)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._log(f"New best model (val_loss: {val_loss:.4f})")
                
                best_model_path = save_dir / "best-model.pt"
                torch.save(checkpoint, best_model_path)
                self._log(f"Saved best model to {best_model_path}")
            
            if self.early_stopping(val_loss):
                self._log("Early stopping triggered!")
                break
            
            if self.scheduler is not None:
                self.scheduler.step(metrics=val_loss)
        
        training_time = datetime.now() - start_time
        self._log(f"Training completed in {training_time}")
        self._log(f"Best validation loss: {best_val_loss:.4f}")
        
        return self.history