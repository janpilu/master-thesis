"""Training module for hate speech classification models."""

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict
from src.utils.config import Config
from src.training.metrics import accuracy_metric, f1_metric
from src.training.early_stopping import EarlyStopping
from src.training.checkpoint import CheckpointManager

class Trainer:
    """Handles model training and evaluation."""

    def __init__(self, model, config: Config):
        """Initialize trainer with model and configuration.
        
        Args:
            model: Neural network model to train
            config: Configuration object
        """
        self.model = model
        self.device = config.device
        self.optimizer = AdamW(model.parameters(), lr=config.training_config['learning_rate'])
        self.criterion = torch.nn.CrossEntropyLoss()
        self.metrics = {'accuracy': accuracy_metric, 'f1_score': f1_metric}
        
        # Initialize training utilities
        self.early_stopping = EarlyStopping(
            patience=config.training_config['early_stopping_patience']
        )
        self.checkpoint_manager = CheckpointManager(config.paths['checkpoint_dir'])
        
        # Setup scheduler if configured
        scheduler_config = config.training_config.get('scheduler', {})
        self.scheduler = None
        if scheduler_config.get('type') == 'reduce_lr_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_config['mode'],
                factor=scheduler_config['factor'],
                patience=scheduler_config['patience']
            )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'metrics': {}
        }

    @classmethod
    def from_params(cls, model, **kwargs):
        """Alternative initialization with manual parameters."""
        instance = cls.__new__(cls)
        instance.model = model
        instance.device = kwargs.get('device', 'cuda')
        instance.optimizer = kwargs.get('optimizer') or AdamW(model.parameters(), lr=2e-5)
        instance.criterion = kwargs.get('criterion') or torch.nn.CrossEntropyLoss()
        instance.metrics = kwargs.get('metrics') or {'accuracy': accuracy_metric, 'f1_score': f1_metric}
        instance.early_stopping = EarlyStopping(patience=kwargs.get('early_stopping_patience', 7))
        instance.checkpoint_manager = CheckpointManager(kwargs.get('checkpoint_dir', 'checkpoints'))
        instance.scheduler = kwargs.get('scheduler')
        instance.history = {'train_loss': [], 'val_loss': [], 'learning_rates': [], 'metrics': {}}
        return instance

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
        metrics_values = {name: 0 for name in self.metrics}
        
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
        log_interval: int = 1
    ) -> Dict:
        """Execute complete training loop with validation and checkpointing.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Maximum number of training epochs
            log_interval: Number of epochs between logging updates
            
        Returns:
            Dictionary containing training history
        """
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
                self.checkpoint_manager.save(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch=epoch,
                    val_loss=val_loss,
                    history=self.history,
                    metrics=val_metrics,
                    is_best=True
                )
            
            # Early stopping check
            if self.early_stopping(val_loss):
                print("Early stopping triggered!")
                break
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(metrics=val_loss)
        
        return self.history