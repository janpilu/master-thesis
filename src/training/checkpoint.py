"""Checkpoint management for model training state."""

from pathlib import Path
import json
from datetime import datetime
import torch
from typing import Dict, Optional

class CheckpointManager:
    """Handles saving and loading of model training state."""
    
    def __init__(self, checkpoint_dir: str):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        val_loss: float,
        history: Dict,
        metrics: Optional[Dict] = None,
        is_best: bool = False
    ) -> str:
        """Save model checkpoint and metadata.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            epoch: Current epoch number
            val_loss: Validation loss value
            history: Training history
            metrics: Optional evaluation metrics
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "best-model" if is_best else f"checkpoint_epoch_{epoch}"
        
        # Save model checkpoint
        checkpoint_path = self.checkpoint_dir / f"{prefix}_{timestamp}.pt"
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'history': history
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata
        metadata_path = checkpoint_path.with_suffix('.json')
        metadata = {
            'epoch': epoch,
            'val_loss': val_loss,
            'metrics': metrics,
            'timestamp': timestamp
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
            
        return str(checkpoint_path)
        
    def load(self, checkpoint_path: str, model, optimizer=None, scheduler=None):
        """Load model and training state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            
        Returns:
            Loaded checkpoint data
        """
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint 