"""Evaluation metrics for hate speech classification model."""

import torch
from typing import Dict
from sklearn.metrics import f1_score, accuracy_score

def accuracy_metric(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate classification accuracy.
    
    Args:
        outputs: Model prediction logits
        targets: Ground truth labels
        
    Returns:
        Classification accuracy score
    """
    predictions = torch.argmax(outputs, dim=1)
    return accuracy_score(targets.cpu(), predictions.cpu())

def f1_metric(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate binary F1 score.
    
    Args:
        outputs: Model prediction logits
        targets: Ground truth labels
        
    Returns:
        F1 score for positive class
    """
    predictions = torch.argmax(outputs, dim=1)
    return f1_score(targets.cpu(), predictions.cpu(), average='binary')

class ModelEvaluator:
    """Comprehensive model evaluation with detailed metrics."""
    
    def __init__(self, model, device):
        """Initialize evaluator with model and device.
        
        Args:
            model: Neural network model to evaluate
            device: Device to run evaluation on ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def evaluate_batch(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate model on entire dataset and return detailed metrics.
        
        Args:
            dataloader: DataLoader containing evaluation data
            
        Returns:
            Dictionary containing various evaluation metrics
        """
        all_outputs = []
        all_targets = []
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = batch["labels"].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            
            all_outputs.append(outputs)
            all_targets.append(targets)
        
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        return {
            'accuracy': accuracy_metric(all_outputs, all_targets),
            'f1_score': f1_metric(all_outputs, all_targets)
        }