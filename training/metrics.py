# metrics.py
import torch
from typing import Tuple, Dict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

class ClassificationMetrics:
    """
    Collection of classification metrics for binary and multi-class classification
    """
    
    @staticmethod
    def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate accuracy for batch"""
        predictions = torch.argmax(outputs, dim=1)
        return (predictions == targets).float().mean()

    @staticmethod
    def precision_recall_f1(
        outputs: torch.Tensor,
        targets: torch.Tensor,
        average: str = 'binary'
    ) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score"""
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets,
            predictions,
            average=average,
            zero_division=0
        )
        return precision, recall, f1

    @staticmethod
    def get_all_metrics(
        outputs: torch.Tensor,
        targets: torch.Tensor,
        average: str = 'binary'
    ) -> Dict[str, float]:
        """Calculate all metrics at once"""
        accuracy = ClassificationMetrics.accuracy(outputs, targets)
        precision, recall, f1 = ClassificationMetrics.precision_recall_f1(
            outputs,
            targets,
            average=average
        )
        
        return {
            'accuracy': accuracy.item(),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

# Example metric functions that can be passed to the Trainer
def accuracy_metric(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Simple accuracy metric for trainer"""
    return ClassificationMetrics.accuracy(outputs, targets).item()

def f1_metric(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """F1 score metric for trainer"""
    _, _, f1 = ClassificationMetrics.precision_recall_f1(outputs, targets)
    return f1

# Usage example with trainer:
def main():
    # Define metrics dictionary
    metrics = {
        'accuracy': accuracy_metric,
        'f1_score': f1_metric
    }
    
    # Initialize trainer with metrics
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        metrics=metrics,
        early_stopping_patience=7,
        checkpoint_dir='checkpoints'
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=30
    )
    
    # The trainer will now track both accuracy and F1 score during training
    # Access metrics from history:
    print("Final validation metrics:")
    print(f"Accuracy: {history['metrics']['accuracy'][-1]:.4f}")
    print(f"F1 Score: {history['metrics']['f1_score'][-1]:.4f}")

# For more detailed evaluation after training
class ModelEvaluator:
    """
    Helper class for detailed model evaluation
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def evaluate_batch(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model on entire dataset and return detailed metrics
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
        
        # Concatenate all batches
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Get detailed metrics
        metrics = ClassificationMetrics.get_all_metrics(all_outputs, all_targets)
        
        return metrics
    
    def get_prediction_probabilities(
        self,
        text: str,
        tokenizer
    ) -> Dict[str, float]:
        """
        Get prediction probabilities for a single text input
        """
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs, dim=1)
        
        return {
            'toxic_probability': probabilities[0][1].item(),
            'non_toxic_probability': probabilities[0][0].item()
        }

# Example usage of the evaluator:
def evaluate_model():
    model.eval()
    evaluator = ModelEvaluator(model, device)
    
    # Get detailed metrics
    metrics = evaluator.evaluate_batch(test_loader)
    print("\nTest Set Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Test individual examples
    test_texts = [
        "You are all wonderful people!",
        "I hate everyone from that country."
    ]
    
    print("\nIndividual Predictions:")
    for text in test_texts:
        probs = evaluator.get_prediction_probabilities(text, tokenizer)
        print(f"\nText: {text}")
        print(f"Toxic probability: {probs['toxic_probability']:.4f}")
        print(f"Non-toxic probability: {probs['non_toxic_probability']:.4f}")