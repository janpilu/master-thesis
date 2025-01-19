"""Early stopping utility to prevent model overfitting."""

import numpy as np

class EarlyStopping:
    """Monitors validation metrics to determine when to stop training.
    
    Tracks a metric (usually validation loss) and signals when to stop training 
    if no improvement is seen for a specified number of epochs.
    """
    
    def __init__(self, patience: int = 7, min_delta: float = 0, mode: str = 'min'):
        """Initialize early stopping handler.
        
        Args:
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = np.inf if mode == 'min' else -np.inf
        
    def __call__(self, value: float) -> bool:
        """Check if training should stop based on monitored value.
        
        Args:
            value: Current value of monitored metric
            
        Returns:
            True if training should stop, False otherwise
        """
        if (self.mode == 'min' and value < self.best_value - self.min_delta) or \
           (self.mode == 'max' and value > self.best_value + self.min_delta):
            self.best_value = value
            self.counter = 0
            return False
            
        self.counter += 1
        return self.counter >= self.patience 