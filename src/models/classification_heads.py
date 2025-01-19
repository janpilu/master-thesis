"""Classification head architectures for the hate speech detection model."""

import torch.nn as nn


class SimpleLinearHead(nn.Module):
    """Single linear layer classification head."""
    
    def __init__(self, input_dim: int, num_classes: int):
        """Initialize linear classifier.
        
        Args:
            input_dim: Dimension of input features
            num_classes: Number of output classes
        """
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """Apply linear transformation to input features."""
        return self.classifier(x)


class MLPHead(nn.Module):
    """Multi-layer perceptron classification head with dropout."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        hidden_dim_2: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        """Initialize MLP classifier.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of first hidden layer
            hidden_dim_2: Dimension of second hidden layer
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim_2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, num_classes),
        )

    def forward(self, x):
        """Apply MLP transformation to input features."""
        return self.classifier(x)
