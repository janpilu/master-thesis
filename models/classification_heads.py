import torch.nn as nn


class SimpleLinearHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


class MLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        hidden_dim_2: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim_2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class LSTMHead(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.classifier(x[:, -1, :])
