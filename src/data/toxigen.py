"""ToxiGen dataset implementation for hate speech classification."""

from typing import Dict, Union, Callable, Optional
from torch.utils.data import DataLoader
import torch
from datasets import load_dataset
from src.data.dataset import BaseDataset
from src.utils.config import Config

class ToxiGenDataset(BaseDataset):
    """Dataset class for loading and processing ToxiGen data.
    
    Handles loading of the ToxiGen dataset and conversion of toxicity scores
    to binary labels using configurable strategies.
    """
    
    def __init__(
        self,
        split: str,
        tokenizer_name: str,
        max_length: int = 128,
        label_strategy: Union[str, Callable] = "toxicity_human",
        label_threshold: float = 0.5,
        preprocessing_fn: Optional[Callable] = None
    ):
        """Initialize ToxiGen dataset for a specific split.
        
        Args:
            split: Dataset split ('train' or 'test')
            tokenizer_name: Name of the pretrained tokenizer
            max_length: Maximum sequence length
            label_strategy: Strategy for converting toxicity scores to labels
            label_threshold: Threshold for binary classification
            preprocessing_fn: Optional text preprocessing function
        """
        super().__init__(tokenizer_name, max_length, preprocessing_fn)
        self.dataset = load_dataset("toxigen/toxigen-data", "annotated")[split]
        self.label_strategy = label_strategy
        self.label_threshold = label_threshold

    def _get_label(self, row: Dict) -> torch.Tensor:
        """Convert dataset row to binary label using the specified strategy.
        
        Args:
            row: Dictionary containing sample data
            
        Returns:
            Binary tensor label (0 or 1)
        """
        if isinstance(self.label_strategy, str):
            label = row[self.label_strategy]
            if isinstance(label, float):
                label = int(label >= self.label_threshold)
        elif callable(self.label_strategy):
            label = self.label_strategy(row)
        else:
            raise ValueError("Invalid label_strategy")
        return torch.tensor(label, dtype=torch.long)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset.
        
        Returns:
            Dictionary containing tokenized input, label, and metadata
        """
        row = self.dataset[idx]
        encoding = self._tokenize(row["text"])
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": self._get_label(row),
            "text": row["text"],
            "metadata": {
                "target_group": row["target_group"],
                "intent": row["intent"],
                "stereotyping": row["stereotyping"],
            },
        }

class ToxiGenDataModule:
    """Data module for managing ToxiGen dataset splits and dataloaders."""
    
    def __init__(self, config: Config):
        """Initialize using Config object."""
        self.tokenizer_name = config.model_config['name']
        self.batch_size = config.training_config['batch_size']
        self.max_length = config.training_config['max_length']
        self.label_strategy = config.data_config['label_strategy']
        self.label_threshold = config.data_config.get('threshold', 0.5)
        self.num_workers = config.training_config['num_workers']
        self.datasets = {}

    @classmethod
    def from_params(
        cls,
        tokenizer_name: str,
        batch_size: int = 32,
        max_length: int = 128,
        label_strategy: Union[str, Callable] = "toxicity_human",
        label_threshold: float = 0.5,
        num_workers: int = 4,
    ):
        """Initialize using individual parameters.
        
        Args:
            tokenizer_name: Name of the pretrained tokenizer
            batch_size: Batch size for dataloaders
            max_length: Maximum sequence length
            label_strategy: Strategy for converting toxicity scores to labels
            label_threshold: Threshold for binary classification
            num_workers: Number of worker processes for data loading
        """
        instance = cls.__new__(cls)
        instance.tokenizer_name = tokenizer_name
        instance.batch_size = batch_size
        instance.max_length = max_length
        instance.label_strategy = label_strategy
        instance.label_threshold = label_threshold
        instance.num_workers = num_workers
        instance.datasets = {}
        return instance

    def setup(self):
        """Initialize train and test dataset splits."""
        for split in ['train', 'test']:
            self.datasets[split] = ToxiGenDataset(
                split=split,
                tokenizer_name=self.tokenizer_name,
                max_length=self.max_length,
                label_strategy=self.label_strategy,
                label_threshold=self.label_threshold,
                preprocessing_fn=self.preprocessing_fn
            )

    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """Create DataLoader instances for each dataset split.
        
        Returns:
            Dictionary mapping split names to DataLoader instances
        """
        return {
            split: DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=(split == 'train'),
                num_workers=self.num_workers,
            )
            for split, dataset in self.datasets.items()
        } 
