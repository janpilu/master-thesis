"""ToxiGen dataset implementation for hate speech classification."""

from typing import Dict, Union, Callable, Optional, List
from torch.utils.data import DataLoader, Subset
import torch
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import KFold
import numpy as np
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
        if split == 'full':
            dataset = load_dataset("toxigen/toxigen-data", "annotated")
            self.dataset = concatenate_datasets([dataset['train'], dataset['test']])
        else:
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
    
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.dataset)

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
        self.n_folds = config.data_config.get('n_folds', None)  # New parameter for k-fold CV
        self.preprocessing_fn = None
        self.datasets = {}
        self.current_fold = None
        self.fold_indices = None

    @classmethod
    def from_params(
        cls,
        tokenizer_name: str,
        batch_size: int = 32,
        max_length: int = 128,
        label_strategy: Union[str, Callable] = "toxicity_human",
        label_threshold: float = 0.5,
        num_workers: int = 4,
        preprocessing_fn: Optional[Callable] = None,
        n_folds: Optional[int] = None,
    ):
        """Initialize using individual parameters.
        
        Args:
            tokenizer_name: Name of the pretrained tokenizer
            batch_size: Batch size for dataloaders
            max_length: Maximum sequence length
            label_strategy: Strategy for converting toxicity scores to labels
            label_threshold: Threshold for binary classification
            num_workers: Number of worker processes for data loading
            n_folds: Number of folds for k-fold cross-validation
        """
        instance = cls.__new__(cls)
        instance.tokenizer_name = tokenizer_name
        instance.batch_size = batch_size
        instance.max_length = max_length
        instance.label_strategy = label_strategy
        instance.label_threshold = label_threshold
        instance.num_workers = num_workers
        instance.preprocessing_fn = preprocessing_fn
        instance.n_folds = n_folds
        instance.datasets = {}
        instance.current_fold = None
        instance.fold_indices = None
        return instance

    def setup(self):
        """Initialize dataset splits."""
        for split in ['train', 'test']:
            self.datasets[split] = ToxiGenDataset(
                split=split,
                tokenizer_name=self.tokenizer_name,
                max_length=self.max_length,
                label_strategy=self.label_strategy,
                label_threshold=self.label_threshold,
                preprocessing_fn=self.preprocessing_fn
            )
        if self.n_folds is not None:
            # Original behavior: use predefined train/test splits
            # K-fold cross-validation setup
            # Load all data into a single dataset
            self.datasets['full'] = ToxiGenDataset(
                split='full',
                tokenizer_name=self.tokenizer_name,
                max_length=self.max_length,
                label_strategy=self.label_strategy,
                label_threshold=self.label_threshold,
                preprocessing_fn=self.preprocessing_fn
            )
            
            # Initialize k-fold splitter
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            
            # Generate and store fold indices
            self.fold_indices = list(kf.split(range(len(self.datasets['full']))))

    def set_fold(self, fold_idx: int):
        """Set the current fold for cross-validation.
        
        Args:
            fold_idx: Index of the fold to use (0 to n_folds-1)
        """
        if self.n_folds is None:
            raise ValueError("K-fold cross-validation not enabled. Initialize with n_folds parameter.")
        
        if not 0 <= fold_idx < self.n_folds:
            raise ValueError(f"Fold index must be between 0 and {self.n_folds-1}")
        
        self.current_fold = fold_idx

    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """Create DataLoader instances for each dataset split.
        
        Returns:
            Dictionary mapping split names to DataLoader instances
        """
        if self.n_folds is None:
            # Original behavior with train/test splits
            return {
                split: DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=(split == 'train'),
                    num_workers=self.num_workers,
                )
                for split, dataset in self.datasets.items()
            }
        else:
            # K-fold cross-validation behavior
            if self.current_fold is None:
                raise ValueError("Must call set_fold() before getting dataloaders in k-fold mode")
            
            # Get fold indices for this fold
            train_idx, val_idx = self.fold_indices[self.current_fold]
            
            # Convert NumPy int64 indices to Python ints to avoid compatibility issues
            train_idx = [int(idx) for idx in train_idx]
            val_idx = [int(idx) for idx in val_idx]
            
            return {
                'train': DataLoader(
                    Subset(self.datasets['full'], train_idx),
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                ),
                'test': DataLoader(
                    Subset(self.datasets['full'], val_idx),
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                )
            } 
