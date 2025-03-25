"""Stormfront dataset implementation for hate speech classification."""

from typing import Dict, Union, Callable, Optional, List
from torch.utils.data import DataLoader
import torch
from datasets import load_dataset, Dataset
from src.data.dataset import BaseDataset
from src.utils.config import Config

class StormfrontDataset(BaseDataset):
    """Dataset class for loading and processing Stormfront hate speech data.
    
    Handles loading of the Stormfront dataset (odegiber/hate_speech18) and
    converting categorical labels to binary classification format.
    """
    
    def __init__(
        self,
        split: str,
        tokenizer_name: str,
        max_length: int = 128,
        label_mapping: Dict[str, int] = None,
        preprocessing_fn: Optional[Callable] = None,
        dataset: Optional[Dataset] = None
    ):
        """Initialize Stormfront dataset for a specific split.
        
        Args:
            split: Dataset split ('train' or 'test')
            tokenizer_name: Name of the pretrained tokenizer
            max_length: Maximum sequence length
            label_mapping: Dictionary mapping original labels to integers
            preprocessing_fn: Optional text preprocessing function
            dataset: Optional pre-loaded dataset (to avoid loading from scratch)
        """
        super().__init__(tokenizer_name, max_length, preprocessing_fn)
        self.split = split
        
        # Allow passing in a pre-loaded dataset to avoid loading issues
        if dataset is not None:
            self.dataset = dataset
        else:
            # Only attempt to load if not provided (and handle potential errors)
            try:
                self.dataset = load_dataset("odegiber/hate_speech18", trust_remote_code=True)[split]
            except KeyError:
                # Initialize with empty dataset if split doesn't exist
                # (it will be set later in StormfrontDataModule.setup())
                self.dataset = None
        
        # Default label mapping (binary classification)
        self.label_mapping = label_mapping or {
            "hate": 1,      # Hate speech
            "noHate": 0,    # Not hate speech
            "relation": 0,  # Consider relation as non-hate for binary classification
            "idk/skip": 0   # Consider unknown/skipped as non-hate for binary classification
        }

    def _get_label(self, row: Dict) -> torch.Tensor:
        """Convert dataset labels to binary format.
        
        Args:
            row: Dictionary containing sample data
            
        Returns:
            Binary tensor label (0 or 1)
        """
        label_str = row["label"]
        label = self.label_mapping.get(label_str, 0)  # Default to 0 for unexpected labels
        return torch.tensor(label, dtype=torch.long)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset.
        
        Returns:
            Dictionary containing tokenized input, label, and metadata
        """
        if self.dataset is None:
            raise RuntimeError(f"Dataset for split '{self.split}' has not been properly initialized")
            
        row = self.dataset[idx]
        encoding = self._tokenize(row["text"])
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": self._get_label(row),
            "text": row["text"],
            "metadata": {
                "user_id": row["user_id"],
                "subforum_id": row["subforum_id"],
                "num_contexts": row["num_contexts"],
                "original_label": row["label"],
            },
        }
    
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        if self.dataset is None:
            return 0
        return len(self.dataset)

class StormfrontDataModule:
    """Data module for managing Stormfront dataset splits and dataloaders."""
    
    def __init__(self, config: Config):
        """Initialize using Config object."""
        self.tokenizer_name = config.model_config['name']
        self.batch_size = config.training_config['batch_size']
        self.max_length = config.training_config['max_length']
        self.num_workers = config.training_config['num_workers']
        
        # Optional custom label mapping from config
        self.label_mapping = config.data_config.get('label_mapping', None)
        
        # Train/test split ratio - default to 80% train, 20% test
        self.test_size = config.data_config.get('test_size', 0.2)
        
        self.preprocessing_fn = None
        self.datasets = {}

    @classmethod
    def from_params(
        cls,
        tokenizer_name: str,
        batch_size: int = 32,
        max_length: int = 128,
        label_mapping: Dict[str, int] = None,
        test_size: float = 0.2,
        num_workers: int = 4,
        preprocessing_fn: Optional[Callable] = None,
    ):
        """Initialize using individual parameters."""
        instance = cls.__new__(cls)
        instance.tokenizer_name = tokenizer_name
        instance.batch_size = batch_size
        instance.max_length = max_length
        instance.label_mapping = label_mapping
        instance.test_size = test_size
        instance.num_workers = num_workers
        instance.preprocessing_fn = preprocessing_fn
        instance.datasets = {}
        return instance

    def setup(self):
        """Initialize dataset splits.
        
        Since the Stormfront dataset only has a 'train' split, we'll load it
        and then split it into train/test according to the test_size ratio.
        """
        # Load the full dataset
        full_dataset = load_dataset("odegiber/hate_speech18", trust_remote_code=True)['train']
        
        # Split into train and test sets
        train_test_split = full_dataset.train_test_split(test_size=self.test_size, seed=42)
        
        # Create our dataset objects with the pre-split datasets
        self.datasets['train'] = StormfrontDataset(
            split='train',
            tokenizer_name=self.tokenizer_name,
            max_length=self.max_length,
            label_mapping=self.label_mapping,
            preprocessing_fn=self.preprocessing_fn,
            dataset=train_test_split['train']  # Pass the dataset directly
        )
        
        self.datasets['test'] = StormfrontDataset(
            split='test',
            tokenizer_name=self.tokenizer_name,
            max_length=self.max_length,
            label_mapping=self.label_mapping,
            preprocessing_fn=self.preprocessing_fn,
            dataset=train_test_split['test']  # Pass the dataset directly
        )

    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """Create DataLoader instances for each dataset split."""
        return {
            split: DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=(split == 'train'),
                num_workers=self.num_workers,
            )
            for split, dataset in self.datasets.items()
        }
