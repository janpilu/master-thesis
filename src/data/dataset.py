"""Base dataset module providing common text tokenization functionality for classification tasks."""

from typing import Dict, Optional, Callable
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

class BaseDataset(Dataset):
    """Abstract base class for text classification datasets.
    
    Provides common tokenization functionality while requiring subclasses
    to implement data loading and label handling.
    """
    
    def __init__(
        self,
        tokenizer_name: str,
        max_length: int = 128,
        preprocessing_fn: Optional[Callable] = None
    ):
        """Initialize tokenizer and preprocessing settings.
        
        Args:
            tokenizer_name: Name/path of the pretrained tokenizer
            max_length: Maximum sequence length for tokenization
            preprocessing_fn: Optional function to preprocess text before tokenization
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.preprocessing_fn = preprocessing_fn

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text using the initialized tokenizer.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Dictionary containing input_ids and attention_mask tensors
        """
        if self.preprocessing_fn:
            text = self.preprocessing_fn(text)
            
        return self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Required abstract method from Dataset class."""
        raise NotImplementedError("Subclasses must implement __getitem__")

    def __len__(self) -> int:
        """Required abstract method from Dataset class."""
        raise NotImplementedError("Subclasses must implement __len__") 