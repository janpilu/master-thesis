from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Optional, Union, Dict, Callable
import torch


def custom_label_strategy(row):
    """Combine human and AI toxicity scores"""
    return int((row["toxicity_human"] + row["toxicity_ai"]) / 2 >= 1.5)


def preprocess_text(text):
    """Basic text preprocessing"""
    return text.lower().strip()


class ToxiGenDataset(Dataset):
    """
    Dataset class for ToxiGen hate speech dataset
    """

    def __init__(
        self,
        split: str,
        tokenizer_name: str,
        max_length: int = 128,
        label_strategy: Union[str, Callable] = "toxicity_human",
        label_threshold: float = 0.5,
        preprocessing_fn: Optional[Callable] = None,
    ):
        self.dataset = load_dataset("toxigen/toxigen-data", "annotated")[split]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.label_strategy = label_strategy
        self.label_threshold = label_threshold
        self.preprocessing_fn = preprocessing_fn

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_label(self, row: Dict) -> torch.Tensor:
        """Extract label based on label_strategy"""
        if isinstance(self.label_strategy, str):
            # Direct column access
            label = row[self.label_strategy]
            # Convert to binary if float
            if isinstance(label, float):
                label = int(label >= self.label_threshold)
        elif callable(self.label_strategy):
            # Custom label extraction
            label = self.label_strategy(row)
        else:
            raise ValueError("Invalid label_strategy")

        return torch.tensor(label, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.dataset[idx]
        text = row["text"]

        # Apply preprocessing if specified
        if self.preprocessing_fn is not None:
            text = self.preprocessing_fn(text)

        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Get label
        label = self._get_label(row)

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": label,
            "text": text,
            "metadata": {
                "target_group": row["target_group"],
                "intent": row["intent"],
                "stereotyping": row["stereotyping"],
            },
        }


class ToxiGenDataModule:
    def __init__(
        self,
        tokenizer_name: str,
        batch_size: int = 32,
        max_length: int = 128,
        label_strategy: Union[str, Callable] = "toxicity_human",
        label_threshold: float = 0.5,
        preprocessing_fn: Optional[Callable] = None,
        num_workers: int = 4,
    ):
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.label_strategy = label_strategy
        self.label_threshold = label_threshold
        self.preprocessing_fn = preprocessing_fn
        self.num_workers = num_workers

    def setup(self):
        """Create train and test datasets"""
        self.train_dataset = ToxiGenDataset(
            split="train",
            tokenizer_name=self.tokenizer_name,
            max_length=self.max_length,
            label_strategy=self.label_strategy,
            label_threshold=self.label_threshold,
            preprocessing_fn=self.preprocessing_fn,
        )

        self.test_dataset = ToxiGenDataset(
            split="test",
            tokenizer_name=self.tokenizer_name,
            max_length=self.max_length,
            label_strategy=self.label_strategy,
            label_threshold=self.label_threshold,
            preprocessing_fn=self.preprocessing_fn,
        )

    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """Return train and test dataloaders"""
        return {
            "train": DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            ),
            "test": DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            ),
        }


def main():
    # Initialize data module with the globally defined functions
    data_module = ToxiGenDataModule(
        tokenizer_name="bert-base-uncased",
        batch_size=32,
        label_strategy=custom_label_strategy,  # Now using the global function
        preprocessing_fn=preprocess_text,  # Now using the global function
        num_workers=2,  # Reduced number of workers for testing
    )

    # Setup datasets
    data_module.setup()

    # Get dataloaders
    dataloaders = data_module.get_dataloaders()

    # Access train and test loaders
    train_loader = dataloaders["train"]
    test_loader = dataloaders["test"]

    # Example of iterating through the data
    for batch in train_loader:
        print(f"Batch size: {batch['input_ids'].shape}")
        print(f"Labels: {batch['labels']}")
        print(f"Sample text: {batch['text'][0]}")
        print(f"Sample metadata: {batch['metadata']['target_group'][0]}")
        break


if __name__ == "__main__":
    main()
