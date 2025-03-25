"""Combined dataset module for training with multiple datasets."""

from typing import Dict, List, Union, Optional, Callable
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from datasets import concatenate_datasets
from src.utils.config import Config
from src.data.stormfront import StormfrontDataModule 
from src.data.toxigen import ToxiGenDataset

class CombinedDataModule:
    """Data module for combining multiple datasets with different strategies.
    
    Supports:
    - Training on one dataset, validating on another
    - Combining datasets for both training and validation
    - Mixing datasets with custom ratios
    """
    
    def __init__(self, config: Config):
        """Initialize combined data module using configuration.
        
        Args:
            config: Configuration object with dataset settings
                Expected to have a 'combined_data' section with:
                - datasets: List of dataset names to use
                - strategy: How to combine datasets ('train_on_a_validate_on_b',
                  'mix_all', or 'proportional_mix')
                - proportions: Optional dict of dataset name to sampling weight
        """
        self.config = config
        self.datasets_config = config.data_config.get('combined_data', {})
        self.strategy = self.datasets_config.get('strategy', 'mix_all')
        self.proportions = self.datasets_config.get('proportions', {})
        
        self.batch_size = config.training_config['batch_size']
        self.num_workers = config.training_config['num_workers']
        
        # Initialize component datasets
        self.dataset_modules = {}
        self.datasets = {'train': {}, 'test': {}}
        self.combined = {'train': None, 'test': None}
        
    def setup(self):
        """Set up all component datasets and combine them according to strategy."""
        # Initialize specified datasets
        dataset_names = self.datasets_config.get('datasets', ['stormfront', 'toxigen'])
        
        for name in dataset_names:
            if name.lower() == 'stormfront':
                module = StormfrontDataModule(self.config)
                module.setup()
                self.dataset_modules['stormfront'] = module
                self.datasets['train']['stormfront'] = module.datasets['train']
                self.datasets['test']['stormfront'] = module.datasets['test']
                
            elif name.lower() == 'toxigen':
                # ToxiGen dataset setup
                toxigen_train = ToxiGenDataset(
                    split='train',
                    tokenizer_name=self.config.model_config['name'],
                    max_length=self.config.training_config['max_length']
                )
                toxigen_test = ToxiGenDataset(
                    split='test',
                    tokenizer_name=self.config.model_config['name'],
                    max_length=self.config.training_config['max_length']
                )
                self.datasets['train']['toxigen'] = toxigen_train
                self.datasets['test']['toxigen'] = toxigen_test
            
            # Add other datasets as needed
        
        # Combine datasets according to strategy
        self._combine_datasets()
        
    def _combine_datasets(self):
        """Combine datasets according to the specified strategy."""
        if self.strategy == 'train_on_a_validate_on_b':
            # First dataset for training, second for validation
            dataset_names = list(self.datasets['train'].keys())
            if len(dataset_names) < 2:
                raise ValueError(f"Strategy '{self.strategy}' requires at least 2 datasets")
                
            self.combined['train'] = self.datasets['train'][dataset_names[0]]
            self.combined['test'] = self.datasets['test'][dataset_names[1]]
            
        elif self.strategy == 'mix_all':
            # Combine all datasets for both training and testing
            self.combined['train'] = ConcatDataset(list(self.datasets['train'].values()))
            self.combined['test'] = ConcatDataset(list(self.datasets['test'].values()))
            
        elif self.strategy == 'proportional_mix':
            # TO DO: Implement weighted sampling across datasets
            # This would require a custom sampler implementation
            self.combined['train'] = ConcatDataset(list(self.datasets['train'].values()))
            self.combined['test'] = ConcatDataset(list(self.datasets['test'].values()))
            
        else:
            raise ValueError(f"Unknown combination strategy: {self.strategy}")
    
    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """Create DataLoader instances for each combined dataset split."""
        if not self.combined['train'] or not self.combined['test']:
            raise RuntimeError("Datasets not properly combined. Call setup() first.")
            
        return {
            split: DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=(split == 'train'),
                num_workers=self.num_workers,
            )
            for split, dataset in self.combined.items()
        }
    
    @classmethod
    def from_datasets(
        cls,
        datasets: Dict[str, Dict[str, Dataset]],
        batch_size: int = 32,
        num_workers: int = 4,
        strategy: str = 'mix_all',
        proportions: Optional[Dict[str, float]] = None
    ):
        """Initialize from existing datasets without using Config.
        
        Args:
            datasets: Dict with 'train' and 'test' keys, each containing a dict 
                     of dataset_name -> dataset
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes
            strategy: How to combine datasets
            proportions: Optional dict of dataset name to sampling weight
        """
        instance = cls.__new__(cls)
        instance.datasets = datasets
        instance.batch_size = batch_size
        instance.num_workers = num_workers
        instance.strategy = strategy
        instance.proportions = proportions or {}
        instance.combined = {'train': None, 'test': None}
        
        # Combine datasets
        instance._combine_datasets()
        return instance 