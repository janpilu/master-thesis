"""Configuration handler for model training and evaluation settings."""

from pathlib import Path
import yaml
from typing import Dict, Any

class Config:
    """Configuration manager for loading and accessing YAML-based settings.
    
    Provides structured access to model, training, and data configuration
    parameters loaded from a YAML file.
    """
    
    def __init__(self, config_path: str):
        """Initialize configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and parse YAML configuration file.
        
        Returns:
            Dictionary containing configuration parameters
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration parameters."""
        return self.config['model']
    
    @property
    def training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration parameters."""
        return self.config['training']
    
    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data processing configuration parameters."""
        return self.config['data']
    
    @property
    def paths(self) -> Dict[str, str]:
        """Get path configurations for data and model artifacts."""
        return self.config['paths']
    
    @property
    def device(self) -> str:
        """Get the computation device ('cuda' if available, else 'cpu')."""
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu" 