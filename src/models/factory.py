"""Factory module for creating model instances based on configuration."""

from typing import Dict
import torch.nn as nn
from src.models.classification_heads import SimpleLinearHead, MLPHead
from src.models.model import HateSpeechClassifier
from src.utils.config import Config

class ModelFactory:
    """Factory class for creating model components and assemblies."""
    
    @staticmethod
    def create_classification_head(config: Dict) -> nn.Module:
        """Create a classification head based on configuration.
        
        Args:
            config: Dictionary containing model configuration parameters
            
        Returns:
            Configured classification head module
            
        Raises:
            ValueError: If specified head type is not recognized
        """
        head_type = config['classification_head']['type']
        input_dim = config['classification_head']['input_dim']
        
        if head_type == 'simple':
            return SimpleLinearHead(input_dim, config['num_classes'])
        elif head_type == 'mlp':
            return MLPHead(
                input_dim=input_dim,
                hidden_dim=config['classification_head']['hidden_dim'],
                hidden_dim_2=config['classification_head']['hidden_dim_2'],
                num_classes=config['num_classes'],
                dropout=config['classification_head']['dropout']
            )
        else:
            raise ValueError(f"Unknown classification head type: {head_type}")

    @staticmethod
    def create_model(config: Config) -> HateSpeechClassifier:
        """Create a complete model instance based on configuration.
        
        Args:
            config: Configuration object containing model settings
            
        Returns:
            Configured HateSpeechClassifier instance
        """
        classification_head = ModelFactory.create_classification_head(config.model_config)
        return HateSpeechClassifier(config, classification_head)