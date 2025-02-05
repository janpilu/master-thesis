"""Factory module for creating model instances based on configuration."""

from typing import Dict
import torch.nn as nn
from .classification_head import DynamicClassificationHead
from .model import HateSpeechClassifier
from ..utils.config import Config

class ModelFactory:
    """Factory class for creating model components and assemblies."""
    
    @staticmethod
    def create_model(config: Config) -> HateSpeechClassifier:
        """Create a complete model instance based on configuration.
        
        Args:
            config: Configuration object containing model settings
            
        Returns:
            Configured HateSpeechClassifier instance
        """
        classification_head = DynamicClassificationHead(config.model_config['classification_head'])
        return HateSpeechClassifier(config, classification_head)