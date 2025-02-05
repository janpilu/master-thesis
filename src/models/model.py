"""Base model architecture for hate speech classification using transformer encoders."""

import torch.nn as nn
from transformers import AutoModel
from ..utils.config import Config
from .classification_head import DynamicClassificationHead

class HateSpeechClassifier(nn.Module):
    """Neural network model for hate speech classification.
    
    Combines a pretrained transformer model with a custom classification head
    for binary hate speech detection.
    """

    def __init__(self, config: Config, classification_head: nn.Module):
        """Initialize using Config object."""
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.model_config['name'])
        self.classification_head = classification_head
        
        if config.model_config['freeze_bert']:
            for param in self.bert.parameters():
                param.requires_grad = False

    @classmethod
    def from_params(
        cls,
        model_name: str,
        classification_head: nn.Module,
        freeze_bert: bool = True
    ):
        """Initialize using individual parameters.
        
        Args:
            model_name: Identifier of the pretrained transformer model
            classification_head: Neural network module for classification
            freeze_bert: Whether to freeze the transformer parameters
        """
        instance = cls.__new__(cls)
        super(cls, instance).__init__()
        instance.bert = AutoModel.from_pretrained(model_name)
        instance.classification_head = classification_head
        
        if freeze_bert:
            for param in instance.bert.parameters():
                param.requires_grad = False
        return instance

    def forward(self, input_ids, attention_mask):
        """Process input through transformer and classification head.
        
        Args:
            input_ids: Tensor of token indices
            attention_mask: Tensor indicating valid input tokens
            
        Returns:
            Tensor of classification logits
        """
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True  # Get all layer outputs
        )
        if isinstance(self.classification_head, DynamicClassificationHead) and \
           getattr(self.classification_head, 'use_all_layers', False):
            return self.classification_head(outputs.last_hidden_state[:, 0, :], outputs.hidden_states)
        return self.classification_head(outputs.last_hidden_state[:, 0, :])
