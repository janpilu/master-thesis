from torch import nn
import torch
from typing import List, Union, Dict
import yaml

class DynamicClassificationHead(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.input_dim = config["input_dim"]
        self.use_all_layers = config.get("use_all_layers", False)
        self.num_layers = config.get("num_layers", 12)  # Default for base models
        self.layers = self._build_layers(config["architecture"])
        
    def _build_layers(self, architecture: List[Dict]) -> nn.Sequential:
        layers = []
        current_dim = self.input_dim * self.num_layers if self.use_all_layers else self.input_dim
        
        for layer_config in architecture:
            layer_type = layer_config["type"].lower()
            
            if layer_type == "linear":
                layers.append(nn.Linear(current_dim, layer_config["out_features"]))
                current_dim = layer_config["out_features"]
                
            elif layer_type == "conv1d":
                layers.append(nn.Conv1d(
                    in_channels=layer_config.get("in_channels", 1),
                    out_channels=layer_config["out_channels"],
                    kernel_size=layer_config["kernel_size"],
                    stride=layer_config.get("stride", 1),
                    padding=layer_config.get("padding", 0)
                ))
                current_dim = layer_config["out_channels"]
                
            elif layer_type == "conv2d":
                layers.append(nn.Conv2d(
                    in_channels=layer_config.get("in_channels", 1),
                    out_channels=layer_config["out_channels"],
                    kernel_size=layer_config["kernel_size"],
                    stride=layer_config.get("stride", 1),
                    padding=layer_config.get("padding", 0)
                ))
                current_dim = layer_config["out_channels"]
                
            elif layer_type == "maxpool1d":
                layers.append(nn.MaxPool1d(
                    kernel_size=layer_config["kernel_size"],
                    stride=layer_config.get("stride", None),
                    padding=layer_config.get("padding", 0)
                ))
                
            elif layer_type == "maxpool2d":
                layers.append(nn.MaxPool2d(
                    kernel_size=layer_config["kernel_size"],
                    stride=layer_config.get("stride", None),
                    padding=layer_config.get("padding", 0)
                ))
                
            elif layer_type == "flatten":
                layers.append(nn.Flatten())
                current_dim = self._calculate_flatten_size(current_dim, layer_config)
                
            elif layer_type == "dropout":
                layers.append(nn.Dropout(layer_config["p"]))
                
            elif layer_type == "relu":
                layers.append(nn.ReLU())
                
            elif layer_type == "tanh":
                layers.append(nn.Tanh())
                
            elif layer_type == "gelu":
                layers.append(nn.GELU())
                
            elif layer_type == "layernorm":
                layers.append(nn.LayerNorm(current_dim))
                
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
                
        return nn.Sequential(*layers)
    
    def _calculate_flatten_size(self, current_dim: int, config: Dict) -> int:
        """Calculate output dimension after flattening."""
        # This would need to track the spatial dimensions through conv/pool operations
        # Simplified version - override with config if needed
        return config.get("out_features", current_dim)
    
    def forward(self, x, hidden_states=None):
        if self.use_all_layers and hidden_states is not None:
            # Extract CLS tokens from each layer and stack them
            # Shape: [batch_size, num_layers, hidden_dim]
            # - Each row represents a layer's CLS token representation
            # - This creates a matrix where we can see how the CLS token evolves
            layer_outputs = torch.stack([states[:, 0, :] for states in hidden_states], dim=1)
            
            # Add channel dimension for 2D convolution
            # Shape: [batch_size, 1, num_layers, hidden_dim]
            # This creates a 2D "image" where:
            # - Each row (height) is a different transformer layer
            # - Each column (width) is a dimension in the hidden state
            # - The single channel (1) treats it like a grayscale image
            x = layer_outputs.unsqueeze(1)
            
            # The resulting matrix captures the evolution of features across layers:
            # - Vertical patterns: How features change through transformer layers
            # - Horizontal patterns: Relationships between different hidden dimensions
            # - Convolution can then find patterns in both directions simultaneously
        return self.layers(x) 