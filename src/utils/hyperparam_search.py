"""Hyperparameter search utilities for running multiple model configurations."""

import itertools
import os
import copy
import yaml
import datetime
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple
import logging
import random
import numpy as np
import json

logger = logging.getLogger(__name__)

class HyperParamSearch:
    """Handles hyperparameter grid search by generating multiple configurations from a base config."""
    
    def __init__(self, base_config_path: Union[str, Path]):
        """Initialize hyperparameter search with a base configuration.
        
        Args:
            base_config_path: Path to base YAML configuration file
        """
        self.base_config_path = Path(base_config_path)
        self.runs_dir = Path("runs")
        
        # Load the base configuration
        with open(self.base_config_path, 'r', encoding='utf-8') as f:
            self.base_config = yaml.safe_load(f)
    
    def _extract_param_variations(self) -> Dict[str, List[Any]]:
        """Extract parameter variations from the config file.
        
        Searches the configuration for lists of values which will be used
        for hyperparameter variations.
        
        Returns:
            Dictionary mapping parameter paths to lists of values
        """
        param_variations = {}
        self._find_list_params(self.base_config, "", param_variations)
        return param_variations
    
    def _find_list_params(self, config_section: Dict, prefix: str, param_variations: Dict[str, List[Any]]):
        """Recursively search for list parameters in the config.
        
        Args:
            config_section: Current section of the config being examined
            prefix: Prefix for parameter paths (used for recursion)
            param_variations: Dictionary to store found parameter variations
        """
        for key, value in config_section.items():
            current_path = f"{prefix}.{key}" if prefix else key
            
            # If it's a list that's not a list of dictionaries, it might be a parameter variation
            if isinstance(value, list) and (not value or not isinstance(value[0], dict)):
                # Only consider it a parameter variation if it has multiple values
                if len(value) > 1:
                    param_variations[current_path] = value
            
            # Recursively search nested dictionaries
            elif isinstance(value, dict):
                self._find_list_params(value, current_path, param_variations)
            
    def _get_param_path(self, param_key: str) -> Tuple[Dict, str]:
        """Get the nested dictionary and last key for a parameter path.
        
        Args:
            param_key: Dot-separated parameter path (e.g., 'training.learning_rate')
            
        Returns:
            Tuple of (parent dict containing the parameter, parameter name)
            
        Raises:
            KeyError: If parameter path is invalid
        """
        path_parts = param_key.split('.')
        current = self.base_config
        
        # Navigate to the parent dictionary
        for part in path_parts[:-1]:
            if part not in current:
                raise KeyError(f"Invalid parameter path: {param_key}. '{part}' not found.")
            current = current[part]
            
        last_key = path_parts[-1]
        if last_key not in current:
            raise KeyError(f"Invalid parameter path: {param_key}. '{last_key}' not found.")
            
        return current, last_key
    
    def _ensure_single_values(self, config: Dict):
        """Recursively ensure all parameters in the config are single values, not lists.
        
        This converts any remaining lists (that weren't part of the parameter variations)
        to their first value to avoid comparison errors during training.
        
        Args:
            config: Configuration dictionary to process
        """
        for key, value in list(config.items()):
            if isinstance(value, list) and (not value or not isinstance(value[0], dict)):
                # Convert list to its first value
                config[key] = value[0]
                logger.warning(f"Converted list parameter '{key}' to single value: {value[0]}")
            elif isinstance(value, dict):
                # Recursively process nested dictionaries
                self._ensure_single_values(value)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # Process lists of dictionaries (e.g., model architecture layers)
                for item in value:
                    if isinstance(item, dict):
                        self._ensure_single_values(item)
            
    def create_grid_search(self, param_variations: Dict[str, List[Any]] = None) -> List[Dict]:
        """Create a list of configurations for a grid search.
        
        Args:
            param_variations: Dictionary mapping parameter paths to lists of values.
                If None, parameter variations will be extracted from the base config.
                e.g., {'training.learning_rate': [1e-3, 1e-4, 1e-5],
                        'training.batch_size': [16, 32, 64]}
                
        Returns:
            List of configurations, each with a specific combination of parameter values
        """
        # If no parameter variations provided, extract them from the config
        if param_variations is None:
            param_variations = self._extract_param_variations()
            
        if not param_variations:
            logger.warning("No parameter variations found or provided. Returning original config.")
            config = copy.deepcopy(self.base_config)
            # Ensure all parameters are single values
            self._ensure_single_values(config)
            return [(config, {})]
            
        # Validate all parameter paths before proceeding
        param_paths = []
        for param_key in param_variations:
            parent_dict, last_key = self._get_param_path(param_key)
            param_paths.append((param_key, parent_dict, last_key))
            
        # Get all parameter combinations
        param_keys = list(param_variations.keys())
        param_values = list(param_variations.values())
        combinations = list(itertools.product(*param_values))
        
        # Create configurations for each combination
        configurations = []
        for combo in combinations:
            config = copy.deepcopy(self.base_config)
            config_params = {}
            
            # Set parameter values for this combination
            for i, (param_key, parent_dict, last_key) in enumerate(param_paths):
                param_value = combo[i]
                
                # Update the new config
                current = config
                path_parts = param_key.split('.')
                for part in path_parts[:-1]:
                    current = current[part]
                current[path_parts[-1]] = param_value
                
                # Store parameters used for this combination
                config_params[param_key] = param_value
            
            # Ensure all parameters are single values
            self._ensure_single_values(config)
                
            configurations.append((config, config_params))
            
        return configurations
        
    def run_grid_search(self, param_variations: Dict[str, List[Any]] = None, run_prefix: str = "run") -> List[str]:
        """Generate and save configurations for a grid search.
        
        Args:
            param_variations: Dictionary mapping parameter paths to lists of values.
                If None, parameter variations will be extracted from the base config.
            run_prefix: Prefix for run directory names
            
        Returns:
            List of created run directory paths
        """
        # If no parameter variations provided, extract them from the config
        if param_variations is None:
            param_variations = self._extract_param_variations()
        
        configurations = self.create_grid_search(param_variations)
        run_dirs = []
        
        if len(configurations) == 1 and not configurations[0][1]:
            logger.warning("No parameter variations found. No runs created.")
            return run_dirs
        
        # Create a parent directory for the entire hyperparameter search
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        parent_dir_name = f"hypersearch_{timestamp}"
        parent_dir = self.runs_dir / parent_dir_name
        os.makedirs(parent_dir, exist_ok=True)
        
        # Save the original config with all variations
        original_config_path = parent_dir / "original_config.yaml"
        with open(original_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.base_config, f, default_flow_style=False)
        
        # Save the parameter variations
        param_variations_path = parent_dir / "param_variations.json"
        with open(param_variations_path, 'w', encoding='utf-8') as f:
            # Convert any non-serializable values to strings
            serializable_variations = {}
            for key, values in param_variations.items():
                serializable_variations[key] = [str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v for v in values]
            json.dump(serializable_variations, f, indent=4)
        
        # Create a run directory for each configuration
        for i, (config, config_params) in enumerate(configurations):
            # Generate a random seed for reproducibility
            random_seed = random.randint(1, 100000)
            
            # Add the random seed to the configuration
            if 'training' not in config:
                config['training'] = {}
            config['training']['random_seed'] = random_seed
            
            # Create a descriptive run name with hyperparameters
            param_str = "_".join([f"{k.split('.')[-1]}_{v}" for k, v in config_params.items()])
            
            # Use a simpler naming scheme for subfolders
            run_name = f"run_{i+1}_{param_str}" if param_str else f"run_{i+1}"
            
            # Create run directory as a subfolder of the parent directory
            run_dir = parent_dir / run_name
            os.makedirs(run_dir, exist_ok=True)
            run_dirs.append(str(run_dir))
            
            # Save the configuration
            config_file = run_dir / "config.yaml"
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Save the parameter variations for this specific run
            run_params_file = run_dir / "run_params.json"
            with open(run_params_file, 'w', encoding='utf-8') as f:
                run_info = {
                    "run_index": i + 1,
                    "total_runs": len(configurations),
                    "parameters": {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v for k, v in config_params.items()},
                    "random_seed": random_seed
                }
                json.dump(run_info, f, indent=4)
            
            logger.info(f"Created configuration {i+1}/{len(configurations)}: {run_name}")
            logger.info(f"Random seed: {random_seed}")
        
        # Save a summary of all runs
        summary_path = parent_dir / "runs_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            summary = {
                "timestamp": timestamp,
                "total_runs": len(configurations),
                "base_config": str(self.base_config_path),
                "runs": [
                    {
                        "run_index": i + 1,
                        "run_dir": os.path.basename(run_dir),
                        "parameters": {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v for k, v in configs[1].items()},
                        "random_seed": configs[0]['training']['random_seed']
                    }
                    for i, (run_dir, configs) in enumerate(zip(run_dirs, configurations))
                ]
            }
            json.dump(summary, f, indent=4)
        
        logger.info(f"Created hyperparameter search directory: {parent_dir}")
        logger.info(f"Total configurations: {len(configurations)}")
        
        return run_dirs 