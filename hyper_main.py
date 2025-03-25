"""Main script for running hyperparameter searches across multiple configurations."""

import torch
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import os
import json
import yaml

from src.data.toxigen import ToxiGenDataModule
from src.models.factory import ModelFactory
from src.training.trainer import Trainer
from src.utils.config import Config
from src.utils.hyperparam_search import HyperParamSearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_device():
    """Get the best available device for training.
    
    Returns:
        str: 'mps', 'cuda', or 'cpu' depending on availability
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def train_model(config_path: Path):
    """Train a model using specified config file.
    
    Args:
        config_path: Path to config file
    """
    logger.info(f"Starting training run with config: {config_path}")
    config = Config(config_path=str(config_path))
    
    # Set random seeds for reproducibility
    if 'random_seed' in config.training_config:
        seed = config.training_config['random_seed']
        logger.info(f"Setting random seed to {seed}")
        import random
        import numpy as np
        import torch
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    
    device = get_device()
    logger.info(f"Using device: {device}")

    data_module = ToxiGenDataModule(config)
    data_module.setup()
    dataloaders = data_module.get_dataloaders()

    model = ModelFactory.create_model(config)
    
    # Get the run directory from the config file path
    run_dir = config_path.parent
    
    # Check for existing checkpoints
    checkpoint_path = None
    checkpoint_dir = run_dir / "checkpoints"
    best_model_path = run_dir / "best-model.pt"
    latest_checkpoint_path = checkpoint_dir / "latest_checkpoint.pt"
    
    # First check for best model
    if best_model_path.exists():
        checkpoint_path = best_model_path
        logger.info(f"Found best model checkpoint: {checkpoint_path}")
    # Then check for latest checkpoint
    elif latest_checkpoint_path.exists():
        checkpoint_path = latest_checkpoint_path
        logger.info(f"Found latest checkpoint: {checkpoint_path}")
    
    # Initialize trainer
    trainer = Trainer(model, config, device, run_dir=run_dir)
    
    # Load checkpoint if available
    start_epoch = 0
    if checkpoint_path:
        logger.info(f"Loading from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and trainer.scheduler:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming from epoch {start_epoch}")
        
        # Update trainer history
        if 'history' in checkpoint:
            trainer.history = checkpoint['history']
            logger.info(f"Loaded training history with {len(trainer.history['train_loss'])} epochs")

    trainer.train(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["test"],
        num_epochs=config.training_config["num_epochs"],
        start_epoch=start_epoch
    )
    logger.info(f"Completed training run with config: {config_path}")

def get_first_config_file():
    """Get the first config file in the config directory.
    
    Returns:
        Path to the first config file found, or None if no config files exist
    """
    config_dir = Path("config")
    if not config_dir.exists():
        logger.warning(f"Config directory {config_dir} does not exist")
        return None
        
    config_files = list(config_dir.glob("*.yaml"))
    if not config_files:
        logger.warning(f"No config files found in {config_dir}")
        return None
        
    # Sort to ensure consistent behavior
    config_files.sort()
    return config_files[0]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run hyperparameter search')
    parser.add_argument('--config', type=str, required=False, 
                        help='Base configuration file path (if not provided, uses first config in config folder)')
    parser.add_argument('--run-prefix', type=str, default='run',
                        help='Prefix for run directory names')
    parser.add_argument('--param-variations', type=str, nargs='+', required=False,
                        help='Parameter variations in format: param_path=val1,val2,val3 (optional, can be defined in config)')
    parser.add_argument('--no-execute', action='store_true',
                        help='Skip executing training runs (by default, runs are executed)')
    parser.add_argument('--continue-from', type=str, required=False,
                        help='Continue from a specific hyperparameter search directory')
    parser.add_argument('--restart-last', action='store_true',
                        help='When continuing, restart the last run that was in progress')
    return parser.parse_args()

def parse_param_variations(param_variations_args: List[str]) -> Dict[str, List[Any]]:
    """Parse parameter variations from command line arguments.
    
    Args:
        param_variations_args: List of parameter variation strings in format:
                               'param_path=val1,val2,val3'
                               
    Returns:
        Dictionary mapping parameter paths to lists of values
    """
    param_variations = {}
    for param_var in param_variations_args:
        parts = param_var.split('=')
        if len(parts) != 2:
            raise ValueError(f"Invalid parameter variation format: {param_var}")
        
        param_path = parts[0]
        values_str = parts[1]
        
        # Try to convert string values to appropriate types
        values = []
        for val_str in values_str.split(','):
            # Try to convert to numeric types if possible
            try:
                # First try int
                values.append(int(val_str))
            except ValueError:
                try:
                    # Then try float
                    values.append(float(val_str))
                except ValueError:
                    # Fall back to string if not numeric
                    values.append(val_str)
        
        param_variations[param_path] = values
    
    return param_variations

def find_incomplete_runs(hypersearch_dir: Path) -> List[Path]:
    """Find runs that did not complete successfully.
    
    Args:
        hypersearch_dir: Path to hyperparameter search directory
        
    Returns:
        List of paths to incomplete run directories
    """
    # Get all run directories
    run_dirs = [d for d in hypersearch_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    incomplete_runs = []
    
    for run_dir in run_dirs:
        # Check if the run has a training.log file with completion message
        log_file = run_dir / "training.log"
        completed = False
        
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                    if "Training completed in" in log_content:
                        completed = True
                        logger.info(f"Run {run_dir.name} has completion message")
            except Exception as e:
                logger.error(f"Error reading log file {log_file}: {e}")
        
        # A run is incomplete if it doesn't have a completion message in the log
        if not completed:
            incomplete_runs.append(run_dir)
            logger.info(f"Run {run_dir.name} is incomplete: no completion message in log")
    
    return incomplete_runs

def continue_hyperparam_search(hypersearch_dir: Path, restart_last: bool = False):
    """Continue a hyperparameter search from a specific directory.
    
    Args:
        hypersearch_dir: Path to hyperparameter search directory
        restart_last: Whether to restart the last run that was in progress
    """
    logger.info(f"Continuing hyperparameter search from: {hypersearch_dir}")
    
    # Find incomplete runs
    incomplete_runs = find_incomplete_runs(hypersearch_dir)
    
    if not incomplete_runs:
        logger.info("All runs completed successfully. Nothing to continue.")
        return
    
    logger.info(f"Found {len(incomplete_runs)} incomplete runs.")
    
    # Sort runs by index
    incomplete_runs.sort(key=lambda x: int(x.name.split('_')[1]))
    
    for run_dir in incomplete_runs:
        config_path = run_dir / "config.yaml"
        
        if restart_last:
            # Delete the run directory and recreate it
            import shutil
            logger.info(f"Restarting run: {run_dir}")
            
            # Load the run parameters to preserve the random seed
            run_params_file = run_dir / "run_params.json"
            if run_params_file.exists():
                with open(run_params_file, 'r', encoding='utf-8') as f:
                    run_params = json.load(f)
                    random_seed = run_params.get("random_seed")
            else:
                random_seed = None
            
            # Backup the original config
            config_backup = None
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_backup = yaml.safe_load(f)
            
            # Delete the run directory
            shutil.rmtree(run_dir)
            
            # Recreate the directory
            os.makedirs(run_dir, exist_ok=True)
            
            # Restore the config with the same random seed
            if config_backup:
                if random_seed and 'training' in config_backup:
                    config_backup['training']['random_seed'] = random_seed
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_backup, f, default_flow_style=False)
                
                # Restore the run parameters file
                if run_params_file.exists():
                    with open(run_params_file, 'w', encoding='utf-8') as f:
                        json.dump(run_params, f, indent=4)
        
        # Train the model
        try:
            logger.info(f"Continuing run: {run_dir}")
            train_model(config_path)
        except Exception as e:
            logger.error(f"Error training with config {config_path}: {str(e)}")
            continue

def main():
    """Run hyperparameter search based on command line arguments."""
    args = parse_args()
    
    # Check if we're continuing from a previous run
    if args.continue_from:
        hypersearch_dir = Path(args.continue_from)
        if not hypersearch_dir.exists() or not hypersearch_dir.is_dir():
            logger.error(f"Hyperparameter search directory not found: {hypersearch_dir}")
            return
        
        continue_hyperparam_search(hypersearch_dir, args.restart_last)
        return
    
    # If no config provided, use the first config file in the config folder
    config_path = args.config
    if not config_path:
        first_config = get_first_config_file()
        if not first_config:
            logger.error("No config file provided and no config files found in config directory")
            return
        config_path = str(first_config)
        logger.info(f"No config file provided, using first config file: {config_path}")
    
    logger.info(f"Running hyperparameter search with base config: {config_path}")
    
    # Create hyperparameter search
    search = HyperParamSearch(config_path)
    
    # Parse parameter variations from command line if provided
    param_variations = None
    if args.param_variations:
        param_variations = parse_param_variations(args.param_variations)
        logger.info(f"Using parameter variations from command line: {param_variations}")
    else:
        logger.info("No command line parameter variations provided, using variations from config file if any")
    
    # Run grid search (will automatically extract variations from config if param_variations is None)
    run_dirs = search.run_grid_search(param_variations, args.run_prefix)
    
    if not run_dirs:
        logger.warning("No runs were created. Ensure your config has parameter variations.")
        return
        
    logger.info(f"Created {len(run_dirs)} configurations")
    
    # Execute training runs by default, unless --no-execute is specified
    if not args.no_execute:
        logger.info("Executing training runs")
        for run_dir in run_dirs:
            config_path = Path(run_dir) / "config.yaml"
            try:
                train_model(config_path)
            except Exception as e:
                logger.error(f"Error training with config {config_path}: {str(e)}")
                continue
    else:
        logger.info("Skipping training runs (--no-execute specified)")
        logger.info(f"Run directories created: {run_dirs}")

if __name__ == "__main__":
    main() 