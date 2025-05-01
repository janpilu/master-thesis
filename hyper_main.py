"""Main script for running hyperparameter searches across multiple configurations."""

import torch
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import os
import json
import yaml
from datetime import datetime

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
    parser.add_argument('--cv', action='store_true',
                        help='Run in cross-validation mode')
    parser.add_argument('--cv-dir', type=str,
                        help='Directory for cross-validation results')
    parser.add_argument('--folds', type=str,
                        help='Comma-separated list of fold indices to run (e.g., "0,2,4"), only used with --cv')
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
    """Main entry point for the script."""
    args = parse_args()
    
    if args.cv and args.config:
        # Run in cross-validation mode
        folds = None
        if args.folds:
            try:
                folds = [int(fold) for fold in args.folds.split(',')]
                logger.info(f"Running specific folds: {folds}")
            except ValueError:
                logger.error(f"Invalid folds format: {args.folds}. Expected comma-separated integers.")
                return
                
        run_cross_validation(Path(args.config), cv_dir=args.cv_dir, folds=folds)
        return
    
    if args.continue_from:
        # Continue from existing run
        continue_hyperparam_search(Path(args.continue_from), args.restart_last)
        return
    
    if not args.config:
        logger.error("Config file not provided")
        parser.print_help()
        return
    
    # Run new hyperparameter search
    config_path = Path(args.config)
    
    variations = []
    if args.param_variations:
        for variation in args.param_variations:
            param_path, values = variation.split("=", 1)
            variations.append((param_path, values.split(",")))
    
    hp_search = HyperParamSearch(config_path, variations)
    hp_search.run(not args.no_execute)

def run_cross_validation(config_path: Path, cv_dir: str = None, folds: List[int] = None):
    """Run cross-validation using the specified config.
    
    Args:
        config_path: Path to the configuration file
        cv_dir: Optional directory for cross-validation results
        folds: Optional list of specific fold indices to run, if None runs all folds
    """
    logger.info(f"Starting cross-validation run with config: {config_path}")
    config = Config(config_path=str(config_path))
    
    if 'n_folds' not in config.data_config:
        logger.error("Cross-validation requires n_folds parameter in data config")
        return
    
    n_folds = config.data_config['n_folds']
    
    if folds:
        # Validate fold indices
        invalid_folds = [f for f in folds if f < 0 or f >= n_folds]
        if invalid_folds:
            logger.error(f"Invalid fold indices: {invalid_folds}. Must be between 0 and {n_folds-1}")
            return
        logger.info(f"Running specific folds: {folds} out of {n_folds} total folds")
    else:
        logger.info(f"Running all {n_folds} folds for cross-validation")
    
    # Set up directories
    if cv_dir:
        run_dir = Path(cv_dir)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        run_dir = Path(config.paths['runs_dir']) / f"cv_{timestamp}"
    
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_copy_path = run_dir / "config.yaml"
    with open(config_copy_path, "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
    
    # Set random seeds for reproducibility
    if 'random_seed' in config.training_config:
        seed = config.training_config['random_seed']
        logger.info(f"Setting random seed to {seed}")
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    
    # Set up model and dataloaders
    device = get_device()
    logger.info(f"Using device: {device}")
    
    data_module = ToxiGenDataModule(config)
    data_module.setup()
    
    model = ModelFactory.create_model(config)
    
    # Initialize trainer
    trainer = Trainer(model, config, device, run_dir=run_dir)
    
    # Run cross-validation
    trainer.train_cv(data_module, config.training_config['num_epochs'], folds=folds)
    
    logger.info(f"Cross-validation completed. Results saved to {run_dir}")
    logger.info("Check cross_validation/cv_results.json for aggregated metrics")

if __name__ == "__main__":
    main() 