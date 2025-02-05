import torch
from src.data.toxigen import ToxiGenDataModule
from src.models.factory import ModelFactory
from src.training.trainer import Trainer
from src.utils.config import Config
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
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
    
    device = get_device()
    logger.info(f"Using device: {device}")

    data_module = ToxiGenDataModule(config)
    data_module.setup()
    dataloaders = data_module.get_dataloaders()

    model = ModelFactory.create_model(config)
    trainer = Trainer(model, config, device)

    trainer.train(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["test"],
        num_epochs=config.training_config["num_epochs"]
    )
    logger.info(f"Completed training run with config: {config_path}")

def main():
    """Run training for all config files in config directory."""
    config_dir = Path("config")
    config_files = list(config_dir.glob("*.yaml"))
    
    if not config_files:
        raise ValueError("No config files found in config directory")
    
    logger.info(f"Found {len(config_files)} config files: {[f.name for f in config_files]}")
    
    for config_path in config_files:
        try:
            train_model(config_path)
        except Exception as e:
            logger.error(f"Error training with config {config_path}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
