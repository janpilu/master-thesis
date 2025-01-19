"""Training script for hate speech classification model.
Handles model initialization, training setup, and execution using configuration from YAML."""

from src.utils.config import Config
from src.models.factory import ModelFactory
from src.data.toxigen import ToxiGenDataModule
from src.training.trainer import Trainer

def main():
    """Initialize and train the hate speech classification model using configuration parameters."""
    config = Config('config/config.yaml')
    
    data_module = ToxiGenDataModule(config)
    data_module.setup()
    dataloaders = data_module.get_dataloaders()
    
    model = ModelFactory.create_model(config).to(config.device)
    trainer = Trainer(model, config)
    
    trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['test'],
        num_epochs=config.training_config['num_epochs']
    )

if __name__ == "__main__":
    main() 