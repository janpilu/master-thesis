import torch
from src.data.toxigen import ToxiGenDataModule
from src.models.factory import ModelFactory
from src.training.trainer import Trainer
from src.utils.config import Config


def main():
    config = Config(config_path="config/config.yaml")

    data_module = ToxiGenDataModule(config)

    # Setup datasets and get dataloaders
    data_module.setup()
    dataloaders = data_module.get_dataloaders()
    train_loader = dataloaders["train"]
    val_loader = dataloaders["test"]

    model = ModelFactory.create_model(config)

    # Initialize trainer
    trainer = Trainer(model, config)

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training_config["num_epochs"]
    )


if __name__ == "__main__":
    main()
