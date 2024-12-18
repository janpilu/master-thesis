import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from data.dataset import HateSpeechDataset
from data.toxigen import ToxiGenDataModule, custom_label_strategy
from models.model import HateSpeechClassifier
from models.classification_heads import SimpleLinearHead, MLPHead
from training.trainer import Trainer


def main():
    config = {
        "model_name": "microsoft/deberta-v3-base",
        "num_classes": 2,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "max_length": 128,
        "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    data_module = ToxiGenDataModule(
        tokenizer_name=config["model_name"],
        batch_size=config["batch_size"],
        max_length=config["max_length"],
        label_strategy=custom_label_strategy,
        num_workers=config["num_workers"],
    )

    # Setup datasets and get dataloaders
    data_module.setup()
    dataloaders = data_module.get_dataloaders()
    train_loader = dataloaders["train"]
    val_loader = dataloaders["test"]

    classification_head = SimpleLinearHead(
        768, config["num_classes"]
    )  # 768 is BERT's hidden size
    model = HateSpeechClassifier(
        config["model_name"], classification_head, freeze_bert=True
    ).to(config["device"])

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize trainer
    trainer = Trainer(
        model=model, optimizer=optimizer, criterion=criterion, device=config["device"]
    )

    # Training loop
    for epoch in range(config["num_epochs"]):
        train_loss = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)

        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
