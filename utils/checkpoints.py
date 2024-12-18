# utils/model_utils.py
import torch
from typing import Dict, Union, Optional
import json
from pathlib import Path
from transformers import AutoTokenizer
from models.model import HateSpeechClassifier
from models.classification_heads import SimpleLinearHead, MLPHead


class ModelCheckpoint:
    """Handles saving and loading of model checkpoints"""

    @staticmethod
    def save_checkpoint(
        model: HateSpeechClassifier,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        metrics: Dict,
        save_dir: str,
        config: Dict,
        name: str = "checkpoint",
    ) -> str:
        """
        Save model checkpoint and metadata
        """
        # Create save directory if it doesn't exist
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Prepare checkpoint data
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "metrics": metrics,
            "config": config,
        }

        # Save model checkpoint
        checkpoint_path = save_dir / f"{name}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save metadata
        metadata = {"epoch": epoch, "loss": loss, "metrics": metrics, "config": config}
        metadata_path = save_dir / f"{name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        return str(checkpoint_path)

    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        model: Optional[HateSpeechClassifier] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> Dict:
        """
        Load model checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if model is not None:
            model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint
