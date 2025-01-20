"""Prediction utilities for hate speech detection."""

import torch
from transformers import AutoTokenizer
from typing import Union, List, Dict
from ..models.model import HateSpeechClassifier


class HateSpeechPredictor:
    """Class for making predictions with a trained model"""

    def __init__(
        self,
        model: HateSpeechClassifier,
        tokenizer_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 128,
    ):
        self.model = model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.device = device
        self.max_length = max_length
        self.model.eval()

    def predict(self, text: Union[str, List[str]]) -> Dict:
        """
        Make prediction for input text(s)
        """
        # Handle single string input
        if isinstance(text, str):
            text = [text]

        # Tokenize
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Move inputs to device
        # inputs = {k: v.to(self.device) for k, v in inputs.items()}

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

        # Prepare results
        results = []
        for i in range(len(text)):
            results.append(
                {
                    "text": text[i],
                    "prediction": predictions[i].item(),
                    "toxic_probability": probabilities[i][1].item(),
                    "non_toxic_probability": probabilities[i][0].item(),
                }
            )

        return results[0] if isinstance(text, str) else results
