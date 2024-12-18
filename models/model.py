import torch.nn as nn
from transformers import AutoModel
from typing import Dict


class HateSpeechClassifier(nn.Module):
    def __init__(
        self, model_name: str, classification_head: nn.Module, freeze_bert: bool = True
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classification_head = classification_head

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classification_head(outputs.last_hidden_state[:, 0, :])
