"""
Improved Model 1: DistillBERT

Run:
    python train.py
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Any, Iterable, List
from transformers import DistilBertModel, DistilBertTokenizerFast


PRETRAINED   = "distilbert-base-uncased"
MAX_LEN      = 64          # headlines are short; 64 tokens is plenty
HIDDEN_DIM   = 768         # DistilBERT hidden size
DROPOUT      = 0.3
NUM_CLASSES  = 2           # 0 = NBC, 1 = FoxNews
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

class Model(nn.Module):
    """
    Template model for the leaderboard.

    Requirements:
    - Must be instantiable with no arguments (called by the evaluator).
    - Must implement `predict(batch)` which receives an iterable of inputs and
      returns a list of predictions (labels).
    - Must implement `eval()` to place the model in evaluation mode.
    - If you use PyTorch, submit a state_dict to be loaded via `load_state_dict`
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder    = DistilBertModel.from_pretrained(PRETRAINED)
        self.tokenizer  = DistilBertTokenizerFast.from_pretrained(PRETRAINED)

        self.classifier = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, 256),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, NUM_CLASSES),
        )
 
        self.to(DEVICE)
 

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids      : (B, L) token ids
            attention_mask : (B, L) attention mask
        Returns:
            logits         : (B, 2) raw class scores
        """
        outputs   = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]   # [CLS] vector
        logits    = self.classifier(cls_token)
        return logits
    
    def _tokenize(self, texts: List[str]) -> dict:
        return self.tokenizer(
            texts,
            padding        = True,
            truncation     = True,
            max_length     = MAX_LEN,
            return_tensors = "pt",
        )

    def eval(self) -> "Model":
        super().eval()
        return self

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        texts = list(batch)
        if not texts:
            return []
 
        self.eval()
        encoded = self._tokenize(texts)
        input_ids      = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)
 
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)   # (B, 2)
            preds  = logits.argmax(dim=-1).cpu().tolist()      # list[int]
 
        return preds
    
        
def get_model() -> Model:
    """
    Factory function required by the evaluator.
    Returns an uninitialized model instance. The evaluator may optionally load
    weights (if provided) before calling predict(...).
    """
    return Model()


