from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from ... import AbstractNNModel, NNModel, NNModelArgs

"""
Generic Encoder-only Transformer for Sequence Classification.
Compatible with SST2, MRPC, DBpedia, IMDB, etc.
"""

class NNModel_TransformerClassification(NNModel):
    def __init__(self):
        super().__init__()
        self.model: nn.Module | None = None
        self.pad_id: int = 0

    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)
        from transformers import AutoModelForSequenceClassification, AutoConfig

        model_name = getattr(args, "pretrained_model", "bert-base-uncased")
        num_labels = getattr(args, "num_classes", 2) # Automatically handles 2 (SST2/MRPC) or 14 (DBpedia)
        
        # Load config first to potentially adjust settings
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            use_safetensors=True,
        )
        
        self.pad_id = getattr(args, "pad_id", 0)
        return self

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor | None = None, **kwargs: Any):
        if attention_mask is None:
            # Dynamically create mask if not provided
            attention_mask = (input_ids != self.pad_id).long()
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if hasattr(outputs, "logits"):
            return outputs.logits
        return outputs[0]
