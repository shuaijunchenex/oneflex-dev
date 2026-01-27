from __future__ import annotations
from typing import Any, Optional
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
        self.model: Optional[nn.Module] = None
        self.pad_id: Optional[int] = None  # use model/tokenizer pad id when possible

    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)
        from transformers import AutoModelForSequenceClassification, AutoConfig

        model_name = getattr(args, "pretrained_model", "bert-base-uncased")
        num_labels = getattr(args, "num_classes", 2)

        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)

        # Don't force safetensors; let HF handle what's available.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            use_safetensors=True,
        )

        # Prefer model-config pad token id (more reliable than args default)
        self.pad_id = getattr(self.model.config, "pad_token_id", None)
        # fallback to args if config missing
        if self.pad_id is None:
            self.pad_id = getattr(args, "pad_id", 0)

        return self

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        **kwargs: Any
    ):
        if attention_mask is None:
            if self.pad_id is None:
                raise ValueError("pad_id is None; provide attention_mask explicitly.")
            attention_mask = (input_ids != int(self.pad_id)).long()

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return outputs.logits