from __future__ import annotations
from pathlib import Path
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

        # Prefer offline models placed under ml_models/hf_models/<model_name>. If not present, download into that folder.
        base_dir = Path(__file__).resolve().parents[2]
        local_dir = base_dir / "hf_models" / model_name
        local_dir.mkdir(parents=True, exist_ok=True)

        # Detect whether usable local weights exist
        has_safetensors = (local_dir / "model.safetensors").exists()
        has_bin = (local_dir / "pytorch_model.bin").exists()
        has_local = has_safetensors or has_bin
        model_source = local_dir if has_local else model_name

        config = AutoConfig.from_pretrained(
            model_source,
            num_labels=num_labels,
            cache_dir=local_dir,
            local_files_only=has_local,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            config=config,
            use_safetensors=has_safetensors or not has_local,  # if only bin exists, allow bin
            cache_dir=local_dir,
            local_files_only=has_local,
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
        # Allow calling with HF BatchEncoding / dict / tokenizers.Encoding
        try:
            from transformers.tokenization_utils_base import BatchEncoding  # type: ignore
        except Exception:
            BatchEncoding = tuple()  # type: ignore

        if isinstance(input_ids, BatchEncoding):
            if attention_mask is None and "attention_mask" in input_ids:
                attention_mask = input_ids["attention_mask"]
            if "input_ids" in input_ids:
                input_ids = input_ids["input_ids"]
            else:
                first = next((v for v in input_ids.values() if torch.is_tensor(v)), None)
                if first is None:
                    raise TypeError("BatchEncoding missing tensor values for input_ids")
                input_ids = first
        elif isinstance(input_ids, dict):
            if attention_mask is None and "attention_mask" in input_ids:
                attention_mask = input_ids["attention_mask"]
            if "input_ids" in input_ids:
                input_ids = input_ids["input_ids"]
            elif "ids" in input_ids:
                input_ids = input_ids["ids"]
        elif hasattr(input_ids, "ids"):
            input_ids = torch.as_tensor(getattr(input_ids, "ids"))

        if not torch.is_tensor(input_ids):
            input_ids = torch.as_tensor(input_ids)
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        if attention_mask is None:
            if self.pad_id is None:
                raise ValueError("pad_id is None; provide attention_mask explicitly.")
            attention_mask = (input_ids != int(self.pad_id)).long()
        else:
            if not torch.is_tensor(attention_mask):
                attention_mask = torch.as_tensor(attention_mask)
            if attention_mask.dtype != torch.long:
                attention_mask = attention_mask.long()

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return outputs.logits