"""
Standalone CoLA classification test (non-federated) using TinyBERT.

Pipeline:
1) Build HF tokenizer (bert-base-uncased) via TokenizerBuilder.
2) Load CoLA with HF collate into train/test DataLoaders.
3) Create TinyBERT classifier (prajjwal1/bert-tiny) via NNModelFactory.
4) Train with AdamW and CrossEntropyLoss for a few epochs.
5) Evaluate on CoLA dev set and print metrics.
"""

from __future__ import annotations

# Init startup path, change current path to startup python file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path

startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from typing import Dict, Any, Tuple
import math

import torch
from transformers import get_linear_schedule_with_warmup

from usyd_learning.ml_algorithms import TokenizerBuilder, LossFunctionBuilder, OptimizerBuilder
from usyd_learning.ml_data_loader.dataset_loader_factory import DatasetLoaderFactory
from usyd_learning.ml_models import NNModelFactory
from usyd_learning.model_trainer.model_evaluator import ModelEvaluator
from usyd_learning.ml_utils import console


def build_tokenizer(tokenizer_cfg: Dict[str, Any]):
	builder = TokenizerBuilder(tokenizer_cfg)
	builder.build()
	hf_tok = builder.meta.get("hf_tokenizer")
	pad_id = int(builder.meta.get("pad_id", 0))
	if hf_tok is None:
		raise RuntimeError("TokenizerBuilder did not return an HF tokenizer; set use_hf_tokenizer=true")
	return hf_tok, pad_id


def build_dataloaders(hf_tokenizer, batch_size: int, max_len: int):
	repo_root = Path(__file__).resolve().parents[2]
	dataset_root = str(repo_root / ".dataset")

	dl_cfg = {
		"data_loader": {
			"name": "cola",
			"root": dataset_root,
			"batch_size": batch_size,
			"test_batch_size": batch_size,
			"shuffle": True,
			"num_workers": 0,
			"is_download": True,
			"train_split": "train",
			"test_split": "dev",
			"task_type": "nlp",
		}
	}

	dl_args = DatasetLoaderFactory.create_args(dl_cfg, is_clone_dict=True)
	dl_args.tokenizer = hf_tokenizer
	dl_args.max_len = max_len

	loader = DatasetLoaderFactory.create(dl_args)
	return loader.data_loader, loader.test_data_loader, loader


def build_model(pad_id: int):
	model_cfg = {
		"nn_model": {
			"name": "tiny_bert",
			"pretrained_model": "prajjwal1/bert-tiny",
			"num_classes": 2,
			"pad_id": pad_id,
		}
	}
	model_args = NNModelFactory.create_args(model_cfg, is_clone_dict=True)
	model = NNModelFactory.create(model_args)
	return model


def build_optimizer(model, lr: float, weight_decay: float):
	opt_cfg = {
		"optimizer": {
			"type": "adamw",
			"lr": lr,
			"weight_decay": weight_decay,
			"betas": [0.9, 0.999],
			"eps": 1e-8,
			"amsgrad": False,
		}
	}
	return OptimizerBuilder(model.parameters(), opt_cfg).build()


def build_loss():
	loss_cfg = {"loss_func": {"type": "CrossEntropyLoss", "reduction": "mean"}}
	return LossFunctionBuilder.build(loss_cfg)


def train_and_eval(
	epochs: int = 3,
	batch_size: int = 32,
	max_len: int = 128,
	lr: float = 2e-5,
	weight_decay: float = 0.01,
	warmup_ratio: float = 0.06,
	max_grad_norm: float = 1.0,
	seed: int = 42,
):
	device = "cuda" if torch.cuda.is_available() else "cpu"
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	# 1) Tokenizer
	tokenizer_cfg = {
		"tokenizer": {
			"type": "bert",
			"use_hf_tokenizer": True,
			"pretrained": "bert-base-uncased",
			"use_fast": True,
			"return_type": "tokens",
			"max_len": max_len,
		}
	}
	hf_tok, pad_id = build_tokenizer(tokenizer_cfg)

	# 2) Data
	train_dl, dev_dl, cola_loader = build_dataloaders(hf_tok, batch_size, max_len)

	# 3) Model / Optimizer / Loss / Scheduler
	model = build_model(pad_id).to(device)
	optimizer = build_optimizer(model, lr, weight_decay)
	loss_fn = build_loss()

	# Steps/epoch estimation: prefer loader metadata, fallback to dataset len, then batch_size heuristic
	steps_per_epoch = None
	data_num = getattr(cola_loader, "data_sample_num", None)
	if isinstance(data_num, int) and data_num > 0:
		steps_per_epoch = math.ceil(data_num / max(batch_size, 1))
	if steps_per_epoch is None:
		try:
			steps_per_epoch = len(train_dl)
		except Exception:
			pass
	if steps_per_epoch is None:
		ds = getattr(train_dl, "dataset", None)
		try:
			steps_per_epoch = math.ceil(len(ds) / max(batch_size, 1)) if ds is not None else None
		except Exception:
			pass
	if steps_per_epoch is None:
		steps_per_epoch = 100  # safe fallback
	total_steps = steps_per_epoch * epochs
	warmup_steps = int(total_steps * warmup_ratio)
	scheduler = get_linear_schedule_with_warmup(
		optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
	)

	# 4) Manual training loop (baseline-style)
	console.info(
		f"Training TinyBERT on CoLA for {epochs} epochs | lr={lr} wd={weight_decay} warmup={warmup_ratio} bs={batch_size} device={device}"
	)
	model.train()
	for epoch in range(1, epochs + 1):
		total_loss = 0.0
		for batch in train_dl:
			inputs, labels = batch
			if isinstance(inputs, dict):
				inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
			else:
				inputs = inputs.to(device)
			labels = labels.to(device)

			optimizer.zero_grad()
			outputs = model(inputs)
			loss = loss_fn(outputs, labels)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
			optimizer.step()
			scheduler.step()

			total_loss += float(loss.detach().item()) * labels.size(0)

		# avg_loss = total_loss / max(train_y0 + train_y1, 1)
		# console.info(f"Epoch {epoch}/{epochs} train_loss={avg_loss:.4f}")

	# 5) Evaluation
	model.eval()
	evaluator = ModelEvaluator(model, dev_dl, criterion=loss_fn, device=device)
	metrics = evaluator.evaluate()
	console.info(f"Dev metrics: {metrics}")
	console.info(
		f"Prediction distribution (dev): {metrics['total_test_samples']} samples | MCC={metrics.get('mcc')} | accuracy={metrics.get('accuracy')}"
	)
	return metrics


if __name__ == "__main__":
	train_and_eval(epochs = 3)
