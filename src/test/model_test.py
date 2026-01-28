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

import torch

from usyd_learning.ml_algorithms import TokenizerBuilder, LossFunctionBuilder, OptimizerBuilder
from usyd_learning.ml_data_loader.dataset_loader_factory import DatasetLoaderFactory
from usyd_learning.ml_models import NNModelFactory
from usyd_learning.model_trainer.model_trainer_factory import ModelTrainerFactory
from usyd_learning.model_trainer.model_trainer_args import ModelTrainerArgs
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


def _count_labels(loader) -> Tuple[int, int]:
	"""Count label distribution in a DataLoader (labels in batch[1])."""
	from collections import Counter
	ctr = Counter()
	for _, labels in loader:
		lbl = labels
		if hasattr(lbl, "tolist"):
			lbl = lbl.tolist()
		ctr.update(lbl)
	return ctr.get(0, 0), ctr.get(1, 0)


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


def train_and_eval(epochs: int = 50, batch_size: int = 32, max_len: int = 128, lr: float = 2e-5, weight_decay: float = 0.01):
	device = "cuda" if torch.cuda.is_available() else "cpu"

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
	train_y0, train_y1 = _count_labels(train_dl)
	dev_y0, dev_y1 = _count_labels(dev_dl)
	console.info(f"CoLA label balance - train: 0={train_y0}, 1={train_y1}; dev: 0={dev_y0}, 1={dev_y1}")

	# 3) Model / Optimizer / Loss
	model = build_model(pad_id).to(device)
	optimizer = build_optimizer(model, lr, weight_decay)
	loss_fn = build_loss()

	# 4) Trainer
	trainer_args = ModelTrainerFactory.create_args({"trainer": {"trainer_type": "imdb", "device": device}}, is_clone_dict=True)
	trainer_args.set_trainer_args(model, optimizer, loss_fn, cola_loader, trainer_type="imdb")
	trainer_args.device = device
	trainer = ModelTrainerFactory.create(trainer_args)

	console.info(f"Training TinyBERT on CoLA for {epochs} epochs (device={device})")
	trainer.train(epochs)

	# 5) Evaluation
	evaluator = ModelEvaluator(model, dev_dl, criterion=loss_fn, device=device)
	metrics = evaluator.evaluate()
	console.info(f"Dev metrics: {metrics}")
	console.info(f"Prediction distribution (dev): {metrics['total_test_samples']} samples | MCC={metrics.get('mcc')} | accuracy={metrics.get('accuracy')}")
	return metrics


if __name__ == "__main__":
	train_and_eval()
