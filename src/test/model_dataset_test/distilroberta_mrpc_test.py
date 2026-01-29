"""
Standalone MRPC (Microsoft Research Paraphrase Corpus) classification test using RoBERTa-base.

Pipeline:
1) Build HF tokenizer (roberta-base) via TokenizerBuilder.
2) Load MRPC with HF collate into train/test DataLoaders.
3) Create RoBERTa-base classifier via NNModelFactory.
4) Train with AdamW and CrossEntropyLoss for a few epochs.
5) Evaluate on MRPC dev set and print metrics.

MRPC Task: Binary classification to determine if two sentences are semantically equivalent.
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
			"name": "mrpc",
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


def build_model(pad_id: int, num_classes: int = 2):
	model_cfg = {
		"nn_model": {
			"name": "transformer_classification",
			"pretrained_model": "distilroberta-base",
			"num_classes": num_classes,
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
	epochs: int = 60,
	batch_size: int = 32,
	max_len: int = 128,
	lr: float = 5e-5,
	weight_decay: float = 0.01,
	warmup_ratio: float = 0.1,
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
			"pretrained": "roberta-base",
			"use_fast": True,
			"return_type": "tokens",
			"max_len": max_len,
		}
	}
	hf_tok, pad_id = build_tokenizer(tokenizer_cfg)

	# 2) Data - MRPC returns IterDataPipe which exhausts after 1 iteration
	train_dl, dev_dl, loader = build_dataloaders(hf_tok, batch_size, max_len)
	
	# Convert IterDataPipe to list for reusable dataset
	train_dataset_list = list(train_dl.dataset) if hasattr(train_dl, 'dataset') else list(train_dl)
	dev_dataset_list = list(dev_dl.dataset) if hasattr(dev_dl, 'dataset') else list(dev_dl)
	console.info(f"Loaded {len(train_dataset_list)} training samples and {len(dev_dataset_list)} dev samples from MRPC")
	
	# Recreate DataLoader with list-based dataset and proper collate_fn
	from torch.utils.data import DataLoader
	from usyd_learning.ml_data_loader.dataset_loader_util import DatasetLoaderUtil
	
	# MRPC format: (label, sent1, sent2, idx) - use idx_label_text format
	collate_fn = lambda batch: DatasetLoaderUtil.text_collate_fn_hf(
		batch,
		hf_tokenizer=hf_tok,
		max_len=max_len,
		label_map=None,
		normalize_int_labels=False,
		tuple_format="auto",  # Auto-detect based on tuple length
		require_labels=True
	)
	
	train_dl = DataLoader(
		train_dataset_list,
		batch_size=batch_size,
		shuffle=True,
		num_workers=0,
		collate_fn=collate_fn
	)
	
	dev_dl = DataLoader(
		dev_dataset_list,
		batch_size=batch_size,
		shuffle=False,
		num_workers=0,
		collate_fn=collate_fn
	)

	# 3) Model / Optimizer / Loss / Scheduler
	model = build_model(pad_id, num_classes=2).to(device)
	optimizer = build_optimizer(model, lr, weight_decay)
	loss_fn = build_loss()

	# Steps/epoch estimation from materialized dataset
	steps_per_epoch = len(train_dl)
	total_steps = steps_per_epoch * epochs
	warmup_steps = int(total_steps * warmup_ratio)
	scheduler = get_linear_schedule_with_warmup(
		optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
	)

	# 4) Manual training loop with validation and early stopping
	console.info(
		f"Training distilroberta-base on MRPC for {epochs} epochs | lr={lr} wd={weight_decay} warmup={warmup_ratio} bs={batch_size} device={device}"
	)
	
	# Early stopping parameters - use F1 for MRPC
	best_f1 = -1.0
	patience = 10  # Larger patience for small dataset
	patience_counter = 0
	best_model_state = None
	
	for epoch in range(1, epochs + 1):
		# Training phase
		model.train()
		total_loss = 0.0
		num_batches = 0
		num_samples = 0
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

			batch_samples = labels.size(0)
			total_loss += float(loss.detach().item()) * batch_samples
			num_batches += 1
			num_samples += batch_samples

		avg_loss = total_loss / max(num_samples, 1)
		
		# Validation phase
		model.eval()
		evaluator = ModelEvaluator(model, dev_dl, criterion=loss_fn, device=device)
		val_metrics = evaluator.evaluate()
		val_mcc = val_metrics.get('mcc', 0.0)
		val_acc = val_metrics.get('accuracy', 0.0)
		val_loss = val_metrics.get('average_loss', 0.0)
		val_f1 = val_metrics.get('f1_score', 0.0)
		
		console.info(
			f"Epoch {epoch}/{epochs} | train_loss={avg_loss:.4f} | "
			f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | val_mcc={val_mcc:.4f}"
		)
		
		# Early stopping logic based on F1 (common metric for MRPC)
		if val_f1 > best_f1:
			best_f1 = val_f1
			patience_counter = 0
			best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
			console.info(f"  *** New best F1: {best_f1:.4f} ***")
		else:
			patience_counter += 1
			console.info(f"  No improvement. Patience: {patience_counter}/{patience}")
			if patience_counter >= patience:
				console.info(f"Early stopping triggered at epoch {epoch}")
				break

	# 5) Final Evaluation with best model
	if best_model_state is not None:
		console.info("\nLoading best model for final evaluation...")
		model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
	
	model.eval()
	evaluator = ModelEvaluator(model, dev_dl, criterion=loss_fn, device=device)
	metrics = evaluator.evaluate()
	console.info("\n" + "="*60)
	console.info("FINAL EVALUATION RESULTS (MRPC):")
	console.info(f"Dev metrics: {metrics}")
	console.info(
		f"Samples: {metrics['total_test_samples']} | "
		f"MCC: {metrics.get('mcc', 0):.4f} | "
		f"Accuracy: {metrics.get('accuracy', 0):.4f} | "
		f"F1: {metrics.get('f1_score', 0):.4f}"
	)
	console.info("="*60)
	return metrics


if __name__ == "__main__":
	# MRPC is a small dataset, use smaller batch size
	metrics = train_and_eval(epochs=50, batch_size=32)
