"""
Standalone AG News topic classification test using TinyBERT.

Pipeline:
1) Build HF tokenizer (bert-base-uncased) via TokenizerBuilder.
2) Load AG News with HF collate into train/test DataLoaders.
3) Create TinyBERT classifier (prajjwal1/bert-tiny) via NNModelFactory.
4) Train with AdamW and CrossEntropyLoss for a few epochs.
5) Evaluate on AG News test set and print metrics.

AG News Task: 4-way news topic classification.
"""

from __future__ import annotations

# Init startup path, change current path to startup python file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path

startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from typing import Dict, Any

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
			"name": "agnews",
			"root": dataset_root,
			"batch_size": batch_size,
			"test_batch_size": batch_size,
			"shuffle": True,
			"num_workers": 0,
			"is_download": True,
			"train_split": "train",
			"test_split": "test",
			"task_type": "nlp",
		}
	}

	dl_args = DatasetLoaderFactory.create_args(dl_cfg, is_clone_dict=True)
	dl_args.tokenizer = hf_tokenizer
	dl_args.max_len = max_len

	loader = DatasetLoaderFactory.create(dl_args)
	return loader.data_loader, loader.test_data_loader, loader


def build_model(pad_id: int, num_classes: int = 4):
	model_cfg = {
		"nn_model": {
			"name": "tiny_bert",
			"pretrained_model": "prajjwal1/bert-tiny",
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
	epochs: int = 12,
	batch_size: int = 32,
	max_len: int = 128,
	lr: float = 2e-5,
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
			"pretrained": "bert-base-uncased",
			"use_fast": True,
			"return_type": "tokens",
			"max_len": max_len,
		}
	}
	hf_tok, pad_id = build_tokenizer(tokenizer_cfg)

	# 2) Data - AG News torchtext returns IterDataPipe-like datasets
	train_dl, test_dl, loader = build_dataloaders(hf_tok, batch_size, max_len)

	# Materialize to list for reuse
	train_dataset_list = list(train_dl.dataset) if hasattr(train_dl, 'dataset') else list(train_dl)
	test_dataset_list = list(test_dl.dataset) if hasattr(test_dl, 'dataset') else list(test_dl)
	console.info(f"Loaded {len(train_dataset_list)} training samples and {len(test_dataset_list)} test samples from AG News")

	from torch.utils.data import DataLoader
	from usyd_learning.ml_data_loader.dataset_loader_util import DatasetLoaderUtil

	# AG News format: (label, text); labels in [1,4] -> map to [0,3]
	collate_fn = lambda batch: DatasetLoaderUtil.text_collate_fn_hf(
		batch,
		hf_tokenizer=hf_tok,
		max_len=max_len,
		label_map={1: 0, 2: 1, 3: 2, 4: 3},
		normalize_int_labels=False,
		tuple_format="auto",
		require_labels=True
	)

	train_dl = DataLoader(
		train_dataset_list,
		batch_size=batch_size,
		shuffle=True,
		num_workers=0,
		collate_fn=collate_fn
	)

	test_dl = DataLoader(
		test_dataset_list,
		batch_size=batch_size,
		shuffle=False,
		num_workers=0,
		collate_fn=collate_fn
	)

	# 3) Model / Optimizer / Loss / Scheduler
	model = build_model(pad_id, num_classes=4).to(device)
	optimizer = build_optimizer(model, lr, weight_decay)
	loss_fn = build_loss()

	steps_per_epoch = len(train_dl)
	total_steps = steps_per_epoch * epochs
	warmup_steps = int(total_steps * warmup_ratio)
	scheduler = get_linear_schedule_with_warmup(
		optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
	)

	console.info(
		f"Training TinyBERT on AG News for {epochs} epochs | lr={lr} wd={weight_decay} warmup={warmup_ratio} bs={batch_size} device={device}"
	)

	best_acc = -1.0
	patience = 3
	patience_counter = 0
	best_model_state = None

	for epoch in range(1, epochs + 1):
		model.train()
		total_loss = 0.0
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

			batch_sz = labels.size(0)
			total_loss += float(loss.detach().item()) * batch_sz
			num_samples += batch_sz

		avg_loss = total_loss / max(num_samples, 1)

		model.eval()
		evaluator = ModelEvaluator(model, test_dl, criterion=loss_fn, device=device)
		test_metrics = evaluator.evaluate()
		test_acc = test_metrics.get('accuracy', 0.0)
		test_loss = test_metrics.get('average_loss', 0.0)

		console.info(
			f"Epoch {epoch}/{epochs} | train_loss={avg_loss:.4f} | test_loss={test_loss:.4f} | test_acc={test_acc:.4f}"
		)

		if test_acc > best_acc:
			best_acc = test_acc
			patience_counter = 0
			best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
			console.info(f"  *** New best Accuracy: {best_acc:.4f} ***")
		else:
			patience_counter += 1
			console.info(f"  No improvement. Patience: {patience_counter}/{patience}")
			if patience_counter >= patience:
				console.info(f"Early stopping triggered at epoch {epoch}")
				break

	if best_model_state is not None:
		console.info("\nLoading best model for final evaluation...")
		model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

	model.eval()
	evaluator = ModelEvaluator(model, test_dl, criterion=loss_fn, device=device)
	metrics = evaluator.evaluate()
	console.info("\n" + "="*60)
	console.info("FINAL EVALUATION RESULTS (AG News - test):")
	for k, v in metrics.items():
		console.info(f"  {k}: {v}")

	return metrics


if __name__ == "__main__":
	metrics = train_and_eval(epochs=20, batch_size=32)
	console.info(metrics)
