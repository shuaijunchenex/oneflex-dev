from typing import Any, Dict
import torch.nn as nn
import torch
import numpy as np
import random
import math
from tqdm.auto import tqdm
from usyd_learning.model_trainer.model_trainer_args import ModelTrainerArgs

from ..model_trainer import ModelTrainer
from ...ml_algorithms import ModelExtractor
from ...ml_utils import console
from ...ml_utils.model_utils import ModelUtils
from ...ml_utils.metric_calculator import MetricCalculator

class ModelTrainer_Standard(ModelTrainer):
    def __init__(self, trainer_args: ModelTrainerArgs):
        super().__init__(trainer_args)

        if trainer_args.model is None:
            raise ValueError("Training Model is None.")
        if trainer_args.optimizer is None:
            raise ValueError("Training optimizer is None.")
        
        self.device = ModelUtils.accelerator_device()
        self.model: nn.Module = trainer_args.model
        self.metrics = MetricCalculator()
        return

    def set_model(self, model: nn.Module):
        self.trainer_args.model = model
        if str(next(model.parameters()).device) != self.trainer_args.device:
            self.trainer_args.model = model.to(self.trainer_args.device)
        self.model = self.trainer_args.model
        return self

    def train_step(self) -> Dict[str, Any]:
        ta = self.trainer_args
        if ta.optimizer is None:
            raise ValueError("Trainer optimizer is None.")
        if ta.model is None:
            raise ValueError("Trainer model is None.")
        if ta.loss_func is None:
            raise ValueError("Trainer loss function is None.")
        if ta.train_loader is None:
            raise ValueError("Trainer train_loader is None.")

        train_dl = ta.train_loader.data_loader
        if not hasattr(train_dl, "__iter__"):
            raise TypeError(f"train_loader must be an iterable DataLoader, got {type(train_dl).__name__}")

        total_epochs = getattr(ta, "total_epochs", getattr(ta, "epochs", None))

        ta.model.to(self.device)
        ta.model.train()

        self.metrics.reset()

        loop = tqdm(
            train_dl,
            desc=f"Training (epoch {self._epoch_idx}{'/' + str(total_epochs) if total_epochs else ''})",
            leave=True, ncols=120, mininterval=0.1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        for inputs, labels in loop:
            inputs = inputs.to(ta.device)
            labels = labels.to(ta.device)

            try:
                batch_size = int(inputs.size(0))
            except Exception:
                batch_size = int(labels.size(0))

            ta.optimizer.zero_grad()
            outputs = ta.model(inputs)
            loss = ta.loss_func(outputs, labels)
            loss.backward()
            ta.optimizer.step()

            loss_scalar = float(loss.item())
            self.metrics.update(loss_scalar, batch_size)

            loop.set_postfix(
                batch=self.metrics.total_batch,
                loss=f"{loss_scalar:.4f}",
                avg_loss=f"{self.metrics.avg_loss:.4f}",
                avg_loss_keras=f"{self.metrics.keras_loss:.4f}",
                lr=ta.optimizer.param_groups[0]["lr"]
            )

        tqdm.write(
            f"[Epoch {self._epoch_idx}{'/' + str(total_epochs) if total_epochs else ''} Finished] "
            f"avg_loss={self.metrics.avg_loss:.6f} | keras_loss={self.metrics.keras_loss:.6f} | "
            f"batches={self.metrics.total_batch} | samples={self.metrics.total_samples} | device={ta.device}"
        )

        return self.metrics.get_stats()

    # def train_step(self) -> float:
    #     ta = self.trainer_args
    #     if ta.optimizer is None:
    #         raise ValueError("Trainer optimizer is None.")
    #     if ta.model is None:
    #         raise ValueError("Trainer model is None.")
    #     if ta.loss_func is None:
    #         raise ValueError("Trainer loss function is None.")
    #     if ta.train_loader is None:
    #         raise ValueError("Trainer train_loader is None.")

    #     train_dl = ta.train_loader.data_loader
    #     if not hasattr(train_dl, "__iter__"):
    #         raise TypeError(f"train_loader must be an iterable DataLoader, got {type(train_dl).__name__}")

    #     total_epochs = getattr(ta, "total_epochs", getattr(ta, "epochs", None))

    #     ta.model.to(self.device)
    #     ta.model.train()
    #     running_loss, total_batch = 0.0, 0

    #     from tqdm.auto import tqdm
    #     loop = tqdm(
    #         train_dl,
    #         desc=f"Training (epoch {self._epoch_idx}{'/' + str(total_epochs) if total_epochs else ''})",
    #         leave=True, ncols=120, mininterval=0.1,
    #         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    #     )

    #     for inputs, labels in loop:
    #         total_batch += 1
    #         inputs = inputs.to(ta.device)
    #         labels = labels.to(ta.device)

    #         ta.optimizer.zero_grad()
    #         outputs = ta.model(inputs)
    #         loss = ta.loss_func(outputs, labels)
    #         loss.backward()
    #         ta.optimizer.step()

    #         running_loss += float(loss.item())

    #         loop.set_postfix(
    #             batch=total_batch,
    #             loss=f"{loss.item():.4f}",
    #             avg_loss=f"{running_loss/total_batch:.4f}",
    #             lr=ta.optimizer.param_groups[0]["lr"]
    #         )

    #     avg_loss = running_loss / max(total_batch, 1)

    #     from tqdm.auto import tqdm as _tqdm
    #     _tqdm.write(
    #         f"[Epoch {self._epoch_idx}{'/' + str(total_epochs) if total_epochs else ''} Finished] "
    #         f"avg_loss={avg_loss:.6f} | batches={total_batch} | device={ta.device}"
    #     )
    #     return avg_loss

    def train(self, epochs: int) -> Any:
        self.trainer_args.total_epochs = epochs
        self._epoch_idx = 0

        train_stats: Dict[str, Any] = {
            "train_loss_sum": 0.0,
            "train_loss_power_two_sum": 0.0,
            "epoch_loss": [],
            "keras_train_loss_sum": 0.0,
            "keras_train_loss_power_two_sum": 0.0,
            "keras_epoch_loss": [],
            "num_batches_sum": 0,
            "num_samples_sum": 0,
        }

        for _ in range(epochs):
            self._epoch_idx += 1

            step_out = self.train_step() 
            avg_loss = float(step_out["avg_loss"])
            keras_loss = float(step_out["keras_loss"])

            num_batches = int(step_out.get("num_batches", 0))
            num_samples = int(step_out.get("num_samples", 0))

            train_stats["train_loss_sum"] += avg_loss
            train_stats["train_loss_power_two_sum"] += avg_loss ** 2
            train_stats["epoch_loss"].append(avg_loss)

            train_stats["keras_train_loss_sum"] += keras_loss
            train_stats["keras_train_loss_power_two_sum"] += keras_loss ** 2
            train_stats["keras_epoch_loss"].append(keras_loss)

            train_stats["num_batches_sum"] += num_batches
            train_stats["num_samples_sum"] += num_samples

        self._epoch_idx = 0

        train_stats["avg_loss"] = train_stats["train_loss_sum"] / max(epochs, 1)
        train_stats["keras_avg_loss"] = train_stats["keras_train_loss_sum"] / max(epochs, 1)

        train_stats["sqrt_train_loss_power_two_sum"] = math.sqrt(train_stats["train_loss_power_two_sum"])
        train_stats["keras_sqrt_train_loss_power_two_sum"] = math.sqrt(train_stats["keras_train_loss_power_two_sum"])

        return self.trainer_args.model.state_dict(), train_stats

    def observe(self, epochs=5) -> Any:
        self.trainer_args.total_epochs = epochs
        train_stats = {"train_loss_sum": 0, "epoch_loss": [], "train_loss_power_two_sum": 0}

        for _ in range(epochs):
            step_out = self.train_step()
            train_loss = float(step_out["avg_loss"])
            train_stats["train_loss_sum"] += train_loss
            train_stats["train_loss_power_two_sum"] += train_loss ** 2
            train_stats["epoch_loss"].append(train_loss)

        train_stats["avg_loss"] = train_stats["train_loss_sum"] / epochs
        train_stats["sqrt_train_loss_power_two_sum"] = math.sqrt(train_stats["train_loss_power_two_sum"])
        return self.trainer_args.model.state_dict(), train_stats