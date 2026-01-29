import math
import contextlib
from typing import Any, Dict
import torch
import torch.nn as nn

from torch.cuda.amp import GradScaler
from usyd_learning.ml_utils.model_utils import ModelUtils
from usyd_learning.model_trainer import ModelTrainer, ModelTrainerArgs
from usyd_learning.ml_utils.metric_calculator import MetricCalculator
from usyd_learning.ml_utils.training_utils import TrainingUtils
from usyd_learning.ml_utils.model_ewma import ModelEWMA


class ModelTrainer_GLUE(ModelTrainer):
    """
    Generic GLUE-style text classification trainer (TinyBERT / RoBERTa, etc.).
    - Optional AMP via TrainingUtils.make_autocast
    - Optional EMA via ModelEWMA
    - Otherwise mirrors ModelTrainer_Standard interface
    """

    def __init__(self, trainer_args: ModelTrainerArgs):
        super().__init__(trainer_args)

        if trainer_args.model is None:
            raise ValueError("Training Model is None.")
        if trainer_args.optimizer is None:
            raise ValueError("Training optimizer is None.")

        self.device = ModelUtils.accelerator_device()
        self.model: nn.Module = trainer_args.model

        ta = self.trainer_args
        self.amp_enabled: bool = bool(getattr(ta, "amp_enabled", False))

        self.use_grad_scaler: bool = bool(getattr(ta, "use_grad_scaler", True))
        self._scaler = None
        if self.amp_enabled and torch.cuda.is_available() and self.use_grad_scaler:
            self._scaler = GradScaler(enabled=True)

        ema_decay = getattr(ta, "ema_decay", None)
        self._ema = None
        if isinstance(ema_decay, (float, int)) and 0.0 < float(ema_decay) < 1.0:
            self._ema = ModelEWMA(self.model, decay=float(ema_decay), device=self.device)

        self.metrics = MetricCalculator()

        # ensure model on target device
        if str(next(self.model.parameters()).device) != str(self.trainer_args.device):
            self.model = self.model.to(self.trainer_args.device)
            self.trainer_args.model = self.model

    def set_model(self, model: nn.Module):
        self.trainer_args.model = model
        if str(next(model.parameters()).device) != self.trainer_args.device:
            self.trainer_args.model = model.to(self.trainer_args.device)
        self.model = self.trainer_args.model
        return self

    def _ctx_model_train(self):
        self.trainer_args.model.train()
        return contextlib.nullcontext()

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
        self.metrics.reset()

        from tqdm.auto import tqdm
        loop = tqdm(
            train_dl,
            desc=f"Training (epoch {self._epoch_idx}{'/' + str(total_epochs) if total_epochs else ''})",
            leave=True, ncols=120, mininterval=0.1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        with self._ctx_model_train():
            for inputs, labels in loop:
                inputs = inputs.to(ta.device) if hasattr(inputs, "to") else inputs
                labels = labels.to(ta.device)

                try:
                    batch_size = int(labels.size(0))
                except Exception:
                    batch_size = 1

                ta.optimizer.zero_grad(set_to_none=True)

                with TrainingUtils.make_autocast(device=self.device, enabled=self.amp_enabled):
                    outputs = ta.model(inputs)
                    loss = ta.loss_func(outputs, labels)

                if self._scaler is not None:
                    self._scaler.scale(loss).backward()
                    self._scaler.step(ta.optimizer)
                    self._scaler.update()
                else:
                    loss.backward()
                    ta.optimizer.step()

                if self._ema is not None:
                    self._ema.update(ta.model)

                loss_scalar = float(loss.detach().item())
                self.metrics.update(loss_scalar, batch_size)

                loop.set_postfix(
                    batch=self.metrics.total_batch,
                    loss=f"{loss_scalar:.4f}",
                    avg_loss=f"{self.metrics.avg_loss:.4f}",
                    avg_loss_keras=f"{self.metrics.keras_loss:.4f}",
                    lr=ta.optimizer.param_groups[0]["lr"]
                )

        from tqdm.auto import tqdm as _tqdm
        _tqdm.write(
            f"[Epoch {self._epoch_idx}{'/' + str(total_epochs) if total_epochs else ''} Finished] "
            f"avg_loss={self.metrics.avg_loss:.6f} | keras_loss={self.metrics.keras_loss:.6f} | "
            f"batches={self.metrics.total_batch} | samples={self.metrics.total_samples} | device={ta.device}"
        )
        return self.metrics.get_stats()

    def train(self, epochs, is_return_wbab=False) -> Any:
        self.trainer_args.total_epochs = epochs
        self._epoch_idx = 0

        stats: Dict[str, Any] = {
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

            stats["train_loss_sum"] += avg_loss
            stats["train_loss_power_two_sum"] += avg_loss ** 2
            stats["epoch_loss"].append(avg_loss)

            stats["keras_train_loss_sum"] += keras_loss
            stats["keras_train_loss_power_two_sum"] += keras_loss ** 2
            stats["keras_epoch_loss"].append(keras_loss)

            stats["num_batches_sum"] += num_batches
            stats["num_samples_sum"] += num_samples

        self._epoch_idx = 0
        stats["avg_loss"] = stats["train_loss_sum"] / max(epochs, 1)
        stats["keras_avg_loss"] = stats["keras_train_loss_sum"] / max(epochs, 1)

        stats["sqrt_train_loss_power_two_sum"] = math.sqrt(stats["train_loss_power_two_sum"])
        stats["keras_sqrt_train_loss_power_two_sum"] = math.sqrt(stats["keras_train_loss_power_two_sum"])

        return self.trainer_args.model.state_dict(), stats

    def observe(self, epochs=5) -> Any:
        self.trainer_args.total_epochs = epochs
        stats = {"train_loss_sum": 0, "epoch_loss": [], "train_loss_power_two_sum": 0}

        for _ in range(epochs):
            step_out = self.train_step()
            loss = float(step_out["avg_loss"])
            stats["train_loss_sum"] += loss
            stats["train_loss_power_two_sum"] += loss ** 2
            stats["epoch_loss"].append(loss)

        stats["avg_loss"] = stats["train_loss_sum"] / max(epochs, 1)
        stats["sqrt_train_loss_power_two_sum"] = math.sqrt(stats["train_loss_power_two_sum"])
        return self.trainer_args.model.state_dict(), stats
