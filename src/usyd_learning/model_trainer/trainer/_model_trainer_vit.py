import math
from typing import Any, Optional

import torch
import torch.nn as nn

from usyd_learning.model_trainer.model_trainer_args import ModelTrainerArgs
from usyd_learning.ml_utils import console
from usyd_learning.ml_utils.model_ewma import ModelEWMA
from usyd_learning.ml_utils.model_utils import ModelUtils
from usyd_learning.model_trainer import ModelTrainer


class ModelTrainer_Vit(ModelTrainer):
    """
    与 ModelTrainer_Standard 相同的对外接口/用法：
      - __init__(trainer_args: ModelTrainerArgs)
      - set_model(model)
      - train_step() -> float
      - train(epochs, is_return_wbab=False) -> (state_dict, stats[, wbab])
      - observe(epochs=5) -> (state_dict, stats)

    增强点（均已解耦成方法）：
      - AMP（CUDA: cuda.amp + GradScaler；MPS: torch.autocast("mps")；CPU：关闭）
      - 梯度累积
      - 梯度裁剪
      - 可选 scheduler.step()（若 ta.scheduler 存在）
      - 可选 EMA（若 ta.use_ewma=True）
    """

    def __init__(self, trainer_args: ModelTrainerArgs):
        super().__init__(trainer_args)

        if trainer_args.model is None:
            raise ValueError("Training Model is None.")
        if trainer_args.optimizer is None:
            raise ValueError("Training optimizer is None.")

        self.device = ModelUtils.accelerator_device()
        self.model: nn.Module = trainer_args.model

        self._epoch_idx: int = 0
        self._scaler = self._make_scaler()         # 仅 CUDA 返回 GradScaler，否则 None
        self._use_amp = self._should_use_amp()     # CUDA/MPS 可用，CPU 关闭

        if self.trainer_args.ema_decay!=None and self.trainer_args.ema_decay>0:
            self._ewma = ModelEWMA(self.trainer_args.model, decay=self.trainer_args.ema_decay, device=self.device)

        return

    def set_model(self, model: nn.Module):
        self.trainer_args.model = model
        if str(next(model.parameters()).device) != self.trainer_args.device:
            self.trainer_args.model = model.to(self.trainer_args.device)
        self.model = self.trainer_args.model
        # 更新 EMA 的结构（如果有）
        if self._ewma is not None:
            self._ewma = self._build_ewma_if_needed(force_rebuild=True)
        return self

    def train_step(self) -> float:
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

        # 准备模型与优化器
        ta.model.to(self.device)
        ta.model.train()
        ta.optimizer.zero_grad(set_to_none=True)

        # 累计器
        grad_accum_steps: int = getattr(ta, "grad_accum_steps", getattr(ta, "accumulation_steps", 1))
        clip_grad_norm: float = float(getattr(ta, "clip_grad_norm", 0.0))

        running_loss, total_batch = 0.0, 0

        from tqdm.auto import tqdm
        loop = tqdm(
            train_dl,
            desc=f"Training (epoch {self._epoch_idx}{'/' + str(total_epochs) if total_epochs else ''})",
            leave=True, ncols=120, mininterval=0.1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        for inputs, labels in loop:
            total_batch += 1
            inputs = inputs.to(ta.device, non_blocking=True)
            labels = labels.to(ta.device, non_blocking=True)

            with self._autocast_context():
                outputs = ta.model(inputs)
                loss = ta.loss_func(outputs, labels)
                loss = loss / max(1, grad_accum_steps)

            # 反向：CUDA 使用 scaler，其他直接 backward
            if self._scaler is not None:
                self._scaler.scale(loss).backward()
            else:
                loss.backward()

            if total_batch % grad_accum_steps == 0:
                # 梯度裁剪
                if clip_grad_norm > 0:
                    if self._scaler is not None:
                        self._scaler.unscale_(ta.optimizer)
                    nn.utils.clip_grad_norm_(ta.model.parameters(), max_norm=clip_grad_norm)

                # step
                if self._scaler is not None:
                    self._scaler.step(ta.optimizer)
                    self._scaler.update()
                else:
                    ta.optimizer.step()

                ta.optimizer.zero_grad(set_to_none=True)

                # scheduler（如果存在）
                self._step_scheduler()

            # EMA（每步）
            self._update_ewma()

            running_loss += float(loss.item()) * max(1, grad_accum_steps)

            loop.set_postfix(
                batch=total_batch,
                loss=f"{loss.item():.4f}",
                avg_loss=f"{running_loss/total_batch:.4f}",
                lr=ta.optimizer.param_groups[0]["lr"]
            )

        avg_loss = running_loss / max(total_batch, 1)

        from tqdm.auto import tqdm as _tqdm
        _tqdm.write(
            f"[Epoch {self._epoch_idx}{'/' + str(total_epochs) if total_epochs else ''} Finished] "
            f"avg_loss={avg_loss:.6f} | batches={total_batch} | device={ta.device}"
        )
        return avg_loss

    def train(self, epochs, is_return_wbab=False) -> Any:
        self.trainer_args.total_epochs = epochs
        self._epoch_idx = 0
        train_stats = {"train_loss_sum": 0, "epoch_loss": [], "train_loss_power_two_sum": 0}

        for _ in range(epochs):
            self._epoch_idx += 1
            train_loss = self.train_step()
            train_stats["train_loss_sum"] += train_loss
            train_stats["train_loss_power_two_sum"] += train_loss ** 2
            train_stats["epoch_loss"].append(train_loss)

        self._epoch_idx = 0
        train_stats["avg_loss"] = train_stats["train_loss_sum"] / epochs
        train_stats["sqrt_train_loss_power_two_sum"] = math.sqrt(train_stats["train_loss_power_two_sum"])

        if is_return_wbab:
            return self.trainer_args.model.state_dict(), train_stats, self.extract_wbab()
        else:
            return self.trainer_args.model.state_dict(), train_stats

    def observe(self, epochs=5) -> Any:
        self.trainer_args.total_epochs = epochs
        train_stats = {"train_loss_sum": 0, "epoch_loss": [], "train_loss_power_two_sum": 0}

        for _ in range(epochs):
            train_loss = self.train_step()
            train_stats["train_loss_sum"] += train_loss
            train_stats["train_loss_power_two_sum"] += train_loss ** 2
            train_stats["epoch_loss"].append(train_loss)

        train_stats["avg_loss"] = train_stats["train_loss_sum"] / epochs
        train_stats["sqrt_train_loss_power_two_sum"] = math.sqrt(train_stats["train_loss_power_two_sum"])
        return self.trainer_args.model.state_dict(), train_stats

    def _should_use_amp(self) -> bool:
        """根据设备与配置决定是否启用 AMP（CUDA/MPS 才考虑）。"""
        ta = self.trainer_args
        want_amp: bool = bool(getattr(ta, "use_amp", getattr(ta, "amp", False)))
        if not want_amp:
            return False
        dev = str(getattr(ta, "device", self.device))
        return dev.startswith("cuda") or dev == "mps"

    def _autocast_context(self):
        """返回正确的 autocast 上下文管理器（CUDA/MPS/CPU）。"""
        if not self._use_amp:
            return _NullCtx()
        if self.device.type == "cuda":
            return torch.cuda.amp.autocast(dtype=torch.float16)
        if self.device.type == "mps":
            return torch.autocast(device_type="mps", dtype=torch.float16)
        return _NullCtx()

    def _make_scaler(self) -> Optional[torch.cuda.amp.GradScaler]:
        """仅在 CUDA + AMP 时使用 GradScaler；MPS/CPU 返回 None。"""
        if self.device.type == "cuda" and self._should_use_amp():
            return torch.cuda.amp.GradScaler(enabled=True)
        return None

    def _step_scheduler(self):
        """如果 ta.scheduler 存在，则每次 optimizer.step() 后推进一次。"""
        ta = self.trainer_args
        if getattr(ta, "scheduler", None) is not None:
            ta.scheduler.step()

    def _update_ewma(self):
        if self._ewma is not None:
            self._ewma.update(self.trainer_args.model)

class _NullCtx:
    """占位空上下文管理器：with _NullCtx(): pass"""
    def __enter__(self): return None
    def __exit__(self, exc_type, exc, tb): return False