from __future__ import annotations

import copy
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from usyd_learning.fed_node.fed_node_vars import FedNodeVars
from usyd_learning.fed_strategy.client_strategy import ClientStrategy
from usyd_learning.fed_strategy.strategy_args import StrategyArgs
from usyd_learning.ml_utils import console
from usyd_learning.ml_utils.model_utils import ModelUtils


class SflClientStrategy(ClientStrategy):
    def __init__(self, args: StrategyArgs, client_node: Any) -> None:
        super().__init__()
        self._args = args
        self._strategy_type = "sfl"
        self._obj = client_node
        self._active_trainer = None
        self._active_model: Optional[nn.Module] = None
        self._active_optimizer = None

    def _create_inner(self, args: StrategyArgs, client_node: Any) -> None:
        self._args = args
        self._obj = client_node
        return

    def run_local_training(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        front_weight, train_record = self.local_training_step()
        return front_weight, {
            "node_id": self._obj.node_id,
            "front_weight": front_weight,
            "train_record": train_record,
            "data_sample_num": getattr(self._obj.node_var, "data_sample_num", 0),
        }

    def local_training_step(self) -> Tuple[dict[str, torch.Tensor], Any]:
        node_vars: FedNodeVars = self._obj.node_var
        cfg: dict = node_vars.config_dict
        device = getattr(node_vars, "device", None) or "cpu"

        client_model = getattr(node_vars, "client_model", None) or node_vars.model
        training_model: nn.Module = copy.deepcopy(client_model).to(device)
        training_model.load_state_dict(node_vars.model_weight, strict=True)

        optimizer = node_vars.optimizer_builder.rebuild(training_model.parameters())
        ModelUtils.clear_all(training_model, optimizer)

        trainer = node_vars.trainer
        trainer.set_model(training_model)
        trainer.set_optimizer(optimizer)
        trainer.trainer_args.device = device

        # Attach server step for split learning
        server_strategy = getattr(self._obj, "server_node", None)
        if server_strategy is None or not hasattr(server_strategy, "strategy"):
            raise RuntimeError("SFL client requires connected server with strategy.")
        trainer.attach_server_step(server_strategy.strategy.train_on_smashed)

        forward_batches = trainer.forward_only()
        activation_grads: list[torch.Tensor] = []
        loss_sum = 0.0
        metric_acc: dict[str, float] = {}

        for smashed_data, labels in forward_batches:
            act_grad, loss_value, metrics = server_strategy.strategy.train_on_smashed(smashed_data, labels, training=True)
            activation_grads.append(act_grad)
            loss_sum += float(loss_value)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    metric_acc[k] = metric_acc.get(k, 0.0) + float(v)

        client_metrics = trainer.backward_only(activation_grads)
        batch_count = max(len(forward_batches), 1)
        if metric_acc:
            metric_acc = {k: v / batch_count for k, v in metric_acc.items()}

        train_record = {
            "server_metrics": metric_acc,
            "server_loss_avg": loss_sum / batch_count,
            "client_metrics": client_metrics,
        }

        updated_weights = copy.deepcopy(training_model.state_dict())
        node_vars.model_weight = updated_weights
        return updated_weights, train_record

    # New: explicit SL client forward (produces smashed activations)
    def run_local_forward(self):
        node_vars: FedNodeVars = self._obj.node_var
        device = getattr(node_vars, "device", None) or "cpu"

        client_model = getattr(node_vars, "client_model", None) or node_vars.model
        training_model: nn.Module = copy.deepcopy(client_model).to(device)
        training_model.load_state_dict(node_vars.model_weight, strict=True)

        optimizer = node_vars.optimizer_builder.rebuild(training_model.parameters())
        ModelUtils.clear_all(training_model, optimizer)

        trainer = node_vars.trainer
        trainer.set_model(training_model)
        trainer.set_optimizer(optimizer)
        trainer.trainer_args.device = device

        forward_batches = trainer.forward_only()

        # cache for backward step
        self._active_trainer = trainer
        self._active_model = training_model
        self._active_optimizer = optimizer

        return forward_batches

    # New: explicit SL client backward (consumes server grads)
    def run_local_backward(self, activation_grads):
        if self._active_trainer is None or self._active_model is None:
            raise RuntimeError("run_local_backward called before run_local_forward.")

        client_metrics = self._active_trainer.backward_only(activation_grads)
        updated_weights = copy.deepcopy(self._active_model.state_dict())
        self._obj.node_var.model_weight = updated_weights

        # clear cache
        self._active_trainer = None
        self._active_model = None
        self._active_optimizer = None

        return updated_weights, client_metrics

    def receive_weight(self, global_weight: dict[str, torch.Tensor]) -> None:
        self._obj.node_var.cache_weight = global_weight

    def set_local_weight(self) -> None:
        node_vars: FedNodeVars = self._obj.node_var
        node_vars.model_weight = node_vars.cache_weight

    def run_observation(self) -> dict[str, Any]:
        console.info(f"\n Observation Client [{self._obj.node_id}] ...\n")
        updated_weights, train_record = self.observation_step()
        return {
            "node_id": self._obj.node_id,
            "train_record": train_record,
            "data_sample_num": getattr(self._obj.node_var, "data_sample_num", 0),
            "updated_weights": updated_weights,
        }

    def observation_step(self) -> Tuple[dict[str, torch.Tensor], Any]:
        node_vars: FedNodeVars = self._obj.node_var
        cfg: dict = node_vars.config_dict
        device = getattr(node_vars, "device", None) or "cpu"

        client_model = getattr(node_vars, "client_model", None) or node_vars.model
        observe_model: nn.Module = copy.deepcopy(client_model).to(device)
        observe_model.load_state_dict(node_vars.model_weight, strict=True)

        optimizer = node_vars.optimizer_builder.rebuild(observe_model.parameters())
        ModelUtils.clear_all(observe_model, optimizer)

        trainer = node_vars.trainer
        trainer.set_model(observe_model)
        trainer.set_optimizer(optimizer)
        trainer.trainer_args.device = device

        local_epochs = int(cfg.get("training", {}).get("local_epochs", 1))
        updated_weights, train_record = trainer.train(local_epochs)

        return copy.deepcopy(updated_weights), train_record


