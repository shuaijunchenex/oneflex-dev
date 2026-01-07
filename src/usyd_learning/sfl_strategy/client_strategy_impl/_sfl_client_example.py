from __future__ import annotations

import copy
from typing import Any, Tuple

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

    def _create_inner(self, args: StrategyArgs, client_node: Any) -> None:
        self._args = args
        self._obj = client_node
        return

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

    def run_local_training(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        updated_weights, train_record = self.local_training_step()
        return updated_weights, {
            "node_id": self._obj.node_id,
            "updated_weights": updated_weights,
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

        local_epochs = int(cfg.get("training", {}).get("epochs", 1))
        updated_weights, train_record = trainer.train(local_epochs)

        node_vars.model_weight = copy.deepcopy(updated_weights)
        return copy.deepcopy(updated_weights), train_record

    def receive_weight(self, global_weight: dict[str, torch.Tensor]) -> None:
        self._obj.node_var.cache_weight = global_weight

    def set_local_weight(self) -> None:
        node_vars: FedNodeVars = self._obj.node_var
        node_vars.model_weight = node_vars.cache_weight

