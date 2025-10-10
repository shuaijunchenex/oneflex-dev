import copy
import torch
from typing import Any, Tuple

from usyd_learning.fed_strategy.strategy_args import StrategyArgs

from ..client_strategy import ClientStrategy
from ...ml_utils.model_utils import ModelUtils
from ...model_trainer import model_trainer_factory
from ...model_trainer.model_trainer_args import ModelTrainerArgs
from ...model_trainer.model_trainer_factory import ModelTrainerFactory
from ...ml_algorithms.optimizer_builder import OptimizerBuilder
from ...ml_data_loader import DatasetLoaderFactory
from ...ml_algorithms.loss_function_builder import LossFunctionBuilder
from ...ml_utils import console
from ...fed_node.fed_node_vars import FedNodeVars

import copy
from typing import Any, Tuple
import torch
import torch.nn as nn

class OortClientTrainingStrategy(ClientStrategy):
    def __init__(self, args, client_node):
        """
        client: a FedNodeClient (or FedNode) that owns a FedNodeVars in `client.node_var`
        config: high-level strategy/trainer config; falls back to `client.node_var.config_dict` when needed
        """
        super().__init__()
        self._args = args
        self._strategy_type = "oort"
        self._obj = client_node

    def _create_inner(self, args, client_node) -> None:
        self._args = args
        self._strategy_type = "oort"
        self._obj = client_node
        return

    # ------------------- Public: Observation wrapper -------------------
    def run_observation(self) -> dict:
        print(f"\n Observation Client [{self._obj.node_id}] ...\n")
        _, train_record = self.observation_step()
        return {
            "node_id": self._obj.node_id,
            "train_record": train_record,
            "data_sample_num": self._obj.node_var.data_sample_num, # TODO: update to sample num
        }

    # ------------------- Observation (no state write-back) -------------------
    def observation_step(self) -> Tuple[dict, Any]:

        node_vars: FedNodeVars = self._obj.node_var
        cfg: dict = self._obj.node_var.config_dict
        device = node_vars.device if hasattr(node_vars, "device") and node_vars.device else "cpu"

        observe_model: nn.Module = copy.deepcopy(node_vars.model)
        observe_model.load_state_dict(node_vars.model_weight, strict=True)
        optimizer = self._obj.node_var.optimizer_builder.rebuild(self, observe_model.parameters())

        ModelUtils.clear_all(observe_model, optimizer)

        self._obj.node_var.trainer.set_optimizer(optimizer)

        local_epochs = int(cfg.get("training", {}).get("local_epochs", 1))
        updated_weights, train_record = self.trainer.train(local_epochs)

        return copy.deepcopy(updated_weights), train_record

    # ------------------- Public: Local training wrapper -------------------
    def run_local_training(self) -> dict:
        updated_weights, train_record = self.local_training_step()
        return updated_weights, {
            "node_id": self._obj.node_id,
            "updated_weights": updated_weights,
            "train_record": train_record,
            "data_sample_num": self._obj.node_var.data_sample_num}

    # ------------------- Full local training (write-back to node_var) -------------------
    def local_training_step(self) -> Tuple[dict, Any]:
        node_vars: FedNodeVars = self._obj.node_var
        cfg: dict = node_vars.config_dict
        device = getattr(node_vars, "device", None) or "cpu"

        training_model: nn.Module = copy.deepcopy(node_vars.model).to(device)
        training_model.load_state_dict(node_vars.model_weight, strict=True)

        optimizer = node_vars.optimizer_builder.rebuild(training_model.parameters())

        ModelUtils.clear_all(training_model, optimizer)

        tr = node_vars.trainer
        tr.set_model(training_model)
        tr.set_optimizer(optimizer)
        tr.trainer_args.device = device

        local_epochs = int(cfg.get("training", {}).get("epochs", 1))
        updated_weights, train_record = tr.train(local_epochs)

        node_vars.model_weight = copy.deepcopy(updated_weights)
        #node_vars.model.load_state_dict(node_vars.model_weight, strict=True)

        return copy.deepcopy(updated_weights), train_record
    
    def receive_weight(self, global_weight) -> dict:
        self._obj.node_var.cache_weight = global_weight
        return
    
    def set_local_weight(self) -> dict:
        self._obj.node_var.model_weight = self._obj.node_var.cache_weight
        return