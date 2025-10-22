from __future__ import annotations
from typing import Dict, List, Any, Optional
from collections import defaultdict

import torch
from usyd_learning.fed_strategy.server_strategy import ServerStrategy
from usyd_learning.fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from usyd_learning.fl_algorithms.selection.fed_client_selector_factory import FedClientSelectorFactory
from usyd_learning.model_trainer.model_evaluator import ModelEvaluator
from usyd_learning.ml_utils import console

class OortServerStrategy(ServerStrategy):

    def __init__(self, args, server_node) -> None:
        super().__init__()
        self._args = args
        self._strategy_type = "oort"
        self._obj = server_node

    def _create_inner(self, args, server_node) -> None:
        self._args = args
        self._strategy_type = "oort"
        self._obj = server_node
        return self

    def aggregation(self) -> dict:
        aggregator = self._obj.node_var.aggregation_method
        aggregated_weights = aggregator.aggregate(self._obj.node_var.client_updates) #TODO: check
        self._obj.node_var.aggregated_weight = aggregated_weights
        return

    def select_clients(self, available_clients) -> list:
        selector = self._obj.node_var.client_selection
        selected_clients = selector.select(available_clients, self._obj.node_var.config_dict["client_selection"]["number"])
        return selected_clients

    def record_evaluation(self)-> None:
        self._obj.node_var.training_logger.record(self._obj.eval_results)
        return

    def receive_client_updates(self, client_updates) -> None:
        self._obj.node_var.client_updates = client_updates #{client1: {weight:"", data_vol:""}, client2: {weight:"", data_vol:""}}
    
    def apply_weight(self):
        self._obj.node_var.model_weight = self._obj.node_var.aggregated_weight
        self._obj.node_var.model_evaluator.update_model(self._obj.node_var.model_weight)
        
        return

    def broadcast(self) -> None:
        for client in self._obj.client_nodes:
            client.receive_weight(self._obj.node_var.model_weight)
            client.set_local_weight()
        return

    def run(self) -> None:
        raise NotImplementedError

    def evaluate(self) -> None:
        self._obj.eval_results = self._obj.node_var.model_evaluator.evaluate()
        self._obj.node_var.model_evaluator.print_results()
        console.info("Server Evaluation Completed.\n")

        return

    def prepare(self, logger_header, client_nodes_in) -> None:
        self._obj.node_var.training_logger.begin(logger_header)
        self._obj.set_client_nodes(client_nodes_in)
        return

    def run(self) -> Dict[str, Any]:
        raise NotImplementedError