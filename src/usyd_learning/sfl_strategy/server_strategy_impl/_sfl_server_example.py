from __future__ import annotations

from typing import Any, Iterable

from usyd_learning.fed_strategy.server_strategy import ServerStrategy
from usyd_learning.fed_strategy.strategy_args import StrategyArgs
from usyd_learning.ml_utils import console


class SflServerStrategy(ServerStrategy):
    def __init__(self, args: StrategyArgs, server_node: Any) -> None:
        super().__init__()
        self._args = args
        self._strategy_type = "sfl"
        self._obj = server_node

    def _create_inner(self, args: StrategyArgs, server_node: Any) -> None:
        self._args = args
        self._obj = server_node
        return

    def aggregation(self) -> None:
        aggregator = getattr(self._obj.node_var, "aggregation_method", None)
        if aggregator is None:
            raise ValueError("SFL server aggregation requires an aggregator instance.")
        aggregated_weights = aggregator.aggregate(self._obj.node_var.client_updates)
        self._obj.node_var.aggregated_weight = aggregated_weights

    def broadcast(self) -> None:
        weight = getattr(self._obj.node_var, "model_weight", None)
        for client in getattr(self._obj, "client_nodes", []):
            client.receive_weight(weight)
            client.set_local_weight()

    def run(self) -> None:
        raise NotImplementedError

    def evaluate(self) -> None:
        evaluator = getattr(self._obj.node_var, "model_evaluator", None)
        if evaluator is None:
            return
        self._obj.eval_results = evaluator.evaluate()
        evaluator.print_results()
        console.info("Server Evaluation Completed.\n")

    def select_clients(self, available_clients) -> list:
        selector = getattr(self._obj.node_var, "client_selection", None)
        if selector is None:
            return list(available_clients)
        config = self._obj.node_var.config_dict.get("client_selection", {})
        number = config.get("number", len(available_clients))
        return selector.select(available_clients, number)

    def record_evaluation(self) -> None:
        logger = getattr(self._obj.node_var, "training_logger", None)
        if logger is not None and hasattr(self._obj, "eval_results"):
            logger.record(self._obj.eval_results)

    def receive_client_updates(self, client_updates) -> None:
        self._obj.node_var.client_updates = client_updates

    def prepare(self, logger_header, client_nodes_in) -> None:
        logger = getattr(self._obj.node_var, "training_logger", None)
        if logger is not None:
            logger.begin(logger_header)
        self._obj.set_client_nodes(client_nodes_in)

    def apply_weight(self) -> None:
        node_vars = self._obj.node_var
        node_vars.model_weight = node_vars.aggregated_weight
        evaluator = getattr(node_vars, "model_evaluator", None)
        if evaluator is not None:
            evaluator.update_model(node_vars.model_weight)

