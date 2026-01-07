from __future__ import annotations

from typing import Any, Iterable, Sequence

from tqdm import tqdm

from usyd_learning.fed_runner.fed_runner import FedRunner
from usyd_learning.fed_strategy.runner_strategy import RunnerStrategy
from usyd_learning.fed_strategy.strategy_args import StrategyArgs
from usyd_learning.ml_utils import console


class SflRunnerStrategy(RunnerStrategy):
    def __init__(self, runner: FedRunner, args: StrategyArgs, client_nodes: Sequence[Any], server_node: Any) -> None:
        super().__init__(runner)
        self._strategy_type = "sfl"
        self.args = args
        self.client_nodes = list(client_nodes)
        self.server_node = server_node
        self.set_node_connection()

    def _create_inner(self, client_nodes: Sequence[Any] | None, server_nodes: Sequence[Any] | None) -> None:
        return

    def prepare(self, logger_header: dict[str, Any]) -> None:
        self.server_node.prepare(logger_header, self.client_nodes)

    def set_node_connection(self) -> None:
        if self.server_node is None:
            return
        self.server_node.set_client_nodes(self.client_nodes)
        for client in self.client_nodes:
            client.set_server_node(self.server_node)

    def simulate_client_local_training_process(self, participants: Iterable[Any]):
        for client in participants:
            console.info(f"\n[{client.node_id}] Local training started")
            updated_weights, train_record = client.run_local_training()
            yield {
                "updated_weights": updated_weights,
                "train_record": train_record,
                "data_sample_num": getattr(client.node_var, "data_sample_num", 0),
            }

    def simulate_server_broadcast_process(self) -> None:
        self.server_node.broadcast()

    def simulate_server_update_process(self) -> None:
        self.server_node.aggregation()

    def run(self) -> None:
        if self.server_node is None:
            raise ValueError("SFL runner strategy requires a server node.")

        console.info("Running [SFL] strategy...")
        header_data = {
            "round": "round",
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1_score": "f1_score",
        }
        self.server_node.prepare(header_data, self.client_nodes)

        total_rounds = int(self.args.get("training_rounds", 0))
        for round_idx in tqdm(range(total_rounds), disable=total_rounds <= 0):
            console.out(
                f"\n{'=' * 10} Training round {round_idx + 1}/{total_rounds}, "
                f"Total participants: {len(self.client_nodes)} {'=' * 10}"
            )

            participants = self.server_node.select_clients(self.client_nodes)
            ids = [str(c.node_id) for c in participants]
            console.info(f"Round: {round_idx + 1}, Select {len(participants)} clients: ").ok(", ".join(ids))

            client_updates = list(self.simulate_client_local_training_process(participants))
            self.server_node.receive_client_updates(client_updates)

            self.simulate_server_update_process()
            self.server_node.apply_weight()
            self.simulate_server_broadcast_process()
            self.server_node.evaluate()
            self.server_node.record_evaluation()

            console.out(
                f"{'=' * 10} Round {round_idx + 1}/{total_rounds} End{'=' * 10}"
            )
