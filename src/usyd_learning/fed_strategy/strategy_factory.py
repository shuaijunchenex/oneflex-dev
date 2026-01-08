from __future__ import annotations
from typing import Callable

from .strategy_args import StrategyArgs
from .client_strategy import ClientStrategy
from .server_strategy import ServerStrategy

class StrategyFactory:
    """
    " Dataset loader factory
    """

    @staticmethod
    def create_args(config_dict: dict, is_clone_dict: bool = False) -> StrategyArgs:
        """
        " Static method to create data loader args
        """
        return StrategyArgs(config_dict, is_clone_dict)

    @staticmethod
    def create_runner_args(config_dict: dict, is_clone_dict: bool = False) -> StrategyArgs:
        """
        " Static method to create runner strategy args
        """
        runner_config = {**config_dict["general"], **config_dict["strategy"]}
        return StrategyArgs(runner_config, is_clone_dict)

    @staticmethod
    def create(args: StrategyArgs, node):
        match args.role.lower():
            case "client":
                return StrategyFactory.create_client_strategy(args, node)
            case "server":
                return StrategyFactory.create_server_strategy(args, node)

    @staticmethod
    def create_runner_strategy(runner_strategy_args: StrategyArgs, runner, client_nodes, server_node) -> ClientStrategy:
        """
        " Static method to create runner strategy
        """
        match runner_strategy_args.strategy_name.lower():
            case "fedavg":
                # Import FedAvgRunnerStrategy from the appropriate module
                from usyd_learning.fed_strategy.runner_strategy_impl._fedavg_runner_strategy import FedAvgRunnerStrategy
                return FedAvgRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "rbla":
                from usyd_learning.fed_strategy.runner_strategy_impl._rbla_runner_strategy import RblaRunnerStrategy
                return RblaRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "sp":
                from usyd_learning.fed_strategy.runner_strategy_impl._sp_runner_strategy import SpRunnerStrategy
                return SpRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "sfl":
                from usyd_learning.sfl_strategy.runner_strategy_impl._sfl_runner_example import SflRunnerStrategy
                return SflRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)

        raise ValueError(f"Runner strategy type '{runner_strategy_args.strategy_name}' not support.")

    @staticmethod
    def create_client_strategy(client_strategy_args: StrategyArgs, client_node_input) -> ClientStrategy:
        """
        " Static method to create data loader
        """
        match client_strategy_args.strategy_name.lower():
            case "fedavg":
                from usyd_learning.fed_strategy.client_strategy_impl._fedavg_client import FedAvgClientTrainingStrategy
                return FedAvgClientTrainingStrategy(client_strategy_args, client_node_input)
            case "rbla":
                from usyd_learning.fed_strategy.client_strategy_impl._rbla_client import RblaClientTrainingStrategy
                return RblaClientTrainingStrategy(client_strategy_args, client_node_input)
            case "sp":
                from usyd_learning.fed_strategy.client_strategy_impl._sp_client import SpClientTrainingStrategy
                return SpClientTrainingStrategy(client_strategy_args, client_node_input)
            case "sfl":
                from usyd_learning.sfl_strategy.client_strategy_impl._sfl_client_example import SflClientStrategy
                return SflClientStrategy(client_strategy_args, client_node_input)

        raise ValueError(f"Client strategy type '{client_strategy_args.strategy_name}' not support.")

    @staticmethod
    def create_server_strategy(server_strategy_args: StrategyArgs, serve_node_input) -> ServerStrategy:
        """
        " Static method to create server strategy
        """
        match server_strategy_args.strategy_name.lower():
            case "fedavg":
                from usyd_learning.fed_strategy.server_strategy_impl._fedavg_server import FedAvgServerStrategy
                return FedAvgServerStrategy(server_strategy_args, serve_node_input)
            case "rbla":
                from usyd_learning.fed_strategy.server_strategy_impl._rbla_server import RblaServerStrategy
                return RblaServerStrategy(server_strategy_args, serve_node_input)
            case "sp":
                from usyd_learning.fed_strategy.server_strategy_impl._sp_server import SpServerStrategy
                return SpServerStrategy(server_strategy_args, serve_node_input)
            case "sfl":
                from usyd_learning.sfl_strategy.server_strategy_impl._sfl_server_example import SflServerStrategy
                return SflServerStrategy(server_strategy_args, serve_node_input)

        raise ValueError(f"Server strategy type '{server_strategy_args.strategy_name}' not support.")