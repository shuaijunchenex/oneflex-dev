from __future__ import annotations

from .fed_aggregator_abc import AbstractFedAggregator
from .fed_aggregator_args import FedAggregatorArgs
from ...ml_utils import console

class FedAggregatorFactory:
    '''
    ' Fed aggregator factory
    '''

    @staticmethod
    def create_args(config_dict: dict, is_clone_dict: bool = False) -> FedAggregatorArgs:
        """
        " Static method to create fed aggregator args
        """
        return FedAggregatorArgs(config_dict, is_clone_dict)

    @staticmethod
    def create_aggregator(args: FedAggregatorArgs) -> AbstractFedAggregator:
        match args.method:
            case "fedavg":
                from .methods._fed_aggregator_fedavg import FedAggregator_FedAvg
                console.debug("Using FedAvg aggregator")
                return FedAggregator_FedAvg(args)
            case "rbla":
                from .methods._fed_aggregator_rbla import FedAggregator_RBLA
                console.debug("Using RBLA aggregator")
                return FedAggregator_RBLA(args)
            case "sp":
                from .methods._fed_aggregator_sp import FedAggregator_SP
                console.debug("Using Sum-Product aggregator")
                return FedAggregator_SP(args)
            case "sflavg":
                from .methods._sfl_aggregator_sflavg import SflAggregator_SflAvg
                console.debug("Using SFLAvg gradient aggregator")
                return SflAggregator_SflAvg(args)
            case _:
                raise ValueError(f"Unsupported aggregation method: {args.method}")
        return

