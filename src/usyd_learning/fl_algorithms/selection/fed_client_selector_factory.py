from __future__ import annotations

from .fed_client_selector_args import FedClientSelectorArgs
from .fed_client_selector_abc import FedClientSelector


class FedClientSelectorFactory:
    '''
    Fed client selector factory
    '''

    @staticmethod
    def create_args(config_dict: dict, is_clone_dict: bool = False) -> FedClientSelectorArgs:
        """
        Static method to create client selector args
        """
        return FedClientSelectorArgs(config_dict, is_clone_dict)
    
    @staticmethod
    def create(args: FedClientSelectorArgs) -> FedClientSelector:#TODO         
        """
        Static method to create client selector
        """
        match args.select_method:
            case "all":
                from .methods._fed_client_selector_all import FedClientSelector_All
                selector = FedClientSelector_All(args)
            case "high_loss":
                from .methods._fed_client_selector_high_loss import FedClientSelector_HighLoss
                selector = FedClientSelector_HighLoss(args)
            case "random":
                from .methods._fed_client_selector_random import FedClientSelector_Random
                selector = FedClientSelector_Random(args)
            case "oort":
                from .methods._fed_client_selector_oort import FedClientSelector_Oort
                selector = FedClientSelector_Oort(args)
            case "pyramidfl":
                from .methods._fed_client_selector_pyramidfl import FedClientSelector_PyramidFL
                selector = FedClientSelector_PyramidFL(args)
            case "afl":
                from .methods._fed_client_selector_afl import FedClientSelector_AFL
                selector = FedClientSelector_AFL(args)
            case "powd":
                from .methods._fed_client_selector_powd import FedClientSelector_Powd
                selector = FedClientSelector_Powd(args)
            case _:
                raise Exception(f"Fed client selector method '{args.select_method}' not found")        

        return selector
