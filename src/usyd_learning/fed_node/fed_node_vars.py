from __future__ import annotations
from typing import Any

import torch.nn as nn
import copy

from .fed_node_event_args import FedNodeEventArgs
from ..ml_utils import TrainingLogger, EventHandler, console, String, ObjectMap, KeyValueArgs
from ..ml_models import NNModelFactory
from ..ml_data_loader import DatasetLoaderArgs, DatasetLoaderFactory, DatasetLoader
from ..ml_algorithms import LossFunctionBuilder, OptimizerBuilder
from ..fl_algorithms import FedClientSelectorFactory, FedClientSelector, FedClientSelectorArgs, FedAggregatorArgs, FedAggregatorFactory
from ..ml_data_process import DataDistribution
from ..fed_strategy.strategy_factory import StrategyFactory
from ..fl_algorithms import FedClientSelectorFactory
from ..model_trainer.model_evaluator import ModelEvaluator
from ..model_trainer.model_trainer_factory import ModelTrainerFactory
from ..model_trainer.model_trainer_args import ModelTrainerArgs

class FedNodeVars(ObjectMap, EventHandler, KeyValueArgs):
    """
    Fed node variables
    """

    # NOTICE: Static share model
    share_model: nn.Module|None = None

    def __init__(self, config_dict: dict|None = None, is_clone_dict:bool = False):
        EventHandler.__init__(self)
        ObjectMap.__init__(self)
        KeyValueArgs.__init__(self, config_dict, is_clone_dict)

        # Computation device (default: cpu)
        self.device = "cpu"
        if config_dict is not None and "general" in config_dict:
            self.device = config_dict["general"].get("device", "cpu")

        # Variables owner node list. One for normal, more means var owned by more than one node
        self.__owner_nodes: list = []
        self.__init_vars()

        # Declare event
        self.declare_events("on_prepare_data_loader", "on_prepare_data_distribution", "on_prepare_data_handler", "on_prepare_model",
                            "on_prepare_optimizer", "on_prepare_loss_func", "on_prepare_client_selection", "on_prepare_trainer",
                            "on_prepare_aggregation", "on_prepare_strategy", "on_prepare_extractor", "on_prepare_training_logger")
        return

    @property
    def config_dict(self) -> dict: return self.key_value_dict.dict

    @property
    def owner_nodes(self):
        return self.__owner_nodes

    @owner_nodes.setter
    def owner_nodes(self, value):
        self.__owner_nodes.append(value)

    @property
    def owner_node_count(self):
        return len(self.__owner_nodes)

    def set_var(self, key: str, var: Any):
        """
        Add extra var
        """
        self.set_object(key, var)
        return

    def get_var(self, key: str):
        self.get_object(key)
        return

    def __init_vars(self):
        self.set_object("data_loader", None)  # Data loader
        self.data_loader_collate_fn = None
        self.data_loader_transform = None

        self.set_object("data_distribution", None)  # Data distribution
        self.set_object("data_handler", None)  # data_handler

        self.set_object("model", None)  # Model
        self.set_object("model_weight", None)  # model weight
        self.set_object("global_model_weight", None)  # global weight

        self.set_object("optimizer", None)  # optimizer
        self.set_object("loss_func", None)  # loss_func
        self.set_object("training", None)  # training

        self.set_object("aggregation", None)  # aggregation
        self.set_object("client_selection", None)  # Client selection

        self.set_object("strategy", None)  # strategy
        self.set_object("extractor", None)  # extractor

        self.set_object("training_logger", None)  # Training logger
        return

    # Properties
    #region
    #---------------------------------------------------------------
    # training_logger property
    @property
    def training_logger(self) -> TrainingLogger:
        return self.get_object("training_logger")

    @training_logger.setter
    def training_logger(self, value):
        self.set_object("training_logger", value)

    # data loader property
    @property
    def data_loader(self) -> DatasetLoader:
        return self.get_object("data_loader")

    @data_loader.setter
    def data_loader(self, value):
        self.set_object("data_loader", value)

    @property
    def model(self) -> nn.Module :
        return self.get_object("model", cast_type=nn.Module)

    @model.setter
    def model(self, value):
        self.set_object("model", value)

    @property
    def model_weight(self):
        return self.get_object("model_weight")

    @model_weight.setter
    def model_weight(self, value):
        self.set_object("model_weight", value)

    @property
    def global_model_weight(self):
        return self.get_object("global_model_weight")

    @global_model_weight.setter
    def global_model_weight(self, value):
        self.set_object("global_model_weight", value)

    @property
    def aggregation(self):
        return self.get_object("aggregation")

    @aggregation.setter
    def aggregation(self, value):
        self.set_object("aggregation", value)

    @property
    def client_selection(self) -> FedClientSelector:
        return self.get_object("client_selection")

    @client_selection.setter
    def client_selection(self, value):
        self.set_object("client_selection", value)

    @property
    def data_distribution(self) -> DataDistribution:
        return self.get_object("data_distribution")

    @data_distribution.setter
    def data_distribution(self, value):
        self.set_object("data_distribution", value)

    @property
    def loss_func(self):
        return self.get_object("loss_func")

    @loss_func.setter
    def loss_func(self, value):
        self.set_object("loss_func", value)

    @property
    def optimizer(self):
        return self.get_object("optimizer")

    @optimizer.setter
    def optimizer(self, value):
        self.set_object("optimizer", value)

    @property
    def training(self):
        return self.get_object("training")

    @training.setter
    def training(self, value):
        self.set_object("training", value)

    @property
    def strategy(self):
        return self.get_object("strategy")

    @strategy.setter
    def strategy(self, value):
        self.set_object("strategy", value)

    @property
    def extractor(self):
        return self.get_object("extractor")

    @extractor.setter
    def extractor(self, value):
        self.set_object("extractor", value)

    #endregion

    # Prepare variables
    # region
    def prepare_data_loader(self):
        if "data_loader" in self.config_dict:
            data_loader_args = DatasetLoaderArgs(self.config_dict["data_loader"])
            data_loader_args.collate_fn = self.data_loader_collate_fn
            data_loader_args.transform = self.data_loader_transform
            self.data_loader = DatasetLoaderFactory.create(data_loader_args)
            data_loader_args.is_train = False  # get test data loader
            self.test_data_loader = DatasetLoaderFactory.create(data_loader_args)
            self.data_sample_num = self.data_loader.data_sample_num
            console.warn("WARN: Missing data loader config in yaml.")

        # Raise event
        args = FedNodeEventArgs("data_loader", self.config_dict).with_sender(self).with_data(self.data_loader)
        self.raise_event("on_prepare_data_loader", args)
        return

    def prepare_data_distribution(self):
        if "data_distribution" in self.config_dict:
            DataDistribution.parse_config(self.config_dict["data_distribution"])
        self.data_distribution = DataDistribution.get()

        args = FedNodeEventArgs("data_distribution", self.config_dict).with_sender(self).with_data(self.data_distribution)
        self.raise_event("on_prepare_data_distribution", args)
        return

    def prepare_data_handler(self):
        args = FedNodeEventArgs("data_handler", self.config_dict).with_sender(self)
        self.raise_event("on_prepare_data_handler", args)
        return

    def prepare_model(self):
        # create model
        if "nn_model" in self.config_dict:
            config = self.config_dict["nn_model"]
        else:
            config = self.config_dict

        name = config["name"]
        if String.is_none_or_empty(name):
            raise ValueError("Error: Missing model name in yaml.")

        is_share_model = config.get("share_model", True)  # NOTICE: Share model
        if is_share_model and FedNodeVars.share_model is not None:
            self.model = FedNodeVars.share_model
            self.model_weight = self.model.state_dict()  # model weight
        else:
            args = NNModelFactory.create_args(config)
            self.model = NNModelFactory.create(args)
            self.model_weight = self.model.state_dict()  # model weight

        if is_share_model and FedNodeVars.share_model is None:
            FedNodeVars.share_model = self.model

        # Raise event
        args = FedNodeEventArgs("model", self.config_dict).with_sender(self).with_data(self.model)
        self.raise_event("on_prepare_model", args)
        return

    def prepare_optimizer(self):
        # build optimizer
        if "optimizer" in self.config_dict:
            self.optimizer_builder = OptimizerBuilder(self.model.parameters(), self.config_dict)
            self.optimizer = self.optimizer_builder.build()

        args = FedNodeEventArgs("optimizer", self.config_dict).with_sender(self).with_data(self.optimizer)
        self.raise_event("on_prepare_optimizer", args)
        return

    def prepare_loss_func(self):
        # build loss function
        if "loss_func" in self.config_dict:
            self.loss_func = LossFunctionBuilder.build(self.config_dict["loss_func"])

        args = FedNodeEventArgs("loss_func", self.config_dict).with_sender(self).with_data(self.loss_func)
        self.raise_event("on_prepare_loss_func", args)
        return

    def prepare_client_selection(self):
        if "client_selection" in self.config_dict:
            client_selection_args = FedClientSelectorArgs(self.config_dict["client_selection"])
            self.client_selection = FedClientSelectorFactory.create(client_selection_args)

        args = FedNodeEventArgs("client_selection", self.config_dict).with_sender(self).with_data(self.client_selection)
        self.raise_event("on_prepare_client_selection", args)
        return

    def prepare_trainer(self):
        args = FedNodeEventArgs("training", self.config_dict).with_sender(self)

        # build trainer
        trainer_args = ModelTrainerFactory.create_args(self.config_dict["trainer"])
        trainer_args.set_trainer_args(self.model, self.optimizer, self.loss_func, self.data_loader, 'standard') #TODO: add trainer type to config
        self.trainer = ModelTrainerFactory.create(trainer_args)

        self.raise_event("on_prepare_trainer", args)
        return

    def prepare_aggregation(self):
        # Raise strategy event
        args = FedNodeEventArgs("model_aggregation", self.config_dict).with_sender(self)

        #########
        if "aggregation" in self.config_dict:
            self.aggregation_method = FedAggregatorFactory.create_aggregator(FedAggregatorArgs(self.config_dict["aggregation"]))

        self.raise_event("on_prepare_aggregation", args)
        return
    
    def prepare_model_evaluator(self):

        self.model_evaluator = ModelEvaluator(self.model, self.data_loader, self.loss_func)

        return

    def prepare_strategy(self):
        if "strategy" in self.config_dict:
            self.strategy = self.config_dict["strategy"]
        # Raise strategy event
        args = FedNodeEventArgs("strategy", self.config_dict).with_sender(self)
        self.strategy = StrategyFactory.create(StrategyFactory.create_args(self.config_dict["strategy"]), self.owner_nodes[0])
        self.raise_event("on_prepare_strategy", args)
        return

    def prepare_extractor(self):
        # Raise extractor event
        args = FedNodeEventArgs("extractor", self.config_dict).with_sender(self)

        #########
        console.error("TODO: prepare_extractor...")

        self.raise_event("on_prepare_extractor", args)

    def prepare_training_logger(self):
        if "training_logger" in self.config_dict:
            self.training_logger = TrainingLogger(self.config_dict["training_logger"])

        # Raise event
        args = FedNodeEventArgs("training_logger", self.config_dict).with_sender(self).with_data(self.training_logger)
        self.raise_event("on_prepare_training_logger", args)
        return

    # endregion

    def prepare(self) -> Any:
        """
        Prepare
        """
        console.info("Prepare data loader...", "")
        self.prepare_data_loader()
        console.ok("OK")

        console.info("Prepare data_distribution...", "")
        self.prepare_data_distribution()
        console.ok("OK")

        console.info("Prepare data handler...", "")
        self.prepare_data_handler()
        console.ok("OK")

        console.info("Prepare NN model...", "")
        self.prepare_model()
        console.ok("OK") #TODO: continue

        console.info("Prepare optimizer...", "")
        self.prepare_optimizer()
        console.ok("OK")

        console.info("Prepare loss function...", "")
        self.prepare_loss_func()
        console.ok("OK")

        console.info("Prepare client selection...", "")
        self.prepare_client_selection()
        console.ok("OK")

        console.info("Prepare aggregation...", "")
        self.prepare_aggregation()
        console.ok("OK")
        
        console.info("Prepare extractor...", "")
        self.prepare_extractor()
        console.ok("OK")

        # Prepare logger
        console.info("Prepare training logger...", "")
        self.prepare_training_logger()
        console.ok("OK")

        console.info("Prepare model evaluator...", "")
        self.prepare_model_evaluator()
        console.ok("OK")
        
        console.info("Prepare trainer...", "")
        self.prepare_trainer()
        console.ok("OK")

        console.info("Prepare completed.")

        return self

    def prepare_strategy_only(self):
        console.info("Prepare strategy...", "")
        self.prepare_strategy()
        console.ok("OK")
        return self
    
    def set_device(self, device: str):
        """
        Set computation device
        """
        self.device = device
        return