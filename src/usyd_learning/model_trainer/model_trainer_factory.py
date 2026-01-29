from __future__ import annotations
from .model_trainer import ModelTrainer, ModelTrainerArgs


class ModelTrainerFactory:
    """
    Model trainer factory
    """
    
    @staticmethod
    def create_args(config_dict: dict, is_clone_dict: bool = False) -> ModelTrainerArgs:
        """
        Static method to create trainer args
        """
        return ModelTrainerArgs(config_dict, is_clone_dict)

    @staticmethod
    def create(args: ModelTrainerArgs) -> ModelTrainer:
        """
        Static method to create trainer
        """
        match args.trainer_type:
            case "standard":
                from .trainer._model_trainer_standard import ModelTrainer_Standard
                return ModelTrainer_Standard(args)
            case "vit":
                from .trainer._model_trainer_vit import ModelTrainer_Vit
                return ModelTrainer_Vit(args)
            case "imdb":
                from .trainer._model_trainer_imdb import ModelTrainer_Imdb
                return ModelTrainer_Imdb(args)
            case "glue":
                from .trainer._model_trainer_glue import ModelTrainer_GLUE
                return ModelTrainer_GLUE(args)
            case "sl_client":
                from .trainer._model_trainer_sl_client import ModelTrainer_SlClient
                return ModelTrainer_SlClient(args)
            case "sl_server":
                from .trainer._model_trainer_sl_server import ModelTrainer_SlServer
                return ModelTrainer_SlServer(args)
            case _:
                raise Exception(f"Undefined trainer type {args.trainer_type}.")

