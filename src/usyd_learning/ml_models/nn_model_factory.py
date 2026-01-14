from __future__ import annotations
from .nn_model import AbstractNNModel, NNModelArgs


class NNModelFactory:
    """
    NN Model factory static class
    """
    @staticmethod
    def create_args(config_dict: dict|None = None, is_clone_dict=False) -> NNModelArgs:
        return NNModelArgs(config_dict, is_clone_dict)

    @staticmethod
    def create(args: NNModelArgs) -> AbstractNNModel:
        match args.model_type:
            case "mnist_nn_brenden":
                from .models._nn_model_mnist_nn_brenden import NNModel_MnistNNBrenden
                return NNModel_MnistNNBrenden().create_model(args)
            case "simple_lora_mlp":
                from .lora._nn_model_simple_lora_mlp import NNModel_SimpleLoRAMLP
                return NNModel_SimpleLoRAMLP().create_model(args)
            case "capstone_mlp":
                from .models._nn_model_capstone_mlp import NNModel_CapstoneMLP
                return NNModel_CapstoneMLP().create_model(args)
            case "simple_mlp":
                from .models._nn_model_simple_mlp import NNModel_SimpleMLP
                return NNModel_SimpleMLP().create_model(args)
            case "cifar_convnet":
                from .models._nn_model_cifar_convnet import NNModel_CifarConvnet
                return NNModel_CifarConvnet().create_model(args)
            case "simple_lora_mlp":
                from .lora._nn_model_simple_lora_mlp import NNModel_SimpleLoRAMLP
                return NNModel_SimpleLoRAMLP().create_model(args)
            case "simple_lora_cnn":
                from .lora._nn_model_simple_lora_cnn import NNModel_SimpleLoRACNN
                return NNModel_SimpleLoRACNN().create_model(args)
            case "cifar_lora_cnn":
                from .lora._nn_model_cifar_lora_cnn import NNModel_CifarLoRACNN
                return NNModel_CifarLoRACNN().create_model(args)
            case "cifar10_lora_vit":
                from .vit._nn_model_cifar10_lora_vit import NNModel_ViTMSLoRACIFAR10
                return NNModel_ViTMSLoRACIFAR10().create_model(args)
            case "imdb_lora_transformer":
                from .lora._nn_model_imdb_lora_transformer import NNModel_ImdbMSLoRATransformer
                return NNModel_ImdbMSLoRATransformer().create_model(args)
            case "peft_transformer":
                from .transformer.encoder_only._nn_model_peft_transformer import NNModel_PeftTransformer
                return NNModel_PeftTransformer().create_model(args)
            case "sfl_mlp_client":
                from .sfl.__nn_model_sfl_mlp_client import NNModel_SflMlpClient
                return NNModel_SflMlpClient().create_model(args)
            case "sfl_mlp_server":
                from .sfl.__nn_model_sfl_mlp_server import NNModel_SflMlpServer
                return NNModel_SflMlpServer().create_model(args)

        raise ValueError(f"Unknown mode type '{args.model_type}'")
