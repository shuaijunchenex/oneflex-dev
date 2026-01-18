import torch
import gc
from .console import console
from torch.optim import Optimizer
from torch import nn

class ModelUtils:
    @staticmethod
    def accelerator_device():
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        console.debug(f"Using device: {device}")
        return device

    @staticmethod
    def clear_all(model: nn.Module, optimizer: Optimizer):
        """
        Clears gradients, resets optimizer state, and releases unused cached GPU memory.
        """
        ModelUtils.clear_model_grads(model)
        ModelUtils.reset_optimizer_state(optimizer)
        ModelUtils.clear_cuda_cache()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ratio = trainable_params / total_params if total_params > 0 else 0.0

        console.ok(
            f"[All Cleared] {model.__class__.__name__} | "
            f"trainable={trainable_params:,}/{total_params:,} ({ratio:.2%}) | "
            f"optimizer={optimizer.__class__.__name__}"
        )

    @staticmethod
    def clear_model_grads(model: nn.Module):
        """
        Clears the gradients of all parameters in the given model by setting .grad to None.
        Also logs parameter statistics.
        """
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ratio = trainable_params / total_params if total_params > 0 else 0.0

        console.info(
            f"[Grads Cleared] {model.__class__.__name__} | id={id(model)} | "
            f"trainable={trainable_params:,} / total={total_params:,} "
            f"({ratio:.2%})"
        )

    @staticmethod
    def clear_cuda_cache():
        """
        Releases unused cached GPU memory to help avoid memory accumulation.
        """
        gc.collect()
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        console.info(f"[Cuda Cache Cleared] allocated={allocated:.2f}MB | reserved={reserved:.2f}MB")

    @staticmethod
    def reset_optimizer_state(optimizer: Optimizer):
        """
        Clears the internal state of an optimizer (e.g., momentum buffers),
        and outputs the current learning rates.
        """
        optimizer.state.clear()
        num_params = sum(
            p.numel() for group in optimizer.param_groups for p in group['params']
        )

        lrs = [group.get("lr", None) for group in optimizer.param_groups]

        console.info(
            f"[Optimizer Reset] {optimizer.__class__.__name__} "
            f"| id={id(optimizer)} | params={num_params:,} | lr={lrs}"
        )

