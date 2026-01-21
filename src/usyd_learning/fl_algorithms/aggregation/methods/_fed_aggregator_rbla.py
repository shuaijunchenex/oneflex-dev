import torch
from collections import OrderedDict

from ..fed_aggregator_abc import AbstractFedAggregator
from ..fed_aggregator_args import FedAggregatorArgs
from ....ml_utils import console


class FedAggregator_RBLA(AbstractFedAggregator):
    """
    RBLA aggregation that is API-compatible with FedAggregator_FedAvg:
      - build_data_list(dict_like) takes values of (state_dict, data_volume)
      - _do_aggregation() aggregates into self._aggregated_weight (OrderedDict)
    """

    def __init__(self, args: FedAggregatorArgs | None = None):
        super().__init__(args)
        self._aggregation_method = "rbla"
        self._lora_suffixes: set[str] = {"lora_A", "lora_B"}
        self._pad_mode: str = "nan"   # 'nan' or 'zero'
        return

    # ---------- Public config ----------
    def set_lora_suffixes(self, lora_suffixes: set[str]) -> None:
        self._lora_suffixes = lora_suffixes

    def set_pad_mode(self, pad_mode: str) -> None:
        assert pad_mode in {"nan", "zero"}, f"Unsupported pad_mode: {pad_mode}"
        self._pad_mode = pad_mode

    # ---------- FedAvg-style data building ----------
    def build_data_list(self, aggregation_data_dict: dict) -> None:
        """
        Make internal list like: [(state_dict, data_volume), ...]
        (Compatible with FedAggregator_FedAvg expectation)
        """
        self._aggregation_data_list = list(aggregation_data_dict.values())
        return

    def build_data_dict(self, aggregation_data_dict: dict) -> None:
        """If you still pass {'state_dicts': [...], 'weights': [...]}, keep it."""
        self._aggregation_data_dict = aggregation_data_dict

    # ---------- Aggregation lifecycle ----------
    def _before_aggregation(self) -> None:
        # console.debug(f"[RBLA] Starting aggregation with {len(self._aggregation_data_list)} clients...")
        return

    def _do_aggregation(self) -> None:
        """
        Aggregate using RBLA. Accept inputs as:
        1) self._aggregation_data_list = [(state_dict, data_volume), ...]
        2) self._aggregation_data_dict = [(state_dict, data_volume), ...]
        3) self._aggregation_data_dict = {'state_dicts': [...], 'weights': [...]}
        """
        state_dicts, weights = None, None

        # 1) Preferred list-on-list
        if hasattr(self, "_aggregation_data_list") and self._aggregation_data_list:
            pairs = self._aggregation_data_list
            state_dicts = [sd for sd, vol in pairs]
            weights    = [float(vol) for sd, vol in pairs]

        # 2) Your current case: _aggregation_data_dict is actually a list of (sd, vol)
        elif hasattr(self, "_aggregation_data_dict") and isinstance(self._aggregation_data_dict, list):
            pairs = self._aggregation_data_dict
            state_dicts = [sd for sd, vol in pairs]
            weights    = [float(vol) for sd, vol in pairs]

        # 3) Legacy dict form
        elif hasattr(self, "_aggregation_data_dict") and isinstance(self._aggregation_data_dict, dict):
            state_dicts = self._aggregation_data_dict["state_dicts"]
            weights     = self._aggregation_data_dict.get("weights", None)

        else:
            raise ValueError("[RBLA] No aggregation data found. Provide a list of (state_dict, data_volume) "
                            "or dict {'state_dicts': [...], 'weights': [...]}.")

        console.debug(f"\n[RBLA] Aggregating {len(state_dicts)} clients...")
        total_data_vol = sum(vol for _, vol in self._aggregation_data_dict)
        for i, (_, vol) in enumerate(self._aggregation_data_dict):
            console.debug(f"  Client {i}: {vol} samples ({vol / total_data_vol * 100:.1f}%)")

        # move to device
        dev = self._device
        sds_on_device = [{k: v.to(dev) for k, v in sd.items()} for sd in state_dicts]

        aggregated = self.aggregate_state_dicts(
            sds_on_device,
            weights=weights,
            lora_suffixes=self._lora_suffixes,
            pad_mode=self._pad_mode,
        )

        # keep key order like the first state_dict
        from collections import OrderedDict
        sample_keys = list(state_dicts[0].keys())
        ordered = OrderedDict((k, aggregated[k]) for k in sample_keys)
        self._aggregated_weight = ordered

        first_param_name = next(iter(ordered.keys()))
        console.debug(f"[RBLA] Aggregated first param mean: {ordered[first_param_name].mean():.6f}")


    def _after_aggregation(self) -> None:
        console.debug("[RBLA] Aggregation completed.")

    # ---------- Core RBLA ops ----------
    @staticmethod
    def get_lora_type(key: str, lora_suffixes: set[str]) -> str | None:
        """
        Check if key contains a LoRA suffix component (e.g. 'lora_A').
        Returns the suffix if found (e.g. 'lora_A'), else None.
        Compatible with HuggingFace PEFT keys like 'base_model.model...lora_A.default.weight'.
        """
        parts = key.split(".")
        for suffix in lora_suffixes:
            if suffix in parts:
                return suffix
        return None

    @staticmethod
    def pad_tensors_to_max_shape(tensors: list[torch.Tensor], pad_mode: str = "nan") -> torch.Tensor:
        """
        Pad 2D tensors to a common shape; return stacked 3D tensor: (N, max_rows, max_cols).
        """
        assert pad_mode in {"nan", "zero"}, f"Unsupported pad_mode: {pad_mode}"
        if len(tensors) == 0:
            raise ValueError("pad_tensors_to_max_shape: empty tensor list")

        # Ensure 2D for LoRA matrices
        for t in tensors:
            if t.dim() != 2:
                raise ValueError(f"LoRA tensor must be 2D, got {t.dim()}D for shape {tuple(t.shape)}")

        max_rows = max(t.shape[0] for t in tensors)
        max_cols = max(t.shape[1] for t in tensors)
        device = tensors[0].device
        dtype = tensors[0].dtype

        fill_val = float("nan") if pad_mode == "nan" else 0.0
        padded_list = []
        for t in tensors:
            pad = torch.full((max_rows, max_cols), fill_val, dtype=dtype, device=device)
            pad[: t.shape[0], : t.shape[1]] = t
            padded_list.append(pad)
        return torch.stack(padded_list, dim=0)

    @staticmethod
    def aggregate_lora_tensors(
        tensors: list[torch.Tensor],
        weights: list[float],
        pad_mode: str = "nan",
    ) -> torch.Tensor:
        """
        Weighted average with padding-aware handling for LoRA matrices.
        """
        if len(tensors) == 0:
            raise ValueError("aggregate_lora_tensors: empty tensor list")

        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=tensors[0].device).view(-1, 1, 1)
        padded = FedAggregator_RBLA.pad_tensors_to_max_shape(tensors, pad_mode=pad_mode)

        if pad_mode == "nan":
            valid_mask = ~torch.isnan(padded)
            padded = torch.nan_to_num(padded, nan=0.0)
            weighted_sum = (padded * weights_tensor).sum(dim=0)
            weight_mask = valid_mask * weights_tensor
            total_weight = weight_mask.sum(dim=0)
            total_weight[total_weight == 0] = 1.0  # avoid div-by-zero
            return weighted_sum / total_weight
        else:  # zero padding
            weighted_sum = (padded * weights_tensor).sum(dim=0)
            total_weight = sum(weights)
            return weighted_sum / total_weight

    @staticmethod
    def aggregate_state_dicts(
        state_dicts: list[dict],
        weights: list[float] | None = None,
        lora_suffixes: set[str] = {"lora_A", "lora_B"},
        pad_mode: str = "nan",
    ) -> dict:
        """
        Aggregate multiple state_dicts with LoRA-aware averaging.
        """
        if len(state_dicts) == 0:
            raise ValueError("aggregate_state_dicts: empty state_dicts")

        if weights is None:
            weights = [1.0] * len(state_dicts)

        # normalize weights to sum=1 for stability
        tw = float(sum(weights))
        weights = [w / tw for w in weights] if tw > 0 else [1.0 / len(weights)] * len(weights)

        keys = list(state_dicts[0].keys())
        aggregated: dict[str, torch.Tensor] = {}

        for key in keys:
            values = [sd[key] for sd in state_dicts]
            lora_type = FedAggregator_RBLA.get_lora_type(key, lora_suffixes)

            if lora_type is not None:
                aggregated[key] = FedAggregator_RBLA.aggregate_lora_tensors(values, weights, pad_mode=pad_mode)
            else:
                stacked = torch.stack(values, dim=0)  # (N, ...)
                # weights reshape: (N, 1, 1, ..., 1)
                view_shape = (len(weights),) + (1,) * (stacked.dim() - 1)
                weight_tensor = torch.as_tensor(weights, dtype=stacked.dtype, device=stacked.device).view(*view_shape)
                weighted_sum = (stacked * weight_tensor).sum(dim=0)
                aggregated[key] = weighted_sum  # weights已归一化

        return aggregated

    @staticmethod
    def broadcast_lora_state_dict(global_sd: dict, local_sd: dict, lora_suffixes: set[str] = {"lora_A", "lora_B"}) -> dict:
        """
        Slice global LoRA matrices back to each client rank, copy non-LoRA tensors directly.
        """
        new_local_sd = {}
        for key, local_tensor in local_sd.items():
            global_tensor = global_sd[key]
            lora_type = FedAggregator_RBLA.get_lora_type(key, lora_suffixes)

            if lora_type is None:
                new_local_sd[key] = global_tensor.clone()
            else:
                if lora_type == "lora_A":       # [r, in]
                    r_local = local_tensor.shape[0]
                    new_local_sd[key] = global_tensor[:r_local, :].clone()
                elif lora_type == "lora_B":     # [out, r]
                    r_local = local_tensor.shape[1]
                    new_local_sd[key] = global_tensor[:, :r_local].clone()
                else:
                    raise ValueError(f"Unrecognized LoRA suffix: {lora_type}")
        return new_local_sd