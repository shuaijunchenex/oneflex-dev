from __future__ import annotations

from typing import Any

import torch.nn as nn
import torch.nn.functional as F

from .. import AbstractNNModel, NNModelArgs, NNModel


class NNModel_SflMlpServer(NNModel):
    def __init__(self) -> None:
        super().__init__()
        self._activation = nn.ReLU()
        self._fc2: nn.Linear
        self._fc3: nn.Linear
        self._softmax_dim = 1

    # override
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)

        hidden_dim = args.hidden_dim or args.get("hidden_dim", 0) or 200
        server_hidden_dim = args.get("server_hidden_dim", hidden_dim) or hidden_dim
        output_dim = args.output_dim or args.get("output_dim", 0) or 10
        self._softmax_dim = args.softmax_dim or args.get("softmax_dim", 0) or 1

        self._fc2 = nn.Linear(200, 200)
        self._fc3 = nn.Linear(200, 10)
        return self

    # override
    def forward(self, x) -> Any:
        x = self._activation(self._fc2(x))
        x = self._fc3(x)
        return F.softmax(x, dim=1)
