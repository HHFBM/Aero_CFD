"""Base model abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class BaseOperatorModel(nn.Module, ABC):
    """Abstract base class for operator surrogate models."""

    @abstractmethod
    def forward(self, branch_inputs: torch.Tensor, query_points: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return a prediction dictionary containing at least field/scalar outputs."""

    def loss_outputs(self, branch_inputs: torch.Tensor, query_points: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward(branch_inputs=branch_inputs, query_points=query_points)

    def predict_fields(self, branch_inputs: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
        return self.forward(branch_inputs=branch_inputs, query_points=query_points)["fields"]

    def predict_scalars(self, branch_inputs: torch.Tensor) -> torch.Tensor:
        dummy_points = torch.zeros(
            branch_inputs.shape[0],
            1,
            2,
            device=branch_inputs.device,
            dtype=branch_inputs.dtype,
        )
        return self.forward(branch_inputs=branch_inputs, query_points=dummy_points)["scalars"]

    def decoder_head_metadata(self) -> dict[str, dict[str, object]]:
        return {}

    def model_metadata(self) -> dict[str, Any]:
        return {
            "class_name": self.__class__.__name__,
            "decoder_heads": self.decoder_head_metadata(),
        }
