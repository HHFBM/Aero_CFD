"""Array normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(slots=True)
class StandardNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "StandardNormalizer":
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < 1.0e-6, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    @classmethod
    def identity(cls, dim: int) -> "StandardNormalizer":
        return cls(
            mean=np.zeros(dim, dtype=np.float32),
            std=np.ones(dim, dtype=np.float32),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return (values * self.std + self.mean).astype(np.float32)

    def transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=values.device, dtype=values.dtype)
        std = torch.as_tensor(self.std, device=values.device, dtype=values.dtype)
        return (values - mean) / std

    def inverse_transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=values.device, dtype=values.dtype)
        std = torch.as_tensor(self.std, device=values.device, dtype=values.dtype)
        return values * std + mean

    def gradient_scale_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(self.std, device=device, dtype=dtype)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> "StandardNormalizer":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )

