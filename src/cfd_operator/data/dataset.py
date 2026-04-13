"""PyTorch datasets for CFD operator training."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from cfd_operator.data.normalization import StandardNormalizer


class CFDOperatorDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset backed by dense per-sample arrays."""

    def __init__(
        self,
        payload: dict[str, Any],
        indices: np.ndarray,
        branch_normalizer: StandardNormalizer,
        coordinate_normalizer: StandardNormalizer,
        field_normalizer: StandardNormalizer,
        scalar_normalizer: StandardNormalizer,
    ) -> None:
        self.payload = payload
        self.indices = indices.astype(np.int64)
        self.branch_normalizer = branch_normalizer
        self.coordinate_normalizer = coordinate_normalizer
        self.field_normalizer = field_normalizer
        self.scalar_normalizer = scalar_normalizer

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str]:
        index = int(self.indices[item])
        branch_inputs = self.payload["branch_inputs"][index].astype(np.float32)
        query_points = self.payload["query_points"][index].astype(np.float32)
        field_targets = self.payload["field_targets"][index].astype(np.float32)
        surface_points = self.payload["surface_points"][index].astype(np.float32)
        surface_cp = self.payload["surface_cp"][index].astype(np.float32)
        scalar_targets = self.payload["scalar_targets"][index].astype(np.float32)

        query_mask = np.ones(query_points.shape[0], dtype=np.float32)
        surface_mask = np.ones(surface_points.shape[0], dtype=np.float32)

        sample: dict[str, torch.Tensor | str] = {
            "airfoil_id": str(self.payload["airfoil_id"][index]),
            "geometry_params": torch.from_numpy(self.payload["geometry_params"][index].astype(np.float32)),
            "flow_conditions": torch.from_numpy(self.payload["flow_conditions"][index].astype(np.float32)),
            "branch_inputs_raw": torch.from_numpy(branch_inputs),
            "branch_inputs": torch.from_numpy(self.branch_normalizer.transform(branch_inputs)),
            "query_points_raw": torch.from_numpy(query_points),
            "query_points": torch.from_numpy(self.coordinate_normalizer.transform(query_points)),
            "field_targets_raw": torch.from_numpy(field_targets),
            "field_targets": torch.from_numpy(self.field_normalizer.transform(field_targets)),
            "surface_points_raw": torch.from_numpy(surface_points),
            "surface_points": torch.from_numpy(self.coordinate_normalizer.transform(surface_points)),
            "surface_cp": torch.from_numpy(surface_cp),
            "scalar_targets_raw": torch.from_numpy(scalar_targets),
            "scalar_targets": torch.from_numpy(self.scalar_normalizer.transform(scalar_targets)),
            "query_mask": torch.from_numpy(query_mask),
            "surface_mask": torch.from_numpy(surface_mask),
            "fidelity_level": torch.tensor(self.payload["fidelity_level"][index], dtype=torch.long),
            "convergence_flag": torch.tensor(self.payload["convergence_flag"][index], dtype=torch.long),
        }
        return sample

