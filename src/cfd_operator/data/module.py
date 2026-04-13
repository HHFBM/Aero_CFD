"""Data module for training, evaluation, and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from cfd_operator.config.schemas import DataConfig
from cfd_operator.data.collate import cfd_collate_fn
from cfd_operator.data.dataset import CFDOperatorDataset
from cfd_operator.data.file_dataset import load_dataset_payload
from cfd_operator.data.normalization import StandardNormalizer
from cfd_operator.data.synthetic import SyntheticAirfoilDatasetGenerator
from cfd_operator.utils.io import ensure_dir


@dataclass(slots=True)
class NormalizerBundle:
    branch: StandardNormalizer
    coordinates: StandardNormalizer
    fields: StandardNormalizer
    scalars: StandardNormalizer

    def to_dict(self) -> dict[str, dict[str, list[float]]]:
        return {
            "branch": self.branch.to_dict(),
            "coordinates": self.coordinates.to_dict(),
            "fields": self.fields.to_dict(),
            "scalars": self.scalars.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NormalizerBundle":
        return cls(
            branch=StandardNormalizer.from_dict(payload["branch"]),
            coordinates=StandardNormalizer.from_dict(payload["coordinates"]),
            fields=StandardNormalizer.from_dict(payload["fields"]),
            scalars=StandardNormalizer.from_dict(payload["scalars"]),
        )


class CFDDataModule:
    """Loads synthetic or file-backed datasets and builds dataloaders."""

    def __init__(self, config: DataConfig, batch_size: int) -> None:
        self.config = config
        self.batch_size = batch_size
        self.payload: dict[str, Any] | None = None
        self.normalizers: NormalizerBundle | None = None
        self.datasets: dict[str, CFDOperatorDataset] = {}

    def prepare_data(self) -> Path:
        dataset_path = Path(self.config.dataset_path)
        if self.config.dataset_type == "synthetic" and not dataset_path.exists():
            ensure_dir(dataset_path.parent)
            generator = SyntheticAirfoilDatasetGenerator(config=self.config)
            return generator.save(dataset_path)
        if self.config.dataset_type == "file" and not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        return dataset_path

    def setup(self) -> None:
        dataset_path = self.prepare_data()
        self.payload = load_dataset_payload(dataset_path)
        train_indices = self.payload.get("train_indices")
        val_indices = self.payload.get("val_indices")
        test_indices = self.payload.get("test_indices")
        if train_indices is None or val_indices is None or test_indices is None:
            num_samples = len(self.payload["airfoil_id"])
            indices = np.arange(num_samples)
            train_end = int(num_samples * self.config.train_ratio)
            val_end = train_end + int(num_samples * self.config.val_ratio)
            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]

        self.normalizers = self._fit_normalizers(np.asarray(train_indices))
        self.datasets = {
            "train": self._make_dataset(np.asarray(train_indices)),
            "val": self._make_dataset(np.asarray(val_indices)),
            "test": self._make_dataset(np.asarray(test_indices)),
        }

    def _fit_normalizers(self, train_indices: np.ndarray) -> NormalizerBundle:
        assert self.payload is not None
        branch_values = self.payload["branch_inputs"][train_indices]
        query_points = self.payload["query_points"][train_indices].reshape(-1, self.payload["query_points"].shape[-1])
        field_values = self.payload["field_targets"][train_indices].reshape(-1, self.payload["field_targets"].shape[-1])
        scalar_values = self.payload["scalar_targets"][train_indices]

        return NormalizerBundle(
            branch=StandardNormalizer.fit(branch_values),
            coordinates=StandardNormalizer.fit(query_points),
            fields=StandardNormalizer.fit(field_values),
            scalars=StandardNormalizer.fit(scalar_values),
        )

    def _make_dataset(self, indices: np.ndarray) -> CFDOperatorDataset:
        assert self.payload is not None
        assert self.normalizers is not None
        return CFDOperatorDataset(
            payload=self.payload,
            indices=indices,
            branch_normalizer=self.normalizers.branch,
            coordinate_normalizer=self.normalizers.coordinates,
            field_normalizer=self.normalizers.fields,
            scalar_normalizer=self.normalizers.scalars,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=cfd_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=cfd_collate_fn,
        )

    def test_dataloader(self, batch_size: int | None = None) -> DataLoader:
        return DataLoader(
            self.datasets["test"],
            batch_size=batch_size or self.batch_size,
            shuffle=False,
            collate_fn=cfd_collate_fn,
        )
