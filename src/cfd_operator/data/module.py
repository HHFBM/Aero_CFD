"""Data module for training, evaluation, and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from cfd_operator.config.schemas import DataConfig
from cfd_operator.data.adapters import AdapterResult, build_adapter
from cfd_operator.data.collate import cfd_collate_fn
from cfd_operator.data.dataset import CFDOperatorDataset
from cfd_operator.data.normalization import StandardNormalizer
from cfd_operator.data.airfrans import AirfRANSDatasetConverter
from cfd_operator.data.airfrans_original import AirfRANSOriginalDatasetConverter
from cfd_operator.data.quality import validate_dataset_payload
from cfd_operator.data.synthetic import SyntheticAirfoilDatasetGenerator
from cfd_operator.schema import CFDSurrogateSample
from cfd_operator.tasks.capabilities import DatasetCapability
from cfd_operator.utils.io import ensure_dir


@dataclass
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
        self.adapter_result: AdapterResult | None = None
        self.unified_samples: list[CFDSurrogateSample] = []
        self.dataset_capability: DatasetCapability | None = None
        self.normalizers: NormalizerBundle | None = None
        self.datasets: dict[str, CFDOperatorDataset] = {}

    def prepare_data(self) -> Path:
        dataset_path = Path(self.config.dataset_path)
        if self.config.dataset_type == "synthetic" and not dataset_path.exists():
            ensure_dir(dataset_path.parent)
            generator = SyntheticAirfoilDatasetGenerator(config=self.config)
            return generator.save(dataset_path)
        if self.config.dataset_type == "airfrans" and not dataset_path.exists():
            ensure_dir(dataset_path.parent)
            converter = AirfRANSDatasetConverter(config=self.config)
            return converter.save(dataset_path)
        if self.config.dataset_type == "airfrans_original" and not dataset_path.exists():
            ensure_dir(dataset_path.parent)
            converter = AirfRANSOriginalDatasetConverter(config=self.config)
            return converter.save(dataset_path)
        if self.config.dataset_type == "file" and not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        return dataset_path

    def setup(self) -> None:
        dataset_path = self.prepare_data()
        adapter = build_adapter(self.config)
        self.adapter_result = adapter.load(dataset_path)
        self.payload = self.adapter_result.payload
        self.unified_samples = self.adapter_result.samples
        self.dataset_capability = self.adapter_result.capability
        if self.config.strict_quality_checks:
            try:
                validate_dataset_payload(self.payload, strict=True)
            except ValueError:
                if self.config.dataset_type != "synthetic":
                    raise
                generator = SyntheticAirfoilDatasetGenerator(config=self.config)
                generator.save(dataset_path)
                self.adapter_result = adapter.load(dataset_path)
                self.payload = self.adapter_result.payload
                self.unified_samples = self.adapter_result.samples
                self.dataset_capability = self.adapter_result.capability
                validate_dataset_payload(self.payload, strict=True)

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
        self.datasets = {}
        for split_name, split_indices in self._discover_splits().items():
            self.datasets[split_name] = self._make_dataset(np.asarray(split_indices))

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

    def test_dataloader(self, batch_size: int | None = None, split_name: str = "test") -> DataLoader:
        return DataLoader(
            self.datasets[split_name],
            batch_size=batch_size or self.batch_size,
            shuffle=False,
            collate_fn=cfd_collate_fn,
        )

    def available_splits(self) -> list[str]:
        return sorted(self.datasets.keys())

    def _discover_splits(self) -> dict[str, np.ndarray]:
        assert self.payload is not None
        split_mapping: dict[str, np.ndarray] = {}
        for key, value in self.payload.items():
            if key.endswith("_indices"):
                split_mapping[key.removesuffix("_indices")] = np.asarray(value, dtype=np.int64)
        return split_mapping
