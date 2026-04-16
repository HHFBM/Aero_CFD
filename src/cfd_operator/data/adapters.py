"""Dataset adapters that normalize source-specific payloads into a unified schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cfd_operator.config.schemas import DataConfig
from cfd_operator.data.analysis import ensure_analysis_payload
from cfd_operator.data.file_dataset import load_dataset_payload
from cfd_operator.geometry.semantics import ensure_geometry_payload_metadata
from cfd_operator.schema import CFDSurrogateSample, sample_from_legacy_payload
from cfd_operator.tasks.capabilities import DatasetCapability, infer_dataset_capability


@dataclass(frozen=True)
class AdapterResult:
    payload: dict[str, Any]
    samples: list[CFDSurrogateSample]
    capability: DatasetCapability
    dataset_name: str
    source_format: str


class BaseCFDAdapter:
    """Read source-specific data and normalize it into the unified sample schema."""

    source_format = "legacy_payload"

    def __init__(self, config: DataConfig) -> None:
        self.config = config

    def load(self, path: str | Path) -> AdapterResult:
        payload = load_dataset_payload(path, config=self.config)
        payload = ensure_geometry_payload_metadata(payload, branch_feature_mode=self.config.branch_feature_mode)
        payload = ensure_analysis_payload(payload)
        samples = self._build_samples(payload)
        capability = infer_dataset_capability(payload, dataset_name=self.config.name)
        return AdapterResult(
            payload=payload,
            samples=samples,
            capability=capability,
            dataset_name=self.config.name,
            source_format=self.source_format,
        )

    def _build_samples(self, payload: dict[str, Any]) -> list[CFDSurrogateSample]:
        split_lookup = self._build_split_lookup(payload)
        return [
            sample_from_legacy_payload(
                payload,
                index,
                dataset_name=self.config.name,
                split=split_lookup.get(index),
                source_format=self.source_format,
            )
            for index in range(len(payload["airfoil_id"]))
        ]

    @staticmethod
    def _build_split_lookup(payload: dict[str, Any]) -> dict[int, str]:
        split_lookup: dict[int, str] = {}
        for key, value in payload.items():
            if not key.endswith("_indices"):
                continue
            split_name = key.removesuffix("_indices")
            for index in value:
                split_lookup[int(index)] = split_name
        return split_lookup


class SyntheticAdapter(BaseCFDAdapter):
    source_format = "synthetic_npz"


class AirfRANSAdapter(BaseCFDAdapter):
    source_format = "airfrans_npz"


class AirfRANSOriginalAdapter(BaseCFDAdapter):
    source_format = "airfrans_original_npz"


class NPZFileAdapter(BaseCFDAdapter):
    source_format = "file_payload"


def build_adapter(config: DataConfig) -> BaseCFDAdapter:
    if config.dataset_type == "synthetic":
        return SyntheticAdapter(config)
    if config.dataset_type == "airfrans":
        return AirfRANSAdapter(config)
    if config.dataset_type == "airfrans_original":
        return AirfRANSOriginalAdapter(config)
    if config.dataset_type == "file":
        return NPZFileAdapter(config)
    raise ValueError(f"Unsupported dataset_type for adapter: {config.dataset_type}")
