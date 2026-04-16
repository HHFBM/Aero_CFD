"""Dataset capability and task-request helpers.

The current project supports multiple output families with different semantics:
some are supervised directly, some are derived from supervised quantities, and
some remain placeholder/experimental.  This module centralizes those
capabilities so trainer/evaluator/inference code can progressively stop
hard-coding dataset-specific assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from cfd_operator.config.schemas import LossConfig


CapabilityCategory = Literal["supervised", "derived", "placeholder", "unavailable"]


@dataclass(frozen=True)
class CapabilityStatus:
    category: CapabilityCategory
    available: bool
    trainable: bool
    evaluable: bool
    exportable: bool
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "available": self.available,
            "trainable": self.trainable,
            "evaluable": self.evaluable,
            "exportable": self.exportable,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CapabilityStatus":
        return cls(
            category=str(payload["category"]),  # type: ignore[arg-type]
            available=bool(payload["available"]),
            trainable=bool(payload["trainable"]),
            evaluable=bool(payload["evaluable"]),
            exportable=bool(payload["exportable"]),
            note=str(payload.get("note", "")),
        )


@dataclass(frozen=True)
class DatasetCapability:
    dataset_name: str
    dimensionality: Literal["2d", "3d"] = "2d"
    target_capabilities: dict[str, CapabilityStatus] = field(default_factory=dict)

    def status(self, name: str) -> CapabilityStatus:
        return self.target_capabilities.get(
            name,
            CapabilityStatus(
                category="unavailable",
                available=False,
                trainable=False,
                evaluable=False,
                exportable=False,
                note="Target not declared by dataset capability.",
            ),
        )

    def available(self, name: str) -> bool:
        return self.status(name).available

    def trainable(self, name: str) -> bool:
        return self.status(name).trainable

    def evaluable(self, name: str) -> bool:
        return self.status(name).evaluable

    def exportable(self, name: str) -> bool:
        return self.status(name).exportable

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "dimensionality": self.dimensionality,
            "target_capabilities": {
                key: value.as_dict() for key, value in self.target_capabilities.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "DatasetCapability":
        target_capabilities = {
            str(key): CapabilityStatus.from_dict(value)  # type: ignore[arg-type]
            for key, value in dict(payload.get("target_capabilities", {})).items()
        }
        return cls(
            dataset_name=str(payload.get("dataset_name", "unknown")),
            dimensionality=str(payload.get("dimensionality", "2d")),  # type: ignore[arg-type]
            target_capabilities=target_capabilities,
        )


@dataclass(frozen=True)
class TaskRequest:
    field: bool = True
    scalar: bool = True
    surface: bool = True
    slice: bool = True
    feature: bool = True
    consistency: bool = False


@dataclass(frozen=True)
class EffectiveTaskSet:
    field: bool
    scalar: bool
    surface: bool
    slice: bool
    feature: bool
    consistency: bool

    def as_dict(self) -> dict[str, bool]:
        return {
            "field": self.field,
            "scalar": self.scalar,
            "surface": self.surface,
            "slice": self.slice,
            "feature": self.feature,
            "consistency": self.consistency,
        }


def _bool_from_payload_mask(payload: dict[str, object], key: str) -> bool:
    if key not in payload:
        return False
    values = np.asarray(payload[key])
    if values.size == 0:
        return False
    if values.dtype.kind in {"U", "S", "O"}:
        return True
    return bool(np.isfinite(values).any() and np.any(np.abs(values) > 0.0)) or bool(np.isfinite(values).all())


def infer_dataset_capability(payload: dict[str, object], *, dataset_name: str) -> DatasetCapability:
    """Infer current dataset capability from the legacy payload."""

    def status(
        *,
        name: str,
        category: CapabilityCategory,
        available: bool,
        trainable: bool | None = None,
        evaluable: bool | None = None,
        exportable: bool | None = None,
        note: str = "",
    ) -> CapabilityStatus:
        return CapabilityStatus(
            category=category,
            available=available,
            trainable=available if trainable is None else trainable,
            evaluable=available if evaluable is None else evaluable,
            exportable=available if exportable is None else exportable,
            note=note,
        )

    target_capabilities = {
        "field_targets": status(name="field_targets", category="supervised", available="field_targets" in payload),
        "scalar_targets": status(name="scalar_targets", category="supervised", available="scalar_targets" in payload),
        "surface_pressure": status(
            name="surface_pressure",
            category="supervised",
            available="surface_pressure" in payload,
            note="Some paths reconstruct raw pressure from cp_reference; do not assume native pressure GT everywhere.",
        ),
        "surface_cp": status(
            name="surface_cp",
            category="derived",
            available="surface_cp" in payload,
            note="Derived from surface pressure/cp_reference semantics even when used as a training target.",
        ),
        "surface_velocity": status(
            name="surface_velocity",
            category="supervised",
            available="surface_velocity" in payload,
        ),
        "surface_nut": status(
            name="surface_nut",
            category="supervised",
            available="surface_nut" in payload,
        ),
        "slice_fields": status(
            name="slice_fields",
            category="derived",
            available="slice_fields" in payload,
        ),
        "pressure_gradient_indicator": status(
            name="pressure_gradient_indicator",
            category="derived",
            available="pressure_gradient_indicator" in payload,
        ),
        "high_gradient_mask": status(
            name="high_gradient_mask",
            category="derived",
            available="high_gradient_mask" in payload,
        ),
        "shock_indicator": status(
            name="shock_indicator",
            category="placeholder",
            available="shock_indicator" in payload,
            trainable=False,
            note="High-gradient approximation only, not real shock supervision.",
        ),
        "shock_location": status(
            name="shock_location",
            category="placeholder",
            available="shock_location" in payload,
            trainable=False,
            note="High-gradient approximation only, not real shock supervision.",
        ),
        "surface_heat_flux": status(
            name="surface_heat_flux",
            category="placeholder",
            available="surface_heat_flux" in payload,
            trainable=False,
            note="Placeholder / approximate output only.",
        ),
        "surface_wall_shear": status(
            name="surface_wall_shear",
            category="placeholder",
            available="surface_wall_shear" in payload,
            trainable=False,
            note="Derived wall shear proxy only.",
        ),
    }

    dimensionality = "3d" if np.asarray(payload["query_points"]).shape[-1] == 3 else "2d"
    return DatasetCapability(
        dataset_name=dataset_name,
        dimensionality=dimensionality,
        target_capabilities=target_capabilities,
    )


def build_task_request_from_loss_config(config: LossConfig) -> TaskRequest:
    return TaskRequest(
        field=config.field_weight > 0.0,
        scalar=config.scalar_weight > 0.0,
        surface=(config.surface_weight > 0.0) or config.use_surface_pressure_loss,
        slice=config.use_slice_loss or config.slice_weight > 0.0,
        feature=config.use_feature_loss or config.feature_weight > 0.0,
        consistency=config.use_physics or config.boundary_weight > 0.0,
    )


def resolve_effective_tasks(capability: DatasetCapability, request: TaskRequest) -> EffectiveTaskSet:
    surface_available = capability.available("surface_pressure") or capability.available("surface_cp")
    feature_available = capability.available("pressure_gradient_indicator") or capability.available("high_gradient_mask")
    return EffectiveTaskSet(
        field=request.field and capability.available("field_targets"),
        scalar=request.scalar and capability.available("scalar_targets"),
        surface=request.surface and surface_available,
        slice=request.slice and capability.available("slice_fields"),
        feature=request.feature and feature_available,
        consistency=request.consistency and capability.available("field_targets"),
    )
