"""Evaluator registry for capability-aware metric dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from cfd_operator.evaluators.metrics import (
    compute_feature_metrics,
    compute_field_metrics,
    compute_scalar_metrics,
    compute_slice_metrics,
    compute_surface_metrics,
)


MetricComputer = Callable[..., dict[str, float]]


@dataclass(frozen=True)
class MetricGroupSpec:
    name: str
    required_targets: tuple[str, ...]
    computer: MetricComputer


class MetricRegistry:
    def __init__(self) -> None:
        self._groups: dict[str, MetricGroupSpec] = {}

    def register(self, spec: MetricGroupSpec) -> None:
        self._groups[spec.name] = spec

    def get(self, name: str) -> MetricGroupSpec:
        return self._groups[name]

    def names(self) -> list[str]:
        return sorted(self._groups.keys())

    def items(self) -> list[tuple[str, MetricGroupSpec]]:
        return sorted(self._groups.items(), key=lambda item: item[0])


def build_default_metric_registry() -> MetricRegistry:
    registry = MetricRegistry()
    registry.register(
        MetricGroupSpec(
            name="field_metrics",
            required_targets=("field_targets",),
            computer=compute_field_metrics,
        )
    )
    registry.register(
        MetricGroupSpec(
            name="scalar_metrics",
            required_targets=("scalar_targets",),
            computer=compute_scalar_metrics,
        )
    )
    registry.register(
        MetricGroupSpec(
            name="surface_metrics",
            required_targets=("surface_pressure", "surface_cp"),
            computer=compute_surface_metrics,
        )
    )
    registry.register(
        MetricGroupSpec(
            name="slice_metrics",
            required_targets=("slice_fields",),
            computer=compute_slice_metrics,
        )
    )
    registry.register(
        MetricGroupSpec(
            name="feature_metrics",
            required_targets=("pressure_gradient_indicator", "high_gradient_mask"),
            computer=compute_feature_metrics,
        )
    )
    return registry
