"""Placeholder Fourier operator implementations."""

from __future__ import annotations

from cfd_operator.models.base import BaseOperatorModel


class FNOModel(BaseOperatorModel):
    def forward(self, branch_inputs, query_points):  # type: ignore[override]
        raise NotImplementedError("FNOModel is reserved for future regular-grid support.")


class GeoFNOModel(BaseOperatorModel):
    def forward(self, branch_inputs, query_points):  # type: ignore[override]
        raise NotImplementedError("GeoFNOModel is reserved for future geometry-aware support.")

