"""Model registry."""

from __future__ import annotations

from cfd_operator.config.schemas import ModelConfig
from cfd_operator.models.base import BaseOperatorModel
from cfd_operator.models.deeponet import DeepONetModel
from cfd_operator.models.fno import FNOModel, GeoFNOModel


MODEL_REGISTRY = {
    "deeponet": DeepONetModel,
    "fno": FNOModel,
    "geofno": GeoFNOModel,
}


def create_model(config: ModelConfig) -> BaseOperatorModel:
    try:
        model_cls = MODEL_REGISTRY[config.name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown model name: {config.name}") from exc
    return model_cls(config)
