"""Future-native geometry backbone interfaces.

This module does not replace the current fixed-dimension branch vector path.
It reserves a clean interface for future variable-length geometry encoders so
checkpoints and inference artifacts can describe that intent without implying
that the training backbone has already switched away from branch_inputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from cfd_operator.config.schemas import ModelConfig


GeometryBackboneMode = Literal["fixed_branch_vector", "native_geometry_latent_reserved"]
GeometryBackboneType = Literal["none", "reserved_interface"]


@dataclass(frozen=True)
class GeometryBackboneContract:
    mode: GeometryBackboneMode
    backbone_type: GeometryBackboneType
    active_in_main_path: bool
    supports_variable_length: bool
    emits_global_latent: bool
    emits_tokens: bool
    latent_dim: int
    token_dim: int
    max_tokens: int
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "backbone_type": self.backbone_type,
            "active_in_main_path": bool(self.active_in_main_path),
            "supports_variable_length": bool(self.supports_variable_length),
            "emits_global_latent": bool(self.emits_global_latent),
            "emits_tokens": bool(self.emits_tokens),
            "latent_dim": int(self.latent_dim),
            "token_dim": int(self.token_dim),
            "max_tokens": int(self.max_tokens),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "GeometryBackboneContract":
        return cls(
            mode=str(payload.get("mode", "fixed_branch_vector")),  # type: ignore[arg-type]
            backbone_type=str(payload.get("backbone_type", "none")),  # type: ignore[arg-type]
            active_in_main_path=bool(payload.get("active_in_main_path", False)),
            supports_variable_length=bool(payload.get("supports_variable_length", False)),
            emits_global_latent=bool(payload.get("emits_global_latent", True)),
            emits_tokens=bool(payload.get("emits_tokens", False)),
            latent_dim=int(payload.get("latent_dim", 0)),
            token_dim=int(payload.get("token_dim", 0)),
            max_tokens=int(payload.get("max_tokens", 0)),
            note=str(payload.get("note", "")),
        )


@dataclass
class GeometryBackboneOutput:
    global_latent: torch.Tensor
    tokens: torch.Tensor | None = None
    token_mask: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None


class BaseGeometryBackbone(nn.Module, ABC):
    """Abstract interface for future native variable-length geometry encoders."""

    def __init__(self, contract: GeometryBackboneContract) -> None:
        super().__init__()
        self.contract = contract

    @abstractmethod
    def encode(
        self,
        surface_points: torch.Tensor,
        surface_mask: torch.Tensor | None = None,
    ) -> GeometryBackboneOutput:
        """Encode raw geometry into a global latent and optional tokens."""

    def backbone_metadata(self) -> dict[str, object]:
        return self.contract.as_dict()


class ReservedNativeGeometryBackbone(BaseGeometryBackbone):
    """Minimal experimental encoder reserved for future native geometry paths.

    This module is intentionally lightweight and not wired into the default
    training path. It only demonstrates the future interface shape:
    raw variable-length surface points -> global latent + optional tokens.
    """

    def __init__(self, contract: GeometryBackboneContract) -> None:
        super().__init__(contract=contract)
        self.token_proj = nn.Linear(2, contract.token_dim)
        self.global_proj = nn.Sequential(
            nn.Linear(4, contract.latent_dim),
            nn.GELU(),
            nn.Linear(contract.latent_dim, contract.latent_dim),
        )

    def encode(
        self,
        surface_points: torch.Tensor,
        surface_mask: torch.Tensor | None = None,
    ) -> GeometryBackboneOutput:
        if surface_points.ndim != 3 or surface_points.shape[-1] != 2:
            raise ValueError("surface_points must have shape [B, N, 2] for ReservedNativeGeometryBackbone.encode().")
        if surface_mask is None:
            surface_mask = torch.ones(
                surface_points.shape[:2],
                device=surface_points.device,
                dtype=torch.bool,
            )
        mask_f = surface_mask.to(dtype=surface_points.dtype).unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        masked_points = surface_points * mask_f
        pooled_mean = masked_points.sum(dim=1) / denom
        fill_value = torch.finfo(surface_points.dtype).min
        masked_for_max = surface_points.masked_fill(~surface_mask.unsqueeze(-1), fill_value)
        pooled_max = masked_for_max.amax(dim=1)
        global_latent = self.global_proj(torch.cat([pooled_mean, pooled_max], dim=-1))

        max_tokens = min(surface_points.shape[1], self.contract.max_tokens)
        token_points = surface_points[:, :max_tokens]
        token_mask = surface_mask[:, :max_tokens]
        tokens = self.token_proj(token_points)
        return GeometryBackboneOutput(
            global_latent=global_latent,
            tokens=tokens,
            token_mask=token_mask,
            metadata={
                "status": "experimental_reserved_interface",
                "note": self.contract.note,
            },
        )


def build_geometry_backbone_contract(config: ModelConfig) -> GeometryBackboneContract:
    if config.geometry_backbone_mode == "native_geometry_latent_reserved":
        return GeometryBackboneContract(
            mode="native_geometry_latent_reserved",
            backbone_type="reserved_interface",
            active_in_main_path=False,
            supports_variable_length=True,
            emits_global_latent=True,
            emits_tokens=True,
            latent_dim=config.native_geometry_latent_dim,
            token_dim=config.native_geometry_token_dim,
            max_tokens=config.native_geometry_max_tokens,
            note=(
                "Reserved interface only. The main training/inference path still uses fixed branch-compatible vectors; "
                "native variable-length geometry conditioning is not active yet."
            ),
        )
    return GeometryBackboneContract(
        mode="fixed_branch_vector",
        backbone_type="none",
        active_in_main_path=True,
        supports_variable_length=False,
        emits_global_latent=False,
        emits_tokens=False,
        latent_dim=0,
        token_dim=0,
        max_tokens=0,
        note="Stable default path: geometry is conditioned through fixed-dimension branch-compatible vectors.",
    )


def create_reserved_geometry_backbone(config: ModelConfig) -> BaseGeometryBackbone | None:
    contract = build_geometry_backbone_contract(config)
    if contract.mode != "native_geometry_latent_reserved":
        return None
    return ReservedNativeGeometryBackbone(contract)
