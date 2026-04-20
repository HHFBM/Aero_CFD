"""Helpers for constructing hard-region masks used in loss weighting and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _valid_points_np(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    if values.size == 0:
        return values.reshape(0, 2)
    return values


def _geometry_reference_np(surface_points: np.ndarray) -> tuple[float, float, float, float]:
    points = _valid_points_np(surface_points)
    if points.shape[0] == 0:
        return 0.0, 1.0, 1.0, 0.0
    x = points[:, 0]
    y = points[:, 1]
    le_x = float(np.min(x))
    te_x = float(np.max(x))
    chord = max(te_x - le_x, 1.0e-6)
    center_y = float(np.mean(y))
    return le_x, te_x, chord, center_y


def query_region_masks_np(
    query_points: np.ndarray,
    surface_points: np.ndarray,
    *,
    high_gradient_mask: np.ndarray | None = None,
    near_wall_distance_fraction: float = 0.08,
    wake_halfwidth_fraction: float = 0.15,
) -> dict[str, np.ndarray]:
    query = _valid_points_np(query_points)
    surface = _valid_points_np(surface_points)
    num_points = query.shape[0]
    le_x, te_x, chord, center_y = _geometry_reference_np(surface)

    if surface.shape[0] == 0 or num_points == 0:
        near_wall = np.zeros((num_points,), dtype=bool)
    else:
        distances = np.linalg.norm(query[:, None, :] - surface[None, :, :], axis=-1)
        near_wall = np.min(distances, axis=1) <= near_wall_distance_fraction * chord

    wake = (query[:, 0] >= te_x) & (np.abs(query[:, 1] - center_y) <= wake_halfwidth_fraction * chord)
    high_gradient = (
        np.asarray(high_gradient_mask, dtype=np.float32).reshape(-1) >= 0.5
        if high_gradient_mask is not None
        else np.zeros((num_points,), dtype=bool)
    )
    return {
        "high_gradient": high_gradient.astype(bool),
        "near_wall": near_wall.astype(bool),
        "wake": wake.astype(bool),
        "leading_edge": (query[:, 0] <= le_x + 0.15 * chord).astype(bool),
    }


def surface_region_masks_np(
    surface_points: np.ndarray,
    *,
    leading_edge_fraction: float = 0.15,
) -> dict[str, np.ndarray]:
    surface = _valid_points_np(surface_points)
    num_points = surface.shape[0]
    le_x, _, chord, _ = _geometry_reference_np(surface)
    leading_edge = surface[:, 0] <= le_x + leading_edge_fraction * chord if num_points > 0 else np.zeros((0,), dtype=bool)
    return {"leading_edge": np.asarray(leading_edge, dtype=bool)}


def _valid_surface_points_torch(surface_points: torch.Tensor, surface_mask: torch.Tensor | None) -> list[torch.Tensor]:
    valid: list[torch.Tensor] = []
    for batch_index in range(surface_points.shape[0]):
        points = surface_points[batch_index]
        if surface_mask is None:
            valid.append(points)
            continue
        mask = surface_mask[batch_index] > 0.5
        valid.append(points[mask])
    return valid


def query_region_masks_torch(
    query_points: torch.Tensor,
    surface_points: torch.Tensor,
    *,
    surface_mask: torch.Tensor | None = None,
    high_gradient_mask: torch.Tensor | None = None,
    near_wall_distance_fraction: float = 0.08,
    wake_halfwidth_fraction: float = 0.15,
) -> dict[str, torch.Tensor]:
    batch_size, num_points, _ = query_points.shape
    device = query_points.device
    high_gradient = torch.zeros((batch_size, num_points), device=device, dtype=query_points.dtype)
    near_wall = torch.zeros((batch_size, num_points), device=device, dtype=query_points.dtype)
    wake = torch.zeros((batch_size, num_points), device=device, dtype=query_points.dtype)
    leading_edge = torch.zeros((batch_size, num_points), device=device, dtype=query_points.dtype)

    if high_gradient_mask is not None:
        high_gradient = (high_gradient_mask.squeeze(-1) >= 0.5).to(dtype=query_points.dtype)

    valid_surface_points = _valid_surface_points_torch(surface_points, surface_mask)
    for batch_index in range(batch_size):
        valid_surface = valid_surface_points[batch_index]
        query = query_points[batch_index]
        if valid_surface.shape[0] == 0:
            continue
        le_x = torch.min(valid_surface[:, 0])
        te_x = torch.max(valid_surface[:, 0])
        chord = torch.clamp(te_x - le_x, min=1.0e-6)
        center_y = torch.mean(valid_surface[:, 1])
        distances = torch.cdist(query.unsqueeze(0), valid_surface.unsqueeze(0)).squeeze(0)
        near_wall[batch_index] = (distances.min(dim=1).values <= near_wall_distance_fraction * chord).to(query_points.dtype)
        wake[batch_index] = (
            (query[:, 0] >= te_x) & (torch.abs(query[:, 1] - center_y) <= wake_halfwidth_fraction * chord)
        ).to(query_points.dtype)
        leading_edge[batch_index] = (query[:, 0] <= le_x + 0.15 * chord).to(query_points.dtype)
    return {
        "high_gradient": high_gradient,
        "near_wall": near_wall,
        "wake": wake,
        "leading_edge": leading_edge,
    }


def surface_region_masks_torch(
    surface_points: torch.Tensor,
    *,
    surface_mask: torch.Tensor | None = None,
    leading_edge_fraction: float = 0.15,
) -> dict[str, torch.Tensor]:
    batch_size, num_points, _ = surface_points.shape
    output = torch.zeros((batch_size, num_points), device=surface_points.device, dtype=surface_points.dtype)
    valid_surface_points = _valid_surface_points_torch(surface_points, surface_mask)
    for batch_index in range(batch_size):
        valid_surface = valid_surface_points[batch_index]
        if valid_surface.shape[0] == 0:
            continue
        le_x = torch.min(valid_surface[:, 0])
        te_x = torch.max(valid_surface[:, 0])
        chord = torch.clamp(te_x - le_x, min=1.0e-6)
        if surface_mask is None:
            output[batch_index] = (surface_points[batch_index, :, 0] <= le_x + leading_edge_fraction * chord).to(surface_points.dtype)
        else:
            mask = surface_mask[batch_index] > 0.5
            output[batch_index, mask] = (
                valid_surface[:, 0] <= le_x + leading_edge_fraction * chord
            ).to(surface_points.dtype)
    return {"leading_edge": output}


def summarize_region_mask(mask: Any) -> float:
    values = np.asarray(mask, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.mean(values > 0.5))
