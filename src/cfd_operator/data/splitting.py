"""Dataset split helpers with generalization-focused holdouts."""

from __future__ import annotations

from typing import Dict

import numpy as np


def condition_holdout_mask(
    primary_values: np.ndarray,
    secondary_values: np.ndarray,
    ratio: float,
) -> np.ndarray:
    if ratio <= 0.0:
        return np.zeros_like(primary_values, dtype=bool)
    primary_threshold = np.quantile(primary_values, 1.0 - ratio)
    secondary_threshold = np.quantile(secondary_values, 1.0 - ratio)
    return (primary_values >= primary_threshold) | (secondary_values >= secondary_threshold)


def split_seen_pool(
    indices: np.ndarray,
    geometry_ids: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    total = indices.shape[0]
    if total < 3:
        raise ValueError("Seen-geometry pool is too small to form train/val/test splits.")

    mandatory_train = []
    remaining = []
    for geometry_id in np.unique(geometry_ids[indices]):
        geometry_indices = indices[geometry_ids[indices] == geometry_id].copy()
        rng.shuffle(geometry_indices)
        mandatory_train.append(int(geometry_indices[0]))
        if geometry_indices.shape[0] > 1:
            remaining.extend(int(value) for value in geometry_indices[1:])

    mandatory_train = np.asarray(mandatory_train, dtype=np.int64)
    remaining = np.asarray(remaining, dtype=np.int64)
    rng.shuffle(remaining)

    desired_train_count = max(mandatory_train.shape[0], int(round(total * train_ratio)))
    extra_train_count = max(0, desired_train_count - mandatory_train.shape[0])
    max_extra_train = max(0, remaining.shape[0] - 2)
    extra_train_count = min(extra_train_count, max_extra_train)
    train_indices = np.sort(np.concatenate([mandatory_train, remaining[:extra_train_count]]))

    residual = remaining[extra_train_count:]
    if residual.shape[0] < 2:
        raise ValueError("Unable to allocate a non-empty IID test split.")
    desired_val_count = max(1, int(round(total * val_ratio)))
    val_count = min(desired_val_count, residual.shape[0] - 1)
    val_indices = np.sort(residual[:val_count])
    test_indices = np.sort(residual[val_count:])

    return {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }


def build_generalization_splits(
    geometry_ids: np.ndarray,
    primary_condition_values: np.ndarray,
    secondary_condition_values: np.ndarray,
    unseen_geometry_ratio: float,
    unseen_condition_ratio: float,
    train_ratio: float,
    val_ratio: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    unique_geometry_ids = np.unique(geometry_ids)
    num_unseen_geometries = max(1, int(round(unique_geometry_ids.shape[0] * unseen_geometry_ratio)))
    unseen_geometry_ids = rng.choice(unique_geometry_ids, size=num_unseen_geometries, replace=False)
    unseen_geometry_mask = np.isin(geometry_ids, unseen_geometry_ids)

    unseen_condition_mask = np.zeros(geometry_ids.shape[0], dtype=bool)
    for geometry_id in unique_geometry_ids:
        if geometry_id in unseen_geometry_ids:
            continue
        geometry_indices = np.where(geometry_ids == geometry_id)[0]
        holdout_candidates = condition_holdout_mask(
            primary_values=primary_condition_values[geometry_indices],
            secondary_values=secondary_condition_values[geometry_indices],
            ratio=unseen_condition_ratio,
        )
        selected = geometry_indices[holdout_candidates]
        if selected.size == 0 and geometry_indices.size > 1 and unseen_condition_ratio > 0.0:
            scores = primary_condition_values[geometry_indices] + 0.1 * secondary_condition_values[geometry_indices]
            selected = geometry_indices[np.argsort(scores)[-1:]]
        if selected.size >= geometry_indices.size:
            scores = primary_condition_values[geometry_indices] + 0.1 * secondary_condition_values[geometry_indices]
            selected = geometry_indices[np.argsort(scores)[-max(1, geometry_indices.size - 1):]]
        unseen_condition_mask[selected] = True

    train_pool = np.where(~unseen_geometry_mask & ~unseen_condition_mask)[0]
    rng.shuffle(train_pool)
    split_payload = split_seen_pool(
        indices=train_pool,
        geometry_ids=geometry_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        rng=rng,
    )

    return {
        **split_payload,
        "test_unseen_geometry_indices": np.sort(np.where(unseen_geometry_mask)[0]),
        "test_unseen_condition_indices": np.sort(np.where(unseen_condition_mask & ~unseen_geometry_mask)[0]),
    }
