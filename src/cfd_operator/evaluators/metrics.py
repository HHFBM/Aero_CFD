"""Evaluation metrics.

The helpers in this module intentionally group metrics by task family so training,
evaluation, report export and downstream tooling can use a consistent naming scheme.
"""

from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def relative_error(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0e-8) -> float:
    numerator = np.linalg.norm(y_true - y_pred)
    denominator = np.linalg.norm(y_true) + eps
    return float(numerator / denominator)


def binary_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    true_label = np.asarray(y_true).reshape(-1) >= threshold
    pred_label = np.asarray(y_pred).reshape(-1) >= threshold
    return float(np.mean(true_label == pred_label))


def binary_f1(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, eps: float = 1.0e-8) -> float:
    true_label = np.asarray(y_true).reshape(-1) >= threshold
    pred_label = np.asarray(y_pred).reshape(-1) >= threshold
    tp = np.sum(true_label & pred_label)
    fp = np.sum(~true_label & pred_label)
    fn = np.sum(true_label & ~pred_label)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return float(2.0 * precision * recall / (precision + recall + eps))


def binary_iou(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, eps: float = 1.0e-8) -> float:
    true_label = np.asarray(y_true).reshape(-1) >= threshold
    pred_label = np.asarray(y_pred).reshape(-1) >= threshold
    intersection = np.sum(true_label & pred_label)
    union = np.sum(true_label | pred_label)
    return float(intersection / (union + eps))


def compute_field_metrics(y_true: np.ndarray, y_pred: np.ndarray, aux_name: str) -> dict[str, float]:
    return {
        "field_mse": mse(y_true, y_pred),
        "field_rmse": rmse(y_true, y_pred),
        "field_relative_error": relative_error(y_true, y_pred),
        f"{aux_name}_rmse": rmse(y_true[:, 3], y_pred[:, 3]),
    }


def compute_scalar_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scalar_names: tuple[str, str] = ("cl", "cd"),
) -> dict[str, float]:
    return {
        f"{scalar_names[0]}_mae": mae(y_true[:, 0], y_pred[:, 0]),
        f"{scalar_names[0]}_relative_error": relative_error(y_true[:, 0], y_pred[:, 0]),
        f"{scalar_names[1]}_mae": mae(y_true[:, 1], y_pred[:, 1]),
        f"{scalar_names[1]}_relative_error": relative_error(y_true[:, 1], y_pred[:, 1]),
    }


def compute_surface_metrics(
    cp_true: np.ndarray,
    cp_pred: np.ndarray,
    pressure_true: np.ndarray,
    pressure_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "cp_surface_rmse": rmse(cp_true, cp_pred),
        "cp_surface_mae": mae(cp_true, cp_pred),
        "pressure_surface_rmse": rmse(pressure_true, pressure_pred),
    }


def compute_slice_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "slice_rmse": rmse(y_true, y_pred),
        "slice_relative_error": relative_error(y_true, y_pred),
    }


def compute_feature_metrics(
    pressure_gradient_true: np.ndarray,
    pressure_gradient_pred: np.ndarray,
    high_true: np.ndarray,
    high_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "pressure_gradient_indicator_accuracy": binary_accuracy(pressure_gradient_true, pressure_gradient_pred),
        "pressure_gradient_indicator_f1": binary_f1(pressure_gradient_true, pressure_gradient_pred),
        "pressure_gradient_indicator_iou": binary_iou(pressure_gradient_true, pressure_gradient_pred),
        "high_gradient_accuracy": binary_accuracy(high_true, high_pred),
        "high_gradient_iou": binary_iou(high_true, high_pred),
        "pressure_gradient_pred_fraction": float(np.mean(pressure_gradient_pred >= 0.5)),
        "high_gradient_pred_fraction": float(np.mean(high_pred >= 0.5)),
    }


def compute_masked_region_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float]:
    region_mask = np.asarray(mask).reshape(-1) >= 0.5
    if not np.any(region_mask):
        return {}
    true_values = np.asarray(y_true)[region_mask]
    pred_values = np.asarray(y_pred)[region_mask]
    return {
        f"{prefix}_rmse": rmse(true_values, pred_values),
        f"{prefix}_relative_error": relative_error(true_values, pred_values),
        f"{prefix}_fraction": float(np.mean(region_mask)),
    }
