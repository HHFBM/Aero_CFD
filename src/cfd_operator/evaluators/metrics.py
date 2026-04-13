"""Evaluation metrics."""

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

