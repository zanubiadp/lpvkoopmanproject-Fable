"""Prediction-error metrics matching Korda & Mezic (2020), eq. (55)."""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def relative_rmse_percent(x_pred: Array, x_true: Array) -> float:
    """100 * ||x_pred - x_true||_F / ||x_true||_F over a whole trajectory.

    This is the paper's prediction-error metric (eq. 55): RMSE over time of
    the state vector error, normalized by the RMS of the true trajectory.
    """
    num = float(np.linalg.norm(x_pred - x_true))
    den = float(np.linalg.norm(x_true))
    return 100.0 * num / den if den > 0 else np.inf


def error_vs_time(x_pred: Array, x_true: Array) -> Array:
    """Pointwise state-error norm ||x_pred(t_k) - x_true(t_k)|| over time."""
    return np.linalg.norm(x_pred - x_true, axis=-1)
