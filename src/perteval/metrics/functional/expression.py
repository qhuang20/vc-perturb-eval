"""Expression-level metrics comparing predicted vs ground-truth mean expression."""

from __future__ import annotations

import numpy as np
from scipy import stats


def pearson_delta(pred: np.ndarray, truth: np.ndarray) -> float:
    """Pearson correlation between predicted and ground-truth expression vectors.
    Returns 0.0 if either array has zero variance."""
    if pred.std() == 0 or truth.std() == 0:
        return 0.0
    r, _ = stats.pearsonr(pred, truth)
    return float(r)


def mse(pred: np.ndarray, truth: np.ndarray) -> float:
    """Mean squared error between expression vectors."""
    return float(np.mean((pred - truth) ** 2))


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    """Mean absolute error between expression vectors."""
    return float(np.mean(np.abs(pred - truth)))
