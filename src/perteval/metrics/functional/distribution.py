"""Distribution-level metrics comparing cell populations."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def edistance(X: np.ndarray, Y: np.ndarray) -> float:
    """Energy distance between two multivariate samples.
    E-distance = 2 * mean(||X_i - Y_j||) - mean(||X_i - X_j||) - mean(||Y_i - Y_j||)"""
    XY = cdist(X, Y, metric="euclidean")
    XX = cdist(X, X, metric="euclidean")
    YY = cdist(Y, Y, metric="euclidean")
    return float(2.0 * XY.mean() - XX.mean() - YY.mean())
