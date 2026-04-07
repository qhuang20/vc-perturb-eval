"""Differential expression metrics comparing predicted vs ground-truth DE gene rankings."""

from __future__ import annotations

import numpy as np


def overlap_at_k(pred_genes: np.ndarray, truth_genes: np.ndarray, k: int = 20) -> float:
    """Fraction of top-k predicted DE genes that appear in top-k ground-truth DE genes."""
    pred_top = set(pred_genes[:k])
    truth_top = set(truth_genes[:k])
    if k == 0:
        return 0.0
    return len(pred_top & truth_top) / k
