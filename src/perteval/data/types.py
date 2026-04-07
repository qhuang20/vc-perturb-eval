"""Core data types for perteval inter-layer communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData


@dataclass(frozen=True)
class PerturbationData:
    """Validated, immutable container for a predicted/ground-truth AnnData pair."""

    predicted: AnnData
    ground_truth: AnnData
    perturbation_key: str = "perturbation"
    control_key: str = "control"
    gene_names: np.ndarray = field(init=False, repr=False)
    perturbation_labels: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        pred, gt = self.predicted, self.ground_truth
        if pred.n_vars != gt.n_vars:
            raise ValueError(
                f"Gene count mismatch: predicted has {pred.n_vars}, ground_truth has {gt.n_vars}"
            )
        pred_genes = pred.var_names.values
        gt_genes = gt.var_names.values
        if not np.array_equal(pred_genes, gt_genes):
            raise ValueError(
                "Gene names do not match between predicted and ground_truth. "
                f"First mismatch: predicted has '{pred_genes[pred_genes != gt_genes][0]}', "
                f"ground_truth has '{gt_genes[pred_genes != gt_genes][0]}'"
            )
        if self.perturbation_key not in gt.obs.columns:
            raise ValueError(
                f"perturbation_key '{self.perturbation_key}' not found in "
                f"ground_truth.obs columns: {list(gt.obs.columns)}"
            )
        if self.perturbation_key not in pred.obs.columns:
            raise ValueError(
                f"perturbation_key '{self.perturbation_key}' not found in "
                f"predicted.obs columns: {list(pred.obs.columns)}"
            )
        object.__setattr__(self, "gene_names", gt_genes)
        gt_labels = gt.obs[self.perturbation_key].unique().tolist()
        labels = [label for label in gt_labels if label != self.control_key]
        object.__setattr__(self, "perturbation_labels", sorted(labels))
