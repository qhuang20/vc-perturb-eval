"""Core evaluator — computes metrics for a single PerturbationData pair."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from perteval.bench.result import EvalResult
from perteval.metrics.base import MetricType
from perteval.metrics.registry import metric_registry

if TYPE_CHECKING:
    from perteval.data.types import PerturbationData

DEFAULT_METRICS = ["pearson_delta", "mse", "overlap_at_k"]


class Evaluator:
    def evaluate(
        self,
        data: PerturbationData,
        metrics: list[str] | str = "default",
        aggregation: str = "average",
    ) -> EvalResult:
        if metrics == "default":
            metrics = DEFAULT_METRICS

        rows: list[dict[str, object]] = []

        for pert in data.perturbation_labels:
            row: dict[str, object] = {"perturbation": pert}
            pred_mask = data.predicted.obs[data.perturbation_key] == pert
            gt_mask = data.ground_truth.obs[data.perturbation_key] == pert
            pred_cells = np.asarray(data.predicted[pred_mask].X)
            gt_cells = np.asarray(data.ground_truth[gt_mask].X)

            if pred_cells.shape[0] == 0 or gt_cells.shape[0] == 0:
                for metric_name in metrics:
                    row[metric_name] = float("nan")
                rows.append(row)
                continue

            pred_mean = pred_cells.mean(axis=0)
            gt_mean = gt_cells.mean(axis=0)

            for metric_name in metrics:
                info = metric_registry.get(metric_name)
                score = self._compute_metric(info, pred_cells, gt_cells, pred_mean, gt_mean, data)
                row[metric_name] = score
            rows.append(row)

        per_pert = pl.DataFrame(rows)
        metric_cols = [c for c in per_pert.columns if c != "perturbation"]
        agg_rows = []
        for stat_name, stat_fn in [("mean", np.nanmean), ("std", np.nanstd),
                                    ("min", np.nanmin), ("max", np.nanmax)]:
            agg_row: dict[str, object] = {"statistic": stat_name}
            for col in metric_cols:
                agg_row[col] = float(stat_fn(per_pert[col].to_numpy()))
            agg_rows.append(agg_row)
        aggregated = pl.DataFrame(agg_rows)

        config = {
            "metrics": metrics,
            "aggregation": aggregation,
            "n_perturbations": len(data.perturbation_labels),
            "n_genes": len(data.gene_names),
        }
        return EvalResult(per_perturbation=per_pert, aggregated=aggregated, config=config)

    @staticmethod
    def _compute_metric(info, pred_cells, gt_cells, pred_mean, gt_mean, data):
        match info.metric_type:
            case MetricType.EXPRESSION:
                return info.func(pred_mean, gt_mean)
            case MetricType.DE:
                diff = np.abs(pred_mean - gt_mean)
                pred_order = np.argsort(-diff)
                pred_genes = data.gene_names[pred_order]
                gt_diff = np.abs(gt_mean - np.zeros_like(gt_mean))
                gt_order = np.argsort(-gt_diff)
                gt_genes = data.gene_names[gt_order]
                return info.func(pred_genes, gt_genes)
            case MetricType.DISTRIBUTION:
                return info.func(pred_cells, gt_cells)
