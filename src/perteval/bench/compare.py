from __future__ import annotations

from collections import defaultdict

import numpy as np
import polars as pl

from perteval.bench.result import EvalResult


class Compare:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    @classmethod
    def from_results(cls, results: dict[str, dict[str, EvalResult]]) -> Compare:
        rows = []
        for bench_name, model_results in results.items():
            for model_name, eval_result in model_results.items():
                row = {"benchmark": bench_name, "model": model_name}
                mean_row = eval_result.aggregated.filter(pl.col("statistic") == "mean")
                if mean_row.height > 0:
                    for col in mean_row.columns:
                        if col != "statistic":
                            row[col] = mean_row[col][0]
                rows.append(row)
        return cls(rows)

    def summary(self) -> pl.DataFrame:
        return pl.DataFrame(self._rows)

    @classmethod
    def evaluate_many(cls, all_results: list[dict[str, dict[str, EvalResult]]]) -> Compare:
        accum: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for results in all_results:
            for bench_name, model_results in results.items():
                for model_name, eval_result in model_results.items():
                    key = (bench_name, model_name)
                    mean_row = eval_result.aggregated.filter(pl.col("statistic") == "mean")
                    if mean_row.height > 0:
                        for col in mean_row.columns:
                            if col != "statistic":
                                accum[key][col].append(float(mean_row[col][0]))
        rows = []
        for (bench, model), metrics in accum.items():
            row: dict[str, object] = {"benchmark": bench, "model": model}
            for metric_name, values in metrics.items():
                arr = np.array(values)
                row[f"{metric_name}_mean"] = float(arr.mean())
                row[f"{metric_name}_std"] = float(arr.std())
            rows.append(row)
        return cls(rows)
