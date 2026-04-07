from __future__ import annotations

import numpy as np
from perteval.bench.evaluator import Evaluator
from perteval.bench.result import EvalResult
from perteval.bench.task_manager import TaskManager
from perteval.data.accessors.local import LocalAccessor
from perteval.data.splitter import Splitter
from perteval.data.types import PerturbationData
from perteval.models.registry import model_registry


class BenchmarkRunner:
    def __init__(
        self,
        benchmarks: list[str],
        models: list[str],
        benchmarks_dir: str = "benchmarks",
        data_dir: str = "data",
    ) -> None:
        self.benchmarks = benchmarks
        self.models = models
        self._task_manager = TaskManager(benchmarks_dir)
        self._data_accessor = LocalAccessor(data_dir)
        self._evaluator = Evaluator()

    def _make_model_keys(self) -> list[str]:
        """Return result keys for each model entry, appending _0/_1/... when names repeat."""
        from collections import Counter

        counts: Counter[str] = Counter(self.models)
        # Only disambiguate names that appear more than once
        seen: dict[str, int] = {}
        keys: list[str] = []
        for name in self.models:
            if counts[name] > 1:
                idx = seen.get(name, 0)
                keys.append(f"{name}_{idx}")
                seen[name] = idx + 1
            else:
                keys.append(name)
        return keys

    def run(self) -> dict[str, dict[str, EvalResult]]:
        results: dict[str, dict[str, EvalResult]] = {}
        model_keys = self._make_model_keys()

        for bench_name in self.benchmarks:
            config = self._task_manager.get(bench_name)
            adata = self._data_accessor.load(config.dataset)
            splits = Splitter.split(
                adata,
                method=config.split_method,
                frac=config.split_frac,
                holdout_key=config.holdout_key,
                seed=config.split_seed,
            )
            results[bench_name] = {}

            for model_name, result_key in zip(self.models, model_keys):
                model_cls = model_registry.get(model_name)
                model = model_cls() if isinstance(model_cls, type) else model_cls

                model.train(
                    splits["train"],
                    perturbation_key="perturbation",
                    control_key="control",
                )

                test_adata = splits["test"]
                control_mask = test_adata.obs["perturbation"] == "control"
                control_cells = test_adata[control_mask]
                test_perts = [
                    p
                    for p in test_adata.obs["perturbation"].unique()
                    if p != "control"
                ]

                if len(test_perts) == 0:
                    continue

                predicted = model.predict(control_cells, perturbations=test_perts)

                gt_mask = test_adata.obs["perturbation"].isin(test_perts)
                ground_truth = test_adata[gt_mask].copy()

                data = PerturbationData(predicted=predicted, ground_truth=ground_truth)
                eval_result = self._evaluator.evaluate(
                    data,
                    metrics=config.metrics,
                    aggregation=config.aggregation,
                )
                eval_result.config["benchmark"] = bench_name
                eval_result.config["model"] = model_name
                results[bench_name][result_key] = eval_result

        return results
