"""Global metric registry with lazy-loaded built-in metrics."""

from __future__ import annotations

import importlib

from perteval._registry import Registry
from perteval.metrics.base import BestValue, MetricInfo, MetricType


class MetricRegistry(Registry[MetricInfo]):
    def get(self, name: str) -> MetricInfo:
        if name not in self._entries:
            raise KeyError(f"'{name}' not found in {self._name} registry")
        info = self._entries[name]
        if isinstance(info.func, str):
            module_path, _, attr_name = info.func.partition(":")
            module = importlib.import_module(module_path)
            resolved_func = getattr(module, attr_name)
            self._entries[name] = MetricInfo(
                name=info.name, func=resolved_func, metric_type=info.metric_type,
                best_value=info.best_value, description=info.description,
            )
        return self._entries[name]

    def register_metric(self, name: str, entry: str, metric_type: MetricType,
                        best_value: BestValue, description: str) -> None:
        info = MetricInfo(name=name, func=entry, metric_type=metric_type,
                          best_value=best_value, description=description)
        self.register(name, info)


metric_registry = MetricRegistry("metric")

metric_registry.register_metric("pearson_delta", "perteval.metrics.functional.expression:pearson_delta",
    MetricType.EXPRESSION, BestValue.ONE, "Pearson correlation of mean expression shift")
metric_registry.register_metric("mse", "perteval.metrics.functional.expression:mse",
    MetricType.EXPRESSION, BestValue.ZERO, "Mean squared error of mean expression")
metric_registry.register_metric("mae", "perteval.metrics.functional.expression:mae",
    MetricType.EXPRESSION, BestValue.ZERO, "Mean absolute error of mean expression")
metric_registry.register_metric("overlap_at_k", "perteval.metrics.functional.de:overlap_at_k",
    MetricType.DE, BestValue.ONE, "Fraction of top-k DE genes overlapping between predicted and ground-truth")
metric_registry.register_metric("edistance", "perteval.metrics.functional.distribution:edistance",
    MetricType.DISTRIBUTION, BestValue.ZERO, "Energy distance between predicted and ground-truth cell distributions")
