import pytest


def test_metric_registry_has_builtins():
    from perteval.metrics.registry import metric_registry
    available = metric_registry.list_available()
    assert "pearson_delta" in available
    assert "mse" in available
    assert "mae" in available
    assert "overlap_at_k" in available
    assert "edistance" in available


def test_metric_registry_get_returns_metric_info():
    from perteval.metrics.base import BestValue, MetricType
    from perteval.metrics.registry import metric_registry
    info = metric_registry.get("pearson_delta")
    assert info.metric_type == MetricType.EXPRESSION
    assert info.best_value == BestValue.ONE
    assert callable(info.func)


def test_metric_registry_lazy_resolution():
    from perteval.metrics.base import BestValue, MetricInfo, MetricType
    from perteval.metrics.registry import MetricRegistry

    fresh_registry = MetricRegistry("metric_test")
    fresh_registry.register_metric(
        "pearson_delta",
        "perteval.metrics.functional.expression:pearson_delta",
        MetricType.EXPRESSION,
        BestValue.ONE,
        "Pearson correlation of mean expression shift",
    )
    raw = fresh_registry._entries["pearson_delta"]
    assert isinstance(raw, MetricInfo)
    assert isinstance(raw.func, str)
    info = fresh_registry.get("pearson_delta")
    assert callable(info.func)


def test_metric_registry_compute():
    import numpy as np
    from perteval.metrics.registry import metric_registry
    info = metric_registry.get("mse")
    pred = np.array([1.0, 2.0, 3.0])
    truth = np.array([1.0, 2.0, 3.0])
    result = info.func(pred, truth)
    assert result == pytest.approx(0.0)
