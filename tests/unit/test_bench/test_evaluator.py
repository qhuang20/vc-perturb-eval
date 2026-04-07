import numpy as np

from tests.conftest import build_random_anndata


def test_evaluator_returns_eval_result():
    from perteval.bench.evaluator import Evaluator
    from perteval.bench.result import EvalResult
    from perteval.data.types import PerturbationData

    rng = np.random.default_rng(42)
    gt = build_random_anndata(rng=rng)
    pred = gt.copy()
    pred.X = gt.X + rng.standard_normal(gt.X.shape).astype(np.float32) * 0.1
    data = PerturbationData(predicted=pred, ground_truth=gt)
    evaluator = Evaluator()
    result = evaluator.evaluate(data, metrics=["pearson_delta", "mse"])
    assert isinstance(result, EvalResult)
    assert "perturbation" in result.per_perturbation.columns
    assert "pearson_delta" in result.per_perturbation.columns
    assert "mse" in result.per_perturbation.columns
    assert result.per_perturbation.height > 0


def test_evaluator_aggregated_has_stats():
    from perteval.bench.evaluator import Evaluator
    from perteval.data.types import PerturbationData

    rng = np.random.default_rng(42)
    gt = build_random_anndata(perturbations=["p1", "p2", "p3"], rng=rng)
    pred = gt.copy()
    pred.X = gt.X + 0.05
    data = PerturbationData(predicted=pred, ground_truth=gt)
    evaluator = Evaluator()
    result = evaluator.evaluate(data, metrics=["mse"])
    stats = result.aggregated["statistic"].to_list()
    assert "mean" in stats
    assert "std" in stats


def test_evaluator_default_metrics():
    from perteval.bench.evaluator import Evaluator
    from perteval.data.types import PerturbationData

    rng = np.random.default_rng(42)
    gt = build_random_anndata(rng=rng)
    pred = gt.copy()
    pred.X = gt.X + 0.1
    data = PerturbationData(predicted=pred, ground_truth=gt)
    evaluator = Evaluator()
    result = evaluator.evaluate(data, metrics="default")
    columns = result.per_perturbation.columns
    assert "pearson_delta" in columns
    assert "mse" in columns
