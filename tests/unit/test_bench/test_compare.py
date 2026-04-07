# tests/unit/test_bench/test_compare.py
import polars as pl

from perteval.bench.result import EvalResult


def _make_result(pearson: float, mse_val: float, model: str, bench: str) -> EvalResult:
    per_pert = pl.DataFrame(
        {
            "perturbation": ["pertA", "pertB"],
            "pearson_delta": [pearson, pearson + 0.02],
            "mse": [mse_val, mse_val - 0.01],
        }
    )
    agg = pl.DataFrame(
        {
            "statistic": ["mean"],
            "pearson_delta": [pearson + 0.01],
            "mse": [mse_val - 0.005],
        }
    )
    return EvalResult(
        per_perturbation=per_pert,
        aggregated=agg,
        config={"model": model, "benchmark": bench, "metrics": ["pearson_delta", "mse"]},
    )


def test_compare_summary():
    from perteval.bench.compare import Compare

    results = {
        "norman19": {
            "mean_control": _make_result(0.5, 0.3, "mean_control", "norman19"),
            "cpa": _make_result(0.8, 0.1, "cpa", "norman19"),
        },
    }
    comparison = Compare.from_results(results)
    summary = comparison.summary()
    assert isinstance(summary, pl.DataFrame)
    assert "model" in summary.columns
    assert "benchmark" in summary.columns
    assert summary.height == 2


def test_compare_evaluate_many_aggregates():
    from perteval.bench.compare import Compare

    all_results = []
    for seed in range(3):
        r = {
            "bench": {
                "model_a": _make_result(0.5 + seed * 0.01, 0.3 - seed * 0.01, "model_a", "bench"),
            }
        }
        all_results.append(r)
    robust = Compare.evaluate_many(all_results)
    summary = robust.summary()
    assert isinstance(summary, pl.DataFrame)
    cols = summary.columns
    assert any("mean" in c or "pearson_delta" in c for c in cols)
