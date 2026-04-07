import json

import polars as pl


def test_eval_result_to_json(tmp_path):
    from perteval.bench.result import EvalResult

    per_pert = pl.DataFrame(
        {
            "perturbation": ["pertA", "pertB"],
            "pearson_delta": [0.85, 0.90],
            "mse": [0.12, 0.08],
        }
    )
    agg = pl.DataFrame(
        {
            "statistic": ["mean", "std"],
            "pearson_delta": [0.875, 0.035],
            "mse": [0.10, 0.028],
        }
    )
    config = {"metrics": ["pearson_delta", "mse"], "aggregation": "average"}
    result = EvalResult(per_perturbation=per_pert, aggregated=agg, config=config)
    out_path = tmp_path / "result.json"
    result.to_json(str(out_path))
    data = json.loads(out_path.read_text())
    assert "config" in data
    assert "per_perturbation" in data
    assert "aggregated" in data
    assert data["config"]["metrics"] == ["pearson_delta", "mse"]


def test_eval_result_to_csv(tmp_path):
    from perteval.bench.result import EvalResult

    per_pert = pl.DataFrame({"perturbation": ["pertA"], "mse": [0.1]})
    agg = pl.DataFrame({"statistic": ["mean"], "mse": [0.1]})
    result = EvalResult(per_perturbation=per_pert, aggregated=agg, config={})
    result.to_csv(str(tmp_path / "out"))
    assert (tmp_path / "out_per_perturbation.csv").exists()
    assert (tmp_path / "out_aggregated.csv").exists()
    loaded = pl.read_csv(tmp_path / "out_per_perturbation.csv")
    assert loaded.shape == (1, 2)
