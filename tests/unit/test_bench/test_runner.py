import numpy as np
import pytest
from tests.conftest import build_random_anndata


def test_benchmark_runner_single(tmp_path):
    from perteval.bench.result import EvalResult
    from perteval.bench.runner import BenchmarkRunner

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    adata = build_random_anndata(n_obs=200, perturbations=["pertA", "pertB"])
    adata.write_h5ad(data_dir / "testdata.h5ad")

    bench_dir = tmp_path / "benchmarks"
    bench_dir.mkdir()
    (bench_dir / "testbench.yaml").write_text(
        "dataset: testdata\nmetrics: [mse, pearson_delta]\n"
        "split:\n  method: random\n  seed: 42\n"
    )

    runner = BenchmarkRunner(
        benchmarks=["testbench"], models=["mean_control"],
        benchmarks_dir=str(bench_dir), data_dir=str(data_dir),
    )
    results = runner.run()
    assert "testbench" in results
    assert "mean_control" in results["testbench"]
    assert isinstance(results["testbench"]["mean_control"], EvalResult)


def test_benchmark_runner_multi_model(tmp_path):
    from perteval.bench.runner import BenchmarkRunner

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    adata = build_random_anndata(n_obs=200, perturbations=["pertA", "pertB"])
    adata.write_h5ad(data_dir / "testdata.h5ad")

    bench_dir = tmp_path / "benchmarks"
    bench_dir.mkdir()
    (bench_dir / "testbench.yaml").write_text("dataset: testdata\nmetrics: [mse]\n")

    runner = BenchmarkRunner(
        benchmarks=["testbench"], models=["mean_control", "mean_control"],
        benchmarks_dir=str(bench_dir), data_dir=str(data_dir),
    )
    results = runner.run()
    assert len(results["testbench"]) == 2
