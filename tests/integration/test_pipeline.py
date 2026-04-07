"""End-to-end integration test: data → model → evaluate → results."""

import numpy as np

from tests.conftest import build_random_anndata


def test_full_pipeline_with_mean_control(tmp_path):
    """Full pipeline: create data → split → train MeanControl → predict → evaluate."""
    from perteval.bench.evaluator import Evaluator
    from perteval.bench.result import EvalResult
    from perteval.data.splitter import Splitter
    from perteval.data.types import PerturbationData
    from perteval.models.baselines.mean_control import MeanControl

    rng = np.random.default_rng(42)
    adata = build_random_anndata(
        n_obs=500, n_vars=100, perturbations=["geneA", "geneB", "geneC", "geneD"], rng=rng
    )

    splits = Splitter.split(
        adata, method="transfer", holdout_key="perturbation", frac=(0.5, 0.25, 0.25), seed=42
    )
    assert "train" in splits
    assert "test" in splits

    model = MeanControl()
    model.train(splits["train"], perturbation_key="perturbation", control_key="control")

    test_adata = splits["test"]
    control_cells = test_adata[test_adata.obs["perturbation"] == "control"]
    test_perts = [p for p in test_adata.obs["perturbation"].unique() if p != "control"]
    assert len(test_perts) > 0

    predicted = model.predict(control_cells, perturbations=test_perts)
    assert predicted.n_obs > 0

    gt_mask = test_adata.obs["perturbation"].isin(test_perts)
    ground_truth = test_adata[gt_mask].copy()

    data = PerturbationData(predicted=predicted, ground_truth=ground_truth)
    evaluator = Evaluator()
    result = evaluator.evaluate(data, metrics=["pearson_delta", "mse", "edistance"])

    assert isinstance(result, EvalResult)
    assert result.per_perturbation.height > 0
    assert "pearson_delta" in result.per_perturbation.columns

    result.to_json(str(tmp_path / "result.json"))
    result.to_csv(str(tmp_path / "result"))
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "result_per_perturbation.csv").exists()


def test_benchmark_runner_end_to_end(tmp_path):
    """BenchmarkRunner: create data + YAML → run → compare."""
    from perteval.bench.compare import Compare
    from perteval.bench.runner import BenchmarkRunner

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    adata = build_random_anndata(n_obs=300, perturbations=["pertA", "pertB", "pertC"])
    adata.write_h5ad(data_dir / "testdata.h5ad")

    bench_dir = tmp_path / "benchmarks"
    bench_dir.mkdir()
    (bench_dir / "test_bench.yaml").write_text(
        "dataset: testdata\nmetrics: [pearson_delta, mse]\nsplit:\n  method: random\n  seed: 42\n"
    )

    runner = BenchmarkRunner(
        benchmarks=["test_bench"],
        models=["mean_control"],
        benchmarks_dir=str(bench_dir),
        data_dir=str(data_dir),
    )
    results = runner.run()

    comparison = Compare.from_results(results)
    summary = comparison.summary()
    assert summary.height == 1
    assert "pearson_delta" in summary.columns
