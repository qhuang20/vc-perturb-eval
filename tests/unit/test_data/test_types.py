import numpy as np
import pytest

from tests.conftest import build_random_anndata


def test_perturbation_data_creation():
    from perteval.data.types import PerturbationData

    rng = np.random.default_rng(42)
    gt = build_random_anndata(rng=rng)
    pred = gt.copy()
    pred.X = pred.X + 0.1
    data = PerturbationData(predicted=pred, ground_truth=gt)
    assert data.perturbation_key == "perturbation"
    assert data.control_key == "control"
    np.testing.assert_array_equal(data.gene_names, gt.var_names.values)


def test_perturbation_data_gene_mismatch_raises():
    from perteval.data.types import PerturbationData

    rng = np.random.default_rng(42)
    gt = build_random_anndata(n_vars=50, rng=rng)
    pred = build_random_anndata(n_vars=50, rng=np.random.default_rng(99))
    pred.var_names = [f"other_gene_{i}" for i in range(50)]
    with pytest.raises(ValueError, match="Gene names do not match"):
        PerturbationData(predicted=pred, ground_truth=gt)


def test_perturbation_data_shape_mismatch_raises():
    from perteval.data.types import PerturbationData

    rng = np.random.default_rng(42)
    gt = build_random_anndata(n_vars=50, rng=rng)
    pred = build_random_anndata(n_vars=30, rng=np.random.default_rng(99))
    with pytest.raises(ValueError, match="Gene count mismatch"):
        PerturbationData(predicted=pred, ground_truth=gt)


def test_perturbation_data_is_frozen():
    from perteval.data.types import PerturbationData

    rng = np.random.default_rng(42)
    gt = build_random_anndata(rng=rng)
    pred = gt.copy()
    data = PerturbationData(predicted=pred, ground_truth=gt)
    with pytest.raises(AttributeError):
        data.predicted = gt


def test_perturbation_data_perturbation_labels():
    from perteval.data.types import PerturbationData

    rng = np.random.default_rng(42)
    gt = build_random_anndata(perturbations=["geneX", "geneY"], rng=rng)
    pred = gt.copy()
    data = PerturbationData(predicted=pred, ground_truth=gt)
    assert set(data.perturbation_labels) == {"geneX", "geneY"}
