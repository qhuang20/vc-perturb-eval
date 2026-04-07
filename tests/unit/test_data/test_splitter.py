import numpy as np
import pytest

from tests.conftest import build_random_anndata


def test_random_split_fractions():
    from perteval.data.splitter import Splitter

    adata = build_random_anndata(n_obs=1000)
    result = Splitter.split(adata, method="random", frac=(0.8, 0.1, 0.1), seed=42)
    assert set(result.keys()) == {"train", "val", "test"}
    total = sum(r.n_obs for r in result.values())
    assert total == 1000
    assert abs(result["train"].n_obs - 800) < 50


def test_random_split_deterministic():
    from perteval.data.splitter import Splitter

    adata = build_random_anndata(n_obs=200)
    r1 = Splitter.split(adata, method="random", seed=42)
    r2 = Splitter.split(adata, method="random", seed=42)
    np.testing.assert_array_equal(r1["train"].obs_names, r2["train"].obs_names)


def test_transfer_split_holdout():
    from perteval.data.splitter import Splitter

    adata = build_random_anndata(n_obs=500, perturbations=["pertA", "pertB", "pertC", "pertD"])
    result = Splitter.split(
        adata, method="transfer", holdout_key="perturbation", frac=(0.5, 0.25, 0.25), seed=42
    )
    train_perts = set(result["train"].obs["perturbation"].unique())
    test_perts = set(result["test"].obs["perturbation"].unique())
    non_control_test = test_perts - {"control"}
    non_control_train = train_perts - {"control"}
    assert len(non_control_test & non_control_train) == 0


def test_split_invalid_method_raises():
    from perteval.data.splitter import Splitter

    adata = build_random_anndata(n_obs=100)
    with pytest.raises(ValueError, match="Unknown split method"):
        Splitter.split(adata, method="unknown")
